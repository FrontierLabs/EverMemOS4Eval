"""
Nanobot EnhancedMem Adapter - bridge EverMemOS4Eval evaluation to nanobot-memory EnhancedMem.

This adapter:
- Uses nanobot's EnhancedMem backend as the memory system
- Reuses nanobot's composed memory context (`EnhancedMemStore.get_memory_context(query)`)
- Generates answers using the same LLM + prompt as other adapters (e.g. answer_prompt_memos)
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.utils.config import load_yaml


@register_adapter("nanobot_enhancedmem")
class NanobotEnhancedMemAdapter(BaseAdapter):
    """Adapter that uses nanobot-memory's EnhancedMem as the memory backend."""

    def __init__(self, config: dict, output_dir: Path | None = None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self._nanobot_imported = False
        self._runner_cls = None
        self._provider_cls = None

        # Answer generation: use evaluation framework's LLM + prompt (same as online adapters)
        from memory_layer.llm.llm_provider import LLMProvider as EvalLLMProvider

        llm_cfg = config.get("llm", {})
        self._answer_llm = EvalLLMProvider(
            provider_type=llm_cfg.get("provider", "openai"),
            model=llm_cfg.get("model", "gpt-4o-mini"),
            api_key=llm_cfg.get("api_key", ""),
            base_url=llm_cfg.get(
                "base_url", os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
            ),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", 32768)),
        )
        eval_root = Path(__file__).resolve().parents[2]
        prompts_path = eval_root / "config" / "prompts.yaml"
        self._prompts = load_yaml(str(prompts_path))

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _ensure_nanobot_imported(self) -> None:
        """Add nanobot-memory repo to sys.path and import runner + CustomProvider only."""
        if self._nanobot_imported:
            return

        repo_path = (
            self.config.get("nanobot_memory_repo_path")
            or os.getenv("NANOBOT_MEMORY_REPO")
        )
        if not repo_path:
            raise ValueError(
                "nanobot_memory_repo_path is not configured. "
                "Set it in the system YAML or via NANOBOT_MEMORY_REPO env."
            )

        repo_path_str = str(Path(repo_path).resolve())
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)

        # Import only runner and CustomProvider (avoids loading litellm, oauth_cli_kit, etc.)
        try:
            runner_mod = importlib.import_module(
                "nanobot.agent.enhancedmem.runner"
            )
            custom_provider_mod = importlib.import_module(
                "nanobot.providers.custom_provider"
            )
        except ImportError as e:
            raise ImportError(
                f"Failed to import nanobot-memory modules from {repo_path_str}: {e}"
            ) from e

        self._runner_cls = getattr(runner_mod, "EnhancedMemRunner")
        self._provider_cls = getattr(custom_provider_mod, "CustomProvider")
        self._runner_message_cls = getattr(runner_mod, "RunnerMessage")
        self._nanobot_imported = True

    def _make_runner(self, workspace: Path) -> Any:
        """Instantiate EnhancedMemRunner for a given workspace."""
        self._ensure_nanobot_imported()

        llm_cfg = self.config.get("llm", {})
        model = llm_cfg.get("model", "openai/gpt-4.1-mini")
        api_key = llm_cfg.get("api_key") or os.getenv("LLM_API_KEY", "")
        base_url = llm_cfg.get(
            "base_url", os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        )

        # CustomProvider(api_key, api_base, default_model) for OpenAI-compatible endpoints
        provider = self._provider_cls(
            api_key=api_key or "no-key",
            api_base=base_url,
            default_model=model,
        )

        # Build EnhancedMem config from YAML (retrieve_method, bm25_*, etc.)
        em_config = SimpleNamespace(
            retrieve_method=self.config.get("retrieve_method", "lightest"),
            bm25_min_score_ratio=self.config.get("bm25_min_score_ratio", 0.1),
            bm25_min_score_absolute=self.config.get("bm25_min_score_absolute", 0.01),
        )
        runner = self._runner_cls(
            workspace=workspace,
            provider=provider,
            model=model,
            memory_window=self.config.get("memory_window", 100),
            memory_consolidate_interval=self.config.get(
                "memory_consolidate_interval", None
            ),
            memory_consolidate_after_turn=self.config.get(
                "memory_consolidate_after_turn", 0
            ),
            config=em_config,
        )
        return runner

    # -------------------------------------------------------------------------
    # BaseAdapter implementation
    # -------------------------------------------------------------------------
    async def add(
        self,
        conversations: List[Conversation],
        output_dir: Path = None,
        checkpoint_manager=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add stage: ingest conversations into EnhancedMem workspaces."""
        output_dir = Path(output_dir) if output_dir else self.output_dir
        workspaces: Dict[str, str] = {}

        for conv in conversations:
            conv_id = conv.conversation_id
            ws = output_dir / "workspaces" / conv_id
            ws.mkdir(parents=True, exist_ok=True)

            runner = self._make_runner(ws)

            # Convert Conversation -> RunnerMessage list
            messages = []
            for msg in conv.messages:
                role = "user"
                if msg.speaker_name.lower() in ("assistant", "system", "bot"):
                    role = "assistant"
                session_key = msg.metadata.get("session")
                messages.append(
                    self._runner_message_cls(
                        role=role,
                        content=msg.content,
                        timestamp=msg.timestamp,
                        session_key=session_key,
                    )
                )

            await runner.ingest(messages)
            await runner.finalize()
            workspaces[conv_id] = str(ws)

        return {"workspaces": workspaces}

    async def search(
        self,
        query: str,
        conversation_id: str,
        index: Any,
        **kwargs,
    ) -> SearchResult:
        """Search stage: get composed memory context from EnhancedMem."""
        workspaces: Dict[str, str] = index.get("workspaces", {})
        if conversation_id not in workspaces:
            raise ValueError(
                f"Workspace for conversation_id={conversation_id} not found in index"
            )

        ws_path = Path(workspaces[conversation_id])
        runner = self._make_runner(ws_path)

        ctx = runner.get_memory_context(query)
        # Keep Nanobot-style wrapper
        formatted_context = f"# Memory\n\n{ctx}" if ctx else ""

        metadata: Dict[str, Any] = {
            "system": "nanobot_enhancedmem",
            "formatted_context": formatted_context,
        }
        results = []
        if formatted_context:
            results.append(
                {
                    "content": formatted_context,
                    "score": 1.0,
                    "metadata": {"type": "nanobot_memory_context"},
                }
            )

        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata=metadata,
        )

    async def answer(self, query: str, context: str, **kwargs) -> str:
        """Generate answer from context using evaluation framework's LLM + prompt."""
        prompt_template = self._prompts.get("online_api", {}).get("default", {}).get(
            "answer_prompt_memos", ""
        )
        if not prompt_template:
            return ""
        prompt = prompt_template.format(context=context, question=query)
        max_retries = self.config.get("answer", {}).get("max_retries", 3)
        for attempt in range(max_retries):
            try:
                result = await self._answer_llm.generate(
                    prompt=prompt, temperature=0, max_tokens=256
                )
                if not result:
                    continue
                result = result.strip()
                if "FINAL ANSWER:" in result:
                    parts = result.split("FINAL ANSWER:")
                    result = parts[-1].strip() if len(parts) > 1 else result
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error: {e}"
                continue
        return ""

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "name": "nanobot_enhancedmem",
            "version": "0.1",
            "description": "Adapter using nanobot-memory EnhancedMem as backend",
            "adapter": "nanobot_enhancedmem_adapter",
        }

