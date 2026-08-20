"""
Centralized Model Router — Switchyard Router
Zero-code configuration-driven model routing for agent nodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..infrastructure.nim_client import NIMClient
from ..infrastructure.rate_limiting import RateLimitMode

logger = logging.getLogger(__name__)


@dataclass
class ModelTarget:
    """Single model target configuration from YAML."""

    name: str
    model: str
    max_rpm: int = 60
    rate_limit_mode: str = "token_bucket"
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.rate_limit_mode not in (RateLimitMode.TOKEN_BUCKET, RateLimitMode.STRICT):
            raise ValueError(
                f"Invalid rate_limit_mode: {self.rate_limit_mode}. "
                f"Must be 'token_bucket' or 'strict'"
            )
        if self.max_rpm <= 0:
            raise ValueError("max_rpm must be positive")


class SwitchyardRouter:
    """
    Centralized model router with cached NIMClient pool.

    Loads model configurations from YAML, maintains a pool of NIMClient
    instances, and provides autonomous routing based on task type.
    """

    def __init__(self, config_path: str = "config/models.yaml") -> None:
        self.config_path = Path(config_path)
        self._targets: dict[str, ModelTarget] = {}
        self._clients: dict[str, NIMClient] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load and parse model configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Model config not found: {self.config_path}")

        with open(self.config_path) as f:
            data = yaml.safe_load(f)

        if not data or "models" not in data:
            raise ValueError("Invalid config: missing 'models' key")

        self._targets.clear()
        self._clients.clear()

        for model_data in data["models"]:
            target = ModelTarget(**model_data)
            if target.enabled:
                self._targets[target.name] = target
                logger.info(f"Loaded model target: {target.name} ({target.model})")
            else:
                logger.info(f"Skipped disabled model target: {target.name}")

        if "default" not in self._targets:
            # Auto-create a default if missing
            first_enabled = next(iter(self._targets.values()), None)
            if first_enabled:
                logger.warning(
                    "No 'default' target found; using first enabled: %s",
                    first_enabled.name,
                )
            else:
                raise ValueError("No enabled model targets in configuration")

    def get_client(self, target_name: str) -> NIMClient:
        """
        Get or create cached NIMClient for a target.

        Args:
            target_name: Name of the model target (e.g., 'reasoning', 'coding')

        Returns:
            NIMClient instance for the target

        Raises:
            KeyError: If target not found or disabled
        """
        if target_name not in self._targets:
            # Fallback to default
            if target_name != "default" and "default" in self._targets:
                logger.warning(
                    "Target '%s' not found, falling back to 'default'", target_name
                )
                target_name = "default"
            else:
                raise KeyError(f"Model target not found: {target_name}")

        if target_name not in self._clients:
            target = self._targets[target_name]
            self._clients[target_name] = NIMClient(
                model=target.model,
                max_rpm=target.max_rpm,
                rate_limit_mode=target.rate_limit_mode,
            )
            logger.debug(f"Created NIMClient for target: {target_name}")

        return self._clients[target_name]

    async def route(
        self,
        prompt: str,
        task_type: str = "auto",
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Route a prompt to the appropriate model and return the response.

        Args:
            prompt: User prompt text
            task_type: Routing strategy - 'auto', 'reasoning', 'coding', 'fast', or target name
            system_prompt: Optional system instructions
            **kwargs: Additional arguments passed to NIMClient.ainvoke_simple

        Returns:
            Generated text response from the routed model
        """
        target_name = self._resolve_target(task_type)
        client = self.get_client(target_name)

        logger.debug(f"Routing to target: {target_name} (task_type={task_type})")
        return await client.ainvoke_simple(prompt, system_prompt=system_prompt, **kwargs)

    def _resolve_target(self, task_type: str) -> str:
        """Resolve task_type to a target name."""
        if task_type == "auto":
            return self._auto_route(task_type)
        # Direct target name or known task type
        return task_type

    def _auto_route(self, prompt: str) -> str:
        """Simple keyword-based routing heuristic."""
        prompt_lower = prompt.lower()

        # Coding keywords
        if any(
            kw in prompt_lower
            for kw in [
                "code",
                "function",
                "class",
                "debug",
                "api",
                "sql",
                "implement",
                "refactor",
                "script",
                "program",
                "algorithm",
                "python",
                "javascript",
                "typescript",
            ]
        ):
            return "coding" if "coding" in self._targets else "default"

        # Reasoning/complex analysis keywords
        if any(
            kw in prompt_lower
            for kw in [
                "reason",
                "analyze",
                "prove",
                "derive",
                "complex",
                "theorem",
                "logic",
                "step by step",
                "think",
                "explain why",
                "tradeoff",
                "architecture",
                "design",
            ]
        ):
            return "reasoning" if "reasoning" in self._targets else "default"

        # Fast/simple keywords
        if any(
            kw in prompt_lower
            for kw in ["summarize", "brief", "tldr", "short", "one sentence", "quick"]
        ):
            return "fast" if "fast" in self._targets else "default"

        return "default"

    def list_targets(self) -> list[str]:
        """List all enabled target names."""
        return list(self._targets.keys())

    def get_target(self, target_name: str) -> ModelTarget | None:
        """Get target configuration by name."""
        return self._targets.get(target_name)

    def reload(self) -> None:
        """Reload configuration from disk (picks up added/removed models)."""
        self._load_config()


# Global singleton instance
_router: SwitchyardRouter | None = None


def get_router(config_path: str = "config/models.yaml") -> SwitchyardRouter:
    """Get or create the global router instance."""
    global _router
    if _router is None:
        _router = SwitchyardRouter(config_path)
    return _router


def set_router(router: SwitchyardRouter) -> None:
    """Set the global router instance (useful for testing)."""
    global _router
    _router = router


__all__ = [
    "ModelTarget",
    "SwitchyardRouter",
    "get_router",
    "set_router",
]
