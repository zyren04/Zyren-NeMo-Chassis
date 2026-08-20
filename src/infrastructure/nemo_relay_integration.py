"""
NeMo Relay Integration Module
Centralizes scope management, middleware registration, and observability configuration.

This module provides a unified interface for integrating NeMo Relay's managed execution
APIs (scopes, tool/LLM lifecycle, guardrails, intercepts) into the infrastructure chassis.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

# NeMo Relay imports with graceful fallback
try:
    import nemo_relay
    from nemo_relay import LLMRequest, ScopeHandle, ScopeType, ToolExecutionInterceptOutcome
    from nemo_relay.codecs import OpenAIChatCodec
    from nemo_relay.integrations.langchain import NemoRelayMiddleware
    from nemo_relay.integrations.langgraph import NemoRelayCallbackHandler
    from nemo_relay.typed import BestEffortAnyCodec

    NEMO_RELAY_AVAILABLE = True
except ImportError:
    nemo_relay = None  # type: ignore
    ScopeType = None  # type: ignore
    ScopeHandle = None  # type: ignore
    LLMRequest = None  # type: ignore
    ToolExecutionInterceptOutcome = None  # type: ignore
    OpenAIChatCodec = None  # type: ignore
    BestEffortAnyCodec = None  # type: ignore
    NemoRelayCallbackHandler = None  # type: ignore
    NemoRelayMiddleware = None  # type: ignore
    NEMO_RELAY_AVAILABLE = False

# Observability imports (optional)
try:
    from ..observability.config import ObservabilityConfig
    from ..observability.plugin import ObservabilityPlugin

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    ObservabilityConfig = None  # type: ignore
    ObservabilityPlugin = None  # type: ignore
    OBSERVABILITY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Context variable for current workflow scope handle
_current_workflow_scope: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "current_workflow_scope", default=None
)

# NeMo Relay scope stack isolation primitives
_scope_stack_var: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "nemo_relay_scope_stack", default=None
)
_propagation_parent_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "nemo_relay_propagation_parent", default=None
)
_propagation_root_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "nemo_relay_propagation_root", default=None
)


class _FallbackScopeStack:
    """Fallback scope stack implementation for when NeMo Relay is not available.

    Provides isolation using Python's contextvars.
    """

    def __init__(self, parent_uuid: str | None = None, root_uuid: str | None = None):
        self.uuid = uuid.uuid4().hex[:8]
        self.parent_uuid = parent_uuid
        self.root_uuid = root_uuid or self.uuid
        self._scopes: list[Any] = []

    def push(
        self,
        name: str,
        scope_type: Any = None,
        data: dict | None = None,
        metadata: dict | None = None,
    ) -> _FallbackScopeHandle:
        handle = _FallbackScopeHandle(
            uuid=self.uuid,
            parent_uuid=self.parent_uuid,
            name=name,
            scope_type=scope_type,
            data=data or {},
            metadata=metadata or {},
        )
        self._scopes.append(handle)
        return handle

    def pop(self, handle: _FallbackScopeHandle, output: dict | None = None) -> None:
        if handle in self._scopes:
            self._scopes.remove(handle)

    def get_current(self) -> _FallbackScopeHandle | None:
        return self._scopes[-1] if self._scopes else None


class _FallbackScopeHandle:
    """Fallback scope handle for when NeMo Relay is not available."""

    def __init__(
        self,
        uuid: str,
        parent_uuid: str | None,
        name: str,
        scope_type: Any = None,
        data: dict | None = None,
        metadata: dict | None = None,
    ):
        self.uuid = uuid
        self.parent_uuid = parent_uuid
        self.name = name
        self.scope_type = scope_type
        self.data = data or {}
        self.metadata = metadata or {}


@dataclass
class NeMoRelayConfig:
    """Configuration for NeMo Relay integration."""

    enabled: bool = True
    scope_name_prefix: str = "workflow"
    register_langgraph_callback: bool = True
    register_langchain_middleware: bool = True
    default_codec: str = "best_effort"  # "best_effort" | "openai_chat"
    global_guardrails: dict[str, Any] = field(default_factory=dict)
    global_intercepts: dict[str, Any] = field(default_factory=dict)

    # NEW: Observability configuration
    observability: ObservabilityConfig | None = None

    @classmethod
    def from_toml(cls, toml_path: str) -> NeMoRelayConfig:
        """Create config from TOML file (matches NeMo Relay plugin schema)."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open(toml_path, "rb") as f:
            toml_dict = tomllib.load(f)

        # Parse observability config if present
        observability = None
        components = toml_dict.get("components", [])
        for comp in components:
            if comp.get("kind") == "observability" and comp.get("enabled", True):
                observability = ObservabilityConfig.from_toml({"components": [comp]})
                break

        return cls(
            enabled=toml_dict.get("enabled", True),
            scope_name_prefix=toml_dict.get("scope_name_prefix", "workflow"),
            register_langgraph_callback=toml_dict.get("register_langgraph_callback", True),
            register_langchain_middleware=toml_dict.get("register_langchain_middleware", True),
            default_codec=toml_dict.get("default_codec", "best_effort"),
            global_guardrails=toml_dict.get("global_guardrails", {}),
            global_intercepts=toml_dict.get("global_intercepts", {}),
            observability=observability,
        )


class NeMoRelayIntegration:
    """Singleton manager for NeMo Relay integration."""

    _instance: NeMoRelayIntegration | None = None
    _initialized: bool = False

    def __new__(cls, config: NeMoRelayConfig | None = None) -> NeMoRelayIntegration:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: NeMoRelayConfig | None = None):
        # Always update config if provided, regardless of singleton state
        # This ensures the singleton instance always has the latest config
        if config is not None:
            self.config = config
            self._callback_handler = None
            self._middleware = None
            self._codec = None
            # Don't reset _initialized - just ensure config is set
            self._initialized = True
            self._initialize()
            return

        # If no config provided, use existing or create default
        if self._initialized:
            return
        self.config = NeMoRelayConfig()
        self._callback_handler = None
        self._initialized = True
        self._middleware = None
        self._codec = None
        self._initialize()

    def _get_codec(self):
        if not NEMO_RELAY_AVAILABLE:
            return None
        if self.config.default_codec == "openai_chat" and OpenAIChatCodec:
            return OpenAIChatCodec()
        return BestEffortAnyCodec()

    def _initialize(self):
        if not NEMO_RELAY_AVAILABLE:
            logger.warning("NeMo Relay not available - running in fallback mode")
            self._initialized = True
            return

        self._codec = self._get_codec()

        # Register LangGraph callback handler
        if self.config.register_langgraph_callback and NemoRelayCallbackHandler:
            self._callback_handler = NemoRelayCallbackHandler()

        # Register LangChain middleware
        if self.config.register_langchain_middleware and NemoRelayMiddleware:
            self._middleware = NemoRelayMiddleware()

        # Apply global guardrails
        for name, guardrail in self.config.global_guardrails.items():
            if hasattr(nemo_relay, "guardrails"):
                nemo_relay.guardrails.register(name, guardrail)

        # Apply global intercepts
        for name, intercept in self.config.global_intercepts.items():
            if hasattr(nemo_relay, "intercepts"):
                nemo_relay.intercepts.register(name, intercept)

        self._initialized = True
        logger.info("NeMo Relay integration initialized")

    # =========================================================================
    # Scope Management
    # =========================================================================

    def create_workflow_scope(self, execution_id: str, entry_point: str) -> Any:
        """Create a new workflow scope."""
        if not NEMO_RELAY_AVAILABLE or not nemo_relay:
            # Fallback: create a mock scope handle
            stack = self.get_scope_stack()
            if stack is None:
                stack = _FallbackScopeStack()
                _scope_stack_var.set(stack)
            return stack.push(
                name=f"{self.config.scope_name_prefix}-{execution_id[:8]}",
                scope_type="Agent",
                data={"execution_id": execution_id, "entry_point": entry_point},
                metadata={"source": "workflow_engine"},
            )
        return nemo_relay.scope.push(
            name=f"{self.config.scope_name_prefix}-{execution_id[:8]}",
            scope_type=ScopeType.Agent,
            data={"execution_id": execution_id, "entry_point": entry_point},
            metadata={"source": "workflow_engine"},
        )

    def close_workflow_scope(self, handle: Any, output: dict | None = None) -> None:
        """Close a workflow scope."""
        if not NEMO_RELAY_AVAILABLE or not nemo_relay or not handle:
            # Fallback: just remove from stack if it's a fallback handle
            if hasattr(handle, "uuid") and hasattr(handle, "parent_uuid"):
                stack = _scope_stack_var.get(None)
                if stack and hasattr(stack, "pop"):
                    stack.pop(handle)
            return
        nemo_relay.scope.pop(handle, output=output)

    @asynccontextmanager
    async def workflow_scope(self, execution_id: str, entry_point: str):
        """Context manager for workflow scope with FULL ISOLATION.

        Creates isolated ScopeStack for this workflow execution.
        Child tasks spawned within must use fork_asyncio_context() for isolation.
        """
        # Ensure we have an isolated scope stack for this workflow
        self.get_scope_stack()

        # Create workflow scope as child of current stack top (or root)
        handle = self.create_workflow_scope(execution_id, entry_point)
        token = _current_workflow_scope.set(handle)

        # Update propagation parent to this workflow's scope for child tasks
        parent_token = None
        if handle and hasattr(handle, "uuid"):
            parent_token = _propagation_parent_var.set(str(handle.uuid))

        try:
            yield handle
        finally:
            self.close_workflow_scope(handle)
            _current_workflow_scope.reset(token)
            if parent_token:
                _propagation_parent_var.reset(parent_token)

    def get_current_scope(self) -> Any:
        """Get current workflow scope handle."""
        return _current_workflow_scope.get()

    # =========================================================================
    # NeMo Relay Scope Stack Isolation Primitives
    # =========================================================================

    def get_scope_stack(self) -> Any:
        """Return current task's active NeMo Relay ScopeStack.

        Lazily creates and synchronizes stack if not present.
        This is the PRIMARY entry point for all NeMo Relay operations.
        """
        if not NEMO_RELAY_AVAILABLE or not nemo_relay:
            # Fallback: create isolated scope stack using contextvars
            stack = _scope_stack_var.get(None)
            if stack is None:
                stack = _FallbackScopeStack()
                _scope_stack_var.set(stack)
            return stack

        stack = _scope_stack_var.get(None)
        if stack is None:
            stack = nemo_relay.create_scope_stack()
            _scope_stack_var.set(stack)
        # Critical: sync Python ContextVar -> native thread-local
        nemo_relay._sync_thread_scope_stack(stack)
        return stack

    def create_isolated_scope_stack(self) -> Any:
        """Create a new isolated ScopeStack (not attached to current context).

        Use for: test fixtures, manual propagation, framework boundaries.
        """
        if not NEMO_RELAY_AVAILABLE or not nemo_relay:
            return _FallbackScopeStack()
        return nemo_relay.create_scope_stack()

    def fork_asyncio_context(self) -> contextvars.Context:
        """Create child asyncio context with isolated ScopeStack preserving ancestry.

        Use when spawning concurrent child tasks that need independent scope stacks
        but correct causal parentage (e.g., parallel agent execution, fan-out).

        Returns:
            contextvars.Context: Pass to asyncio.create_task(..., context=...)
        """
        if not NEMO_RELAY_AVAILABLE or not nemo_relay:
            # Fallback: create child context with new fallback stack
            parent_stack = _scope_stack_var.get(None)
            parent_uuid = (
                parent_stack.uuid if parent_stack and hasattr(parent_stack, "uuid") else None
            )
            root_uuid = (
                parent_stack.root_uuid
                if parent_stack and hasattr(parent_stack, "root_uuid")
                else None
            )

            child_stack = _FallbackScopeStack(parent_uuid=parent_uuid, root_uuid=root_uuid)
            child_context = contextvars.copy_context()
            child_context.run(_scope_stack_var.set, child_stack)
            if parent_uuid:
                child_context.run(_propagation_parent_var.set, parent_uuid)
            if root_uuid:
                child_context.run(_propagation_root_var.set, root_uuid)
            return child_context

        # Capture current propagation context (parent_uuid, root_uuid)
        propagation = nemo_relay.capture_propagation_context()
        # Create new isolated stack seeded from propagation context
        child_stack = nemo_relay.create_scope_stack_from_propagation(propagation)
        # Copy current context and install child stack
        child_context = contextvars.copy_context()
        child_context.run(_scope_stack_var.set, child_stack)
        # Also propagate parent/root vars for distributed tracing continuity
        if propagation.parent_uuid:
            child_context.run(_propagation_parent_var.set, propagation.parent_uuid)
        if propagation.root_uuid:
            child_context.run(_propagation_root_var.set, propagation.root_uuid)
        return child_context

    @asynccontextmanager
    async def use_scope_stack(self, stack: Any) -> AsyncIterator[Any]:
        """Temporarily install a ScopeStack in the current context.

        Restores previous stack on exit. Use for explicit stack management
        (e.g., worker threads, test fixtures, framework integration).
        """
        if not NEMO_RELAY_AVAILABLE or not nemo_relay or not stack:
            yield None
            return

        current_stack = _scope_stack_var.get(None)
        if current_stack is not None:
            nemo_relay._sync_thread_scope_stack(current_stack)

        previous_native_stack = nemo_relay._capture_thread_scope_stack()
        token = _scope_stack_var.set(stack)
        nemo_relay._sync_thread_scope_stack(stack)

        root_token = None
        try:
            # Capture root UUID for propagation context
            try:
                propagation_ctx = nemo_relay.capture_propagation_context()
                root_uuid = propagation_ctx.root_uuid
            except (RuntimeError, AttributeError):
                root_uuid = None
            root_token = _propagation_root_var.set(root_uuid)

            yield stack
        finally:
            if root_token is not None:
                _propagation_root_var.reset(root_token)
            _scope_stack_var.reset(token)
            nemo_relay._restore_thread_scope_stack(previous_native_stack)

    def propagate_scope_to_thread(self) -> Any:
        """Capture active ScopeStack for use in another thread.

        Returns stack reference to pass to set_thread_scope_stack() in worker.
        Does NOT clone — worker shares same logical trace.
        """
        if not NEMO_RELAY_AVAILABLE or not nemo_relay:
            # Fallback: return the current fallback stack
            stack = _scope_stack_var.get(None)
            return stack
        return nemo_relay.propagate_scope_to_thread()

    def set_thread_scope_stack(self, stack: Any) -> None:
        """Install ScopeStack in current thread's native runtime.

        Call at START of worker thread function.
        """
        if not NEMO_RELAY_AVAILABLE or not nemo_relay or not stack:
            # Fallback: install the stack in the current thread's contextvar
            if stack is not None:
                _scope_stack_var.set(stack)
            return
        nemo_relay.set_thread_scope_stack(stack)

    def capture_propagation_context(self) -> Any:
        """Capture current causal context for distributed tracing export.

        Returns PropagationContext with parent_uuid and root_uuid.
        """
        if not NEMO_RELAY_AVAILABLE or not nemo_relay:
            return None
        return nemo_relay.capture_propagation_context()

    def create_scope_stack_from_propagation(self, context: Any) -> Any:
        """Create isolated ScopeStack from received PropagationContext.

        Use when receiving trace context from upstream service.
        """
        if not NEMO_RELAY_AVAILABLE or not nemo_relay or not context:
            return None
        return nemo_relay.create_scope_stack_from_propagation(context)

    # =========================================================================
    # Managed execution helpers
    # =========================================================================

    def _ensure_stack_synced(self, handle: Any | None = None) -> Any:
        """Ensure scope stack is synced and return effective handle."""
        if not NEMO_RELAY_AVAILABLE or not nemo_relay:
            return handle
        # This calls ensure_scope_stack() internally via get_scope_stack()
        stack = self.get_scope_stack()
        return handle or stack  # nemo_relay APIs accept stack as handle fallback

    async def execute_tool(
        self, name: str, args: dict, func: Callable, handle: Any | None = None, **kwargs
    ) -> Any:
        """Execute a tool through NeMo Relay managed pipeline."""
        if not NEMO_RELAY_AVAILABLE or not nemo_relay:
            # Fallback: execute directly
            return await func(args)

        effective_handle = self._ensure_stack_synced(handle)
        return await nemo_relay.tools.execute(
            name=name,
            args=args,
            func=func,
            handle=effective_handle,
            codec=self._codec,
            result_codec=self._codec,
            **kwargs,
        )

    async def execute_llm(
        self,
        name: str,
        request: Any,
        func: Callable,
        handle: Any | None = None,
        model_name: str | None = None,
        **kwargs,
    ) -> dict:
        """Execute an LLM call through NeMo Relay managed pipeline."""
        if not NEMO_RELAY_AVAILABLE or not nemo_relay:
            # Fallback: execute directly
            return await func(request)

        effective_handle = self._ensure_stack_synced(handle)
        return await nemo_relay.llm.execute(
            name=name,
            request=request,
            func=func,
            handle=effective_handle,
            model_name=model_name,
            codec=self._codec,
            response_codec=self._codec,
            **kwargs,
        )

    def get_callback_handler(self) -> Any | None:
        return self._callback_handler

    def get_middleware(self) -> Any | None:
        return self._middleware

    # =========================================================================
    # Utility: Spawn isolated child tasks / thread pool execution
    # =========================================================================

    async def spawn_isolated_task(self, coro, *args, **kwargs):
        """Spawn a child task with isolated ScopeStack preserving ancestry.

        Usage:
            await integration.spawn_isolated_task(worker_coro, arg1, arg2)
        """
        if not NEMO_RELAY_AVAILABLE or not nemo_relay:
            return await coro(*args, **kwargs)

        child_context = self.fork_asyncio_context()
        if sys.version_info >= (3, 11):
            task = asyncio.create_task(coro(*args, **kwargs), context=child_context)
        else:
            task = child_context.run(asyncio.create_task, coro(*args, **kwargs))
        return await task

    async def run_in_thread_pool(self, executor, func, *args, **kwargs):
        """Run function in ThreadPoolExecutor with propagated scope stack.

        Usage:
            result = await integration.run_in_thread_pool(executor, blocking_func, arg)
        """
        if not NEMO_RELAY_AVAILABLE or not nemo_relay:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

        stack = self.propagate_scope_to_thread()
        loop = asyncio.get_event_loop()

        def wrapped():
            nemo_relay.set_thread_scope_stack(stack)
            return func(*args, **kwargs)

        return await loop.run_in_executor(executor, wrapped)

    # =========================================================================
    # Observability Integration
    # =========================================================================

    def _initialize(self):
        if not NEMO_RELAY_AVAILABLE:
            logger.warning("NeMo Relay not available - running in fallback mode")
            return

        self._codec = self._get_codec()

        # Register LangGraph callback handler
        if self.config.register_langgraph_callback and NemoRelayCallbackHandler:
            self._callback_handler = NemoRelayCallbackHandler()

        # Register LangChain middleware
        if self.config.register_langchain_middleware and NemoRelayMiddleware:
            self._middleware = NemoRelayMiddleware()

        # Apply global guardrails
        for name, guardrail in self.config.global_guardrails.items():
            if hasattr(nemo_relay, "guardrails"):
                nemo_relay.guardrails.register(name, guardrail)

        # Apply global intercepts
        for name, intercept in self.config.global_intercepts.items():
            if hasattr(nemo_relay, "intercepts"):
                nemo_relay.intercepts.register(name, intercept)

        # NEW: Initialize observability if configured
        if self.config.observability and self.config.observability.enabled:
            self._init_observability()

        self._initialized = True
        logger.info("NeMo Relay integration initialized")

    def _init_observability(self):
        """Initialize observability plugin for current execution."""
        if not OBSERVABILITY_AVAILABLE:
            logger.warning("Observability module not available")
            return

        from ..state.event_store import get_event_store

        event_store = get_event_store()
        execution_id = f"exec-{uuid.uuid4().hex[:8]}"

        self._observability_plugin = ObservabilityPlugin(
            config=self.config.observability,
            event_store=event_store,
            execution_id=execution_id,
        )

    async def activate_observability(self, execution_id: str):
        """Activate observability for a specific execution."""
        from ..state.event_store import get_event_store

        event_store = await get_event_store()
        obs_config = self.config.observability

        # If observability is disabled, return None
        if obs_config is None or not obs_config.enabled:
            return None

        plugin = ObservabilityPlugin(
            config=obs_config,
            event_store=event_store,
            execution_id=execution_id,
        )
        await plugin.activate()
        return plugin

    async def deactivate_observability(self, plugin):
        """Deactivate and flush observability."""
        if plugin:
            await plugin.deactivate()

    def get_observability_plugin(self):
        return getattr(self, "_observability_plugin", None)


# Global instance getter
def get_nemo_relay_integration(config: NeMoRelayConfig | None = None) -> NeMoRelayIntegration:
    return NeMoRelayIntegration(config)


def set_nemo_relay_integration(integration: NeMoRelayIntegration) -> None:
    NeMoRelayIntegration._instance = integration
    NeMoRelayIntegration._initialized = False
    integration._initialize()


__all__ = [
    "NeMoRelayConfig",
    "NeMoRelayIntegration",
    "get_nemo_relay_integration",
    "set_nemo_relay_integration",
    "NEMO_RELAY_AVAILABLE",
    # NEW: Observability exports
    "ObservabilityConfig",
    "ObservabilityPlugin",
    "ATOFConfig",
    "ATIFConfig",
    "OpenTelemetryConfig",
]
