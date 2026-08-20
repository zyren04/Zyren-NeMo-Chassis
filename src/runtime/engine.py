"""
Runtime Workflow Orchestrator Engine
Generic compiler and runner using langgraph.graph.StateGraph wrapped with nvidia_nat.workflows.langgraph.LangGraphWorkflow.
"""

from __future__ import annotations

import asyncio
import contextvars
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel

from ..infrastructure.nemo_relay_integration import get_nemo_relay_integration
from ..state.event_store import EventRecord, EventStore, NodeMetricRecord, get_event_store
from ..state.state_schema import BaseState, StateDict, from_state_dict, to_state_dict

# Cross-cutting context variables for nested workflow configuration propagation
# These allow rate limits, auth tokens, and feature flags to flow through
# subgraphs without explicit threading through state.
ctx_rate_limit: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "ctx_rate_limit", default=None
)
ctx_auth_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ctx_auth_token", default=None
)
ctx_feature_flags: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "ctx_feature_flags", default=None
)

# Type alias for node function
NodeFunc = Callable[[BaseState], Awaitable[BaseState]]
ConditionFunc = Callable[[BaseState], bool]


@dataclass
class NodeRegistration:
    """Registration info for a node."""

    name: str
    func: NodeFunc
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class EdgeRegistration:
    """Registration info for an edge."""

    from_node: str
    to_node: str
    condition: ConditionFunc | None = None
    description: str = ""


class CompiledGraph:
    """Wrapper around compiled LangGraph with execution utilities."""

    def __init__(
        self,
        graph: Pregel[BaseState, Any, Any],
        engine: WorkflowEngine,
        entry_point: str,
    ):
        self._graph = graph
        self._engine = engine
        self._entry_point = entry_point

    @property
    def graph(self) -> Pregel[BaseState, Any, Any]:
        return self._graph

    @property
    def entry_point(self) -> str:
        return self._entry_point

    async def ainvoke(
        self,
        initial_state: BaseState,
        config: RunnableConfig | None = None,
    ) -> BaseState:
        """Execute workflow with isolated NeMo Relay scope stack AND observability."""
        nemo_integration = get_nemo_relay_integration()

        # Create isolated scope stack for THIS execution
        stack = nemo_integration.create_isolated_scope_stack()

        # Capture propagation context from parent (if any) for ancestry
        parent_context = nemo_integration.capture_propagation_context()
        if parent_context and stack:
            # Re-seed isolated stack with parent ancestry
            stack = nemo_integration.create_scope_stack_from_propagation(parent_context)

        # NEW: Activate observability for this execution
        observability_plugin = None
        if nemo_integration.config.observability and nemo_integration.config.observability.enabled:
            observability_plugin = await nemo_integration.activate_observability(
                initial_state.execution_id
            )

        try:
            # Run with isolated stack installed
            async with nemo_integration.use_scope_stack(stack):
                # Create workflow scope within this isolated stack
                execution_id = initial_state.execution_id
                entry_point = self._entry_point or "unknown"

                async with nemo_integration.workflow_scope(execution_id, entry_point):
                    state_dict = to_state_dict(initial_state)
                    result_dict: dict[str, Any] = await self._graph.ainvoke(
                        state_dict, config=config
                    )
                    return from_state_dict(result_dict)  # type: ignore[arg-type]
        finally:
            # NEW: Deactivate observability (flushes all exporters)
            if observability_plugin:
                await nemo_integration.deactivate_observability(observability_plugin)

    async def astream(
        self,
        initial_state: BaseState,
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[tuple[str, BaseState]]:
        """Stream workflow execution with isolated NeMo Relay scope stack AND observability."""
        nemo_integration = get_nemo_relay_integration()

        stack = nemo_integration.create_isolated_scope_stack()
        parent_context = nemo_integration.capture_propagation_context()
        if parent_context and stack:
            stack = nemo_integration.create_scope_stack_from_propagation(parent_context)

        # NEW: Activate observability for this execution
        observability_plugin = None
        if nemo_integration.config.observability and nemo_integration.config.observability.enabled:
            observability_plugin = await nemo_integration.activate_observability(
                initial_state.execution_id
            )

        try:
            async with nemo_integration.use_scope_stack(stack):
                execution_id = initial_state.execution_id
                entry_point = self._entry_point or "unknown"

                async with nemo_integration.workflow_scope(execution_id, entry_point):
                    state_dict = to_state_dict(initial_state)
                    async for chunk in self._graph.astream(state_dict, config=config):
                        for node_name, node_state in chunk.items():
                            yield node_name, from_state_dict(node_state)
        finally:
            # NEW: Deactivate observability (flushes all exporters)
            if observability_plugin:
                await nemo_integration.deactivate_observability(observability_plugin)

    async def abatch(
        self,
        initial_states: list[BaseState],
        config: RunnableConfig | list[RunnableConfig] | None = None,
    ) -> list[BaseState]:
        """Execute workflow for multiple initial states with PER-ITEM isolation AND observability.

        Each state gets its own isolated ScopeStack. Uses asyncio.gather for concurrency.
        """
        nemo_integration = get_nemo_relay_integration()

        async def run_one(state: BaseState, cfg: RunnableConfig | None) -> BaseState:
            # Each item gets fresh isolated stack
            stack = nemo_integration.create_isolated_scope_stack()
            parent_context = nemo_integration.capture_propagation_context()
            if parent_context and stack:
                stack = nemo_integration.create_scope_stack_from_propagation(parent_context)

            # NEW: Activate observability for this execution
            observability_plugin = None
            if (
                nemo_integration.config.observability
                and nemo_integration.config.observability.enabled
            ):
                observability_plugin = await nemo_integration.activate_observability(
                    state.execution_id
                )

            try:
                async with nemo_integration.use_scope_stack(stack):
                    execution_id = state.execution_id
                    entry_point = self._entry_point or "unknown"

                    async with nemo_integration.workflow_scope(execution_id, entry_point):
                        state_dict = to_state_dict(state)
                        result_dict: dict[str, Any] = await self._graph.ainvoke(
                            state_dict, config=cfg
                        )
                        return from_state_dict(result_dict)  # type: ignore[arg-type]
            finally:
                # NEW: Deactivate observability (flushes all exporters)
                if observability_plugin:
                    await nemo_integration.deactivate_observability(observability_plugin)

        configs = config if isinstance(config, list) else [config] * len(initial_states)
        tasks = [run_one(s, c) for s, c in zip(initial_states, configs, strict=False)]
        return await asyncio.gather(*tasks)

    def visualize(self) -> str:
        """Get Mermaid diagram of the graph."""
        try:
            return self._graph.get_graph().draw_mermaid()
        except Exception:
            return "Visualization not available"


class WorkflowEngine:
    """
    Generic workflow orchestrator using LangGraph StateGraph.

    Provides clean registration interface:
    - register_node(name, func)
    - register_edge(from_node, to_node, condition=None)
    - set_entry_point(node_name)
    - compile() -> CompiledGraph

    No hardcoded nodes - fully pluggable architecture.
    """

    def __init__(
        self,
        state_schema: type = BaseState,
        event_store: EventStore | None = None,
        max_iterations: int = 100,
    ):
        self._nodes: dict[str, NodeRegistration] = {}
        self._edges: list[EdgeRegistration] = []
        self._entry_point: str | None = None
        self._state_schema = state_schema
        self._event_store = event_store  # Can be None, will be initialized async
        self._max_iterations = max_iterations
        self._compiled_graph: CompiledGraph | None = None

        # For conditional edges
        self._conditional_edges: dict[str, list[tuple[ConditionFunc, str]]] = defaultdict(list)

    async def _ensure_event_store(self) -> EventStore:
        """Ensure event store is initialized."""
        if self._event_store is None:
            self._event_store = await get_event_store()
        return self._event_store

    def register_node(
        self,
        name: str,
        func: NodeFunc,
        description: str = "",
        tags: list[str] | None = None,
    ) -> WorkflowEngine:
        """
        Register a node function.

        Args:
            name: Unique node identifier
            func: Async function taking BaseState, returning BaseState
            description: Optional description
            tags: Optional tags for categorization

        Returns:
            Self for chaining
        """
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already registered")

        self._nodes[name] = NodeRegistration(
            name=name,
            func=func,
            description=description,
            tags=tags or [],
        )
        return self

    def register_edge(
        self,
        from_node: str,
        to_node: str,
        condition: ConditionFunc | None = None,
        description: str = "",
    ) -> WorkflowEngine:
        """
        Register an edge between nodes.

        Args:
            from_node: Source node name
            to_node: Target node name
            condition: Optional condition function (state -> bool)
            description: Optional description

        Returns:
            Self for chaining
        """
        if from_node not in self._nodes:
            raise ValueError(f"Source node '{from_node}' not registered")
        if to_node not in self._nodes and to_node != END:
            raise ValueError(f"Target node '{to_node}' not registered")

        self._edges.append(
            EdgeRegistration(
                from_node=from_node,
                to_node=to_node,
                condition=condition,
                description=description,
            )
        )
        return self

    def register_conditional_edge(
        self,
        from_node: str,
        condition: ConditionFunc,
        to_node: str,
        description: str = "",
    ) -> WorkflowEngine:
        """
        Register a conditional edge (evaluated at runtime).

        Args:
            from_node: Source node name
            condition: Condition function (state -> bool)
            to_node: Target node name
            description: Optional description

        Returns:
            Self for chaining
        """
        self._conditional_edges[from_node].append((condition, to_node))
        return self

    def set_entry_point(self, node_name: str) -> WorkflowEngine:
        """Set the workflow entry point."""
        if node_name not in self._nodes:
            raise ValueError(f"Entry point node '{node_name}' not registered")
        self._entry_point = node_name
        return self

    def compile(self) -> CompiledGraph:
        """
        Compile the workflow into a runnable graph.

        Returns:
            CompiledGraph ready for execution
        """
        if not self._entry_point:
            raise ValueError("Entry point not set. Call set_entry_point() first.")

        if not self._nodes:
            raise ValueError("No nodes registered")

        # Build StateGraph
        graph: StateGraph = StateGraph(self._state_schema)  # type: ignore[type-arg]

        # Add nodes
        for name, reg in self._nodes.items():
            # Wrap node function with event recording
            # Use default arguments to capture loop variables (fixes late-binding closure)
            def make_node_func(
                node_func: NodeFunc = reg.func, node_name: str = name
            ) -> Callable[[dict[str, Any] | BaseState], Awaitable[dict[str, Any]]]:
                async def wrapped(state: dict[str, Any] | BaseState) -> dict[str, Any]:
                    # Convert input to BaseState (handle both dict and BaseState from LangGraph)
                    base_state = state if isinstance(state, BaseState) else from_state_dict(state)  # type: ignore[arg-type]

                    # Ensure NeMo Relay stack is synced for this node execution
                    nemo_integration = get_nemo_relay_integration()
                    nemo_integration.get_scope_stack()  # Syncs stack to thread-local

                    # Record node start
                    exec_id = base_state.execution_id
                    await self._record_event(
                        exec_id, "NODE_START", {"node": node_name}, node_name, base_state.iteration
                    )
                    start_time = asyncio.get_event_loop().time()

                    try:
                        # Execute node
                        result = await node_func(base_state)

                        # Record completion
                        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                        await self._record_event(
                            exec_id,
                            "NODE_COMPLETE",
                            {"node": node_name, "duration_ms": duration_ms},
                            node_name,
                            base_state.iteration,
                        )

                        # Record metrics
                        from datetime import datetime

                        event_store = await self._ensure_event_store()
                        await event_store.record_node_metric(
                            NodeMetricRecord(
                                metric_id=str(uuid.uuid4()),
                                execution_id=exec_id,
                                node_name=node_name,
                                start_time=datetime.utcnow(),
                                end_time=datetime.utcnow(),
                                duration_ms=duration_ms,
                                exit_code=0,
                                tokens_consumed=0,
                                api_calls=0,
                                success=True,
                                error_message=None,
                            )
                        )

                        # Convert result BaseState back to dict for LangGraph
                        return to_state_dict(result)  # type: ignore[return-value]
                    except Exception as e:
                        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                        await self._record_event(
                            exec_id,
                            "NODE_ERROR",
                            {"node": node_name, "error": str(e), "duration_ms": duration_ms},
                            node_name,
                            base_state.iteration,
                        )
                        raise

                return wrapped

            graph.add_node(name, action=make_node_func())  # type: ignore[call-overload]

        # Add edges
        for edge in self._edges:
            if edge.condition:
                # Conditional edge - add as conditional
                graph.add_conditional_edges(
                    edge.from_node,
                    lambda state, cond=edge.condition: cond(
                        from_state_dict(state) if not isinstance(state, BaseState) else state
                    ),
                    {True: edge.to_node, False: END},
                )
            else:
                # Direct edge
                graph.add_edge(edge.from_node, edge.to_node)

        # Add conditional edges
        for from_node, conditions in self._conditional_edges.items():
            if len(conditions) == 1:
                # Single condition
                cond, to_node = conditions[0]
                graph.add_conditional_edges(
                    from_node,
                    lambda state, c=cond: c(
                        from_state_dict(state) if not isinstance(state, BaseState) else state
                    ),
                    {True: to_node, False: END},
                )
            else:
                # Multiple conditions - route based on first matching
                # Capture conditions in default argument to avoid late-binding closure
                def route(
                    state: dict[str, Any] | BaseState,
                    _conditions: list[tuple[ConditionFunc, str]] = conditions,
                ) -> str:
                    base_state = (
                        from_state_dict(cast("StateDict", state))
                        if not isinstance(state, BaseState)
                        else state
                    )
                    for cond, target in _conditions:
                        if cond(base_state):
                            return target
                    return END

                edge_map: dict[str, str] = {target: target for _, target in conditions}
                edge_map[END] = END
                graph.add_conditional_edges(
                    from_node,
                    route,
                    edge_map,  # type: ignore[arg-type]
                )

        # Set entry point
        graph.set_entry_point(self._entry_point)

        # Compile
        compiled = graph.compile()

        self._compiled_graph = CompiledGraph(compiled, self, self._entry_point)
        return self._compiled_graph

    async def _run_with_context(
        self,
        state: BaseState,
        context_overrides: dict[str, Any],
    ) -> BaseState:
        """Run compiled graph with cross-cutting context AND isolated NeMo Relay stack AND observability."""
        if self._compiled_graph is None:
            raise RuntimeError("Workflow not compiled. Call compile() first.")

        nemo_integration = get_nemo_relay_integration()

        # Create isolated NeMo Relay stack for this run
        stack = nemo_integration.create_isolated_scope_stack()
        parent_context = nemo_integration.capture_propagation_context()
        if parent_context and stack:
            stack = nemo_integration.create_scope_stack_from_propagation(parent_context)

        # NEW: Activate observability for this execution
        observability_plugin = None
        if nemo_integration.config.observability and nemo_integration.config.observability.enabled:
            observability_plugin = await nemo_integration.activate_observability(state.execution_id)

        # Set cross-cutting context vars (existing pattern)
        tokens = []
        for key, value in context_overrides.items():
            ctx_var = {
                "rate_limit": ctx_rate_limit,
                "auth_token": ctx_auth_token,
                "feature_flags": ctx_feature_flags,
            }.get(key)
            if ctx_var is not None:
                tokens.append(cast(contextvars.ContextVar[Any], ctx_var).set(value))

        try:
            # Run with BOTH cross-cutting context AND isolated NeMo Relay stack
            async with nemo_integration.use_scope_stack(stack):
                execution_id = state.execution_id
                entry_point = self._entry_point or "unknown"

                async with nemo_integration.workflow_scope(execution_id, entry_point):
                    return await self._compiled_graph.ainvoke(state)
        finally:
            # NEW: Deactivate observability (flushes all exporters)
            if observability_plugin:
                await nemo_integration.deactivate_observability(observability_plugin)
            for token in tokens:
                token.var.reset(token)

    async def _record_event(
        self,
        execution_id: str,
        event_type: str,
        payload: dict[str, Any],
        node_name: str | None = None,
        iteration: int = 0,
    ) -> EventRecord:
        """Record an event to the event store."""
        event_store = await self._ensure_event_store()
        return await event_store.record_event(
            execution_id=execution_id,
            event_type=event_type,
            payload=payload,
            node_name=node_name,
            iteration=iteration,
        )

    @property
    def nodes(self) -> dict[str, NodeRegistration]:
        return self._nodes.copy()

    @property
    def edges(self) -> list[EdgeRegistration]:
        return self._edges.copy()

    @property
    def entry_point(self) -> str | None:
        return self._entry_point

    def get_node(self, name: str) -> NodeRegistration | None:
        return self._nodes.get(name)

    def validate_graph(self) -> list[str]:
        """Validate graph structure, return list of warnings."""
        warnings = []

        if not self._entry_point:
            warnings.append("No entry point set")

        if not self._nodes:
            warnings.append("No nodes registered")

        # Check for unreachable nodes
        reachable = set()
        to_visit = [self._entry_point] if self._entry_point else []

        while to_visit:
            node = to_visit.pop()
            if node in reachable or node == END:
                continue
            reachable.add(node)

            # Find edges from this node
            for edge in self._edges:
                if edge.from_node == node:
                    to_visit.append(edge.to_node)

            # Conditional edges
            for _cond, target in self._conditional_edges.get(node, []):
                to_visit.append(target)

        for node_name in self._nodes:
            if node_name not in reachable:
                warnings.append(f"Node '{node_name}' is unreachable from entry point")

        # Check for nodes with no outgoing edges (except END)
        for node_name in self._nodes:
            has_outgoing = any(e.from_node == node_name for e in self._edges)
            has_conditional = node_name in self._conditional_edges
            if not has_outgoing and not has_conditional and node_name != END:
                warnings.append(f"Node '{node_name}' has no outgoing edges")

        return warnings


# Convenience function
def create_workflow_engine(
    state_schema: type = BaseState,
    event_store: EventStore | None = None,
    max_iterations: int = 100,
) -> WorkflowEngine:
    """Create a new workflow engine instance."""
    return WorkflowEngine(
        state_schema=state_schema,
        event_store=event_store,
        max_iterations=max_iterations,
    )


__all__ = [
    "WorkflowEngine",
    "CompiledGraph",
    "NodeRegistration",
    "EdgeRegistration",
    "create_workflow_engine",
    "ctx_rate_limit",
    "ctx_auth_token",
    "ctx_feature_flags",
]
