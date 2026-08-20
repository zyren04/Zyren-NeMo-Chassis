"""
Generic State Schema - TypedDict & Pydantic v2 BaseState
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

from ..contracts.base_contracts import (
    ArtifactItem,
    ExecutionSignal,
    NodePayload,
    SignalType,
    ValidationDecision,
)


class BaseState(BaseModel):
    """
    Generic execution state for multi-node state machine.
    All nodes read/write through this shared state object.
    """

    execution_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique execution identifier"
    )
    iteration: int = Field(default=0, ge=0, description="Global iteration counter")
    current_node: str | None = Field(default=None, description="Currently executing node")
    previous_node: str | None = Field(default=None, description="Last completed node")

    artifacts: dict[str, ArtifactItem] = Field(
        default_factory=dict, description="Artifact registry by artifact_id"
    )
    artifact_index: dict[str, set[str]] = Field(
        default_factory=dict, description="Reverse index: type -> {artifact_ids}"
    )

    pending_payloads: list[NodePayload] = Field(
        default_factory=list, description="Payloads awaiting processing"
    )
    processed_payload_ids: set[str] = Field(
        default_factory=set, description="Deduplication set for payload IDs"
    )

    signals: list[ExecutionSignal] = Field(
        default_factory=list, description="Emitted control signals"
    )
    validation_decisions: list[ValidationDecision] = Field(
        default_factory=list, description="Guardrails decisions"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary execution metadata"
    )
    node_metadata: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Per-node metadata"
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)

    total_tokens_consumed: int = Field(default=0, ge=0)
    total_api_calls: int = Field(default=0, ge=0)
    node_execution_times: dict[str, float] = Field(
        default_factory=dict, description="Node -> total_ms"
    )
    node_call_counts: dict[str, int] = Field(default_factory=dict, description="Node -> call count")
    error_count: int = Field(default=0, ge=0)

    max_iterations: int = Field(default=100, ge=1, description="Hard iteration ceiling")
    max_tokens_per_execution: int = Field(default=100000, ge=1, description="Token budget")

    @model_validator(mode="after")
    def update_timestamp(self) -> BaseState:
        self.updated_at = datetime.utcnow()
        return self

    def increment_iteration(self) -> BaseState:
        return self.model_copy(
            update={
                "iteration": self.iteration + 1,
                "updated_at": datetime.utcnow(),
            }
        )

    def set_current_node(self, node_name: str) -> BaseState:
        return self.model_copy(
            update={
                "previous_node": self.current_node,
                "current_node": node_name,
                "updated_at": datetime.utcnow(),
            }
        )

    def add_artifact(self, artifact: ArtifactItem) -> BaseState:
        new_artifacts = {**self.artifacts, artifact.artifact_id: artifact}
        new_index = {k: set(v) for k, v in self.artifact_index.items()}
        if artifact.artifact_type not in new_index:
            new_index[artifact.artifact_type] = set()
        new_index[artifact.artifact_type].add(artifact.artifact_id)
        return self.model_copy(
            update={
                "artifacts": new_artifacts,
                "artifact_index": new_index,
                "updated_at": datetime.utcnow(),
            }
        )

    def get_artifacts_by_type(self, artifact_type: str) -> list[ArtifactItem]:
        ids = self.artifact_index.get(artifact_type, set())
        return [self.artifacts[aid] for aid in ids if aid in self.artifacts]

    def enqueue_payload(self, payload: NodePayload) -> BaseState:
        if payload.payload_id in self.processed_payload_ids:
            return self
        new_pending = self.pending_payloads + [payload]
        new_processed = self.processed_payload_ids | {payload.payload_id}
        return self.model_copy(
            update={
                "pending_payloads": new_pending,
                "processed_payload_ids": new_processed,
                "updated_at": datetime.utcnow(),
            }
        )

    def dequeue_payload(self) -> tuple[NodePayload | None, BaseState]:
        if not self.pending_payloads:
            return None, self
        payload = self.pending_payloads[0]
        new_pending = self.pending_payloads[1:]
        return payload, self.model_copy(
            update={
                "pending_payloads": new_pending,
                "updated_at": datetime.utcnow(),
            }
        )

    def emit_signal(self, signal: ExecutionSignal) -> BaseState:
        return self.model_copy(
            update={
                "signals": self.signals + [signal],
                "updated_at": datetime.utcnow(),
            }
        )

    def record_validation(self, decision: ValidationDecision) -> BaseState:
        return self.model_copy(
            update={
                "validation_decisions": self.validation_decisions + [decision],
                "updated_at": datetime.utcnow(),
            }
        )

    def update_node_metrics(
        self, node_name: str, execution_time_ms: float, tokens: int = 0, api_calls: int = 0
    ) -> BaseState:
        new_times = {
            **self.node_execution_times,
            node_name: self.node_execution_times.get(node_name, 0) + execution_time_ms,
        }
        new_counts = {
            **self.node_call_counts,
            node_name: self.node_call_counts.get(node_name, 0) + 1,
        }
        return self.model_copy(
            update={
                "node_execution_times": new_times,
                "node_call_counts": new_counts,
                "total_tokens_consumed": self.total_tokens_consumed + tokens,
                "total_api_calls": self.total_api_calls + api_calls,
                "updated_at": datetime.utcnow(),
            }
        )

    def record_error(self) -> BaseState:
        return self.model_copy(
            update={
                "error_count": self.error_count + 1,
                "updated_at": datetime.utcnow(),
            }
        )

    def check_iteration_limit(self) -> bool:
        return self.iteration >= self.max_iterations

    def check_token_budget(self) -> bool:
        return self.total_tokens_consumed >= self.max_tokens_per_execution

    def is_terminal(self) -> bool:
        return (
            self.check_iteration_limit()
            or self.check_token_budget()
            or any(s.signal_type == SignalType.HALT for s in self.signals)
            or any(
                not d.valid and d.suggested_action == SignalType.HALT
                for d in self.validation_decisions
            )
        )


class StateDict(TypedDict):
    execution_id: str
    iteration: int
    current_node: str | None
    previous_node: str | None
    artifacts: dict[str, ArtifactItem]
    artifact_index: dict[str, list[str]]
    pending_payloads: list[NodePayload]
    processed_payload_ids: list[str]
    signals: list[ExecutionSignal]
    validation_decisions: list[ValidationDecision]
    metadata: dict[str, Any]
    node_metadata: dict[str, dict[str, Any]]
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None
    total_tokens_consumed: int
    total_api_calls: int
    node_execution_times: dict[str, float]
    node_call_counts: dict[str, int]
    error_count: int
    max_iterations: int
    max_tokens_per_execution: int


def to_state_dict(state: BaseState) -> StateDict:
    return StateDict(
        execution_id=state.execution_id,
        iteration=state.iteration,
        current_node=state.current_node,
        previous_node=state.previous_node,
        artifacts=state.artifacts,
        artifact_index={k: list(v) for k, v in state.artifact_index.items()},
        pending_payloads=state.pending_payloads,
        processed_payload_ids=list(state.processed_payload_ids),
        signals=state.signals,
        validation_decisions=state.validation_decisions,
        metadata=state.metadata,
        node_metadata=state.node_metadata,
        created_at=state.created_at.isoformat(),
        updated_at=state.updated_at.isoformat(),
        started_at=state.started_at.isoformat() if state.started_at else None,
        completed_at=state.completed_at.isoformat() if state.completed_at else None,
        total_tokens_consumed=state.total_tokens_consumed,
        total_api_calls=state.total_api_calls,
        node_execution_times=state.node_execution_times,
        node_call_counts=state.node_call_counts,
        error_count=state.error_count,
        max_iterations=state.max_iterations,
        max_tokens_per_execution=state.max_tokens_per_execution,
    )


def from_state_dict(d: StateDict) -> BaseState:
    return BaseState(
        execution_id=d["execution_id"],
        iteration=d["iteration"],
        current_node=d["current_node"],
        previous_node=d["previous_node"],
        artifacts=d["artifacts"],
        artifact_index={k: set(v) for k, v in d["artifact_index"].items()},
        pending_payloads=d["pending_payloads"],
        processed_payload_ids=set(d["processed_payload_ids"]),
        signals=d["signals"],
        validation_decisions=d["validation_decisions"],
        metadata=d["metadata"],
        node_metadata=d["node_metadata"],
        created_at=datetime.fromisoformat(d["created_at"]),
        updated_at=datetime.fromisoformat(d["updated_at"]),
        started_at=datetime.fromisoformat(d["started_at"]) if d["started_at"] else None,
        completed_at=datetime.fromisoformat(d["completed_at"]) if d["completed_at"] else None,
        total_tokens_consumed=d["total_tokens_consumed"],
        total_api_calls=d["total_api_calls"],
        node_execution_times=d["node_execution_times"],
        node_call_counts=d["node_call_counts"],
        error_count=d["error_count"],
        max_iterations=d["max_iterations"],
        max_tokens_per_execution=d["max_tokens_per_execution"],
    )


__all__ = [
    "BaseState",
    "StateDict",
    "to_state_dict",
    "from_state_dict",
]
