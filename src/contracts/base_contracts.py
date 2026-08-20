"""
Type-Safe Contract Engine - Base Contracts
Strict Pydantic v2 schemas for generic inter-node message passing.
No business logic, pure infrastructure plumbing.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import TypedDict


# ruff: noqa: UP042 - StrEnum not available in Python 3.10
class SignalType(str, Enum):
    """Execution control signals between nodes."""

    CONTINUE = "CONTINUE"
    HALT = "HALT"
    RETRY = "RETRY"
    BRANCH = "BRANCH"


class NodePayload(BaseModel):
    """
    Generic message envelope for inter-node communication.
    All nodes communicate exclusively through this contract.
    Immutable (frozen) for content-addressing guarantees.
    """

    model_config = {"frozen": True}

    payload_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_node: str = Field(..., min_length=1, description="Originating node identifier")
    target_node: str = Field(..., min_length=1, description="Destination node identifier")
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Tracks request-response chains"
    )
    data: dict[str, Any] = Field(default_factory=dict, description="Arbitrary payload data")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Routing and processing metadata"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    iteration: int = Field(default=0, ge=0, description="Current iteration number")

    @field_validator("source_node", "target_node")
    @classmethod
    def validate_node_names(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Node name cannot be empty")
        return v.strip()

    def with_updated_target(self, new_target: str) -> NodePayload:
        """Create a new payload routed to a different target node."""
        return self.model_copy(
            update={
                "target_node": new_target,
                "payload_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow(),
            }
        )

    def with_incremented_iteration(self) -> NodePayload:
        """Create a new payload with incremented iteration counter."""
        return self.model_copy(
            update={
                "iteration": self.iteration + 1,
                "payload_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow(),
            }
        )


class ArtifactItem(BaseModel):
    """
    Represents a versioned, immutable artifact produced during execution.
    Artifacts are content-addressed via SHA256 checksum.
    Immutable (frozen) for content-addressing guarantees.
    """

    model_config = {"frozen": True}

    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    artifact_type: str = Field(
        ..., min_length=1, description="Logical type: code, test, doc, binary, etc."
    )
    content: bytes = Field(..., description="Raw artifact content")
    mime_type: str = Field(default="application/octet-stream", description="MIME type of content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extensible metadata")
    checksum: str = Field(..., min_length=64, max_length=64, description="SHA256 hex digest")
    size_bytes: int = Field(..., ge=0, description="Content size in bytes")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by_node: str = Field(..., min_length=1, description="Node that produced this artifact")

    @model_validator(mode="after")
    def validate_checksum(self) -> ArtifactItem:
        import hashlib

        computed = hashlib.sha256(self.content).hexdigest()
        if self.checksum != computed:
            raise ValueError(f"Checksum mismatch: expected {computed}, got {self.checksum}")
        if self.size_bytes != len(self.content):
            raise ValueError(f"Size mismatch: expected {len(self.content)}, got {self.size_bytes}")
        return self

    @classmethod
    def create(
        cls,
        artifact_type: str,
        content: bytes,
        created_by_node: str,
        mime_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactItem:
        """Factory method to create artifact with auto-computed checksum."""
        import hashlib

        checksum = hashlib.sha256(content).hexdigest()
        return cls(
            artifact_type=artifact_type,
            content=content,
            mime_type=mime_type,
            metadata=metadata or {},
            checksum=checksum,
            size_bytes=len(content),
            created_by_node=created_by_node,
        )


class ExecutionSignal(BaseModel):
    """
    Control signal emitted by nodes to direct workflow execution.
    Replaces imperative control flow with declarative signals.
    Immutable (frozen) for content-addressing guarantees.
    """

    model_config = {"frozen": True}

    signal_type: SignalType = Field(..., description="Control flow directive")
    reason: str = Field(..., min_length=1, description="Human-readable justification")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Signal-specific parameters")
    originating_node: str = Field(..., min_length=1, description="Node that emitted this signal")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    target_node: str | None = Field(default=None, description="Explicit target for BRANCH signals")

    @model_validator(mode="after")
    def validate_branch_target(self) -> ExecutionSignal:
        if self.signal_type == SignalType.BRANCH and not self.target_node:
            raise ValueError("BRANCH signal requires target_node")
        return self


class Justification(BaseModel):
    """
    Generic justification/verdict for any decision node.

    Provides structured rationale for decisions, enabling traceability
    and human-readable explanations. Domain-agnostic - applicable to
    any node that makes a decision requiring justification.
    """

    model_config = {"frozen": True}

    label: str = Field(
        ...,
        description="Categorical classification (e.g., 'code_not_reachable', 'false_positive', 'policy_violation')",
    )
    reason: str = Field(..., description="Human-readable explanation")
    status: Literal["TRUE", "FALSE", "UNKNOWN"] = Field(..., description="Ternary verdict")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in the verdict")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")


class ValidationDecision(BaseModel):
    """
    Result of guardrails validation - determines if execution proceeds.
    Immutable (frozen) for content-addressing guarantees.
    """

    model_config = {"frozen": True}

    valid: bool = Field(..., description="Whether the input/output passes validation")
    errors: list[str] = Field(default_factory=list, description="Blocking validation errors")
    warnings: list[str] = Field(default_factory=list, description="Non-blocking warnings")
    suggested_action: SignalType = Field(
        default=SignalType.CONTINUE, description="Recommended control flow"
    )
    validator_name: str = Field(
        ..., min_length=1, description="Name of validator that produced this decision"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    justification: Justification | None = Field(
        default=None, description="Structured justification for the decision"
    )

    def merge(self, other: ValidationDecision) -> ValidationDecision:
        """Combine two decisions - invalid if either is invalid."""
        return ValidationDecision(
            valid=self.valid and other.valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            suggested_action=(
                SignalType.HALT
                if (
                    self.suggested_action == SignalType.HALT
                    or other.suggested_action == SignalType.HALT
                )
                else SignalType.CONTINUE
            ),
            validator_name=f"{self.validator_name}+{other.validator_name}",
            metadata={**self.metadata, **other.metadata},
            justification=self.justification or other.justification,
        )


# TypedDict variants for LangGraph StateGraph compatibility
class NodePayloadDict(TypedDict):
    payload_id: str
    source_node: str
    target_node: str
    correlation_id: str
    data: dict[str, Any]
    metadata: dict[str, Any]
    timestamp: str  # ISO format
    iteration: int


class ArtifactItemDict(TypedDict):
    artifact_id: str
    artifact_type: str
    content: bytes
    mime_type: str
    metadata: dict[str, Any]
    checksum: str
    size_bytes: int
    created_at: str
    created_by_node: str


class ExecutionSignalDict(TypedDict):
    signal_type: str
    reason: str
    metadata: dict[str, Any]
    originating_node: str
    timestamp: str
    target_node: str | None


class ValidationDecisionDict(TypedDict):
    valid: bool
    errors: list[str]
    warnings: list[str]
    suggested_action: str
    validator_name: str
    metadata: dict[str, Any]


# Conversion utilities
def to_payload_dict(payload: NodePayload) -> NodePayloadDict:
    return NodePayloadDict(
        payload_id=payload.payload_id,
        source_node=payload.source_node,
        target_node=payload.target_node,
        correlation_id=payload.correlation_id,
        data=payload.data,
        metadata=payload.metadata,
        timestamp=payload.timestamp.isoformat(),
        iteration=payload.iteration,
    )


def from_payload_dict(d: NodePayloadDict) -> NodePayload:
    return NodePayload(
        payload_id=d["payload_id"],
        source_node=d["source_node"],
        target_node=d["target_node"],
        correlation_id=d["correlation_id"],
        data=d["data"],
        metadata=d["metadata"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
        iteration=d["iteration"],
    )


def to_artifact_dict(artifact: ArtifactItem) -> ArtifactItemDict:
    return ArtifactItemDict(
        artifact_id=artifact.artifact_id,
        artifact_type=artifact.artifact_type,
        content=artifact.content,
        mime_type=artifact.mime_type,
        metadata=artifact.metadata,
        checksum=artifact.checksum,
        size_bytes=artifact.size_bytes,
        created_at=artifact.created_at.isoformat(),
        created_by_node=artifact.created_by_node,
    )


def from_artifact_dict(d: ArtifactItemDict) -> ArtifactItem:
    return ArtifactItem(
        artifact_id=d["artifact_id"],
        artifact_type=d["artifact_type"],
        content=d["content"],
        mime_type=d["mime_type"],
        metadata=d["metadata"],
        checksum=d["checksum"],
        size_bytes=d["size_bytes"],
        created_at=datetime.fromisoformat(d["created_at"]),
        created_by_node=d["created_by_node"],
    )


def to_signal_dict(signal: ExecutionSignal) -> ExecutionSignalDict:
    return ExecutionSignalDict(
        signal_type=signal.signal_type.value,
        reason=signal.reason,
        metadata=signal.metadata,
        originating_node=signal.originating_node,
        timestamp=signal.timestamp.isoformat(),
        target_node=signal.target_node,
    )


def from_signal_dict(d: ExecutionSignalDict) -> ExecutionSignal:
    return ExecutionSignal(
        signal_type=SignalType(d["signal_type"]),
        reason=d["reason"],
        metadata=d["metadata"],
        originating_node=d["originating_node"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
        target_node=d["target_node"],
    )


def to_decision_dict(decision: ValidationDecision) -> ValidationDecisionDict:
    return ValidationDecisionDict(
        valid=decision.valid,
        errors=decision.errors,
        warnings=decision.warnings,
        suggested_action=decision.suggested_action.value,
        validator_name=decision.validator_name,
        metadata=decision.metadata,
    )


def from_decision_dict(d: ValidationDecisionDict) -> ValidationDecision:
    return ValidationDecision(
        valid=d["valid"],
        errors=d["errors"],
        warnings=d["warnings"],
        suggested_action=SignalType(d["suggested_action"]),
        validator_name=d["validator_name"],
        metadata=d["metadata"],
    )


__all__ = [
    "SignalType",
    "NodePayload",
    "ArtifactItem",
    "ExecutionSignal",
    "ValidationDecision",
    "Justification",
    "NodePayloadDict",
    "ArtifactItemDict",
    "ExecutionSignalDict",
    "ValidationDecisionDict",
    "to_payload_dict",
    "from_payload_dict",
    "to_artifact_dict",
    "from_artifact_dict",
    "to_signal_dict",
    "from_signal_dict",
    "to_decision_dict",
    "from_decision_dict",
]
