"""Translation layer between NeMo Relay canonical events and EventStore records."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class RelayEvent:
    """Canonical NeMo Relay event structure (ATOF-compatible)."""

    uuid: str
    parent_uuid: str | None
    event_type: str  # scope_start, scope_end, tool_start, tool_end, llm_start, llm_end, mark
    scope_type: str  # agent, workflow, tool, llm
    name: str
    timestamp: datetime
    data: dict[str, Any]
    metadata: dict[str, Any]


class EventStoreBridge:
    """Bidirectional bridge: NeMo Relay events <-> EventStore records."""

    def __init__(self, event_store):
        self.event_store = event_store
        self._execution_id_map: dict[str, str] = {}  # relay_uuid -> execution_id

    def relay_to_event_record(self, relay_event: RelayEvent, execution_id: str):
        """Convert NeMo Relay event to EventStore EventRecord."""
        # Import here to avoid circular dependency
        from ..state.event_store import EventRecord

        # Map Relay event_type to our event_type taxonomy
        event_type_map = {
            "scope_start": "workflow_start",
            "scope_end": "workflow_end",
            "tool_start": "tool_invocation_start",
            "tool_end": "tool_invocation_end",
            "llm_start": "llm_request_start",
            "llm_end": "llm_request_end",
            "mark": "checkpoint",
        }

        return EventRecord(
            event_id=relay_event.uuid,
            execution_id=execution_id,
            event_type=event_type_map.get(relay_event.event_type, relay_event.event_type),
            node_name=relay_event.name,
            payload={
                "relay_uuid": relay_event.uuid,
                "relay_parent_uuid": relay_event.parent_uuid,
                "scope_type": relay_event.scope_type,
                "data": relay_event.data,
                "metadata": relay_event.metadata,
                # Preserve OTel correlation IDs
                "trace_id": relay_event.metadata.get("trace_id"),
                "span_id": relay_event.metadata.get("span_id"),
            },
            timestamp=relay_event.timestamp,
            iteration=relay_event.metadata.get("iteration", 0),
        )

    def event_record_to_relay(self, record) -> RelayEvent:
        """Convert EventStore record back to Relay event (for ATIF export)."""
        payload = record.payload
        # Get trace_id and span_id from record fields, or fall back to payload metadata
        trace_id = record.trace_id or payload.get("metadata", {}).get("trace_id")
        span_id = record.span_id or payload.get("metadata", {}).get("span_id")
        return RelayEvent(
            uuid=payload.get("relay_uuid", record.event_id),
            parent_uuid=payload.get("relay_parent_uuid"),
            event_type=record.event_type,
            scope_type=payload.get("scope_type", "unknown"),
            name=record.node_name or "unknown",
            timestamp=record.timestamp,
            data=payload.get("data", {}),
            metadata={
                **payload.get("metadata", {}),
                "iteration": record.iteration,
                "trace_id": trace_id,
                "span_id": span_id,
            },
        )
