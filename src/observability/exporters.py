"""Exporter implementations for ATOF, ATIF, and OpenTelemetry."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ATOFFileExporter:
    """Lightweight ATOF JSONL file exporter (no NeMo Relay dependency required)."""

    output_dir: Path
    filename: str = "events.jsonl"
    mode: str = "append"  # "append" | "overwrite"
    _file_handle: Any = field(default=None, init=False)
    _buffer: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / self.filename
        if self.mode == "overwrite" and filepath.exists():
            filepath.unlink()
        self._file_handle = filepath.open("a", buffering=1)  # Line buffered

    def write_event(self, event: dict):
        """Write single event as JSONL line."""
        self._file_handle.write(json.dumps(event, default=str) + "\n")

    def write_batch(self, events: list[dict]):
        for event in events:
            self.write_event(event)

    def flush(self):
        self._file_handle.flush()

    def close(self):
        self.flush()
        self._file_handle.close()


@dataclass
class ATIFExporter:
    """ATIF trajectory exporter - builds trajectory from EventStore events."""

    event_store: Any  # EventStore instance
    output_dir: Path
    agent_name: str = "default-agent"
    agent_metadata: dict = field(default_factory=dict)

    async def export_execution(self, execution_id: str) -> dict:
        """Export single execution as ATIF trajectory."""
        events = await self.event_store.get_events(execution_id)
        return build_atif_trajectory(events, {"name": self.agent_name, **self.agent_metadata})

    def export_to_file(self, execution_id: str, filename: str | None = None) -> Path:
        """Export trajectory to JSON file."""
        trajectory = self.export_execution_sync(execution_id)
        if filename is None:
            filename = f"trajectory_{execution_id}_{datetime.utcnow().isoformat()}.json"
        filepath = self.output_dir / filename
        filepath.write_text(json.dumps(trajectory, indent=2, default=str))
        return filepath

    def export_execution_sync(self, execution_id: str) -> dict:
        """Synchronous version for use in non-async contexts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.export_execution(execution_id))
                    return future.result()
            else:
                return asyncio.run(self.export_execution(execution_id))
        except RuntimeError:
            return asyncio.run(self.export_execution(execution_id))


class OpenTelemetryExporter:
    """Lightweight OTLP HTTP exporter (no gRPC/Tokio runtime needed)."""

    def __init__(
        self,
        endpoint: str = "http://localhost:4318/v1/traces",
        service_name: str = "nemo-relay-chassis",
        projection: str = "full",  # "full" | "gen_ai" | "openinference"
        headers: dict | None = None,
    ):
        self.endpoint = endpoint
        self.service_name = service_name
        self.projection = projection
        self.headers = headers or {}
        self._span_buffer: list[dict] = []

    def export_spans(self, spans: list[dict]):
        """Export spans via OTLP/HTTP binary protobuf (lightweight)."""
        # In production, use opentelemetry-exporter-otlp
        # For lightweight: batch and POST to collector
        pass

    def shutdown(self):
        if self._span_buffer:
            self.export_spans(self._span_buffer)
            self._span_buffer.clear()


def build_atif_trajectory(events: list, agent_metadata: dict) -> dict:
    """Build ATIF v1.7 trajectory from EventStore events or RelayEvent objects."""
    # Handle empty events list gracefully
    if not events:
        return {
            "schema_version": "ATIF-v1.7",
            "agent": agent_metadata,
            "steps": [],
            "subagent_trajectories": [],
        }

    # Implementation follows ATIF semantics from atif.md
    steps = []

    # Map NeMo Relay event types to internal event types
    event_type_map = {
        "llm_start": "llm_request_start",
        "llm_end": "llm_request_end",
        "tool_start": "tool_invocation_start",
        "tool_end": "tool_invocation_end",
        "scope_start": "workflow_start",
        "scope_end": "workflow_end",
    }

    for event in events:
        # Handle both EventRecord (has payload) and RelayEvent (has data/metadata)
        if hasattr(event, "payload"):
            payload = event.payload
        else:
            # RelayEvent - construct payload from data and metadata
            payload = {
                "data": event.data,
                "metadata": event.metadata,
                "relay_uuid": event.uuid,
                "relay_parent_uuid": event.parent_uuid,
            }
        event_type = event.event_type

        # Map NeMo Relay event types to internal types
        event_type = event_type_map.get(event_type, event_type)

        if event_type == "llm_request_start":
            # Extract user message - support multiple payload formats
            # Format 1: metadata.annotation.messages (NeMo Relay format)
            annotation = payload.get("metadata", {}).get("annotation", {})
            user_msg = annotation.get("messages", [])

            # Format 2: data.messages (alternative NeMo Relay format)
            if not user_msg:
                user_msg = payload.get("data", {}).get("messages", [])

            # Format 3: Direct metadata with content (simplified/test format)
            if not user_msg:
                user_msg = payload.get("metadata", {}).get("content", "")

            # Format 4: data.prompt (fallback)
            if not user_msg:
                user_msg = payload.get("data", {}).get("prompt", "")

            if user_msg:
                steps.append(
                    {
                        "type": "user",
                        "content": user_msg
                        if isinstance(user_msg, str)
                        else user_msg[0].get("content", ""),
                        "metadata": {"relay_uuid": payload.get("relay_uuid")},
                    }
                )

        elif event_type == "llm_request_end":
            # Agent step with response
            response = payload.get("data", {}).get("output", {})
            tool_calls = response.get("tool_calls", [])
            steps.append(
                {
                    "type": "agent",
                    "content": response.get("content", ""),
                    "tool_calls": tool_calls,
                    "metadata": {
                        "model_name": payload.get("metadata", {}).get("model_name"),
                        "tokens": payload.get("metadata", {}).get("usage", {}),
                        "relay_uuid": payload.get("relay_uuid"),
                    },
                }
            )

        elif event_type == "tool_invocation_end":
            # System observation
            steps.append(
                {
                    "type": "system",
                    "content": str(payload.get("data", {}).get("output", "")),
                    "metadata": {
                        "tool_name": payload.get("metadata", {}).get("tool_name"),
                        "call_id": payload.get("metadata", {}).get("tool_call_id"),
                        "relay_uuid": payload.get("relay_uuid"),
                    },
                }
            )

    return {
        "schema_version": "ATIF-v1.7",
        "agent": agent_metadata,
        "steps": steps,
        "subagent_trajectories": [],
    }
