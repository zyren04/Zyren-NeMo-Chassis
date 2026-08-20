"""Custom NeMo Relay subscribers that write directly to EventStore."""

import asyncio
from typing import Any

try:
    from nemo_relay import ScopeHandle
    from nemo_relay.subscribers import Subscriber, deregister, register

    NEMO_RELAY_AVAILABLE = True
except ImportError:
    Subscriber = object
    register = deregister = None
    ScopeHandle = None
    NEMO_RELAY_AVAILABLE = False

from ..state.event_store import EventStore
from .event_bridge import EventStoreBridge, RelayEvent


class EventStoreSubscriber(Subscriber):
    """Subscriber that forwards all NeMo Relay events to EventStore."""

    def __init__(self, event_store: EventStore, execution_id: str):
        self.event_store = event_store
        self.bridge = EventStoreBridge(event_store)
        self.execution_id = execution_id
        self._buffer: list[RelayEvent] = []
        self._flush_task: asyncio.Task | None = None

    def on_event(self, event: Any) -> None:
        """Called by NeMo Relay for each lifecycle event."""
        # Convert NeMo Relay event to our RelayEvent format
        relay_event = self._convert_nemo_event(event)
        self._buffer.append(relay_event)

        # Batch flush every 100 events or 1 second
        if len(self._buffer) >= 100:
            self._schedule_flush()

    def _convert_nemo_event(self, event: Any) -> RelayEvent:
        """Extract fields from NeMo Relay event object."""
        # NeMo Relay 0.7 event structure
        return RelayEvent(
            uuid=getattr(event, "uuid", str(event.get("uuid", ""))),
            parent_uuid=getattr(event, "parent_uuid", None),
            event_type=getattr(event, "event_type", event.get("event_type", "unknown")),
            scope_type=getattr(event, "scope_type", event.get("scope_type", "unknown")),
            name=getattr(event, "name", event.get("name", "unknown")),
            timestamp=getattr(event, "timestamp", event.get("timestamp")),
            data=getattr(event, "data", event.get("data", {})),
            metadata=getattr(event, "metadata", event.get("metadata", {})),
        )

    def _schedule_flush(self):
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush())

    async def _flush(self):
        if not self._buffer:
            return
        events = self._buffer[:]
        self._buffer.clear()

        for relay_event in events:
            record = self.bridge.relay_to_event_record(relay_event, self.execution_id)
            await self.event_store.record_event(
                execution_id=record.execution_id,
                event_type=record.event_type,
                payload=record.payload,
                node_name=record.node_name,
                iteration=record.iteration,
            )

    async def force_flush(self):
        await self._flush()

    async def shutdown(self):
        await self.force_flush()


class ATIFTrajectorySubscriber(Subscriber):
    """Subscriber that builds ATIF trajectories from Relay events."""

    def __init__(self, event_store: EventStore, execution_id: str, agent_metadata: dict):
        self.event_store = event_store
        self.execution_id = execution_id
        self.agent_metadata = agent_metadata
        self._events: list[RelayEvent] = []

    def on_event(self, event: Any) -> None:
        relay_event = self._convert_nemo_event(event)
        self._events.append(relay_event)

    def _convert_nemo_event(self, event: Any) -> RelayEvent:
        # Same as EventStoreSubscriber
        return RelayEvent(
            uuid=getattr(event, "uuid", str(event.get("uuid", ""))),
            parent_uuid=getattr(event, "parent_uuid", None),
            event_type=getattr(event, "event_type", event.get("event_type", "unknown")),
            scope_type=getattr(event, "scope_type", event.get("scope_type", "unknown")),
            name=getattr(event, "name", event.get("name", "unknown")),
            timestamp=getattr(event, "timestamp", event.get("timestamp")),
            data=getattr(event, "data", event.get("data", {})),
            metadata=getattr(event, "metadata", event.get("metadata", {})),
        )

    def export_atif(self) -> dict:
        """Export collected events as ATIF v1.7 trajectory."""
        from .exporters import build_atif_trajectory

        # Convert RelayEvent objects to format expected by build_atif_trajectory
        # The build_atif_trajectory function handles both EventRecord and RelayEvent objects
        # and supports both annotation.messages and data.messages formats
        return build_atif_trajectory(self._events, self.agent_metadata)

    def clear(self):
        self._events.clear()

    async def force_flush(self):
        pass  # ATIF is exported on demand

    async def shutdown(self):
        pass
