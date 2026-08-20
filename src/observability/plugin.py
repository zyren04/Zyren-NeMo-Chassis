"""Observability plugin for NeMo Relay integration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..state.event_store import EventStore
from .config import ObservabilityConfig
from .exporters import ATIFExporter, ATOFFileExporter, OpenTelemetryExporter
from .subscribers import ATIFTrajectorySubscriber, EventStoreSubscriber

# Check NeMo Relay availability
try:
    from nemo_relay.observability import OpenTelemetrySubscriber
    from nemo_relay.subscribers import deregister, register

    NEMO_RELAY_AVAILABLE = True
except ImportError:
    register = deregister = None
    OpenTelemetrySubscriber = None
    NEMO_RELAY_AVAILABLE = False


@dataclass
class ObservabilityPlugin:
    """Manages observability lifecycle for a workflow execution."""

    config: ObservabilityConfig
    event_store: EventStore
    execution_id: str

    # Active components
    _eventstore_subscriber: EventStoreSubscriber | None = None
    _atif_subscriber: ATIFTrajectorySubscriber | None = None
    _atof_exporter: ATOFFileExporter | None = None
    _otel_exporter: OpenTelemetryExporter | None = None
    _otel_subscriber: Any = None
    _registered_names: list[str] = field(default_factory=list)

    async def activate(self):
        """Activate all configured exporters/subscribers."""
        # Guard against config being None
        if self.config is None:
            return
        # 1. EventStore subscriber (always active for durability)
        self._eventstore_subscriber = EventStoreSubscriber(self.event_store, self.execution_id)
        if NEMO_RELAY_AVAILABLE and register:
            name = f"eventstore-{self.execution_id}"
            register(name, self._eventstore_subscriber)
            self._registered_names.append(name)

        # 2. ATOF file exporter (if configured)
        if self.config.atof.enabled:
            self._atof_exporter = ATOFFileExporter(
                output_dir=Path(self.config.atof.output_directory),
                filename=self.config.atof.filename,
                mode=self.config.atof.mode,
            )

        # 3. ATIF trajectory subscriber (if configured)
        if self.config.atif.enabled:
            self._atif_subscriber = ATIFTrajectorySubscriber(
                self.event_store,
                self.execution_id,
                self.config.atif.agent_metadata,
            )
            if NEMO_RELAY_AVAILABLE and register:
                name = f"atif-{self.execution_id}"
                register(name, self._atif_subscriber)
                self._registered_names.append(name)

        # 4. OpenTelemetry exporter (if configured)
        if self.config.opentelemetry.enabled:
            self._otel_exporter = OpenTelemetryExporter(
                endpoint=self.config.opentelemetry.endpoint,
                service_name=self.config.opentelemetry.service_name,
                projection=self.config.opentelemetry.projection,
                headers=self.config.opentelemetry.headers,
            )
            if NEMO_RELAY_AVAILABLE and OpenTelemetrySubscriber:
                subscriber = OpenTelemetrySubscriber(
                    endpoint=self.config.opentelemetry.endpoint,
                    service_name=self.config.opentelemetry.service_name,
                    projection=self.config.opentelemetry.projection,
                )
                name = f"otel-{self.execution_id}"
                register(name, subscriber)
                self._registered_names.append(name)
                self._otel_subscriber = subscriber

    async def deactivate(self):
        """Graceful shutdown: flush, deregister, close."""
        # Guard against config being None
        if self.config is None:
            return
        # Flush all
        if self._eventstore_subscriber:
            await self._eventstore_subscriber.force_flush()
        if self._atif_subscriber:
            await self._atif_subscriber.force_flush()
        if self._otel_exporter:
            self._otel_exporter.shutdown()
        if self._atof_exporter:
            self._atof_exporter.flush()

        # Deregister NeMo Relay subscribers
        if NEMO_RELAY_AVAILABLE and deregister:
            for name in self._registered_names:
                deregister(name)

        # Close file handles
        if self._atof_exporter:
            self._atof_exporter.close()

    def export_atif(self) -> dict:
        """Export ATIF trajectory for current execution."""
        if self._atif_subscriber:
            return self._atif_subscriber.export_atif()
        # Fallback: build from EventStore
        import asyncio

        from .exporters import build_atif_trajectory

        try:
            events = asyncio.run(self.event_store.get_events(self.execution_id))
            return build_atif_trajectory(events, self.config.atif.agent_metadata)
        except RuntimeError:
            # If already in async context, can't run
            return {
                "schema_version": "ATIF-v1.7",
                "agent": self.config.atif.agent_metadata,
                "steps": [],
            }

    def export_atif_to_file(self, filename: str | None = None) -> Path:
        if self._atif_subscriber:
            # This would need the ATIFExporter
            pass
        # Fallback
        atif_exporter = ATIFExporter(self.event_store, Path(self.config.atif.output_directory))
        return atif_exporter.export_to_file(self.execution_id, filename)
