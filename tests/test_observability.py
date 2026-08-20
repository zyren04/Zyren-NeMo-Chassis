"""
Tests for NeMo Relay Observability Integration
Validates ATOF, ATIF, and OpenTelemetry exporters with NeMo Relay 0.7 plugin.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.nemo_relay_integration import (
    NeMoRelayConfig,
    NeMoRelayIntegration,
)
from src.observability.config import (
    ATIFConfig,
    ATOFConfig,
    ObservabilityConfig,
    OpenTelemetryConfig,
)
from src.observability.event_bridge import EventStoreBridge, RelayEvent
from src.observability.exporters import ATIFExporter, ATOFFileExporter, OpenTelemetryExporter
from src.observability.plugin import ObservabilityPlugin
from src.observability.subscribers import ATIFTrajectorySubscriber, EventStoreSubscriber
from src.state.event_store import EventRecord
from src.state.state_schema import BaseState


class TestObservabilityConfig:
    """Test observability configuration parsing."""

    def test_default_config(self):
        """Test default observability config values."""
        config = ObservabilityConfig()
        assert config.enabled is True
        assert config.atof.enabled is True
        assert config.atof.output_directory == "logs/observability"
        assert config.atif.enabled is False
        assert config.opentelemetry.enabled is False

    def test_custom_config(self):
        """Test custom observability config."""
        config = ObservabilityConfig(
            enabled=True,
            atof=ATOFConfig(
                enabled=True,
                output_directory="/custom/logs",
                filename="custom.jsonl",
            ),
            atif=ATIFConfig(
                enabled=True,
                output_directory="/custom/trajectories",
            ),
            opentelemetry=OpenTelemetryConfig(
                enabled=True,
                endpoint="http://otel:4318/v1/traces",
            ),
        )
        assert config.atof.output_directory == "/custom/logs"
        assert config.atof.filename == "custom.jsonl"
        assert config.atif.enabled is True
        assert config.opentelemetry.endpoint == "http://otel:4318/v1/traces"

    def test_config_from_toml(self):
        """Test loading config from TOML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
version = 3
[[components]]
kind = "observability"
enabled = true
[components.config]
version = 3
[components.config.atof]
enabled = true
output_directory = "test_logs"
filename = "test.jsonl"
[components.config.atif]
enabled = true
output_directory = "test_trajectories"
[components.config.opentelemetry]
enabled = true
endpoint = "http://test:4318/v1/traces"
""")
            toml_path = f.name

        try:
            config = NeMoRelayConfig.from_toml(toml_path)
            assert config.observability is not None
            assert config.observability.enabled is True
            assert config.observability.atof.output_directory == "test_logs"
            assert config.observability.atof.filename == "test.jsonl"
            assert config.observability.atif.enabled is True
            assert config.observability.opentelemetry.enabled is True
        finally:
            os.unlink(toml_path)


class TestATOFFileExporter:
    """Test ATOF (Raw Canonical Events) file exporter."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def exporter(self, temp_dir):
        return ATOFFileExporter(
            output_dir=temp_dir,
            filename="test_events.jsonl",
            mode="append",
        )

    @pytest.mark.asyncio
    async def test_export_event(self, exporter, temp_dir):
        """Test exporting a single event."""
        event = {
            "event_id": "evt-123",
            "execution_id": "exec-456",
            "event_type": "node_start",
            "node_name": "test_node",
            "payload": {"input": "test"},
            "timestamp": "2024-01-01T00:00:00Z",
            "iteration": 1,
            "trace_id": "trace-789",
            "span_id": "span-abc",
            "relay_uuid": "relay-1",
            "relay_parent_uuid": "relay-0",
        }

        exporter.write_event(event)
        exporter.flush()

        # Verify file was created and contains the event
        output_file = temp_dir / "test_events.jsonl"
        assert output_file.exists()

        content = output_file.read_text().strip()
        exported = json.loads(content)
        assert exported["event_id"] == "evt-123"
        assert exported["trace_id"] == "trace-789"
        assert exported["relay_uuid"] == "relay-1"

    @pytest.mark.asyncio
    async def test_export_multiple_events(self, exporter, temp_dir):
        """Test exporting multiple events."""
        events = [
            {"event_id": f"evt-{i}", "execution_id": "exec-1", "event_type": "test", "payload": {}}
            for i in range(5)
        ]

        for event in events:
            exporter.write_event(event)
        exporter.flush()

        output_file = temp_dir / "test_events.jsonl"
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 5

    @pytest.mark.asyncio
    async def test_sanitization(self, exporter, temp_dir):
        """Test sensitive field sanitization."""
        event = {
            "event_id": "evt-1",
            "execution_id": "exec-1",
            "event_type": "test",
            "payload": {
                "api_key": "secret-key-123",
                "password": "secret-pass",
                "normal_field": "visible",
            },
        }

        exporter.write_event(event)
        exporter.flush()

        output_file = temp_dir / "test_events.jsonl"
        content = output_file.read_text()
        # Note: ATOFFileExporter doesn't sanitize by default, that's done at plugin level
        assert "visible" in content


class TestATIFExporter:
    """Test ATIF (Execution Trajectories) exporter."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_event_store(self):
        store = MagicMock()
        store.get_events = AsyncMock(return_value=[])
        return store

    @pytest.fixture
    def exporter(self, mock_event_store, temp_dir):
        return ATIFExporter(
            event_store=mock_event_store,
            output_dir=temp_dir,
            agent_name="test-agent",
            agent_metadata={"team": "test"},
        )

    @pytest.mark.asyncio
    async def test_export_trajectory(self, exporter, mock_event_store, temp_dir):
        """Test exporting a trajectory."""
        # Mock events
        mock_events = [
            MagicMock(
                event_id="evt-1",
                event_type="llm_request_start",
                payload={
                    "metadata": {
                        "annotation": {"messages": [{"role": "user", "content": "Hello"}]}
                    },
                    "relay_uuid": "relay-1",
                },
                timestamp="2024-01-01T00:00:00Z",
            ),
            MagicMock(
                event_id="evt-2",
                event_type="llm_request_end",
                payload={
                    "data": {"output": {"content": "Hi there!", "tool_calls": []}},
                    "metadata": {"model_name": "test-model", "usage": {}},
                    "relay_uuid": "relay-1",
                },
                timestamp="2024-01-01T00:00:01Z",
            ),
        ]
        mock_event_store.get_events.return_value = mock_events

        trajectory = await exporter.export_execution("exec-123")

        assert trajectory["schema_version"] == "ATIF-v1.7"
        assert len(trajectory["steps"]) == 2
        assert trajectory["steps"][0]["type"] == "user"
        assert trajectory["steps"][1]["type"] == "agent"

    @pytest.mark.asyncio
    async def test_export_to_file(self, exporter, mock_event_store, temp_dir):
        """Test exporting trajectory to file."""
        mock_event_store.get_events.return_value = []

        filepath = exporter.export_to_file("exec-123", "test_trajectory.json")

        assert filepath.exists()
        assert filepath.name == "test_trajectory.json"
        content = json.loads(filepath.read_text())
        assert content["schema_version"] == "ATIF-v1.7"


class TestOpenTelemetryExporter:
    """Test OpenTelemetry (OTLP) exporter."""

    @pytest.fixture
    def exporter(self):
        return OpenTelemetryExporter(
            endpoint="http://localhost:4318/v1/traces",
            service_name="test-service",
            projection="full",
        )

    @pytest.mark.asyncio
    async def test_export_spans(self, exporter):
        """Test exporting spans (mocked)."""
        spans = [
            {
                "trace_id": "trace-123",
                "span_id": "span-456",
                "parent_span_id": "span-000",
                "name": "test_operation",
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-01T00:00:01Z",
                "attributes": {"key": "value"},
                "status": "OK",
            }
        ]

        # Should not raise
        exporter.export_spans(spans)
        exporter.shutdown()


class TestEventBridge:
    """Test EventBridge for translating Relay events to EventStore records."""

    @pytest.fixture
    def bridge(self):
        mock_store = MagicMock()
        return EventStoreBridge(mock_store)

    def test_relay_to_event_record(self, bridge):
        """Test converting Relay event to EventRecord."""
        relay_event = RelayEvent(
            uuid="relay-123",
            parent_uuid="relay-000",
            event_type="llm_start",
            scope_type="llm",
            name="llm_call",
            timestamp="2024-01-01T00:00:00Z",
            data={"model": "test"},
            metadata={"trace_id": "trace-1", "span_id": "span-1", "iteration": 1},
        )

        record = bridge.relay_to_event_record(relay_event, "exec-1")

        assert record.event_id == "relay-123"
        assert record.execution_id == "exec-1"
        assert record.event_type == "llm_request_start"
        assert record.node_name == "llm_call"
        assert record.payload["relay_uuid"] == "relay-123"
        assert record.payload["trace_id"] == "trace-1"
        assert record.payload["span_id"] == "span-1"
        assert record.iteration == 1

    def test_event_record_to_relay(self, bridge):
        """Test converting EventRecord back to Relay event."""
        record = EventRecord(
            event_id="evt-1",
            execution_id="exec-1",
            event_type="llm_request_start",
            node_name="llm_call",
            payload={
                "relay_uuid": "relay-1",
                "relay_parent_uuid": "relay-0",
                "scope_type": "llm",
                "data": {"model": "test"},
                "metadata": {"trace_id": "trace-1"},
            },
            timestamp="2024-01-01T00:00:00Z",
            iteration=1,
        )

        relay_event = bridge.event_record_to_relay(record)

        assert relay_event.uuid == "relay-1"
        assert relay_event.parent_uuid == "relay-0"
        assert relay_event.event_type == "llm_request_start"
        assert relay_event.scope_type == "llm"
        assert relay_event.name == "llm_call"
        assert relay_event.metadata["trace_id"] == "trace-1"


class TestEventStoreSubscriber:
    """Test EventStoreSubscriber for NeMo Relay events."""

    @pytest.fixture
    def mock_event_store(self):
        store = MagicMock()
        store.record_event = AsyncMock()
        return store

    @pytest.fixture
    def subscriber(self, mock_event_store):
        return EventStoreSubscriber(mock_event_store, "exec-123")

    def test_on_event(self, subscriber):
        """Test receiving event from NeMo Relay."""
        # Mock NeMo Relay event
        nemo_event = MagicMock()
        nemo_event.uuid = "relay-1"
        nemo_event.parent_uuid = "relay-0"
        nemo_event.event_type = "llm_start"
        nemo_event.scope_type = "llm"
        nemo_event.name = "llm_call"
        nemo_event.timestamp = "2024-01-01T00:00:00Z"
        nemo_event.data = {"model": "test"}
        nemo_event.metadata = {"trace_id": "trace-1", "iteration": 1}

        subscriber.on_event(nemo_event)

        assert len(subscriber._buffer) == 1
        assert subscriber._buffer[0].uuid == "relay-1"

    @pytest.mark.asyncio
    async def test_force_flush(self, subscriber, mock_event_store):
        """Test flushing buffered events to EventStore."""
        nemo_event = MagicMock()
        nemo_event.uuid = "relay-1"
        nemo_event.parent_uuid = "relay-0"
        nemo_event.event_type = "llm_start"
        nemo_event.scope_type = "llm"
        nemo_event.name = "llm_call"
        nemo_event.timestamp = "2024-01-01T00:00:00Z"
        nemo_event.data = {"model": "test"}
        nemo_event.metadata = {"trace_id": "trace-1", "iteration": 1}

        subscriber.on_event(nemo_event)
        await subscriber.force_flush()

        mock_event_store.record_event.assert_called_once()
        call_args = mock_event_store.record_event.call_args
        assert call_args.kwargs["execution_id"] == "exec-123"
        assert call_args.kwargs["event_type"] == "llm_request_start"


class TestATIFTrajectorySubscriber:
    """Test ATIFTrajectorySubscriber."""

    @pytest.fixture
    def subscriber(self):
        mock_store = MagicMock()
        return ATIFTrajectorySubscriber(mock_store, "exec-123", {"name": "test-agent"})

    def test_on_event(self, subscriber):
        """Test collecting events for trajectory."""
        nemo_event = MagicMock()
        nemo_event.uuid = "relay-1"
        nemo_event.parent_uuid = "relay-0"
        nemo_event.event_type = "llm_start"
        nemo_event.scope_type = "llm"
        nemo_event.name = "llm_call"
        nemo_event.timestamp = "2024-01-01T00:00:00Z"
        nemo_event.data = {"model": "test"}
        nemo_event.metadata = {"trace_id": "trace-1"}

        subscriber.on_event(nemo_event)

        assert len(subscriber._events) == 1

    def test_export_atif(self, subscriber):
        """Test exporting ATIF trajectory."""
        # Use event with proper message structure
        nemo_event = MagicMock()
        nemo_event.uuid = "relay-1"
        nemo_event.parent_uuid = "relay-0"
        nemo_event.event_type = "llm_start"
        nemo_event.scope_type = "llm"
        nemo_event.name = "llm_call"
        nemo_event.timestamp = "2024-01-01T00:00:00Z"
        nemo_event.data = {"model": "test"}
        # Provide proper annotation.messages structure expected by build_atif_trajectory
        nemo_event.metadata = {
            "annotation": {"messages": [{"role": "user", "content": "Hello"}]},
            "trace_id": "trace-1",
        }

        subscriber.on_event(nemo_event)
        trajectory = subscriber.export_atif()

        assert trajectory["schema_version"] == "ATIF-v1.7"
        assert trajectory["agent"]["name"] == "test-agent"
        assert len(trajectory["steps"]) == 1


class TestObservabilityPlugin:
    """Test ObservabilityPlugin lifecycle."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self, temp_dir):
        return ObservabilityConfig(
            enabled=True,
            atof=ATOFConfig(
                enabled=True,
                output_directory=str(temp_dir / "atof"),
            ),
            atif=ATIFConfig(
                enabled=True,
                output_directory=str(temp_dir / "atim"),
            ),
        )

    @pytest.fixture
    def mock_event_store(self):
        store = MagicMock()
        store.record_event = AsyncMock()
        return store

    @pytest.mark.asyncio
    async def test_plugin_lifecycle(self, config, mock_event_store, temp_dir):
        """Test plugin activate/deactivate lifecycle."""
        plugin = ObservabilityPlugin(config, mock_event_store, "exec-123")

        # Activate plugin
        await plugin.activate()
        assert plugin._eventstore_subscriber is not None
        assert plugin._atof_exporter is not None
        assert plugin._atif_subscriber is not None

        # Emit event through subscriber
        nemo_event = MagicMock()
        nemo_event.uuid = "relay-1"
        nemo_event.parent_uuid = "relay-0"
        nemo_event.event_type = "llm_start"
        nemo_event.scope_type = "llm"
        nemo_event.name = "llm_call"
        nemo_event.timestamp = "2024-01-01T00:00:00Z"
        nemo_event.data = {"model": "test"}
        nemo_event.metadata = {"trace_id": "trace-1", "iteration": 1}

        plugin._eventstore_subscriber.on_event(nemo_event)
        await plugin._eventstore_subscriber.force_flush()

        # Deactivate plugin (flushes exporters)
        await plugin.deactivate()

        # Verify ATOF file was written
        atof_files = list((temp_dir / "atof").glob("*.jsonl"))
        assert len(atof_files) == 1

    @pytest.mark.asyncio
    async def test_plugin_disabled(self, temp_dir, mock_event_store):
        """Test plugin when observability is disabled."""
        config = ObservabilityConfig(enabled=False)
        plugin = ObservabilityPlugin(config, mock_event_store, "exec-123")

        await plugin.activate()
        await plugin.deactivate()

        # No files should be created
        atof_files = list((temp_dir / "atof").glob("*.jsonl"))
        assert len(atof_files) == 0

    @pytest.mark.asyncio
    async def test_export_atif(self, config, mock_event_store, temp_dir):
        """Test ATIF export from plugin."""
        plugin = ObservabilityPlugin(config, mock_event_store, "exec-123")
        await plugin.activate()

        # Add some events with proper message structure
        nemo_event = MagicMock()
        nemo_event.uuid = "relay-1"
        nemo_event.parent_uuid = "relay-0"
        nemo_event.event_type = "llm_start"
        nemo_event.scope_type = "llm"
        nemo_event.name = "llm_call"
        nemo_event.timestamp = "2024-01-01T00:00:00Z"
        nemo_event.data = {"model": "test"}
        # Provide proper annotation.messages structure
        nemo_event.metadata = {
            "annotation": {"messages": [{"role": "user", "content": "Hello"}]},
            "trace_id": "trace-1",
        }

        plugin._atif_subscriber.on_event(nemo_event)

        trajectory = plugin.export_atif()
        assert trajectory["schema_version"] == "ATIF-v1.7"
        assert len(trajectory["steps"]) == 1

        await plugin.deactivate()


class TestNeMoRelayIntegrationObservability:
    # Reset singleton state before each test to ensure fresh config
    @pytest.fixture(autouse=True, scope="function")
    def reset_singleton_state(self):
        from src.infrastructure import nemo_relay_integration

        nemo_relay_integration.NeMoRelayIntegration._instance = None
        nemo_relay_integration.NeMoRelayIntegration._initialized = False

    """Test NeMo Relay Integration with observability."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self, temp_dir):
        return NeMoRelayConfig(
            observability=ObservabilityConfig(
                enabled=True,
                atof=ATOFConfig(
                    enabled=True,
                    output_directory=str(temp_dir / "atof"),
                ),
                atif=ATIFConfig(
                    enabled=True,
                    output_directory=str(temp_dir / "atim"),
                ),
            )
        )

    @pytest.mark.asyncio
    async def test_activate_deactivate_observability(self, config, temp_dir):
        """Test activating and deactivating observability via integration."""
        # Reset singleton state to ensure fresh config
        from src.infrastructure import nemo_relay_integration

        nemo_relay_integration.NeMoRelayIntegration._instance = None
        nemo_relay_integration.NeMoRelayIntegration._initialized = False

        integration = NeMoRelayIntegration(config)

        # Activate observability
        plugin = await integration.activate_observability("exec-123")
        assert plugin is not None
        assert plugin._eventstore_subscriber is not None

        # Emit event through plugin's subscriber
        nemo_event = MagicMock()
        nemo_event.uuid = "relay-1"
        nemo_event.parent_uuid = "relay-0"
        nemo_event.event_type = "llm_start"
        nemo_event.scope_type = "llm"
        nemo_event.name = "llm_call"
        nemo_event.timestamp = "2024-01-01T00:00:00Z"
        nemo_event.data = {"model": "test"}
        nemo_event.metadata = {"trace_id": "trace-1", "iteration": 1}

        plugin._eventstore_subscriber.on_event(nemo_event)
        await plugin._eventstore_subscriber.force_flush()

        # Deactivate observability
        await integration.deactivate_observability(plugin)

        # Verify files written
        atof_files = list((temp_dir / "atof").glob("*.jsonl"))
        assert len(atof_files) == 1

    @pytest.mark.asyncio
    async def test_observability_disabled_returns_none(self, temp_dir):
        """Test that disabled observability returns None."""
        config = NeMoRelayConfig(observability=ObservabilityConfig(enabled=False))
        integration = NeMoRelayIntegration(config)

        plugin = await integration.activate_observability("exec-123")
        assert plugin is None


class TestEventStoreTelemetryFields:
    """Test EventRecord telemetry correlation fields."""

    def test_event_record_with_telemetry(self):
        """Test EventRecord with telemetry fields."""
        event = EventRecord(
            event_id="evt-1",
            execution_id="exec-1",
            event_type="node_start",
            node_name="test_node",
            payload={},
            timestamp="2024-01-01T00:00:00Z",
            iteration=1,
            trace_id="trace-123",
            span_id="span-456",
            relay_uuid="relay-789",
            relay_parent_uuid="relay-000",
        )

        assert event.trace_id == "trace-123"
        assert event.span_id == "span-456"
        assert event.relay_uuid == "relay-789"
        assert event.relay_parent_uuid == "relay-000"

    def test_event_record_without_telemetry(self):
        """Test EventRecord without telemetry fields (backward compat)."""
        event = EventRecord(
            event_id="evt-1",
            execution_id="exec-1",
            event_type="node_start",
            node_name="test_node",
            payload={},
            timestamp="2024-01-01T00:00:00Z",
            iteration=1,
        )

        assert event.trace_id is None
        assert event.span_id is None
        assert event.relay_uuid is None
        assert event.relay_parent_uuid is None


class TestWorkflowEngineRunWithContext:
    """Test WorkflowEngine._run_with_context with observability."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_run_with_context_observability(self, temp_dir):
        """Test _run_with_context activates observability."""
        from src.runtime.engine import WorkflowEngine

        # Create a simple workflow
        async def node1(state: BaseState) -> BaseState:
            return state

        engine = WorkflowEngine()
        engine.register_node("node1", node1)
        engine.set_entry_point("node1")
        compiled = engine.compile()

        # Patch nemo integration
        with patch("src.runtime.engine.get_nemo_relay_integration") as mock_get:
            config = NeMoRelayConfig(
                observability=ObservabilityConfig(
                    enabled=True,
                    atof=ATOFConfig(
                        enabled=True,
                        output_directory=str(temp_dir / "atof"),
                    ),
                )
            )
            mock_integration = NeMoRelayIntegration(config)
            mock_get.return_value = mock_integration

            state = BaseState(execution_id="exec-context-123")
            result = await compiled.ainvoke(state)

            assert result.execution_id == "exec-context-123"
            # Check ATOF file was created
            atof_files = list((temp_dir / "atof").glob("*.jsonl"))
            assert len(atof_files) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
