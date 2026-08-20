#!/usr/bin/env python3
"""Test singleton behavior with test-like scenario."""

from src.infrastructure.nemo_relay_integration import NeMoRelayIntegration, NeMoRelayConfig, get_nemo_relay_integration
from src.observability.config import ObservabilityConfig, ATOFConfig, ATIFConfig
import tempfile
from pathlib import Path
from unittest.mock import patch

# Simulate fixture calling get_nemo_relay_integration() without config
print("=== Simulating fixture: isolated_nemo_relay_context ===")
integration_fixture = get_nemo_relay_integration()
print(f"Created integration: {integration_fixture}")
print(f"  observability: {integration_fixture.config.observability}")
print(f"  _initialized: {integration_fixture._initialized}")

# Simulate test_run_with_context_observability
print("\n=== Simulating test_run_with_context_observability ===")
with tempfile.TemporaryDirectory() as tmpdir:
    config = NeMoRelayConfig(
        observability=ObservabilityConfig(
            enabled=True,
            atof=ATOFConfig(enabled=True, output_directory=str(Path(tmpdir) / "atof")),
        )
    )
    mock_integration = NeMoRelayIntegration(config)
    print(f"Created mock_integration: {mock_integration}")
    print(f"  observability: {mock_integration.config.observability}")
    print(f"  _initialized: {mock_integration._initialized}")
    print(f"  Same as fixture: {mock_integration is integration_fixture}")

# Now simulate test_activate_deactivate_observability
print("\n=== Simulating test_activate_deactivate_observability ===")
with tempfile.TemporaryDirectory() as tmpdir:
    config = NeMoRelayConfig(
        observability=ObservabilityConfig(
            enabled=True,
            atof=ATOFConfig(enabled=True, output_directory=str(Path(tmpdir) / "atof")),
            atif=ATIFConfig(enabled=True, output_directory=str(Path(tmpdir) / "atim")),
        )
    )
    integration = NeMoRelayIntegration(config)
    print(f"Created integration: {integration}")
    print(f"  observability: {integration.config.observability}")
    print(f"  _initialized: {integration._initialized}")
    print(f"  Same as mock_integration: {integration is mock_integration}")