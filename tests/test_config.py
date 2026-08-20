#!/usr/bin/env python3
"""Test config attribute access."""

from src.infrastructure.nemo_relay_integration import NeMoRelayConfig
from src.observability.config import ObservabilityConfig, ATOFConfig, ATIFConfig
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    config = NeMoRelayConfig(
        observability=ObservabilityConfig(
            enabled=True,
            atof=ATOFConfig(enabled=True, output_directory=str(Path(tmpdir) / 'atof')),
            atif=ATIFConfig(enabled=True, output_directory=str(Path(tmpdir) / 'atim')),
        )
    )
    print(f'config.observability = {config.observability}')
    print(f'config.__dict__["observability"] = {config.__dict__["observability"]}')
    print(f'type(config.observability) = {type(config.observability)}')
    print(f'hasattr(config, "observability") = {hasattr(config, "observability")}')
    print(f'getattr(config, "observability", "NOT_FOUND") = {getattr(config, "observability", "NOT_FOUND")}')