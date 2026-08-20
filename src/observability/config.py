"""Observability configuration (TOML-compatible)."""

from dataclasses import dataclass, field


@dataclass
class ATOFConfig:
    enabled: bool = True
    output_directory: str = "logs/observability"
    filename: str = "events.jsonl"
    mode: str = "append"  # "append" | "overwrite"
    # Stream sink for remote delivery (optional)
    stream_url: str | None = None
    stream_transport: str = "http_post"
    stream_header_env: dict[str, str] = field(default_factory=dict)


@dataclass
class ATIFConfig:
    enabled: bool = False
    output_directory: str = "logs/trajectories"
    agent_name: str = "default-agent"
    agent_metadata: dict = field(default_factory=dict)
    filename_template: str = "trajectory_{session_id}.json"


@dataclass
class OpenTelemetryConfig:
    enabled: bool = False
    endpoint: str = "http://localhost:4318/v1/traces"
    service_name: str = "nemo-relay-chassis"
    projection: str = "full"  # "full" | "gen_ai" | "openinference"
    transport: str = "http_binary"  # "http_binary" | "grpc"
    headers: dict[str, str] = field(default_factory=dict)
    resource_attributes: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 10


@dataclass
class ObservabilityConfig:
    """Root observability configuration matching NeMo Relay plugin schema."""

    # Version: 2 for NeMo Relay 0.6, 3 for 0.7
    version: int = 3

    # Global toggle
    enabled: bool = True

    # Exporter configurations
    atof: ATOFConfig = field(default_factory=ATOFConfig)
    atif: ATIFConfig = field(default_factory=ATIFConfig)
    opentelemetry: OpenTelemetryConfig = field(default_factory=OpenTelemetryConfig)

    # Sanitization (applies to all exporters)
    sanitize_payloads: bool = True
    sensitive_fields: list[str] = field(
        default_factory=lambda: ["api_key", "authorization", "password", "secret", "token"]
    )

    @classmethod
    def from_toml(cls, toml_dict: dict) -> "ObservabilityConfig":
        """Create config from parsed TOML (matches plugin schema)."""
        # Handle multiple TOML structures defensively
        components = toml_dict.get("components", []) or []

        # Find observability component
        obs_config = {}
        for comp in components:
            if comp.get("kind") == "observability" and comp.get("enabled", True):
                # Support both nested config and direct config
                obs_config = comp.get("config", {}) or comp.get("observability", {}) or comp
                break
        else:
            # Fallback: check if config is at root level (for test fixtures)
            if (
                "version" in toml_dict
                or "atof" in toml_dict
                or "atif" in toml_dict
                or "opentelemetry" in toml_dict
            ):
                obs_config = toml_dict

        # Safely extract nested configs with defaults
        atof_config = obs_config.get("atof", {}) or {}
        atif_config = obs_config.get("atif", {}) or {}
        otel_config = obs_config.get("opentelemetry", {}) or {}

        return cls(
            version=obs_config.get("version", 3),
            enabled=obs_config.get("enabled", True),
            atof=ATOFConfig(**atof_config) if atof_config else ATOFConfig(),
            atif=ATIFConfig(**atif_config) if atif_config else ATIFConfig(),
            opentelemetry=OpenTelemetryConfig(**otel_config)
            if otel_config
            else OpenTelemetryConfig(),
        )

    def to_toml(self) -> dict:
        """Export as TOML-compatible dict."""
        return {
            "version": self.version,
            "enabled": self.enabled,
            "atof": self.atof.__dict__,
            "atif": self.atif.__dict__,
            "opentelemetry": self.opentelemetry.__dict__,
        }
