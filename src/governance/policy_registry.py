"""Policy Registry — Load, validate, and serve Nemotron policy taxonomies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml


class PolicyRegistry:
    """Registry for Nemotron content-safety policies."""

    def __init__(self, policy_dir: Path | None = None, schema_path: Path | None = None):
        self.policy_dir = policy_dir or Path(__file__).parent.parent.parent / "config" / "policies"
        self.schema_path = schema_path or self.policy_dir / "policy_json_schema.json"
        self._schema: dict[str, Any] | None = None
        self._policies: dict[str, dict[str, Any]] = {}
        self._load_schema()

    def _load_schema(self) -> None:
        with open(self.schema_path) as f:
            self._schema = json.load(f)

    def validate(self, policy: dict[str, Any]) -> None:
        """Validate policy against JSON schema. Raises ValidationError on failure."""
        jsonschema.validate(instance=policy, schema=self._schema)

    def load_policy(self, policy_name: str) -> dict[str, Any]:
        """Load a policy by name (without extension)."""
        if policy_name in self._policies:
            return self._policies[policy_name]

        for ext in (".json", ".yaml", ".yml"):
            path = self.policy_dir / f"{policy_name}{ext}"
            if path.exists():
                with open(path) as f:
                    policy = json.load(f) if ext == ".json" else yaml.safe_load(f)
                self.validate(policy)
                self._policies[policy_name] = policy
                return policy

        raise FileNotFoundError(f"Policy not found: {policy_name}")

    def get_category(self, policy_name: str, category_name: str) -> dict[str, Any] | None:
        """Get a specific category from a loaded policy."""
        policy = self.load_policy(policy_name)
        for cat in policy.get("categories", []):
            if cat["name"] == category_name:
                return cat
        return None

    def get_severity(self, policy_name: str, category_name: str) -> str | None:
        """Get severity band for a category."""
        cat = self.get_category(policy_name, category_name)
        return cat["severity"] if cat else None

    def list_policies(self) -> list[str]:
        """List all available policy names."""
        names = set()
        for ext in (".json", ".yaml", ".yml"):
            for path in self.policy_dir.glob(f"*{ext}"):
                names.add(path.stem)
        return sorted(names)
