from __future__ import annotations

"""
Deprecation Registry - Deprecated NIM Policy Enforcement
Loads deprecated NIM identifiers from config and provides substring matching.
"""

from pathlib import Path  # noqa: E402

import yaml  # noqa: E402


class DeprecatedNimList:
    """Container for deprecated NIM list loaded from YAML."""

    def __init__(self, version: str = "1.0", deprecated: list[str] | None = None):
        self.version = version
        self.deprecated = deprecated or []

    @classmethod
    def load(cls, path: Path) -> DeprecatedNimList:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            version=data.get("version", "1.0"),
            deprecated=data.get("deprecated", []),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(
                {"version": self.version, "deprecated": self.deprecated},
                f,
                sort_keys=False,
            )


class DeprecationRegistry:
    """Registry for deprecated NIM identifiers with case-insensitive substring matching."""

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "nims.deprecated.yaml"
        self.config_path = config_path
        self._deprecated: list[str] = []
        self._load()

    def _load(self) -> None:
        if self.config_path.exists():
            data = DeprecatedNimList.load(self.config_path)
            self._deprecated = data.deprecated
        else:
            self._deprecated = []

    def reload(self) -> None:
        """Reload the deprecated list from disk."""
        self._load()

    def check(self, nim_identifier: str) -> list[str]:
        """Return list of deprecated patterns that match (case-insensitive substring)."""
        matches = []
        nim_lower = nim_identifier.lower()
        for pattern in self._deprecated:
            if pattern.lower() in nim_lower:
                matches.append(pattern)
        return matches

    def is_deprecated(self, nim_identifier: str) -> bool:
        """Check if a NIM identifier matches any deprecated pattern."""
        return len(self.check(nim_identifier)) > 0

    def get_all_deprecated(self) -> list[str]:
        """Get all deprecated patterns."""
        return self._deprecated.copy()

    def add_deprecated(self, pattern: str) -> None:
        """Add a new deprecated pattern (in memory only, call save() to persist)."""
        if pattern not in self._deprecated:
            self._deprecated.append(pattern)

    def remove_deprecated(self, pattern: str) -> bool:
        """Remove a deprecated pattern (in memory only, call save() to persist)."""
        if pattern in self._deprecated:
            self._deprecated.remove(pattern)
            return True
        return False

    def save(self) -> None:
        """Persist current deprecated list to disk."""
        data = DeprecatedNimList(version="1.0", deprecated=self._deprecated)
        data.save(self.config_path)


# Global default instance
_default_registry: DeprecationRegistry | None = None


def get_deprecation_registry(config_path: Path | None = None) -> DeprecationRegistry:
    """Get or create the default deprecation registry instance."""
    global _default_registry
    if _default_registry is None:
        _default_registry = DeprecationRegistry(config_path)
    return _default_registry


def set_deprecation_registry(registry: DeprecationRegistry) -> None:
    """Set the default deprecation registry instance (useful for testing)."""
    global _default_registry
    _default_registry = registry


__all__ = [
    "DeprecatedNimList",
    "DeprecationRegistry",
    "get_deprecation_registry",
    "set_deprecation_registry",
]
