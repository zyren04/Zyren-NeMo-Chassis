"""
Repository Registry Configuration
Multi-category repository configuration (active/github-only/deprecated) with blueprint metadata.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class BlueprintMeta(BaseModel):
    """Blueprint metadata from Build catalog."""

    name: str
    url: str
    category: str  # "Enterprise Blueprint" | "Developer Example" | "Partner Example" | "NemoClaw"


class RepoConfig(BaseModel):
    """Configuration for a single repository."""

    name: str = Field(..., min_length=1)
    url: str = Field(..., pattern=r"^(https?://|git@|ssh://)")
    branch: str | None = None
    depth: int | None = Field(None, ge=1)
    enabled: bool = True
    blueprints: list[BlueprintMeta] = Field(default_factory=list)

    @field_validator("url")
    @classmethod
    def validate_git_url(cls, v: str) -> str:
        if not (
            v.startswith("https://")
            or v.startswith("http://")
            or v.startswith("git@")
            or v.startswith("ssh://")
        ):
            raise ValueError("Invalid Git URL format")
        return v

    def branch_or_default(self, default: str = "main") -> str:
        return self.branch or default

    def depth_or_default(self, default: int = 1) -> int:
        return self.depth or default


class Defaults(BaseModel):
    """Default values for repository settings."""

    branch: str = "main"
    depth: int = 1


class RepoRegistry(BaseModel):
    """Top-level repository registry configuration."""

    version: str = "1.0"
    defaults: Defaults = Field(default_factory=Defaults)
    repos_active: list[RepoConfig] = Field(default_factory=list)
    repos_github_only: list[RepoConfig] = Field(default_factory=list)
    repos_deprecated: list[str] = Field(default_factory=list)

    def scannable_repos(self) -> list[RepoConfig]:
        """Get all repositories that should be scanned (active + github_only)."""
        return [*self.repos_active, *self.repos_github_only]

    def enabled_scannable_repos(self) -> list[RepoConfig]:
        """Get enabled repositories that should be scanned."""
        return [r for r in self.scannable_repos() if r.enabled]

    def get_repo_by_name(self, name: str) -> RepoConfig | None:
        """Find a repository by name across all categories."""
        for repo in self.scannable_repos():
            if repo.name == name:
                return repo
        return None

    def validate_unique_names(self) -> list[str]:
        """Validate that all repository names are unique across categories.

        Returns:
            List of error messages (empty if valid).
        """
        errors = []
        seen = set()
        for repo in self.scannable_repos():
            if repo.name in seen:
                errors.append(f"Duplicate repository name: {repo.name}")
            seen.add(repo.name)
        return errors

    @classmethod
    def load(cls, path: Path) -> RepoRegistry:
        """Load registry from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save registry to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(exclude_none=True), f, sort_keys=False)


__all__ = [
    "BlueprintMeta",
    "RepoConfig",
    "Defaults",
    "RepoRegistry",
]
