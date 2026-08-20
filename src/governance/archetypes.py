"""Policy Archetypes — Pre-seeded category sets for deployment contexts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class DeploymentContext(StrEnum):
    CONSUMER_CHATBOT = "consumer_chatbot"
    ENTERPRISE_RAG = "enterprise_rag"
    KIDS_EDUCATION = "kids_education"
    HEALTHCARE = "healthcare"
    FINANCIAL_SERVICES = "financial_services"
    CODE_ASSISTANT = "code_assistant"
    GOVERNMENT_SOVEREIGN = "government_sovereign"
    SYNTHETIC_LABELING = "synthetic_labeling"


@dataclass
class Archetype:
    name: str
    description: str
    categories: list[str] = field(default_factory=list)  # V2 category names
    severity_overrides: dict[str, str] = field(default_factory=dict)  # category -> S0-S4
    custom_categories: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""


class ArchetypeLoader:
    """Loads policy archetypes from configuration."""

    def __init__(self, archetypes_path: Path | None = None):
        self.archetypes_path = (
            archetypes_path
            or Path(__file__).parent.parent.parent / "config" / "policies" / "archetypes.yaml"
        )
        self._archetypes: dict[DeploymentContext, Archetype] = {}
        self._load_archetypes()

    def _load_archetypes(self) -> None:
        with open(self.archetypes_path) as f:
            data = yaml.safe_load(f)

        for ctx_name, arch_data in data.get("archetypes", {}).items():
            ctx = DeploymentContext(ctx_name)
            self._archetypes[ctx] = Archetype(
                name=arch_data["name"],
                description=arch_data["description"],
                categories=arch_data.get("categories", []),
                severity_overrides=arch_data.get("severity_overrides", {}),
                custom_categories=arch_data.get("custom_categories", []),
                notes=arch_data.get("notes", ""),
            )

    def get_archetype(self, context: DeploymentContext | None) -> Archetype:
        """Get archetype for context, defaulting to consumer_chatbot."""
        if context and context in self._archetypes:
            return self._archetypes[context]
        return self._archetypes[DeploymentContext.CONSUMER_CHATBOT]

    def list_contexts(self) -> list[DeploymentContext]:
        return list(self._archetypes.keys())
