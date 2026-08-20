"""Taxonomy Mapper — Map rough words to Nemotron V2 categories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class MappingMode(StrEnum):
    CLEAN_V2 = "clean_v2"
    V2_PLUS_CUSTOM = "v2_plus_custom"
    MOSTLY_CUSTOM = "mostly_custom"


@dataclass
class MappingResult:
    mode: MappingMode
    match_ratio: float
    matched_categories: dict[str, list[str]]  # V2 category -> matched synonyms
    unmapped_words: list[str]
    warnings: list[str]


class TaxonomyMapper:
    """Maps rough requirement words to canonical V2 categories."""

    def __init__(self, taxonomy_path: Path | None = None):
        self.taxonomy_path = (
            taxonomy_path
            or Path(__file__).parent.parent.parent / "config" / "policies" / "v2_taxonomy.yaml"
        )
        self._taxonomy: dict[str, Any] = {}
        self._synonym_index: dict[str, str] = {}  # synonym -> category_name
        self._load_taxonomy()

    def _load_taxonomy(self) -> None:
        with open(self.taxonomy_path) as f:
            self._taxonomy = yaml.safe_load(f)

        # Build synonym index
        for cat in self._taxonomy.get("categories", []):
            name = cat["name"]
            for synonym in cat.get("synonyms", []):
                self._synonym_index[synonym.lower()] = name

    def map_rough_words(
        self, rough_words: str, archetype_categories: list[str] | None = None
    ) -> MappingResult:
        """Map rough words to V2 categories."""
        # Tokenize rough words (simple split on commas, semicolons, "and")
        words = re.split(r"[,;]|\band\b", rough_words.lower())
        words = [w.strip() for w in words if w.strip()]

        matched = {}
        unmapped = []

        for word in words:
            # Direct synonym match
            if word in self._synonym_index:
                cat = self._synonym_index[word]
                matched.setdefault(cat, []).append(word)
            # Fuzzy match: check if word contains or is contained in synonym
            elif any(word in syn or syn in word for syn in self._synonym_index):
                for syn, cat in self._synonym_index.items():
                    if word in syn or syn in word:
                        matched.setdefault(cat, []).append(word)
                        break
            else:
                unmapped.append(word)

        # Filter to archetype categories if provided
        if archetype_categories:
            matched = {k: v for k, v in matched.items() if k in archetype_categories}

        total = len(words)
        matched_count = sum(len(v) for v in matched.values())
        ratio = matched_count / total if total > 0 else 0.0

        if ratio >= 0.8:
            mode = MappingMode.CLEAN_V2
        elif ratio >= 0.4:
            mode = MappingMode.V2_PLUS_CUSTOM
        else:
            mode = MappingMode.MOSTLY_CUSTOM

        warnings = []
        if unmapped:
            warnings.append(f"{len(unmapped)} rough words unmapped: {', '.join(unmapped[:5])}")

        return MappingResult(
            mode=mode,
            match_ratio=ratio,
            matched_categories=matched,
            unmapped_words=unmapped,
            warnings=warnings,
        )
