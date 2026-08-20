"""Nemotron Policy Generator — BYO custom safety policies from rough requirements."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .archetypes import ArchetypeLoader, DeploymentContext
from .nemotron_prompts import NemotronPrompts, PromptMode, TargetModel
from .policy_registry import PolicyRegistry
from .taxonomy_mapper import MappingMode, TaxonomyMapper


@dataclass
class PolicyGenerationRequest:
    """Input request for policy generation."""

    rough_words: str  # User's rough requirements
    target_model: TargetModel = TargetModel.NCS_REASONING_4B
    deployment_context: DeploymentContext | None = None
    custom_categories: list[dict[str, Any]] = field(default_factory=list)
    allow_list: list[str] = field(default_factory=list)
    severity_overrides: dict[str, str] = field(default_factory=dict)  # category -> S0-S4
    locale: str = "en-US"
    jurisdiction: str | None = None
    output_formats: list[str] = field(default_factory=lambda: ["markdown", "json", "prompt"])


@dataclass
class PolicyGenerationResult:
    """Generated policy artifacts."""

    markdown: str
    json_taxonomy: dict[str, Any]
    system_prompt: str
    assumptions: list[str]
    taxonomy_mode: MappingMode
    warnings: list[str] = field(default_factory=list)


class PolicyGenerator:
    """Generates Nemotron-compatible safety policies from rough requirements."""

    NON_NEGOTIABLE_FLOOR = {"sexual_minor": "S7"}  # Can never be carved out

    def __init__(
        self,
        taxonomy_mapper: TaxonomyMapper | None = None,
        archetype_loader: ArchetypeLoader | None = None,
        nemotron_prompts: NemotronPrompts | None = None,
        policy_registry: PolicyRegistry | None = None,
    ):
        self.taxonomy_mapper = taxonomy_mapper or TaxonomyMapper()
        self.archetype_loader = archetype_loader or ArchetypeLoader()
        self.nemotron_prompts = nemotron_prompts or NemotronPrompts()
        self.policy_registry = policy_registry or PolicyRegistry()

    def generate(self, request: PolicyGenerationRequest) -> PolicyGenerationResult:
        """Generate complete policy artifacts from rough requirements."""
        # 1. Determine starting archetype
        archetype = self.archetype_loader.get_archetype(request.deployment_context)
        assumptions = [f"Starting archetype: {archetype.name}"]

        # 2. Map rough words to V2 categories
        mapping_result = self.taxonomy_mapper.map_rough_words(
            request.rough_words, archetype.categories
        )
        assumptions.append(
            f"Taxonomy mode: {mapping_result.mode.value} "
            f"({mapping_result.match_ratio:.0%} rough words mapped)"
        )

        # 3. Build category list (archetype + mapped + custom)
        categories = self._build_categories(
            archetype,
            mapping_result,
            request.custom_categories,
            request.severity_overrides,
            request.allow_list,
        )

        # 4. Enforce non-negotiable floor
        assumptions.extend(self._enforce_non_negotiable_floor(categories))

        # 5. Generate JSON taxonomy (validated against schema)
        json_taxonomy = self._build_json_taxonomy(
            request, categories, mapping_result.mode, assumptions
        )
        self.policy_registry.validate(json_taxonomy)

        # 6. Generate Markdown policy
        markdown = self._render_markdown(json_taxonomy, request)

        # 7. Generate system prompt for target model
        system_prompt = self.nemotron_prompts.render(
            target_model=request.target_model,
            policy=json_taxonomy,
            mode=PromptMode.NO_THINK,  # Default; configurable
        )

        return PolicyGenerationResult(
            markdown=markdown,
            json_taxonomy=json_taxonomy,
            system_prompt=system_prompt,
            assumptions=assumptions,
            taxonomy_mode=mapping_result.mode,
            warnings=mapping_result.warnings,
        )

    def _build_categories(
        self,
        archetype: Archetype,
        mapping_result: MappingResult,
        custom_categories: list[dict[str, Any]],
        severity_overrides: dict[str, str],
        allow_list: list[str],
    ) -> list[dict[str, Any]]:
        """Build complete category list from archetype, mapping, and custom."""
        categories = []
        custom_counter = 23  # Start custom categories at S23

        # Add archetype categories (canonical V2)
        for cat_name in archetype.categories:
            v2_cat = self._get_v2_category(cat_name)
            if v2_cat:
                cat = self._v2_to_policy_category(v2_cat)
                # Apply severity override if specified
                if cat_name in severity_overrides:
                    cat["severity"] = severity_overrides[cat_name]
                # Apply archetype severity override
                elif cat_name in archetype.severity_overrides:
                    cat["severity"] = archetype.severity_overrides[cat_name]
                categories.append(cat)

        # Add mapped categories from rough words (if not already in archetype)
        for cat_name, _matched_synonyms in mapping_result.matched_categories.items():
            if not any(c["name"] == cat_name for c in categories):
                v2_cat = self._get_v2_category(cat_name)
                if v2_cat:
                    cat = self._v2_to_policy_category(v2_cat)
                    if cat_name in severity_overrides:
                        cat["severity"] = severity_overrides[cat_name]
                    categories.append(cat)

        # Add archetype custom categories
        for custom_cat in archetype.custom_categories:
            cat = custom_cat.copy()
            cat["name"] = custom_cat["name"]
            cat["display_name"] = custom_cat["display_name"]
            cat["definition"] = custom_cat["definition"]
            cat["severity"] = custom_cat["severity"]
            cat["custom"] = True
            cat["aegis_parent"] = ""
            cat.setdefault("in_scope", [custom_cat["definition"]])
            cat.setdefault("out_of_scope", ["None specified"])
            cat.setdefault("examples_safe", ["Safe example for " + custom_cat["display_name"]])
            cat.setdefault("examples_unsafe", ["Unsafe example for " + custom_cat["display_name"]])
            cat.setdefault("edge_cases", [])
            cat.setdefault("modality_notes", "")
            categories.append(cat)

        # Add user-provided custom categories
        for custom_cat in custom_categories:
            cat = custom_cat.copy()
            cat.setdefault("name", f"custom_{custom_counter}")
            cat.setdefault("display_name", cat["name"].replace("_", " ").title())
            cat.setdefault("custom", True)
            cat.setdefault("aegis_parent", "")
            cat.setdefault("in_scope", [cat.get("definition", "Custom category")])
            cat.setdefault("out_of_scope", ["None specified"])
            cat.setdefault(
                "examples_safe", ["Safe example for " + cat.get("display_name", cat["name"])]
            )
            cat.setdefault(
                "examples_unsafe", ["Unsafe example for " + cat.get("display_name", cat["name"])]
            )
            cat.setdefault("edge_cases", [])
            cat.setdefault("modality_notes", "")
            if "severity" not in cat:
                cat["severity"] = "S2"
            categories.append(cat)
            custom_counter += 1

        return categories

    def _get_v2_category(self, name: str) -> dict[str, Any] | None:
        """Get V2 category definition by name."""
        taxonomy_path = (
            Path(__file__).parent.parent.parent / "config" / "policies" / "v2_taxonomy.yaml"
        )
        with open(taxonomy_path) as f:
            taxonomy = yaml.safe_load(f)
        for cat in taxonomy.get("categories", []):
            if cat["name"] == name:
                return cat
        return None

    def _v2_to_policy_category(self, v2_cat: dict[str, Any]) -> dict[str, Any]:
        """Convert V2 taxonomy category to policy category format."""
        return {
            "name": v2_cat["name"],
            "display_name": v2_cat["display_name"],
            "definition": v2_cat["definition"],
            "severity": v2_cat["default_severity"],
            "custom": False,
            "aegis_parent": v2_cat.get("aegis_label", ""),
            "in_scope": [v2_cat["definition"]],
            "out_of_scope": v2_cat.get("out_of_scope", ["None specified"]),
            "examples_safe": [f"Safe example for {v2_cat['display_name']}"],
            "examples_unsafe": [f"Unsafe example for {v2_cat['display_name']}"],
            "edge_cases": [],
            "modality_notes": "",
        }

    def _enforce_non_negotiable_floor(self, categories: list[dict[str, Any]]) -> list[str]:
        """Enforce non-negotiable floor (S7 Sexual Minor can never be carved out)."""
        assumptions = []
        for cat in categories:
            if cat["name"] == "sexual_minor" and cat.get("severity") != "S4":
                assumptions.append(
                    "NON-NEGOTIABLE FLOOR: S7 Sexual (Minor) severity forced to S4 "
                    "(cannot be relaxed, carved out, or allowed)"
                )
                cat["severity"] = "S4"
            if cat["name"] == "sexual_minor" and cat.get("custom") is True:
                assumptions.append(
                    "NON-NEGOTIABLE FLOOR: Custom category cannot override S7 Sexual (Minor). "
                    "Custom sexual_minor category rejected."
                )
                # Remove the custom category
                categories.remove(cat)
        return assumptions

    def _build_json_taxonomy(
        self,
        request: PolicyGenerationRequest,
        categories: list[dict[str, Any]],
        taxonomy_mode: MappingMode,
        assumptions: list[str],
    ) -> dict[str, Any]:
        """Build JSON taxonomy from categories."""
        target_model_map = {
            TargetModel.NCS_REASONING_4B: ["ncs-reasoning"],
            TargetModel.NCS_VL: ["ncs-vl"],
            TargetModel.NEMO_GUARDRAILS: ["nemo-guardrails"],
        }

        return {
            "policy_name": f"generated-policy-{date.today().isoformat()}",
            "version": "0.1.0",
            "date": date.today().isoformat(),
            "owner": "nemotron-policy-generator",
            "target_models": target_model_map.get(request.target_model, ["ncs-reasoning"]),
            "use_cases": ["runtime_guardrails"],
            "taxonomy_mode": taxonomy_mode.value,
            "severity_model": "graded_s0_s4",
            "assumptions": assumptions,
            "allow_list": request.allow_list,
            "response_guidance": {
                "S0": "Proceed normally",
                "S1": "Proceed with caveat",
                "S2": "Refuse + brief explanation",
                "S3": "Refuse + redirect to resources where applicable",
                "S4": "Refuse + log for human review",
            },
            "jurisdiction_notes": request.jurisdiction or f"Default: {request.locale}",
            "calibration_notes": "Default calibration; adjust based on deployment requirements.",
            "cot_rules": "Standard CoT rules apply for reasoning models.",
            "categories": categories,
        }

    def _render_markdown(self, policy: dict[str, Any], request: PolicyGenerationRequest) -> str:
        """Render Markdown policy from JSON taxonomy."""
        template_path = (
            Path(__file__).parent.parent.parent / "config" / "policies" / "policy_md_template.md"
        )
        template = template_path.read_text()

        # Build category blocks
        category_blocks = []
        for i, cat in enumerate(policy["categories"], 1):
            block = f"""### {i}. {cat["display_name"]} (`{cat["name"]}`)

**Severity:** {cat["severity"]} | **Custom:** {cat["custom"]}

**Definition:** {cat["definition"]}

**In scope:**
"""
            for item in cat.get("in_scope", []):
                block += f"- {item}\n"

            block += "\n**Out of scope (carve-outs):**\n"
            for item in cat.get("out_of_scope", []):
                block += f"- {item}\n"

            block += "\n**Safe examples (should NOT trigger):**\n"
            for j, ex in enumerate(cat.get("examples_safe", []), 1):
                block += f"{j}. {ex}\n"

            block += "\n**Unsafe examples (clear violations):**\n"
            for j, ex in enumerate(cat.get("examples_unsafe", []), 1):
                block += f"{j}. {ex}\n"

            if cat.get("edge_cases"):
                block += "\n**Edge cases:**\n"
                for ec in cat["edge_cases"]:
                    block += f"- *{ec['case']}* — Resolution: {ec['resolution']}. Reasoning: {ec['reasoning']}.\n"

            block += f"\n**Modality notes:** {cat.get('modality_notes', 'N/A')}\n"
            category_blocks.append(block)

        # Build assumptions block
        assumptions_block = "\n".join(f"- {a}" for a in policy["assumptions"])

        # Build allow list
        allow_list = "\n".join(f"- {item}" for item in policy.get("allow_list", [])) or "None."

        # Build response guidance
        response_guidance = ""
        for sev in ["S0", "S1", "S2", "S3", "S4"]:
            response_guidance += f"- {sev}: {policy['response_guidance'].get(sev, 'N/A')}\n"

        return (
            template.replace("{{POLICY_NAME}}", policy["policy_name"])
            .replace("{{VERSION}}", policy["version"])
            .replace("{{DATE}}", policy["date"])
            .replace("{{OWNER}}", policy["owner"])
            .replace("{{TARGET_MODELS}}", ", ".join(policy["target_models"]))
            .replace("{{USE_CASES}}", ", ".join(policy["use_cases"]))
            .replace("{{TAXONOMY_MODE}}", policy["taxonomy_mode"])
            .replace("{{ASSUMPTIONS_BLOCK}}", assumptions_block)
            .replace("{{ALLOW_LIST}}", allow_list)
            .replace("{{RESPONSE_GUIDANCE}}", response_guidance)
            .replace("{{JURISDICTION}}", policy["jurisdiction_notes"])
            .replace("{{CALIBRATION}}", policy["calibration_notes"])
            .replace("{{COT_RULES}}", policy["cot_rules"])
            .replace("{{CATEGORY_BLOCKS}}", "\n\n---\n\n".join(category_blocks))
        )
