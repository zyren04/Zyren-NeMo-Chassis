"""Nemotron System Prompt Templates — Dual-target prompt rendering."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from string import Template
from typing import Any


class TargetModel(StrEnum):
    NCS_REASONING_4B = "ncs-reasoning-4b"  # Nemotron-Content-Safety-Reasoning-4B
    NCS_VL = "ncs-vl"  # Nemotron-3-Content-Safety (multimodal)
    NEMO_GUARDRAILS = "nemo-guardrails"  # NeMo Guardrails integration


class PromptMode(StrEnum):
    THINK = "think"  # /think — reasoning trace
    NO_THINK = "no_think"  # /no_think — low latency
    CATEGORIES = "categories"  # /categories — emit category list (Nemotron-3)
    NO_CATEGORIES = "no_categories"  # /no_categories — binary only (Nemotron-3)


class NemotronPrompts:
    """Renders Nemotron system prompts from policy taxonomy."""

    def __init__(self, template_path: Path | None = None):
        self.template_path = (
            template_path
            or Path(__file__).parent.parent.parent
            / "config"
            / "policies"
            / "nemotron_system_prompt_template.txt"
        )
        self._templates: dict[str, str] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        content = self.template_path.read_text()
        # Parse sections marked with === PATTERN X ===
        self._templates = {
            "ncs_reasoning_vanilla": self._extract_pattern(content, "PATTERN A"),
            "ncs_reasoning_custom": self._extract_pattern(content, "PATTERN B"),
            "ncs_reasoning_topic": self._extract_pattern(content, "PATTERN C"),
            "nemotron3_vanilla": self._extract_pattern(content, "PATTERN D"),
            "nemotron3_custom": self._extract_pattern(content, "PATTERN E"),
            "nemotron3_multimodal": self._extract_pattern(content, "PATTERN F"),
        }

    def _extract_pattern(self, content: str, pattern_name: str) -> str:
        """Extract a pattern section from the template file."""
        # Find the pattern section
        start_marker = f"=== {pattern_name} "
        start_idx = content.find(start_marker)
        if start_idx == -1:
            return ""

        # Find the next pattern or end of file
        next_pattern_idx = content.find("=== PATTERN ", start_idx + len(start_marker))
        if next_pattern_idx == -1:
            next_pattern_idx = content.find("### ==", start_idx + len(start_marker))
        if next_pattern_idx == -1:
            next_pattern_idx = len(content)

        section = content[start_idx:next_pattern_idx]
        # Remove the pattern header line
        lines = section.split("\n")
        if lines and lines[0].startswith("==="):
            lines = lines[1:]
        return "\n".join(lines).strip()

    def render(
        self,
        target_model: TargetModel,
        policy: dict[str, Any],
        mode: PromptMode = PromptMode.NO_THINK,
        categories_mode: PromptMode = PromptMode.NO_CATEGORIES,
    ) -> str:
        """Render system prompt for target model and mode."""

        # Build taxonomy block
        taxonomy_block = self._build_taxonomy_block(policy, target_model)
        allow_list_block = self._build_allow_list_block(policy)

        if target_model == TargetModel.NCS_REASONING_4B:
            template = self._templates["ncs_reasoning_custom"]  # Always custom for BYO
            if not template:
                template = self._get_default_ncs_reasoning_template()
            return Template(template).safe_substitute(
                TAXONOMY_BLOCK=taxonomy_block,
                ALLOW_LIST_BLOCK=allow_list_block,
                MODE=f"/{mode.value}" if mode in (PromptMode.THINK, PromptMode.NO_THINK) else "",
            )

        elif target_model in (TargetModel.NCS_VL, TargetModel.NEMO_GUARDRAILS):
            # Nemotron-3 uses chat template with category names
            template = self._templates["nemotron3_custom"]
            if not template:
                template = self._get_default_nemotron3_template()
            return Template(template).safe_substitute(
                CATEGORY_NAMES_BLOCK=self._build_category_names_block(policy),
                ALLOW_LIST_BLOCK=allow_list_block,
                REQUEST_CATEGORIES=f"/{categories_mode.value}",
                REQUEST_REASONING=f"/{mode.value}"
                if mode in (PromptMode.THINK, PromptMode.NO_THINK)
                else "",
            )

        raise ValueError(f"Unsupported target model: {target_model}")

    def _get_default_ncs_reasoning_template(self) -> str:
        """Default template for NCS Reasoning 4B custom policy."""
        return """You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine 1) whether the human user's request is harmful, 2) whether the AI assistant's response is a refusal, and 3) whether the AI assistant's response is harmful.

Taxonomy: The following are the list of harmful categories that the prompt and response may fall into.

${TAXONOMY_BLOCK}

Allow-list (explicitly permitted even if adjacent to a category above):
${ALLOW_LIST_BLOCK}

Human user:
${USER_PROMPT}

AI assistant:
${AI_RESPONSE}

Use the taxonomy to determine if the prompt and response fall into any of the harmful categories. Output your answer in the following format:

"Prompt harm: <harmful/unharmful>
Response harm: <harmful/unharmful>"

Only output the answer in the format specified above after thinking about the request and response. If there is no response, output "Response harm: <harmful/unharmful>" as None. ${MODE}"""

    def _get_default_nemotron3_template(self) -> str:
        """Default template for Nemotron-3 custom policy."""
        return '''messages = [
    {"role": "system", "content": """You are a safety classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine 1) whether the human user's request is safe or unsafe, 2) whether the AI assistant's response is safe or unsafe, and 3) if unsafe, which safety categories apply.

Safety Categories:
${CATEGORY_NAMES_BLOCK}

Allow-list (explicitly permitted even if adjacent to a category above):
${ALLOW_LIST_BLOCK}"""},
    {"role": "user", "content": "${USER_PROMPT}"},
    {"role": "assistant", "content": "${AI_RESPONSE}"}
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
    request_categories="${REQUEST_CATEGORIES}"
)'''

    def _build_taxonomy_block(self, policy: dict, target: TargetModel) -> str:
        """Build taxonomy block with S1..Sn labels (Reasoning-4B) or names (Nemotron-3)."""
        lines = []
        for cat in policy.get("categories", []):
            if target == TargetModel.NCS_REASONING_4B:
                # Use Sn label if canonical, else generate from custom index
                label = cat.get("aegis_parent", cat["name"])
                lines.append(f"{label}: {cat['display_name']}.")
            else:
                lines.append(f"{cat['display_name']}: {cat['definition']}")
        return "\n".join(lines)

    def _build_category_names_block(self, policy: dict) -> str:
        """Build comma-separated category names for Nemotron-3."""
        names = [cat["display_name"] for cat in policy.get("categories", [])]
        return ", ".join(names)

    def _build_allow_list_block(self, policy: dict) -> str:
        allow_list = policy.get("allow_list", [])
        if not allow_list:
            return "None."
        return "\n".join(f"- {item}" for item in allow_list)
