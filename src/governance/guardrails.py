# ruff: noqa: I001
"""
Deterministic Guardrails Layer
Integration wrapper for nemoguardrails using RunnableRails.
Extended with Nemotron Content Safety Policy support.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# nemoguardrails doesn't have stubs - use type: ignore for imports
from nemoguardrails import RailsConfig  # type: ignore[import-untyped]
from nemoguardrails.actions import action  # type: ignore[import-untyped]
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails  # type: ignore[import-untyped]

from ..contracts.base_contracts import SignalType, ValidationDecision
from ..state.state_schema import BaseState
from .deprecation_registry import DeprecationRegistry, get_deprecation_registry
from .policy_registry import PolicyRegistry
from .policy_generator import PolicyGenerator, PolicyGenerationRequest
from .nemotron_prompts import TargetModel, PromptMode
from .archetypes import DeploymentContext


@dataclass
class ValidationResult:
    """Result of guardrails validation."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    suggested_action: SignalType
    validator_name: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        # Ensure proper type initialization
        self.errors = self.errors or []
        self.warnings = self.warnings or []
        self.metadata = self.metadata or {}


# Module-level action registration to avoid duplicate registration
# when multiple GuardrailsEngine instances are created
_action_registry_initialized = False


def _register_colang_actions() -> None:
    """Register all Colang user-defined functions as Python actions (module-level, runs once)."""
    global _action_registry_initialized
    if _action_registry_initialized:
        return

    # Use type: ignore to suppress mypy errors from untyped nemoguardrails decorators
    @action(name="get_iteration_count")  # type: ignore[untyped-decorator]
    def get_iteration_count() -> int:
        # This will be called with context from the rails runtime
        return 0  # Placeholder - actual value comes from context

    @action(name="get_user_input")  # type: ignore[untyped-decorator]
    def get_user_input() -> dict[str, Any] | None:
        return None  # Placeholder

    @action(name="has_required_fields")  # type: ignore[untyped-decorator]
    def has_required_fields(input_data: dict[str, Any], fields: list[str]) -> bool:
        return all(field in input_data for field in fields)

    @action(name="is_valid_json")  # type: ignore[untyped-decorator]
    def is_valid_json(text: str) -> bool:
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    @action(name="count_tokens")  # type: ignore[untyped-decorator]
    def count_tokens(text: str) -> int:
        # Rough estimate: ~4 chars per token
        return len(text) // 4

    @action(name="get_bot_response")  # type: ignore[untyped-decorator]
    def get_bot_response() -> str:
        return ""  # Placeholder - actual value comes from context

    @action(name="get_current_iteration")  # type: ignore[untyped-decorator]
    def get_current_iteration() -> int:
        return 0  # Placeholder

    @action(name="get_max_iterations")  # type: ignore[untyped-decorator]
    def get_max_iterations() -> int:
        # This will be overridden by context in the rails runtime
        return 100  # Default

    @action(name="get_previous_node")  # type: ignore[untyped-decorator]
    def get_previous_node() -> str:
        return ""  # Placeholder

    @action(name="get_next_node")  # type: ignore[untyped-decorator]
    def get_next_node() -> str:
        return ""  # Placeholder

    @action(name="get_valid_transitions")  # type: ignore[untyped-decorator]
    def get_valid_transitions() -> list[dict[str, list[str]]]:
        # Return as list of single-key dicts to match Colang expectation
        return [
            {"entry": ["processor", "validator", "end"]},
            {"processor": ["validator", "processor", "end"]},
            {"validator": ["processor", "end"]},
            {"end": []},
        ]

    @action(name="is_valid_transition")  # type: ignore[untyped-decorator]
    def is_valid_transition(
        from_node: str, to_node: str, valid_transitions: list[dict[str, list[str]]]
    ) -> bool:
        # Flatten the list of dicts to a single dict for lookup
        flattened: dict[str, list[str]] = {}
        for d in valid_transitions:
            flattened.update(d)
        allowed = flattened.get(from_node, [])
        return to_node in allowed

    @action(name="get_execution_time_seconds")  # type: ignore[untyped-decorator]
    def get_execution_time_seconds() -> float:
        return 0.0  # Placeholder

    @action(name="get_memory_usage_mb")  # type: ignore[untyped-decorator]
    def get_memory_usage_mb() -> float:
        return 0.0  # Placeholder

    @action(name="execute_halt")  # type: ignore[untyped-decorator]
    def execute_halt(reason: str) -> dict[str, Any]:
        """Execute halt action - returns a decision dict for the rails runtime."""
        return {
            "type": "halt",
            "reason": reason,
            "valid": False,
            "errors": [reason],
            "warnings": [],
            "suggested_action": "HALT",
            "validator_name": "guardrails",
        }

    _action_registry_initialized = True


class GuardrailsEngine:
    """
    Deterministic guardrails engine wrapping nemoguardrails RunnableRails.

    Features:
    - Loads config from config/rails/config.yml and config/rails/rails.co
    - Enforces iteration bounds (max loop ceiling)
    - Execution safety policies
    - Integration with BaseState for context
    - Registers all Colang user functions as Python actions (module-level)
    - Deprecated NIM detection via DeprecationRegistry
    - Nemotron Content Safety Policy integration
    """

    def __init__(
        self,
        config_path: str | None = None,
        rails_path: str | None = None,
        max_iterations: int = 100,
        max_tokens_per_call: int = 8192,
        max_total_tokens: int = 100000,
        deprecation_registry: DeprecationRegistry | None = None,
    ):
        self.config_path = config_path or "config/rails/config.yml"
        self.rails_path = rails_path or "config/rails/rails.co"
        self.max_iterations = max_iterations
        self.max_tokens_per_call = max_tokens_per_call
        self.max_total_tokens = max_total_tokens
        self.deprecation_registry = deprecation_registry or get_deprecation_registry()

        # Load configuration
        self.config = self._load_config()

        # Register all Colang user functions as Python actions (module-level, runs once)
        _register_colang_actions()

        # Create RunnableRails with the enhanced config
        self.rails = RunnableRails(self.config)

        # Nemotron Policy Integration
        self.policy_registry = PolicyRegistry()
        self.policy_generator = PolicyGenerator(policy_registry=self.policy_registry)
        self._active_policy: dict[str, Any] | None = None

    def _load_config(self) -> RailsConfig:
        """Load guardrails configuration from YAML and Colang files."""
        config = RailsConfig.from_path(self.config_path)
        return config

    async def validate_input(
        self,
        input_data: dict[str, Any],
        state: BaseState | None = None,
    ) -> ValidationResult:
        """
        Validate input before node execution.

        Args:
            input_data: Input payload to validate
            state: Current execution state for context

        Returns:
            ValidationResult with validation decision
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check iteration limit
        if state and state.iteration >= self.max_iterations:
            errors.append(f"Iteration limit exceeded: {state.iteration} >= {self.max_iterations}")

        # Check token budget
        if state and state.total_tokens_consumed >= self.max_total_tokens:
            errors.append(
                f"Token budget exceeded: {state.total_tokens_consumed} >= {self.max_total_tokens}"
            )

        # Validate required fields
        if "execution_id" not in input_data:
            errors.append("Missing required field: execution_id")

        if "payload" not in input_data:
            errors.append("Missing required field: payload")

        # Check for forbidden patterns in string inputs
        forbidden_patterns = ["rm -rf", "sudo ", "chmod 777", "__import__", "eval(", "exec("]
        input_str = str(input_data)
        for pattern in forbidden_patterns:
            if pattern in input_str:
                errors.append(f"Forbidden pattern detected: {pattern}")

        # Check for deprecated NIM references
        deprecation_errors, deprecation_warnings = self._check_deprecated_nims(input_str)
        errors.extend(deprecation_errors)
        warnings.extend(deprecation_warnings)

        valid = len(errors) == 0
        suggested_action = SignalType.HALT if not valid else SignalType.CONTINUE

        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            suggested_action=suggested_action,
            validator_name="input_validator",
            metadata={"iteration": state.iteration if state else 0},
        )

    def _check_deprecated_nims(self, text: str) -> tuple[list[str], list[str]]:
        """Check text for deprecated NIM references.

        Returns:
            Tuple of (errors, warnings) - errors for deprecated NIMs that should halt,
            warnings for deprecated NIMs that are flagged but allowed.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Common NIM identifier patterns to search for
        # Matches patterns like: nvidia/model-name, org/model-name, nvcr.io/nim/org/model:tag
        nim_patterns = [
            r"nvcr\.io/nim/([a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+)(?::[a-zA-Z0-9._-]+)?",
            r"\b([a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+)\b",  # org/model pattern
        ]

        found_nims = set()
        for pattern in nim_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                if match:
                    found_nims.add(match)

        for nim in found_nims:
            if self.deprecation_registry.is_deprecated(nim):
                matched_patterns = self.deprecation_registry.check(nim)
                for pattern in matched_patterns:
                    errors.append(f"Deprecated NIM detected: {nim} (matches: {pattern})")

        return errors, warnings

    async def validate_output(
        self,
        output_data: dict[str, Any],
        state: BaseState | None = None,
    ) -> ValidationResult:
        """
        Validate output after node execution.

        Args:
            output_data: Output payload to validate
            state: Current execution state for context

        Returns:
            ValidationResult with validation decision
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check token limits in output
        output_str = str(output_data)
        # Rough token estimate: ~4 chars per token
        estimated_tokens = len(output_str) // 4

        if estimated_tokens > self.max_tokens_per_call:
            errors.append(
                f"Output token estimate exceeds limit: {estimated_tokens} > {self.max_tokens_per_call}"
            )

        # Validate JSON serializability
        try:
            import json

            json.dumps(output_data)
        except (TypeError, ValueError) as e:
            errors.append(f"Output not JSON serializable: {e}")

        # Check for deprecated NIM references in output
        deprecation_errors, deprecation_warnings = self._check_deprecated_nims(output_str)
        errors.extend(deprecation_errors)
        warnings.extend(deprecation_warnings)

        valid = len(errors) == 0
        suggested_action = SignalType.HALT if not valid else SignalType.CONTINUE

        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            suggested_action=suggested_action,
            validator_name="output_validator",
            metadata={"estimated_tokens": estimated_tokens},
        )

    async def validate_transition(
        self,
        from_node: str,
        to_node: str,
        state: BaseState,
    ) -> ValidationResult:
        """
        Validate node transition.

        Args:
            from_node: Source node name
            to_node: Target node name
            state: Current execution state

        Returns:
            ValidationResult with validation decision
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Define valid transitions (could be loaded from config)
        valid_transitions = {
            "entry": ["processor", "validator", "end"],
            "processor": ["validator", "processor", "end"],
            "validator": ["processor", "end"],
            "end": [],
        }

        allowed = valid_transitions.get(from_node, [])
        if to_node not in allowed:
            errors.append(f"Invalid transition: {from_node} -> {to_node} (allowed: {allowed})")

        valid = len(errors) == 0
        suggested_action = SignalType.HALT if not valid else SignalType.CONTINUE

        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            suggested_action=suggested_action,
            validator_name="transition_validator",
            metadata={"from_node": from_node, "to_node": to_node},
        )

    async def validate_resource_limits(
        self,
        state: BaseState,
    ) -> ValidationResult:
        """
        Validate resource limits (time, memory).

        Args:
            state: Current execution state

        Returns:
            ValidationResult with validation decision
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check execution time (would need actual timing)
        # This is a placeholder for actual implementation

        valid = len(errors) == 0
        suggested_action = SignalType.HALT if not valid else SignalType.CONTINUE

        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            suggested_action=suggested_action,
            validator_name="resource_validator",
            metadata={},
        )

    async def run_full_validation(
        self,
        input_data: dict[str, Any],
        output_data: dict[str, Any] | None = None,
        state: BaseState | None = None,
        from_node: str | None = None,
        to_node: str | None = None,
    ) -> ValidationDecision:
        """
        Run all validations and combine results.

        Args:
            input_data: Input to validate
            output_data: Optional output to validate
            state: Current execution state
            from_node: Source node (for transition validation)
            to_node: Target node (for transition validation)

        Returns:
            Combined ValidationDecision
        """
        # Run input validation
        input_result = await self.validate_input(input_data, state)

        # Run output validation if provided
        output_result = None
        if output_data:
            output_result = await self.validate_output(output_data, state)

        # Run transition validation if nodes provided
        transition_result = None
        if from_node and to_node and state:
            transition_result = await self.validate_transition(from_node, to_node, state)

        # Run resource validation
        resource_result = None
        if state:
            resource_result = await self.validate_resource_limits(state)

        # Combine all decisions
        decisions = [input_result]
        if output_result:
            decisions.append(output_result)
        if transition_result:
            decisions.append(transition_result)
        if resource_result:
            decisions.append(resource_result)

        return self._combine_decisions(decisions)

    async def run_rails_validation(
        self,
        messages: list[dict[str, Any]],
        state: BaseState | None = None,
    ) -> ValidationDecision:
        """
        Run the nemoguardrails RunnableRails validation pipeline.

        Args:
            messages: List of messages to validate through the rails
            state: Current execution state for context

        Returns:
            ValidationDecision from the rails pipeline
        """
        # Prepare context for rails
        context = {
            "iteration": state.iteration if state else 0,
            "max_iterations": self.max_iterations,
            "max_tokens_per_call": self.max_tokens_per_call,
            "max_total_tokens": self.max_total_tokens,
            "total_tokens_consumed": state.total_tokens_consumed if state else 0,
        }

        try:
            # Run the rails validation
            result = await self.rails.ainvoke(
                input=messages, config={"configurable": {"context": context}}
            )

            # Parse result into ValidationDecision
            if isinstance(result, dict):
                return ValidationDecision(
                    valid=result.get("valid", True),
                    errors=result.get("errors", []),
                    warnings=result.get("warnings", []),
                    suggested_action=SignalType(result.get("suggested_action", "CONTINUE")),
                    validator_name="nemoguardrails_rails",
                    metadata=result.get("metadata", {}),
                )
            else:
                return ValidationDecision(
                    valid=True,
                    errors=[],
                    warnings=[],
                    suggested_action=SignalType.CONTINUE,
                    validator_name="nemoguardrails_rails",
                    metadata={},
                )
        except Exception as e:
            return ValidationDecision(
                valid=False,
                errors=[f"Rails validation error: {str(e)}"],
                warnings=[],
                suggested_action=SignalType.HALT,
                validator_name="nemoguardrails_rails",
                metadata={"error": str(e)},
            )

    def _combine_decisions(self, results: list[ValidationResult]) -> ValidationDecision:
        """Combine multiple validation results into a single decision."""
        all_errors = []
        all_warnings = []
        all_valid = True
        halt_requested = False

        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            if not result.valid:
                all_valid = False
            if result.suggested_action == SignalType.HALT:
                halt_requested = True

        suggested_action = SignalType.HALT if halt_requested else SignalType.CONTINUE

        return ValidationDecision(
            valid=all_valid,
            errors=all_errors,
            warnings=all_warnings,
            suggested_action=suggested_action,
            validator_name="combined",
            metadata={"validators_run": len(results)},
        )

    # ============================================================
    # Nemotron Content Safety Policy Integration
    # ============================================================

    def load_nemotron_policy(
        self,
        policy_name: str,
        target_model: TargetModel = TargetModel.NCS_REASONING_4B,
        mode: PromptMode = PromptMode.NO_THINK,
    ) -> str:
        """Load a Nemotron policy and return the rendered system prompt."""
        policy = self.policy_registry.load_policy(policy_name)
        self._active_policy = policy
        return self.policy_generator.nemotron_prompts.render(
            target_model=target_model, policy=policy, mode=mode
        )

    def generate_policy_from_rough(
        self,
        rough_words: str,
        target_model: TargetModel = TargetModel.NCS_REASONING_4B,
        deployment_context: DeploymentContext | None = None,
        **kwargs,
    ) -> PolicyGenerationResult:
        """Generate a new policy from rough requirements."""
        request = PolicyGenerationRequest(
            rough_words=rough_words,
            target_model=target_model,
            deployment_context=deployment_context,
            **kwargs,
        )
        result = self.policy_generator.generate(request)
        self._active_policy = result.json_taxonomy
        return result

    def validate_with_policy(self, state: BaseState) -> ValidationDecision:
        """Enhanced validation using active Nemotron policy."""
        # Run existing validations
        base_decision = self.validate(state)

        # If policy loaded, add content-safety checks
        if self._active_policy:
            policy_decision = self._validate_content_safety(state)
            return self._combine_decisions([base_decision, policy_decision])

        return base_decision

    def _validate_content_safety(self, state: BaseState) -> ValidationDecision:
        """Validate against active Nemotron policy categories."""
        # This would integrate with NIM API for actual Nemotron inference
        # For now, returns a placeholder that can be extended
        return ValidationDecision(
            valid=True,
            errors=[],
            warnings=["Nemotron policy validation requires NIM API integration"],
            suggested_action=SignalType.CONTINUE,
            validator_name="nemotron_content_safety",
            metadata={
                "policy": self._active_policy.get("policy_name") if self._active_policy else None
            },
        )

    def to_decision(self, result: ValidationResult) -> ValidationDecision:
        """Convert ValidationResult to ValidationDecision."""
        return ValidationDecision(
            valid=result.valid,
            errors=result.errors,
            warnings=result.warnings,
            suggested_action=result.suggested_action,
            validator_name=result.validator_name,
            metadata=result.metadata,
        )


# Convenience functions
_default_engine: GuardrailsEngine | None = None


def get_guardrails_engine(
    config_path: str | None = None,
    rails_path: str | None = None,
    max_iterations: int = 100,
    max_tokens_per_call: int = 8192,
    max_total_tokens: int = 100000,
) -> GuardrailsEngine:
    """Get or create the default guardrails engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = GuardrailsEngine(
            config_path=config_path,
            rails_path=rails_path,
            max_iterations=max_iterations,
            max_tokens_per_call=max_tokens_per_call,
            max_total_tokens=max_total_tokens,
        )
    return _default_engine


def set_guardrails_engine(engine: GuardrailsEngine) -> None:
    """Set the default guardrails engine (useful for testing)."""
    global _default_engine
    _default_engine = engine


__all__ = [
    "GuardrailsEngine",
    "ValidationResult",
    "get_guardrails_engine",
    "set_guardrails_engine",
]
