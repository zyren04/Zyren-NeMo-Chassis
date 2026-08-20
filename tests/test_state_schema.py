"""
Tests for state.state_schema module
"""

from src.contracts.base_contracts import (
    ArtifactItem,
    ExecutionSignal,
    NodePayload,
    SignalType,
    ValidationDecision,
)
from src.state.state_schema import BaseState, from_state_dict, to_state_dict


class TestBaseState:
    def test_create_default(self):
        state = BaseState()
        assert state.execution_id is not None
        assert state.iteration == 0
        assert state.current_node is None
        assert state.max_iterations == 100
        assert state.max_tokens_per_execution == 100000

    def test_create_with_custom_limits(self):
        state = BaseState(max_iterations=50, max_tokens_per_execution=50000)
        assert state.max_iterations == 50
        assert state.max_tokens_per_execution == 50000

    def test_increment_iteration(self):
        state = BaseState()
        new_state = state.increment_iteration()
        assert new_state.iteration == 1
        assert state.iteration == 0  # Original unchanged

    def test_set_current_node(self):
        state = BaseState()
        new_state = state.set_current_node("node1")
        assert new_state.current_node == "node1"
        assert new_state.previous_node is None

        newer_state = new_state.set_current_node("node2")
        assert newer_state.current_node == "node2"
        assert newer_state.previous_node == "node1"

    def test_add_artifact(self):
        state = BaseState()
        artifact = ArtifactItem.create(
            artifact_type="code",
            content=b"test",
            created_by_node="generator",
        )
        new_state = state.add_artifact(artifact)

        assert artifact.artifact_id in new_state.artifacts
        assert "code" in new_state.artifact_index
        assert artifact.artifact_id in new_state.artifact_index["code"]

    def test_get_artifacts_by_type(self):
        state = BaseState()
        artifact1 = ArtifactItem.create(artifact_type="code", content=b"a", created_by_node="n1")
        artifact2 = ArtifactItem.create(artifact_type="code", content=b"b", created_by_node="n2")
        artifact3 = ArtifactItem.create(artifact_type="test", content=b"c", created_by_node="n3")

        state = state.add_artifact(artifact1).add_artifact(artifact2).add_artifact(artifact3)

        code_artifacts = state.get_artifacts_by_type("code")
        assert len(code_artifacts) == 2

        test_artifacts = state.get_artifacts_by_type("test")
        assert len(test_artifacts) == 1

    def test_enqueue_dequeue_payload(self):
        state = BaseState()
        payload = NodePayload(source_node="a", target_node="b", data={"x": 1})

        state = state.enqueue_payload(payload)
        assert len(state.pending_payloads) == 1

        dequeued, new_state = state.dequeue_payload()
        assert dequeued is not None
        assert dequeued.payload_id == payload.payload_id
        assert len(new_state.pending_payloads) == 0

        # Dequeue from empty
        none_payload, _ = new_state.dequeue_payload()
        assert none_payload is None

    def test_payload_deduplication(self):
        state = BaseState()
        payload = NodePayload(source_node="a", target_node="b")

        state = state.enqueue_payload(payload)
        state = state.enqueue_payload(payload)  # Same ID

        assert len(state.pending_payloads) == 1

    def test_emit_signal(self):
        state = BaseState()
        signal = ExecutionSignal(
            signal_type=SignalType.CONTINUE,
            reason="test",
            originating_node="node1",
        )
        new_state = state.emit_signal(signal)

        assert len(new_state.signals) == 1
        assert new_state.signals[0].signal_type == SignalType.CONTINUE

    def test_record_validation(self):
        state = BaseState()
        decision = ValidationDecision(valid=True, validator_name="test")
        new_state = state.record_validation(decision)

        assert len(new_state.validation_decisions) == 1

    def test_update_node_metrics(self):
        state = BaseState()
        new_state = state.update_node_metrics("node1", 100.0, tokens=50, api_calls=1)

        assert new_state.node_execution_times["node1"] == 100.0
        assert new_state.node_call_counts["node1"] == 1
        assert new_state.total_tokens_consumed == 50
        assert new_state.total_api_calls == 1

    def test_check_iteration_limit(self):
        state = BaseState(max_iterations=5)
        assert not state.check_iteration_limit()

        state = state.model_copy(update={"iteration": 5})
        assert state.check_iteration_limit()

    def test_check_token_budget(self):
        state = BaseState(max_tokens_per_execution=100)
        assert not state.check_token_budget()

        state = state.model_copy(update={"total_tokens_consumed": 100})
        assert state.check_token_budget()

    def test_is_terminal(self):
        state = BaseState()
        assert not state.is_terminal()

        # Terminal via iteration limit
        state = state.model_copy(update={"iteration": 100, "max_iterations": 100})
        assert state.is_terminal()

        # Terminal via token budget
        state = BaseState()
        state = state.model_copy(
            update={"total_tokens_consumed": 100000, "max_tokens_per_execution": 100000}
        )
        assert state.is_terminal()

        # Terminal via HALT signal
        state = BaseState()
        signal = ExecutionSignal(signal_type=SignalType.HALT, reason="stop", originating_node="n1")
        state = state.emit_signal(signal)
        assert state.is_terminal()

        # Terminal via validation decision
        state = BaseState()
        decision = ValidationDecision(
            valid=False, suggested_action=SignalType.HALT, validator_name="v"
        )
        state = state.record_validation(decision)
        assert state.is_terminal()

    def test_record_error(self):
        state = BaseState()
        new_state = state.record_error()
        assert new_state.error_count == 1


class TestStateDictConversion:
    def test_roundtrip(self):
        state = BaseState(
            execution_id="test-exec",
            iteration=3,
            current_node="node1",
            total_tokens_consumed=500,
            total_api_calls=10,
            error_count=2,
        )

        state_dict = to_state_dict(state)
        restored = from_state_dict(state_dict)

        assert restored.execution_id == state.execution_id
        assert restored.iteration == state.iteration
        assert restored.current_node == state.current_node
        assert restored.total_tokens_consumed == state.total_tokens_consumed
        assert restored.total_api_calls == state.total_api_calls
        assert restored.error_count == state.error_count
