"""
Tests for contracts.base_contracts module
"""

from datetime import datetime

import pytest

from src.contracts.base_contracts import (
    ArtifactItem,
    ExecutionSignal,
    NodePayload,
    SignalType,
    ValidationDecision,
    from_artifact_dict,
    from_decision_dict,
    from_payload_dict,
    from_signal_dict,
    to_artifact_dict,
    to_decision_dict,
    to_payload_dict,
    to_signal_dict,
)


class TestSignalType:
    def test_signal_types_exist(self):
        assert SignalType.CONTINUE == "CONTINUE"
        assert SignalType.HALT == "HALT"
        assert SignalType.RETRY == "RETRY"
        assert SignalType.BRANCH == "BRANCH"


class TestNodePayload:
    def test_create_minimal(self):
        payload = NodePayload(source_node="a", target_node="b")
        assert payload.source_node == "a"
        assert payload.target_node == "b"
        assert payload.payload_id is not None
        assert payload.correlation_id is not None
        assert isinstance(payload.timestamp, datetime)
        assert payload.iteration == 0

    def test_create_with_data(self):
        payload = NodePayload(
            source_node="a",
            target_node="b",
            data={"key": "value"},
            metadata={"meta": "data"},
            iteration=5,
        )
        assert payload.data == {"key": "value"}
        assert payload.metadata == {"meta": "data"}
        assert payload.iteration == 5

    def test_validate_node_names(self):
        with pytest.raises(ValueError):
            NodePayload(source_node="", target_node="b")
        with pytest.raises(ValueError):
            NodePayload(source_node="a", target_node="")
        with pytest.raises(ValueError):
            NodePayload(source_node="  ", target_node="b")

    def test_with_updated_target(self):
        payload = NodePayload(source_node="a", target_node="b", data={"x": 1})
        new_payload = payload.with_updated_target("c")
        assert new_payload.target_node == "c"
        assert new_payload.source_node == "a"
        assert new_payload.data == {"x": 1}
        assert new_payload.payload_id != payload.payload_id

    def test_with_incremented_iteration(self):
        payload = NodePayload(source_node="a", target_node="b", iteration=3)
        new_payload = payload.with_incremented_iteration()
        assert new_payload.iteration == 4
        assert new_payload.payload_id != payload.payload_id


class TestArtifactItem:
    def test_create_valid(self):
        artifact = ArtifactItem.create(
            artifact_type="code",
            content=b"print('hello')",
            created_by_node="generator",
            mime_type="text/plain",
            metadata={"language": "python"},
        )
        assert artifact.artifact_type == "code"
        assert artifact.content == b"print('hello')"
        assert artifact.mime_type == "text/plain"
        assert artifact.checksum is not None
        assert len(artifact.checksum) == 64
        assert artifact.size_bytes == len(b"print('hello')")
        assert artifact.created_by_node == "generator"
        assert artifact.metadata == {"language": "python"}

    def test_validate_checksum_mismatch(self):
        with pytest.raises(ValueError, match="Checksum mismatch"):
            ArtifactItem(
                artifact_type="test",
                content=b"content",
                mime_type="text/plain",
                metadata={},
                checksum="0" * 64,
                size_bytes=7,
                created_by_node="test",
            )

    def test_validate_size_mismatch(self):
        with pytest.raises(ValueError, match="Checksum mismatch"):
            ArtifactItem(
                artifact_type="test",
                content=b"content",
                mime_type="text/plain",
                metadata={},
                checksum="a" * 64,
                size_bytes=999,
                created_by_node="test",
            )


class TestExecutionSignal:
    def test_continue_signal(self):
        signal = ExecutionSignal(
            signal_type=SignalType.CONTINUE,
            reason="Continue processing",
            originating_node="node1",
        )
        assert signal.signal_type == SignalType.CONTINUE
        assert signal.target_node is None

    def test_halt_signal(self):
        signal = ExecutionSignal(
            signal_type=SignalType.HALT,
            reason="Error occurred",
            originating_node="node1",
        )
        assert signal.signal_type == SignalType.HALT

    def test_branch_signal_requires_target(self):
        with pytest.raises(ValueError, match="BRANCH signal requires target_node"):
            ExecutionSignal(
                signal_type=SignalType.BRANCH,
                reason="Branch to other",
                originating_node="node1",
            )

    def test_branch_signal_with_target(self):
        signal = ExecutionSignal(
            signal_type=SignalType.BRANCH,
            reason="Branch to other",
            originating_node="node1",
            target_node="node2",
        )
        assert signal.target_node == "node2"


class TestValidationDecision:
    def test_valid_decision(self):
        decision = ValidationDecision(
            valid=True,
            validator_name="test_validator",
        )
        assert decision.valid is True
        assert decision.errors == []
        assert decision.suggested_action == SignalType.CONTINUE

    def test_invalid_decision(self):
        decision = ValidationDecision(
            valid=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            suggested_action=SignalType.HALT,
            validator_name="test_validator",
        )
        assert decision.valid is False
        assert decision.errors == ["Error 1", "Error 2"]
        assert decision.warnings == ["Warning 1"]
        assert decision.suggested_action == SignalType.HALT

    def test_merge_decisions(self):
        d1 = ValidationDecision(
            valid=True,
            errors=[],
            warnings=["w1"],
            suggested_action=SignalType.CONTINUE,
            validator_name="v1",
        )
        d2 = ValidationDecision(
            valid=False,
            errors=["e1"],
            warnings=["w2"],
            suggested_action=SignalType.HALT,
            validator_name="v2",
        )
        merged = d1.merge(d2)
        assert merged.valid is False
        assert merged.errors == ["e1"]
        assert merged.warnings == ["w1", "w2"]
        assert merged.suggested_action == SignalType.HALT
        assert merged.validator_name == "v1+v2"

    def test_merge_both_valid(self):
        d1 = ValidationDecision(valid=True, validator_name="v1")
        d2 = ValidationDecision(valid=True, validator_name="v2")
        merged = d1.merge(d2)
        assert merged.valid is True
        assert merged.suggested_action == SignalType.CONTINUE


class TestTypedDictConversions:
    def test_payload_roundtrip(self):
        payload = NodePayload(source_node="a", target_node="b", data={"x": 1}, iteration=3)
        d = to_payload_dict(payload)
        assert isinstance(d, dict)
        assert d["source_node"] == "a"
        assert d["iteration"] == 3

        restored = from_payload_dict(d)
        assert restored.source_node == "a"
        assert restored.target_node == "b"
        assert restored.data == {"x": 1}
        assert restored.iteration == 3

    def test_artifact_roundtrip(self):
        artifact = ArtifactItem.create(
            artifact_type="test",
            content=b"content",
            created_by_node="node1",
        )
        d = to_artifact_dict(artifact)
        restored = from_artifact_dict(d)
        assert restored.artifact_id == artifact.artifact_id
        assert restored.content == artifact.content
        assert restored.checksum == artifact.checksum

    def test_signal_roundtrip(self):
        signal = ExecutionSignal(
            signal_type=SignalType.CONTINUE,
            reason="test",
            originating_node="node1",
        )
        d = to_signal_dict(signal)
        restored = from_signal_dict(d)
        assert restored.signal_type == SignalType.CONTINUE
        assert restored.reason == "test"

    def test_decision_roundtrip(self):
        decision = ValidationDecision(
            valid=True,
            errors=[],
            warnings=["warn"],
            suggested_action=SignalType.CONTINUE,
            validator_name="test",
        )
        d = to_decision_dict(decision)
        restored = from_decision_dict(d)
        assert restored.valid is True
        assert restored.warnings == ["warn"]
        assert restored.suggested_action == SignalType.CONTINUE
