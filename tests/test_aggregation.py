from specialist_factory.aggregation import aggregate, validate_signal
from specialist_factory.schemas import (
    AggregationConfig,
    ClassDefinition,
    SignalType,
    TaskDefinition,
    TeacherSignal,
)


def _vote(teacher_id: str, predicted_label: str) -> TeacherSignal:
    return TeacherSignal(
        sample_id="1",
        teacher_id=teacher_id,
        teacher_model="m",
        probabilities={"a": 1.0, "b": 1.0},
        predicted_label=predicted_label,
        confidence=1.0,
        signal_type=SignalType.MODEL_PROBABILITY,
    )


def task():
    return TaskDefinition(
        name="demo",
        modality="text",
        classes=[ClassDefinition(name="a", description="a"), ClassDefinition(name="b", description="b")],
    )


def test_weighted_aggregation_and_abstention():
    one = TeacherSignal(
        sample_id="1",
        teacher_id="t1",
        teacher_model="m",
        probabilities={"a": 0.9, "b": 0.1},
        predicted_label="a",
        confidence=0.9,
        signal_type=SignalType.MODEL_PROBABILITY,
    )
    two = TeacherSignal(
        sample_id="1",
        teacher_id="t2",
        teacher_model="m",
        probabilities={"a": 0.8, "b": 0.2},
        predicted_label="a",
        confidence=0.8,
        signal_type=SignalType.MODEL_PROBABILITY,
    )
    result = aggregate("1", [one, two], task(), AggregationConfig(min_confidence=0.6), {"t1": 1, "t2": 1})
    assert result.predicted_label == "a" and not result.abstained and result.teacher_agreement == 1


def test_signal_rejects_wrong_label_set():
    signal = TeacherSignal(
        sample_id="1",
        teacher_id="t",
        teacher_model="m",
        probabilities={"a": 1},
        predicted_label="a",
        confidence=1,
        signal_type=SignalType.MODEL_PROBABILITY,
    )
    try:
        validate_signal(signal, task())
    except ValueError as error:
        assert "exactly" in str(error)
    else:
        raise AssertionError("invalid signal accepted")


def test_majority_vote_weights_the_heavier_teacher_and_matches_plain_count_when_equal():
    votes = [_vote("heavy", "a"), _vote("light1", "b"), _vote("light2", "b")]
    config = AggregationConfig(method="majority_vote", min_confidence=0, min_teacher_agreement=0)
    weighted = aggregate("1", votes, task(), config, {"heavy": 3.0, "light1": 1.0, "light2": 1.0})
    assert weighted.predicted_label == "a"
    unweighted = aggregate("1", votes, task(), config, {"heavy": 1.0, "light1": 1.0, "light2": 1.0})
    assert unweighted.predicted_label == "b"
