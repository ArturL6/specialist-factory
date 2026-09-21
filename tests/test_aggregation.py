from specialist_factory.aggregation import aggregate, validate_signal
from specialist_factory.schemas import (
    AggregationConfig,
    ClassDefinition,
    SignalType,
    TaskDefinition,
    TeacherSignal,
)


def task():
    return TaskDefinition(name="demo", modality="text", classes=[ClassDefinition(name="a", description="a"), ClassDefinition(name="b", description="b")])


def test_weighted_aggregation_and_abstention():
    one = TeacherSignal(sample_id="1", teacher_id="t1", teacher_model="m", probabilities={"a": .9, "b": .1}, predicted_label="a", confidence=.9, signal_type=SignalType.MODEL_PROBABILITY)
    two = TeacherSignal(sample_id="1", teacher_id="t2", teacher_model="m", probabilities={"a": .8, "b": .2}, predicted_label="a", confidence=.8, signal_type=SignalType.MODEL_PROBABILITY)
    result = aggregate("1", [one, two], task(), AggregationConfig(min_confidence=.6), {"t1": 1, "t2": 1})
    assert result.predicted_label == "a" and not result.abstained and result.teacher_agreement == 1


def test_signal_rejects_wrong_label_set():
    signal = TeacherSignal(sample_id="1", teacher_id="t", teacher_model="m", probabilities={"a": 1}, predicted_label="a", confidence=1, signal_type=SignalType.MODEL_PROBABILITY)
    try: validate_signal(signal, task())
    except ValueError as error: assert "exactly" in str(error)
    else: raise AssertionError("invalid signal accepted")
