from pathlib import Path

from specialist_factory.cache import TeacherCache
from specialist_factory.schemas import (
    ClassDefinition,
    Sample,
    SignalType,
    TaskDefinition,
    TeacherConfig,
    TeacherSignal,
)


def test_cache_roundtrip(tmp_path: Path):
    cache = TeacherCache(tmp_path / "cache.sqlite")
    sample = Sample(id="x", text="hello")
    task = TaskDefinition(name="t", modality="text", classes=[ClassDefinition(name="yes", description="yes"), ClassDefinition(name="no", description="no")])
    cfg = TeacherConfig(id="mock", type="mock")
    key = cache.key(sample, task, cfg)
    signal = TeacherSignal(sample_id="x", teacher_id="mock", teacher_model="mock", probabilities={"yes": .7, "no": .3}, predicted_label="yes", confidence=.7, signal_type=SignalType.MODEL_PROBABILITY)
    cache.put(key, signal)
    assert cache.get(key) == signal
