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
    task = TaskDefinition(
        name="t",
        modality="text",
        classes=[ClassDefinition(name="yes", description="yes"), ClassDefinition(name="no", description="no")],
    )
    cfg = TeacherConfig(id="mock", type="mock")
    key = cache.key(sample, task, cfg)
    signal = TeacherSignal(
        sample_id="x",
        teacher_id="mock",
        teacher_model="mock",
        probabilities={"yes": 0.7, "no": 0.3},
        predicted_label="yes",
        confidence=0.7,
        signal_type=SignalType.MODEL_PROBABILITY,
    )
    cache.put(key, signal)
    assert cache.get(key) == signal


def test_key_ignores_weight_but_not_model_name_or_version(tmp_path: Path):
    cache = TeacherCache(tmp_path / "cache.sqlite")
    sample = Sample(id="x", text="hello")
    task = TaskDefinition(
        name="t",
        modality="text",
        classes=[ClassDefinition(name="yes", description="yes"), ClassDefinition(name="no", description="no")],
    )
    base = TeacherConfig(id="mock", type="mock", model_name="m1", version="v1", weight=1.0)
    reweighted = base.model_copy(update={"weight": 0.9})
    renamed = base.model_copy(update={"model_name": "m2"})
    bumped = base.model_copy(update={"version": "v2"})
    assert cache.key(sample, task, base) == cache.key(sample, task, reweighted)
    assert cache.key(sample, task, base) != cache.key(sample, task, renamed)
    assert cache.key(sample, task, base) != cache.key(sample, task, bumped)
