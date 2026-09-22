from pathlib import Path
from types import SimpleNamespace

import specialist_factory.annotate as annotate_module
from specialist_factory.config import load_config, write_jsonl
from specialist_factory.schemas import Sample, SignalType, TeacherSignal

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "text_demo.yaml"


def _skip_predict(samples, task):
    # Mirrors HuggingFaceTextTeacher/JevTeacher: silently skip samples without text.
    signals = []
    for sample in samples:
        if not sample.text:
            continue
        signals.append(
            TeacherSignal(
                sample_id=sample.id,
                teacher_id="skip",
                teacher_model="skip-test",
                probabilities={label: 1.0 for label in task.labels},
                predicted_label=task.labels[0],
                confidence=1.0,
                signal_type=SignalType.MODEL_PROBABILITY,
            )
        )
    return signals


def test_annotate_skips_samples_missing_text(tmp_path: Path):
    cfg = load_config(CONFIG_PATH).model_copy(
        update={
            "data_path": str(tmp_path / "samples.jsonl"),
            "run_dir": str(tmp_path / "run"),
            "cache_path": str(tmp_path / "cache.sqlite"),
        }
    )
    write_jsonl(cfg.data_path, [Sample(id=f"s{i}") for i in range(3)])
    original_build_teacher = annotate_module.build_teacher
    annotate_module.build_teacher = lambda teacher_cfg: SimpleNamespace(
        teacher_id=teacher_cfg.id, predict=_skip_predict
    )
    try:
        signals, labels = annotate_module.annotate(cfg)
    finally:
        annotate_module.build_teacher = original_build_teacher
    assert signals == []
    assert len(labels) == 3
    assert all(label.abstained for label in labels)
