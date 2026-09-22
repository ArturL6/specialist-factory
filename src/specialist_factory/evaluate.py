from __future__ import annotations

import json
from pathlib import Path

from .annotate import load_annotations
from .config import read_jsonl
from .schemas import AppConfig
from .training.engine import load_student, predict, split_samples


def evaluate(config: AppConfig, checkpoint: str) -> dict:
    module = load_student(checkpoint)
    samples = read_jsonl(config.data_path)
    _train, validation = split_samples(samples, config.training.seed)
    annotations = Path(config.run_dir, "aggregated_labels.jsonl")
    _signals, aggregated = load_annotations(config) if annotations.exists() else ([], {})
    output = predict(module, samples)
    by_id = {row["sample_id"]: row for row in output}
    human = [sample for sample in validation if sample.human_label]
    human_accuracy = (
        sum(by_id[sample.id]["predicted_label"] == sample.human_label for sample in human) / len(human)
        if human
        else None
    )
    validation_ids = {sample.id for sample in validation}
    accepted = [item for item in aggregated.values() if not item.abstained and item.sample_id in validation_ids]
    # Abstention is a teacher-only property: no student involved, so no leakage, so measure it
    # over the full dataset instead of the noisier holdout slice.
    abstained_all = sum(item.abstained for item in aggregated.values())
    teacher_agreement = (
        sum(by_id[item.sample_id]["predicted_label"] == item.predicted_label for item in accepted) / len(accepted)
        if accepted
        else None
    )
    report = {
        "checkpoint": checkpoint,
        "split": "validation",
        "samples": len(samples),
        "holdout_size": len(validation),
        "human_labeled": len(human),
        "human_accuracy": human_accuracy,
        "teacher_accepted": len(accepted),
        "student_teacher_agreement": teacher_agreement,
        "teacher_abstention_rate": abstained_all / len(aggregated) if aggregated else None,
        "student_predictions": output,
    }
    path = Path(config.run_dir) / "evaluation.json"
    path.write_text(json.dumps(report, indent=2))
    return report
