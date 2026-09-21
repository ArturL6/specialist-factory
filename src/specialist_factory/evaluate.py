from __future__ import annotations

import json
from pathlib import Path

from .annotate import load_annotations
from .config import read_jsonl
from .schemas import AppConfig
from .training.engine import load_student, predict


def evaluate(config: AppConfig, checkpoint: str) -> dict:
    module = load_student(checkpoint); samples = read_jsonl(config.data_path); _signals, aggregated = load_annotations(config); output = predict(module, samples)
    by_id = {row["sample_id"]: row for row in output}; human = [sample for sample in samples if sample.human_label]
    human_accuracy = sum(by_id[sample.id]["predicted_label"] == sample.human_label for sample in human) / len(human) if human else None
    accepted = [item for item in aggregated.values() if not item.abstained]
    teacher_agreement = sum(by_id[item.sample_id]["predicted_label"] == item.predicted_label for item in accepted) / len(accepted) if accepted else None
    report = {"checkpoint": checkpoint, "samples": len(samples), "human_labeled": len(human), "human_accuracy": human_accuracy, "teacher_accepted": len(accepted), "student_teacher_agreement": teacher_agreement, "teacher_abstention_rate": 1 - len(accepted)/len(samples), "student_predictions": output}
    path = Path(config.run_dir) / "evaluation.json"; path.write_text(json.dumps(report, indent=2)); return report
