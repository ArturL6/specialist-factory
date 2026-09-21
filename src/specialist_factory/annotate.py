from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .aggregation import aggregate, validate_signal
from .cache import TeacherCache
from .config import read_jsonl
from .schemas import AggregatedLabel, AppConfig, Sample, TeacherSignal
from .teachers import build_teacher


def annotate(config: AppConfig) -> tuple[list[TeacherSignal], list[AggregatedLabel]]:
    samples = read_jsonl(config.data_path); cache = TeacherCache(config.cache_path)
    all_signals: list[TeacherSignal] = []
    for teacher_cfg in config.teachers:
        teacher = build_teacher(teacher_cfg); misses: list[Sample] = []
        for sample in samples:
            key = cache.key(sample, config.task, teacher_cfg); cached = cache.get(key)
            if cached: all_signals.append(cached)
            else: misses.append(sample)
        if misses:
            returned = teacher.predict(misses, config.task)
            by_id = {signal.sample_id: signal for signal in returned}
            for sample in misses:
                key = cache.key(sample, config.task, teacher_cfg)
                try:
                    signal = by_id[sample.id]; validate_signal(signal, config.task)
                    cache.put(key, signal, latency_ms=signal.metadata.get("latency_ms", 0), cost=signal.metadata.get("cost", 0)); all_signals.append(signal)
                except Exception as exc:
                    cache.put(key, None, error=str(exc)); raise
    run = Path(config.run_dir); run.mkdir(parents=True, exist_ok=True)
    (run / "teacher_signals.jsonl").write_text("".join(signal.model_dump_json() + "\n" for signal in all_signals))
    grouped: dict[str, list[TeacherSignal]] = defaultdict(list)
    for signal in all_signals: grouped[signal.sample_id].append(signal)
    weights = {teacher.id: teacher.weight for teacher in config.teachers}
    labels = [aggregate(sample.id, grouped[sample.id], config.task, config.aggregation, weights) for sample in samples]
    (run / "aggregated_labels.jsonl").write_text("".join(label.model_dump_json() + "\n" for label in labels))
    return all_signals, labels


def load_annotations(config: AppConfig) -> tuple[list[TeacherSignal], dict[str, AggregatedLabel]]:
    run = Path(config.run_dir)
    signals = [TeacherSignal.model_validate_json(line) for line in (run / "teacher_signals.jsonl").read_text().splitlines()]
    labels = {row.sample_id: row for row in (AggregatedLabel.model_validate_json(line) for line in (run / "aggregated_labels.jsonl").read_text().splitlines())}
    return signals, labels
