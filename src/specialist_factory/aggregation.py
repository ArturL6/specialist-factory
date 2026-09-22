from __future__ import annotations

import math

from .schemas import (
    AggregatedLabel,
    AggregationConfig,
    SignalType,
    TaskDefinition,
    TeacherSignal,
)


def validate_signal(signal: TeacherSignal, task: TaskDefinition) -> None:
    if signal.predicted_label not in task.labels:
        raise ValueError(f"Unknown prediction {signal.predicted_label}")
    if signal.probabilities and set(signal.probabilities) != set(task.labels):
        raise ValueError("Teacher probability map must contain exactly the task labels")
    if (
        signal.signal_type == SignalType.SELF_REPORTED_PROBABILITY
        and signal.metadata.get("self_reported_confidence") is None
    ):
        raise ValueError("Self-reported probabilities need provenance metadata")


def aggregate(
    sample_id: str,
    signals: list[TeacherSignal],
    task: TaskDefinition,
    config: AggregationConfig,
    weights: dict[str, float],
) -> AggregatedLabel:
    usable = [item for item in signals if item.probabilities]
    if not usable:
        return AggregatedLabel(
            sample_id=sample_id,
            probabilities={label: 0 for label in task.labels},
            predicted_label=None,
            confidence=0,
            teacher_agreement=0,
            entropy=0,
            margin=0,
            abstained=True,
            teacher_count=0,
        )
    scores = {label: 0.0 for label in task.labels}
    if config.method == "majority_vote":
        for item in usable:
            scores[item.predicted_label] += weights.get(item.teacher_id, 1.0)
    elif config.method == "log_probability_mean":
        for label in task.labels:
            scores[label] = math.exp(
                sum(
                    weights.get(item.teacher_id, 1.0) * math.log(max(item.probabilities[label], 1e-8))
                    for item in usable
                )
                / sum(weights.get(item.teacher_id, 1.0) for item in usable)
            )
    else:
        for item in usable:
            for label, probability in item.probabilities.items():
                scores[label] += weights.get(item.teacher_id, 1.0) * probability
    total = sum(scores.values())
    probs = {label: value / total for label, value in scores.items()}
    ordered = sorted(probs.values(), reverse=True)
    label = max(probs, key=probs.get)
    confidence = probs[label]
    agreement = sum(item.predicted_label == label for item in usable) / len(usable)
    entropy = -sum(value * math.log(value + 1e-12) for value in probs.values()) / math.log(len(probs))
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    abstained = (
        confidence < config.min_confidence
        or agreement < config.min_teacher_agreement
        or (config.max_entropy is not None and entropy > config.max_entropy)
    )
    return AggregatedLabel(
        sample_id=sample_id,
        probabilities=probs,
        predicted_label=None if abstained else label,
        confidence=confidence,
        teacher_agreement=agreement,
        entropy=entropy,
        margin=margin,
        abstained=abstained,
        teacher_count=len(usable),
    )
