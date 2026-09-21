from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from ..schemas import (
    Sample,
    SignalType,
    TaskDefinition,
    TeacherConfig,
    TeacherSignal,
)


def _signal(sample: Sample, cfg: TeacherConfig, probabilities: dict[str, float], signal_type: SignalType = SignalType.MODEL_PROBABILITY, **metadata: Any) -> TeacherSignal:
    probabilities = {key: max(0.0, float(value)) for key, value in probabilities.items()}
    total = sum(probabilities.values())
    probabilities = {key: value / total for key, value in probabilities.items()} if total else {key: 1 / len(probabilities) for key in probabilities}
    label = max(probabilities, key=probabilities.get)
    return TeacherSignal(sample_id=sample.id, teacher_id=cfg.id, teacher_model=cfg.model_name or cfg.model or cfg.type, probabilities=probabilities, predicted_label=label, confidence=probabilities[label], signal_type=signal_type, metadata=metadata)


class MockTeacher:
    """Deterministic offline teacher used for reproducible demo flows."""
    def __init__(self, cfg: TeacherConfig): self.cfg, self.teacher_id = cfg, cfg.id

    def predict(self, samples: list[Sample], task: TaskDefinition) -> list[TeacherSignal]:
        signals = []
        for sample in samples:
            haystack = (sample.text or "").lower()
            scores = {label: 0.05 for label in task.labels}
            if "cancel" in haystack and "cancel_order" in scores: scores["cancel_order"] = .75
            elif any(word in haystack for word in ("refund", "money back", "return")) and "refund" in scores: scores["refund"] = .75
            elif any(word in haystack for word in ("where", "track", "package", "order")) and "order_status" in scores: scores["order_status"] = .75
            elif sample.image and {"scratch", "clean"}.issubset(scores):
                from PIL import Image
                red, green = __import__("numpy").asarray(Image.open(sample.image).convert("RGB"), dtype=float).mean(axis=(0, 1))[:2]
                scores["scratch" if red > green else "clean"] = .8
            elif "other" in scores: scores["other"] = .65
            signals.append(_signal(sample, self.cfg, scores, source="deterministic_mock", model_revision=self.cfg.version, latency_ms=0, cost=0))
        return signals


class HuggingFaceTextTeacher:
    def __init__(self, cfg: TeacherConfig): self.cfg, self.teacher_id, self._pipe = cfg, cfg.id, None

    def _pipeline(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("zero-shot-classification", model=self.cfg.model_name, device=-1)
        return self._pipe

    def predict(self, samples: list[Sample], task: TaskDefinition) -> list[TeacherSignal]:
        labels = [item.description for item in task.classes]
        description_to_name = {item.description: item.name for item in task.classes}
        output = []
        for sample in samples:
            if not sample.text: continue
            started = time.perf_counter()
            result = self._pipeline()(sample.text, candidate_labels=labels, multi_label=False, hypothesis_template="This request is about {}.")
            probabilities = {description_to_name[label]: score for label, score in zip(result["labels"], result["scores"])}
            output.append(_signal(sample, self.cfg, probabilities, latency_ms=round((time.perf_counter()-started)*1000, 2), source="transformers.zero-shot-classification", model_revision=self.cfg.version, cost=0))
        return output


class HuggingFaceVisionTeacher:
    def __init__(self, cfg: TeacherConfig): self.cfg, self.teacher_id, self._pipe = cfg, cfg.id, None

    def _pipeline(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("zero-shot-image-classification", model=self.cfg.model_name, device=-1)
        return self._pipe

    def predict(self, samples: list[Sample], task: TaskDefinition) -> list[TeacherSignal]:
        output = []
        candidates = [item.description for item in task.classes]
        mapping = {item.description: item.name for item in task.classes}
        for sample in samples:
            if not sample.image: continue
            started = time.perf_counter()
            result = self._pipeline()(sample.image, candidate_labels=candidates)
            probs = {mapping[row["label"]]: row["score"] for row in result}
            output.append(_signal(sample, self.cfg, probs, latency_ms=round((time.perf_counter()-started)*1000, 2), source="transformers.zero-shot-image-classification", model_revision=self.cfg.version, cost=0))
        return output


class JevTeacher:
    """TypeSafe Jev NOUL adapter; every class gets the same evidence and a binary decision."""

    def __init__(self, cfg: TeacherConfig):
        self.cfg, self.teacher_id = cfg, cfg.id

    def predict(self, samples: list[Sample], task: TaskDefinition) -> list[TeacherSignal]:
        key = os.getenv(self.cfg.api_key_env or "TYPESAFE_API_KEY")
        if not key:
            raise RuntimeError("Jev key missing; set the configured api_key_env")
        base = self.cfg.base_url or os.getenv("TYPESAFE_BASE_URL", "https://openrouter.ai/api")
        url = base.rstrip("/") + "/v1/systemone"
        output: list[TeacherSignal] = []
        for sample in samples:
            if not sample.text:
                continue
            questions = {
                item.name: {
                    "type": "noul",
                    "instructions": (
                        f"Does the request match exactly this intent? {item.description} "
                        "Answer from the request only; do not infer an unstated intent."
                    ),
                }
                for item in task.classes
            }
            payload = {
                "model": self.cfg.model or self.cfg.model_name or "jev-1.13",
                "state": {"request": sample.text, "metadata": sample.metadata},
                "questions": questions,
            }
            started = time.perf_counter()
            last_error: Exception | None = None
            for attempt in range(4):
                try:
                    response = requests.post(
                        url,
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json=payload,
                        timeout=60,
                    )
                    if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                        time.sleep(2**attempt)
                        continue
                    response.raise_for_status()
                    raw = response.json()
                    probabilities = {label: float(raw["answers"][label]["noul"]) for label in task.labels}
                    output.append(
                        _signal(
                            sample,
                            self.cfg,
                            probabilities,
                            SignalType.MODEL_PROBABILITY,
                            latency_ms=round((time.perf_counter() - started) * 1000, 2),
                            cost=float(raw.get("usage", {}).get("cost", 0) or 0),
                            input_tokens=int(raw.get("usage", {}).get("input_tokens", 0) or 0),
                            provider=raw.get("provider", "unknown"),
                            served_model=raw.get("model", self.cfg.model or self.cfg.model_name),
                            model_revision=self.cfg.version,
                            decision_type="noul",
                        )
                    )
                    break
                except requests.HTTPError as exc:
                    last_error = exc
                    status = exc.response.status_code if exc.response is not None else 0
                    if status not in {429, 500, 502, 503, 504} or attempt == 3:
                        raise RuntimeError(f"Jev request failed with HTTP {status}") from exc
                    time.sleep(2**attempt)
                except requests.RequestException as exc:
                    last_error = exc
                    if attempt == 3:
                        raise RuntimeError("Jev request failed after four attempts") from exc
                    time.sleep(2**attempt)
            else:
                raise RuntimeError("Jev did not return a response") from last_error
        return output


class OpenRouterVisionTeacher:
    """Documented OpenRouter chat-completions adapter. Self-reported confidence stays distinct."""
    def __init__(self, cfg: TeacherConfig): self.cfg, self.teacher_id = cfg, cfg.id

    def predict(self, samples: list[Sample], task: TaskDefinition) -> list[TeacherSignal]:
        key = os.getenv(self.cfg.api_key_env or "OPENROUTER_API_KEY")
        if not key: raise RuntimeError("OpenRouter key missing; set the configured api_key_env")
        url = (self.cfg.base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/") + "/chat/completions"
        classes = [{"name": item.name, "description": item.description} for item in task.classes]
        output = []
        for sample in samples:
            parts: list[dict[str, Any]] = [{"type": "text", "text": f"Classify against {classes}. Return JSON only: {{'label': one class name, 'confidence': 0..1}}. Text: {sample.text or ''}"}]
            if sample.image:
                mime = "image/png" if sample.image.lower().endswith(".png") else "image/jpeg"
                data = base64.b64encode(Path(sample.image).read_bytes()).decode()
                parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
            started = time.perf_counter()
            response = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json={"model": self.cfg.model, "messages": [{"role": "user", "content": parts}], "response_format": {"type": "json_object"}}, timeout=60)
            response.raise_for_status(); raw = response.json()
            parsed = json.loads(raw["choices"][0]["message"]["content"])
            label = parsed["label"]
            if label not in task.labels: raise ValueError(f"VLM returned invalid label {label!r}")
            confidence = float(parsed.get("confidence", 1.0))
            probs = {item: 0.0 for item in task.labels}; probs[label] = 1.0
            output.append(_signal(sample, self.cfg, probs, SignalType.SELF_REPORTED_PROBABILITY, latency_ms=round((time.perf_counter()-started)*1000,2), cost=float(raw.get("usage", {}).get("cost", 0) or 0), provider="openrouter", raw_label=label, self_reported_confidence=confidence, model_revision=self.cfg.version))
        return output


def build_teacher(cfg: TeacherConfig):
    types = {
        "mock": MockTeacher,
        "huggingface_text": HuggingFaceTextTeacher,
        "huggingface_vision": HuggingFaceVisionTeacher,
        "jev": JevTeacher,
        "openrouter_vision": OpenRouterVisionTeacher,
    }
    if cfg.type not in types: raise ValueError(f"Unsupported teacher type: {cfg.type}")
    return types[cfg.type](cfg)
