from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class SignalType(str, Enum):
    MODEL_PROBABILITY = "model_probability"
    SELF_REPORTED_PROBABILITY = "self_reported_probability"
    HARD_LABEL = "hard_label"
    HUMAN_LABEL = "human_label"


class ClassDefinition(BaseModel):
    name: str
    description: str


class TaskDefinition(BaseModel):
    name: str
    modality: Modality
    classes: list[ClassDefinition]
    available_modalities: list[Modality] = Field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        return [item.name for item in self.classes]

    def description_for(self, label: str) -> str:
        return next(item.description for item in self.classes if item.name == label)


class Sample(BaseModel):
    id: str
    text: str | None = None
    image: str | None = None
    human_label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TeacherSignal(BaseModel):
    sample_id: str
    teacher_id: str
    teacher_model: str
    probabilities: dict[str, float] | None = None
    predicted_label: str
    confidence: float = 1.0
    signal_type: SignalType
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("probabilities")
    @classmethod
    def normalize(cls, values: dict[str, float] | None) -> dict[str, float] | None:
        if values is None:
            return None
        if not values or any(value < 0 for value in values.values()):
            raise ValueError("probabilities must be non-empty and non-negative")
        total = sum(values.values())
        if total <= 0:
            raise ValueError("probabilities must sum to a positive number")
        return {key: float(value / total) for key, value in values.items()}


class AggregatedLabel(BaseModel):
    sample_id: str
    probabilities: dict[str, float]
    predicted_label: str | None
    confidence: float
    teacher_agreement: float
    entropy: float
    margin: float
    abstained: bool = False
    teacher_count: int


class TeacherConfig(BaseModel):
    id: str
    type: str
    model_name: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    weight: float = 1.0
    modalities: list[Modality] = Field(default_factory=list)
    version: str = "v1"


class AggregationConfig(BaseModel):
    method: Literal["weighted_mean", "log_probability_mean", "majority_vote"] = "weighted_mean"
    min_teacher_agreement: float = 0.5
    min_confidence: float = 0.6
    max_entropy: float | None = None


class StudentConfig(BaseModel):
    input_modalities: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])
    encoder: dict[str, Any] = Field(default_factory=lambda: {"type": "tiny_text", "output_dim": 32})
    head: dict[str, Any] = Field(default_factory=lambda: {"type": "mlp", "hidden_dim": 32, "dropout": 0.1})


class TrainingConfig(BaseModel):
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-3
    alpha_human: float = 1.0
    beta_teacher: float = 1.0
    distillation_temperature: float = 2.0
    accelerator: str = "auto"
    seed: int = 7


class AppConfig(BaseModel):
    task: TaskDefinition
    data_path: str
    run_dir: str
    cache_path: str = "data/teacher_cache.sqlite"
    teachers: list[TeacherConfig] = Field(default_factory=list)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    student: StudentConfig = Field(default_factory=StudentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    generator: dict[str, Any] = Field(default_factory=dict)
