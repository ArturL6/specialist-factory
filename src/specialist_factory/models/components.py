from __future__ import annotations

import hashlib

import numpy as np
import torch
from PIL import Image
from torch import nn

from ..schemas import Modality, Sample, StudentConfig


def text_features(text: str | None, dimension: int) -> torch.Tensor:
    vector = torch.zeros(dimension)
    for token in (text or "").lower().split():
        index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big") % dimension
        vector[index] += 1
    return vector / max(vector.norm(), torch.tensor(1.0))


def image_features(path: str | None, dimension: int) -> torch.Tensor:
    if not path:
        return torch.zeros(dimension)
    image = Image.open(path).convert("RGB").resize((32, 32))
    array = np.asarray(image, dtype=np.float32) / 255.0
    base = torch.tensor([*array.mean(axis=(0, 1)), *array.std(axis=(0, 1)), array.mean()])
    return base.repeat((dimension + len(base) - 1) // len(base))[:dimension]


class TinyTextEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, texts: list[str | None]) -> torch.Tensor:
        return torch.stack([text_features(text, self.output_dim) for text in texts])


class TinyVisionEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, images: list[str | None]) -> torch.Tensor:
        return torch.stack([image_features(path, self.output_dim) for path in images])


class FeatureEncoder(nn.Module):
    def __init__(self, config: StudentConfig):
        super().__init__()
        self.modalities = config.input_modalities
        dim = int(config.encoder.get("output_dim", 32))
        self.text = TinyTextEncoder(dim) if Modality.TEXT in self.modalities else None
        self.image = TinyVisionEncoder(dim) if Modality.IMAGE in self.modalities else None
        self.output_dim = dim * len(self.modalities)

    def forward(self, samples: list[Sample]) -> torch.Tensor:
        parts = []
        if self.text:
            parts.append(self.text([sample.text for sample in samples]))
        if self.image:
            parts.append(self.image([sample.image for sample in samples]))
        return torch.cat(parts, dim=-1)


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, class_count: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, class_count)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)
