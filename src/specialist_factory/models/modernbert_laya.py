from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ..schemas import ClassDefinition, Sample


class LayaStyleDecisionHead(nn.Module):
    """Laya-inspired typed decision head: type embedding → head transformer → marker scoring.

    This POC has one question type (multi-class classification), so the action-cost head
    used by Laya's interactive decision agent is deliberately out of scope.
    """

    def __init__(self, hidden_size: int, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            hidden_size,
            nhead=max(1, hidden_size // 64),
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.contextualizer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.type_embedding = nn.Embedding(1, hidden_size)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor, marker_pos: torch.Tensor) -> torch.Tensor:
        qtype = torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
        hidden = hidden + self.type_embedding(qtype)[:, None, :]
        hidden = self.contextualizer(hidden, src_key_padding_mask=~attention_mask.bool())
        positions = marker_pos[:, :, None].expand(-1, -1, hidden.shape[-1])
        markers = torch.gather(hidden, 1, positions)
        return self.scorer(markers).squeeze(-1).float()


class ModernBertLayaStudent(nn.Module):
    def __init__(
        self,
        model_name: str,
        classes: Sequence[ClassDefinition],
        *,
        max_length: int = 256,
        head_layers: int = 1,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.classes = list(classes)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
        self.head = LayaStyleDecisionHead(self.encoder.config.hidden_size, layers=head_layers, dropout=dropout)
        if freeze_encoder:
            self.encoder.requires_grad_(False)

    def _render(self, sample: Sample) -> str:
        options = " ".join(f"{item.name}: {item.description} {self.tokenizer.sep_token}" for item in self.classes)
        return f"Request: {sample.text or ''} {self.tokenizer.sep_token} Options: {options}"

    def _encode(self, samples: list[Sample], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        batch = self.tokenizer(
            [self._render(sample) for sample in samples],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items() if key != "token_type_ids"}
        sep_id = self.tokenizer.sep_token_id
        marker_rows = []
        for ids in batch["input_ids"]:
            separators = (ids == sep_id).nonzero(as_tuple=False).flatten().tolist()
            if len(separators) < len(self.classes):
                raise ValueError("Prompt truncation removed option markers; increase max_length")
            marker_rows.append(separators[-len(self.classes) :])
        return batch, torch.tensor(marker_rows, dtype=torch.long, device=device)

    def forward(self, samples: list[Sample]) -> torch.Tensor:
        device = next(self.head.parameters()).device
        encoded, marker_pos = self._encode(samples, device)
        hidden = self.encoder(**encoded).last_hidden_state
        return self.head(hidden, encoded["attention_mask"], marker_pos)
