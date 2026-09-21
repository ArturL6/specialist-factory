from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..config import read_jsonl
from ..models import FeatureEncoder, MLPHead, ModernBertLayaStudent
from ..schemas import AggregatedLabel, AppConfig, Sample


@dataclass
class Batch:
    samples: list[Sample]
    human: torch.Tensor
    teacher: torch.Tensor
    teacher_mask: torch.Tensor


class SpecialistDataset(Dataset):
    def __init__(self, samples: list[Sample], labels: dict[str, AggregatedLabel], class_names: list[str]): self.samples, self.labels, self.class_names = samples, labels, class_names
    def __len__(self): return len(self.samples)
    def __getitem__(self, index):
        sample = self.samples[index]; aggregate = self.labels.get(sample.id); human = self.class_names.index(sample.human_label) if sample.human_label in self.class_names else -1
        probs = [aggregate.probabilities[name] for name in self.class_names] if aggregate and not aggregate.abstained else [0.0] * len(self.class_names)
        return sample, human, probs, int(bool(aggregate and not aggregate.abstained))


def collate(rows) -> Batch:
    samples, human, teacher, mask = zip(*rows)
    return Batch(list(samples), torch.tensor(human), torch.tensor(teacher, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool))


class SpecialistModule(L.LightningModule):
    def __init__(self, config: AppConfig | dict):
        super().__init__()
        config = AppConfig.model_validate(config)
        self.save_hyperparameters({"config": config.model_dump()})
        self.config = config
        encoder = config.student.encoder
        self.decision_student = None
        if encoder.get("type") == "modernbert":
            if config.task.modality.value != "text":
                raise ValueError("ModernBERT decision student supports text classification only")
            self.decision_student = ModernBertLayaStudent(
                str(encoder.get("model_name", "answerdotai/ModernBERT-base")),
                config.task.classes,
                max_length=int(encoder.get("max_length", 256)),
                head_layers=int(config.student.head.get("layers", 1)),
                dropout=float(config.student.head.get("dropout", 0.1)),
                freeze_encoder=bool(encoder.get("freeze_encoder", True)),
            )
        else:
            self.encoder = FeatureEncoder(config.student)
            head = config.student.head
            self.head = MLPHead(
                self.encoder.output_dim,
                len(config.task.labels),
                hidden_dim=int(head.get("hidden_dim", 32)),
                dropout=float(head.get("dropout", 0.1)),
            )

    def forward(self, samples: list[Sample]):
        if self.decision_student is not None:
            return self.decision_student(samples)
        return self.head(self.encoder(samples))

    def _loss(self, batch: Batch):
        logits = self(batch.samples); losses = []
        valid = batch.human >= 0
        if valid.any(): losses.append(self.config.training.alpha_human * F.cross_entropy(logits[valid], batch.human[valid]))
        if batch.teacher_mask.any():
            temperature = self.config.training.distillation_temperature
            soft = F.log_softmax(logits[batch.teacher_mask] / temperature, dim=-1)
            losses.append(self.config.training.beta_teacher * F.kl_div(soft, batch.teacher[batch.teacher_mask], reduction="batchmean") * temperature**2)
        return sum(losses) if losses else logits.sum() * 0

    def training_step(self, batch, _):
        loss = self._loss(batch); self.log("train_loss", loss, prog_bar=True); return loss
    def validation_step(self, batch, _):
        loss = self._loss(batch); self.log("val_loss", loss, prog_bar=True); return loss
    def configure_optimizers(self): return torch.optim.AdamW(self.parameters(), lr=self.config.training.learning_rate)


def train(config: AppConfig) -> Path:
    L.seed_everything(config.training.seed, workers=True)
    samples = read_jsonl(config.data_path)
    rows = Path(config.run_dir, "aggregated_labels.jsonl").read_text().splitlines()
    labels = {item.sample_id: item for item in (AggregatedLabel.model_validate_json(row) for row in rows)}
    random.Random(config.training.seed).shuffle(samples); cut = max(1, int(.8 * len(samples))); train_data = SpecialistDataset(samples[:cut], labels, config.task.labels); validation_data = SpecialistDataset(samples[cut:], labels, config.task.labels)
    module = SpecialistModule(config); run = Path(config.run_dir); ckpt = L.pytorch.callbacks.ModelCheckpoint(dirpath=run, filename="best", monitor="val_loss", mode="min", save_top_k=1)
    trainer = L.Trainer(max_epochs=config.training.epochs, accelerator=config.training.accelerator, devices=1, logger=False, enable_progress_bar=False, callbacks=[ckpt], deterministic=True, enable_model_summary=False)
    trainer.fit(module, DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True, collate_fn=collate), DataLoader(validation_data, batch_size=config.training.batch_size, collate_fn=collate))
    checkpoint = Path(ckpt.best_model_path)
    (run / "training_manifest.json").write_text(json.dumps({"checkpoint": str(checkpoint), "labels": config.task.labels, "student": config.student.model_dump(), "task": config.task.model_dump()}, indent=2))
    return checkpoint


def load_student(checkpoint: str | Path) -> SpecialistModule:
    return SpecialistModule.load_from_checkpoint(str(checkpoint), map_location="cpu", weights_only=False)


def predict(module: SpecialistModule, samples: list[Sample]) -> list[dict]:
    module.eval()
    with torch.no_grad(): probabilities = torch.softmax(module(samples), dim=-1).tolist()
    labels = module.config.task.labels
    return [{"sample_id": sample.id, "probabilities": dict(zip(labels, row)), "predicted_label": labels[max(range(len(row)), key=row.__getitem__)], "confidence": max(row), "teacher_dependency": False} for sample, row in zip(samples, probabilities)]
