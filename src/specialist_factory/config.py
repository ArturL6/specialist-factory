from __future__ import annotations

import json
from pathlib import Path

import yaml

from .schemas import AppConfig, Sample


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AppConfig.model_validate(raw)


def read_jsonl(path: str | Path) -> list[Sample]:
    source = Path(path)
    if not source.exists():
        return []
    return [Sample.model_validate_json(line) for line in source.read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[Sample] | list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row.model_dump() if hasattr(row, "model_dump") else row) + "\n" for row in rows)
    )
