from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from .schemas import Sample, TaskDefinition, TeacherConfig, TeacherSignal


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class TeacherCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS teacher_cache (
            cache_key TEXT PRIMARY KEY, signal_json TEXT, latency_ms REAL, cost REAL,
            error TEXT, attempts INTEGER NOT NULL DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        self.conn.commit()

    def key(self, sample: Sample, task: TaskDefinition, teacher: TeacherConfig) -> str:
        return stable_hash({"sample": sample.model_dump(), "task": task.model_dump(), "teacher": teacher.model_dump()})

    def get(self, key: str) -> TeacherSignal | None:
        row = self.conn.execute("SELECT signal_json FROM teacher_cache WHERE cache_key=? AND error IS NULL", (key,)).fetchone()
        return TeacherSignal.model_validate_json(row[0]) if row else None

    def put(self, key: str, signal: TeacherSignal | None, *, latency_ms: float = 0, cost: float = 0, error: str | None = None, attempts: int = 1) -> None:
        self.conn.execute("INSERT OR REPLACE INTO teacher_cache VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)", (key, signal.model_dump_json() if signal else None, latency_ms, cost, error, attempts))
        self.conn.commit()
