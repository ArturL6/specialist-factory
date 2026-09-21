from __future__ import annotations

from typing import Protocol

from ..schemas import Sample, TaskDefinition, TeacherSignal


class Teacher(Protocol):
    teacher_id: str

    def predict(self, samples: list[Sample], task: TaskDefinition) -> list[TeacherSignal]: ...
