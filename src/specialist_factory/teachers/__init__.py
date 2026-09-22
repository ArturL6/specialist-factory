"""Teacher implementations and factory."""

from .base import Teacher
from .implementations import (
    HuggingFaceTextTeacher,
    HuggingFaceVisionTeacher,
    JevTeacher,
    MockTeacher,
    OpenRouterVisionTeacher,
    build_teacher,
)

__all__ = [
    "HuggingFaceTextTeacher",
    "HuggingFaceVisionTeacher",
    "JevTeacher",
    "MockTeacher",
    "OpenRouterVisionTeacher",
    "Teacher",
    "build_teacher",
]
