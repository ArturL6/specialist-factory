"""Teacher implementations and factory."""
from .base import Teacher
from .implementations import (
    HuggingFaceTextTeacher,
    HuggingFaceVisionTeacher,
    MockTeacher,
    OpenRouterVisionTeacher,
    build_teacher,
)

__all__ = ["HuggingFaceTextTeacher", "HuggingFaceVisionTeacher", "MockTeacher", "OpenRouterVisionTeacher", "Teacher", "build_teacher"]
