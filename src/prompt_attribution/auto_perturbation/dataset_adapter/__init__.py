"""Stage 1: Dataset auto-adapter — zero-config onboarding from HuggingFace."""

from .dataset_adapter import (
    TaskType,
    DatasetProfile,
    DatasetDetector,
    DatasetAdapter,
    AdaptedExample,
)
from .answer_parser import (
    extract_features,
    extract_features_async,
    AnswerParser,
    GenericVerifier,  # backwards compat alias
)
from .label_ideation import AnswerLabel, LabelIdeator

__all__ = [
    "TaskType",
    "DatasetProfile",
    "DatasetDetector",
    "DatasetAdapter",
    "AdaptedExample",
    "extract_features",
    "extract_features_async",
    "AnswerParser",
    "GenericVerifier",
    "AnswerLabel",
    "LabelIdeator",
]
