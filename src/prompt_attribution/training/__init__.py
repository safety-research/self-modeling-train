"""
Module: prompt_attribution/training/__init__.py

GRPO training pipeline (Tinker LoRA): single-task and multitask training
for self-modeling tasks.
"""

from .config import (
    DataConfig,
    GRPOConfig,
    ModelFormat,
    MultitaskDataConfig,
    TrainingSchedule,
)

__all__ = [
    "DataConfig",
    "GRPOConfig",
    "ModelFormat",
    "MultitaskDataConfig",
    "TrainingSchedule",
]
