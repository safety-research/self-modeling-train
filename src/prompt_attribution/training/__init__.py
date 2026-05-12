"""
Module: prompt_attribution/training/__init__.py

RL training pipeline (Tinker LoRA): single-task and multitask training
for self-modeling tasks.
"""

from .config import (
    DataConfig,
    RLConfig,
    ModelFormat,
    MultitaskDataConfig,
    TrainingSchedule,
)

__all__ = [
    "DataConfig",
    "RLConfig",
    "ModelFormat",
    "MultitaskDataConfig",
    "TrainingSchedule",
]
