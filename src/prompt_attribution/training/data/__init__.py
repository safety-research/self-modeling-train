"""
Module: prompt_attribution/training/data/__init__.py

Data pipeline for training: loading, splitting, prompt building.
"""

from .dataset import TrainingDataset, TrainingRecord
from .prompt_builder import TrainingPromptBuilder

__all__ = [
    "TrainingDataset",
    "TrainingRecord",
    "TrainingPromptBuilder",
]
