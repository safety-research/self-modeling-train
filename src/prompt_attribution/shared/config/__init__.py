"""
Configuration module — dataclasses + provider detection.

Structure:
- experiment_config.py: ModelConfig, PerturbationConfig, get_provider
- sweep_defaults.py:    Provider-detection prefixes
"""

from .experiment_config import (
    ModelConfig,
    PerturbationConfig,
    get_provider,
)
from .sweep_defaults import TOGETHER_PREFIXES

__all__ = [
    "ModelConfig",
    "PerturbationConfig",
    "get_provider",
    "TOGETHER_PREFIXES",
]
