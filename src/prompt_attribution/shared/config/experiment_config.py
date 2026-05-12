"""
Module: prompt_attribution/shared/config/experiment_config.py

Structure:
- ModelConfig:        Configuration for a model to test
- PerturbationConfig: Configuration for a perturbation
- get_provider:       Map a model_id to its provider string
"""

from dataclasses import dataclass
from typing import Optional

from .sweep_defaults import TOGETHER_PREFIXES


@dataclass
class ModelConfig:
    """Configuration for a model to test."""

    model_id: str
    temperature: float = 0.0
    max_tokens: int = 1024
    thinking_budget: Optional[int] = None
    system_prompt: Optional[str] = None
    force_provider: Optional[str] = None  # e.g. "together" to override auto-detect


@dataclass
class PerturbationConfig:
    """Configuration for a perturbation."""

    perturbation_id: str
    description: str
    baseline: str
    lever: str
    # E9 baseline behavioral feature (benchmark-level, e.g. response length)
    feature_description: Optional[str] = None
    target_features: Optional[list[str]] = None
    feature_target_value: Optional[str] = None
    # E1/E3 flip feature: what the perturbation changes (perturbation-driven)
    flip_feature_description: Optional[str] = None
    flip_target_features: Optional[list[str]] = None
    is_control: bool = False


def get_provider(model_id: str) -> str:
    """Determine provider from model ID."""
    if model_id.startswith("claude"):
        return "anthropic"
    if model_id.startswith("gpt") or model_id.startswith(("o1", "o3", "o4")):
        return "openai"
    if model_id.startswith("gemini"):
        return "gemini"
    if model_id.startswith("openrouter/"):
        return "openrouter"
    if model_id.startswith(("moonshotai/", "x-ai/")):
        return "openrouter"  # Kimi and Grok via OpenRouter
    if any(model_id.startswith(prefix) for prefix in TOGETHER_PREFIXES):
        return "together"
    raise ValueError(f"Unknown provider for model: {model_id}")
