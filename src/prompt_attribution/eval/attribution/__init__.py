"""
Module: prompt_attribution/phase2

Phase 2 prompt engineering system for attribution queries.
This module provides configurable templates for generating Phase 2 prompts.

Exports:
- FewShotExample: A single few-shot example
- Phase2PromptConfig: Configuration dataclass for prompt generation
- Phase2Template: Template class that generates attribution prompts
- create_phase2_template: Factory function to create templates
"""

from .config import FewShotExample, Phase2PromptConfig
from .templates import Phase2Template, create_phase2_template

__all__ = [
    "FewShotExample",
    "Phase2PromptConfig",
    "Phase2Template",
    "create_phase2_template",
]
