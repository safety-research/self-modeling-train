"""
Module: prompt_attribution/phase2/config.py

Configuration dataclasses for Phase 2 prompt generation.

Structure:
- FewShotExample: A single few-shot example for attribution prompts
- Phase2PromptConfig: Main configuration for attribution prompts
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class FewShotExample:
    """A single few-shot example for Phase 2 attribution prompts.

    Attributes:
        problem_summary: Brief description of the problem (1-2 sentences)
        answer: The model's answer to display
        instruction_change: Description of the instruction change
        score: Expected 1-5 score
        reasoning: Brief reasoning for the score (optional)
    """
    problem_summary: str
    answer: str
    instruction_change: str
    score: int
    reasoning: str = ""


@dataclass
class Phase2PromptConfig:
    """Configuration for Phase 2 attribution prompt generation.

    Attributes:
        symmetric_variant: Which answer to show (baseline_shown or lever_shown).
            - baseline_shown: Shows baseline answer, asks about counterfactual with lever
            - lever_shown: Shows lever answer, asks about counterfactual without lever
        reasoning_order: Order of reasoning and score in the response.
            - explain_first: {"reasoning": "...", "score": N} (default)
            - answer_first: {"score": N, "reasoning": "..."}
            - direct: {"score": N} (no explanation required)
        show_answer: Whether to display the answer in the prompt.
            - True: Show the answer (default)
            - False: Hide the answer (model must infer from context)
        answer_format: How to format the answer when show_answer=True.
            - parsed_only: Show only the parsed answer (e.g., "4") (default)
            - full_response: Show the full model response
            - label_only: Show only the label (e.g., "A" for MCQ)
        few_shot_examples: List of few-shot examples to include in the prompt.
        system_prompt: System prompt to use for Phase 2 queries (None = no system prompt).
    """

    # Symmetric variant (baseline_shown or lever_shown)
    # This is set per-run based on template_variants in ExperimentConfig
    symmetric_variant: Literal["baseline_shown", "lever_shown"] = "baseline_shown"

    # Reasoning order for response format
    reasoning_order: Literal["explain_first", "answer_first", "direct"] = "explain_first"

    # Answer display options
    show_answer: bool = True
    answer_format: Literal["full_response", "parsed_only", "label_only"] = "parsed_only"

    # Few-shot examples (domain-specific)
    # When provided, these examples are shown before asking for the response
    few_shot_examples: list[FewShotExample] | None = None

    # System prompt for Phase 2 queries
    # When set, this overrides any default system prompt
    system_prompt: str | None = None

    # Show full Phase 1 prompt with instruction marked
    # When True, shows the complete prompt structure instead of just the instruction
    # This helps the model understand WHERE the instruction was placed
    show_full_phase1_prompt: bool = False

    # Show both Phase 1 prompts (baseline AND lever) side by side
    # When True, the attribution prompt shows both full prompts so the model
    # can see exactly what changed, instead of describing the change abstractly.
    # Takes priority over show_full_phase1_prompt when both are True.
    show_both_phase1_prompts: bool = False

    # Future fields (will be added in subsequent plans):
    # counterfactual_style: Literal["suppose", "what_if", "imagine"] = "suppose"
