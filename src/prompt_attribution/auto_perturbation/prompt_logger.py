"""
Module: prompt_attribution/auto_perturbation/prompt_logger.py

Logs all LLM prompts and responses to a readable markdown file.
Each component (decomposer, generator, critic) writes its prompts
with clear visual separation for easy debugging in VSCode.

Usage:
    logger = PromptLogger(run_dir / "prompts.md")
    logger.log("decomposer", "problem_0", system_prompt, user_prompt, response)
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PromptLogger:
    """Logs LLM prompts and responses to a markdown file.

    Creates a single prompts.md file with all LLM interactions,
    organized by component and problem index. Renders well in VSCode.
    """

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._count = 0
        # Write header
        with open(self.output_path, "w") as f:
            f.write("# Pipeline Prompt Log\n\n")
            f.write("All LLM calls in execution order.\n\n")
            f.write("---\n\n")

    def log(
        self,
        component: str,
        label: str,
        system_prompt: str = "",
        user_prompt: str = "",
        response: str = "",
        extra: Optional[dict] = None,
    ) -> None:
        """Log a single LLM call.

        Args:
            component: Which pipeline component (decomposer, generator, critic, label_ideation)
            label: Identifier (e.g., "problem_0", "problem_2_flip_inducing")
            system_prompt: System message content
            user_prompt: User message content
            response: LLM response text
            extra: Optional metadata (model, temperature, etc.)
        """
        self._count += 1

        with open(self.output_path, "a") as f:
            f.write(f"## [{self._count}] {component} — {label}\n\n")

            if extra:
                f.write("| Key | Value |\n|-----|-------|\n")
                for k, v in extra.items():
                    f.write(f"| {k} | {v} |\n")
                f.write("\n")

            if system_prompt:
                f.write("### System Prompt\n\n")
                f.write("```\n")
                f.write(system_prompt)
                f.write("\n```\n\n")

            if user_prompt:
                f.write("### User Prompt\n\n")
                f.write("```\n")
                f.write(user_prompt)
                f.write("\n```\n\n")

            if response:
                f.write("### Response\n\n")
                f.write("```\n")
                f.write(response[:5000])  # Cap at 5k chars
                if len(response) > 5000:
                    f.write(f"\n... ({len(response)} chars total, truncated)")
                f.write("\n```\n\n")

            f.write("---\n\n")
