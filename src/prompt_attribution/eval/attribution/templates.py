"""
Module: prompt_attribution/phase2/templates.py

Phase 2 attribution template that generates prompts for testing model attribution ability.

Structure:
- Phase2Template: Main template class for attribution prompts
- create_phase2_template: Factory function to create templates
"""

from typing import Any

from .config import Phase2PromptConfig


class Phase2Template:
    """Phase 2 attribution template.

    Two symmetric variants (controlled by config.symmetric_variant):
    - baseline_shown: Shows baseline answer, asks about counterfactual with lever
    - lever_shown: Shows lever answer, asks about counterfactual without lever

    Three perturbation types:
    - Replacement: When baseline_instruction is not empty, lever replaces baseline
    - Add/Remove: When baseline_instruction is empty, lever is added/removed
    - Problem Edit: When perturbation_metadata has perturbation_type="problem_edit"

    Configuration options:
    - reasoning_order: Control response format (explain_first, answer_first, direct)
    - show_answer: Whether to display the answer in the prompt
    - few_shot_examples: Optional few-shot examples
    - system_prompt: Optional custom system prompt
    - show_full_phase1_prompt: Show full Phase 1 prompt with instruction marked
    """

    def __init__(self, config: Phase2PromptConfig):
        """Initialize template with configuration.

        Args:
            config: Phase2PromptConfig with symmetric_variant setting
        """
        self.config = config

    @property
    def variant(self) -> str:
        """Returns symmetric_variant for convenient access."""
        return self.config.symmetric_variant

    def make_attribution_prompt(
        self,
        problem: str,
        answer: str,
        lever_instruction: str,
        baseline_instruction: str,
        attribution_question: str,
        yes_explanation: str,
        no_explanation: str,
        domain: Any = None,
        full_phase1_prompt: str | None = None,
        perturbation_metadata: dict | None = None,
        **kwargs: Any,
    ) -> str:
        """Alias for build_prompt() for convenience."""
        return self.build_prompt(
            problem=problem,
            answer=answer,
            lever_instruction=lever_instruction,
            baseline_instruction=baseline_instruction,
            attribution_question=attribution_question,
            yes_explanation=yes_explanation,
            no_explanation=no_explanation,
            domain=domain,
            full_phase1_prompt=full_phase1_prompt,
            perturbation_metadata=perturbation_metadata,
            **kwargs,
        )

    def build_prompt(
        self,
        problem: str,
        answer: str,
        lever_instruction: str,
        baseline_instruction: str,
        attribution_question: str,
        yes_explanation: str,
        no_explanation: str,
        domain: Any = None,
        full_phase1_prompt: str | None = None,
        perturbation_metadata: dict | None = None,
        prompt_baseline: str | None = None,
        prompt_lever: str | None = None,
    ) -> str:
        """Build attribution prompt for an example.

        Args:
            problem: The problem/question text
            answer: The model's answer (from baseline or lever condition)
            lever_instruction: The lever instruction text
            baseline_instruction: The baseline instruction text (empty for add/remove type)
            attribution_question: Domain-specific question (from config)
            yes_explanation: What YES means (domain-specific, built by caller)
            no_explanation: What NO means (domain-specific, built by caller)
            domain: Domain object for custom prompt formatting (optional)
            full_phase1_prompt: Full Phase 1 prompt for show_full_phase1_prompt mode (optional)
            perturbation_metadata: Optional dict for problem-edit perturbations with:
                - "perturbation_type": "problem_edit"
                - "problem_edits": list of ProblemEdit dicts
                - "edited_problem": The problem text after edits applied
                - "lever_text": Human-readable edit summary
            prompt_baseline: Full Phase 1 baseline prompt (for show_both_phase1_prompts)
            prompt_lever: Full Phase 1 lever prompt (for show_both_phase1_prompts)

        Returns:
            Attribution prompt string
        """
        # Show both full Phase 1 prompts side by side (highest priority —
        # works for all perturbation types including problem_edit)
        if self.config.show_both_phase1_prompts and prompt_baseline and prompt_lever:
            return self._build_both_prompts_variant(
                prompt_baseline=prompt_baseline,
                prompt_lever=prompt_lever,
                attribution_question=attribution_question,
                yes_explanation=yes_explanation,
                no_explanation=no_explanation,
            )

        # Handle problem-edit perturbations (when not using both-prompts mode)
        if perturbation_metadata and perturbation_metadata.get("perturbation_type") == "problem_edit":
            return self._build_problem_edit_prompt(
                problem=problem,
                answer=answer,
                attribution_question=attribution_question,
                yes_explanation=yes_explanation,
                no_explanation=no_explanation,
                perturbation_metadata=perturbation_metadata,
            )

        # Check if we should use full prompt variant
        if self.config.show_full_phase1_prompt and full_phase1_prompt:
            return self._build_full_prompt_variant(
                full_phase1_prompt=full_phase1_prompt,
                lever_instruction=lever_instruction,
                baseline_instruction=baseline_instruction,
                answer=answer,
                attribution_question=attribution_question,
                yes_explanation=yes_explanation,
                no_explanation=no_explanation,
            )

        # Determine perturbation type
        is_replacement = bool(baseline_instruction.strip())

        if self.config.symmetric_variant == "baseline_shown":
            return self._make_baseline_shown_prompt(
                problem, answer, lever_instruction, baseline_instruction,
                attribution_question, yes_explanation, no_explanation, is_replacement,
                domain=domain
            )
        else:
            return self._make_lever_shown_prompt(
                problem, answer, lever_instruction, baseline_instruction,
                attribution_question, yes_explanation, no_explanation, is_replacement,
                domain=domain
            )

    def _make_baseline_shown_prompt(
        self,
        problem: str,
        answer: str,
        lever_instruction: str,
        baseline_instruction: str,
        attribution_question: str,
        yes_explanation: str,
        no_explanation: str,
        is_replacement: bool,
        domain: Any = None,
    ) -> str:
        """Create baseline_shown attribution prompt.

        Shows baseline answer, asks about counterfactual with lever.
        """
        # Format the baseline prompt block (shows baseline instruction if present)
        # Use domain's method if available, otherwise use default
        if domain is not None and hasattr(domain, 'format_prompt_for_attribution'):
            prompt_block = domain.format_prompt_for_attribution(problem, baseline_instruction)
        else:
            prompt_block = self._format_prompt_block(problem, baseline_instruction)

        # Build the counterfactual description based on perturbation type
        counterfactual = self._format_counterfactual_add(
            lever_instruction, baseline_instruction, is_replacement
        )

        # Build YES/NO explanation
        yes_no_block = self._format_yes_no_block(yes_explanation, no_explanation)

        # Build answer block (conditional on show_answer)
        answer_block = self._format_answer_block(answer)

        return f"""Consider the following problem and your answer:

{prompt_block}
{answer_block}
{counterfactual}

{attribution_question}

{yes_no_block}"""

    def _make_lever_shown_prompt(
        self,
        problem: str,
        answer: str,
        lever_instruction: str,
        baseline_instruction: str,
        attribution_question: str,
        yes_explanation: str,
        no_explanation: str,
        is_replacement: bool,
        domain: Any = None,
    ) -> str:
        """Create lever_shown attribution prompt.

        Shows lever answer, asks about counterfactual without lever.
        """
        # Format the lever prompt block (shows lever instruction)
        # Use domain's method if available, otherwise use default
        if domain is not None and hasattr(domain, 'format_prompt_for_attribution'):
            prompt_block = domain.format_prompt_for_attribution(problem, lever_instruction)
        else:
            prompt_block = self._format_prompt_block(problem, lever_instruction)

        # Build the counterfactual description based on perturbation type
        counterfactual = self._format_counterfactual_remove(
            lever_instruction, baseline_instruction, is_replacement
        )

        # Build YES/NO explanation
        yes_no_block = self._format_yes_no_block(yes_explanation, no_explanation)

        # Build answer block (conditional on show_answer)
        answer_block = self._format_answer_block(answer)

        return f"""Consider the following problem and your answer:

{prompt_block}
{answer_block}
{counterfactual}

{attribution_question}

{yes_no_block}"""

    def _build_full_prompt_variant(
        self,
        full_phase1_prompt: str,
        lever_instruction: str,
        baseline_instruction: str,
        answer: str,
        attribution_question: str,
        yes_explanation: str,
        no_explanation: str,
    ) -> str:
        """Build prompt showing full Phase 1 context with instruction marked.

        This variant shows the complete Phase 1 prompt with the relevant instruction
        highlighted using >>> ... <<< markers, so the model can see exactly WHERE
        the instruction was placed in the original prompt.

        The prompt shown matches what produced the answer:
        - baseline_shown + replacement: Show baseline prompt with baseline instruction marked
        - baseline_shown + add/remove: Show lever prompt with lever instruction marked
        - lever_shown: Show lever prompt with lever instruction marked
        """
        # Determine perturbation type
        is_replacement = bool(baseline_instruction.strip())

        # Determine which instruction to mark based on variant and perturbation type
        if self.config.symmetric_variant == "baseline_shown" and is_replacement:
            # Mark the baseline instruction in the baseline prompt
            instruction_to_mark = baseline_instruction
        else:
            # Mark the lever instruction in the lever prompt
            instruction_to_mark = lever_instruction

        # Mark the instruction in the prompt
        marked_prompt = full_phase1_prompt.replace(
            instruction_to_mark,
            f">>> {instruction_to_mark} <<<"
        )

        # Build YES/NO explanation block
        yes_no_block = self._format_yes_no_block(yes_explanation, no_explanation)

        # Build answer block (conditional on show_answer)
        answer_block = self._format_answer_block(answer)

        # Build counterfactual description based on variant and perturbation type
        if self.config.symmetric_variant == "baseline_shown":
            # baseline_shown: Answer was produced from this prompt (baseline)
            # Ask: "what if lever instruction was added/replaced?"
            if is_replacement:
                # Baseline had different instruction, ask about replacing with lever
                counterfactual = f"""Your answer was produced from this prompt.

Now suppose the marked instruction (>>> ... <<<) was replaced with:
"{lever_instruction}\""""
            else:
                # Baseline had no instruction, ask about adding lever
                # (We show lever prompt to indicate WHERE it would go)
                counterfactual = """Your answer was produced from a prompt WITHOUT the marked instruction.

Now suppose the marked instruction (>>> ... <<<) WAS included."""
        else:
            # lever_shown: Answer was produced from this prompt (lever)
            # Ask: "what if lever instruction was removed/replaced?"
            intro = "Your answer was produced from this prompt."
            if is_replacement:
                counterfactual = f"""{intro}

Now suppose the marked instruction (>>> ... <<<) was replaced with:
"{baseline_instruction}\""""
            else:
                counterfactual = f"""{intro}

Now suppose the marked instruction (>>> ... <<<) was NOT included."""

        return f"""Consider the following problem and your answer.

The prompt structure was:
---
{marked_prompt}
---
{answer_block}
{counterfactual}

{attribution_question}

{yes_no_block}"""

    def _build_both_prompts_variant(
        self,
        prompt_baseline: str,
        prompt_lever: str,
        attribution_question: str,
        yes_explanation: str,
        no_explanation: str,
    ) -> str:
        """Build prompt showing both full Phase 1 prompts side by side.

        Instead of describing the change abstractly ("Now suppose..."), this
        shows the model exactly what both versions of the prompt look like.
        The model can directly compare Version A and Version B to reason
        about the effect of the change.
        """
        yes_no_block = self._format_yes_no_block(yes_explanation, no_explanation)

        return f"""Consider the following two versions of a problem prompt.

Version A (original):
---
{prompt_baseline}
---

Version B (modified):
---
{prompt_lever}
---

{attribution_question}

{yes_no_block}"""

    def _format_counterfactual_add(
        self,
        lever_instruction: str,
        baseline_instruction: str,
        is_replacement: bool,
    ) -> str:
        """Format counterfactual description for baseline_shown (adding lever).

        For replacement: "instruction X was replaced with Y"
        For add: "prompt included this instruction"
        """
        if is_replacement:
            return f"""Now suppose the instruction:
"{baseline_instruction}"

was replaced with:
"{lever_instruction}\""""
        else:
            return f"""Now suppose the prompt included this instruction:
"{lever_instruction}\""""

    def _format_counterfactual_remove(
        self,
        lever_instruction: str,
        baseline_instruction: str,
        is_replacement: bool,
    ) -> str:
        """Format counterfactual description for lever_shown (removing lever).

        For replacement: "instruction Y was replaced with X"
        For remove: "prompt did NOT include the instruction"
        """
        if is_replacement:
            return f"""Now suppose the instruction:
"{lever_instruction}"

was replaced with:
"{baseline_instruction}\""""
        else:
            return f"""Now suppose the prompt did NOT include the instruction:
"{lever_instruction}\""""

    def _build_problem_edit_prompt(
        self,
        problem: str,
        answer: str,
        attribution_question: str,
        yes_explanation: str,
        no_explanation: str,
        perturbation_metadata: dict,
    ) -> str:
        """Build attribution prompt for problem-edit perturbations.

        Shows the original/edited problem and describes the edit, then asks
        whether the edit would change the answer.
        """
        edits = perturbation_metadata.get("problem_edits", [])
        edited_problem = perturbation_metadata.get("edited_problem", "")
        edit_description = self._format_edit_description(edits)

        yes_no_block = self._format_yes_no_block(yes_explanation, no_explanation)
        answer_block = self._format_answer_block(answer)

        if self.config.symmetric_variant == "baseline_shown":
            # Show original problem + answer, ask about edited version
            prompt_block = self._format_prompt_block(problem, "")
            counterfactual = f"""Now suppose the problem was modified as follows:
{edit_description}

The modified problem would read:
---
{edited_problem}
---"""
        else:
            # Show edited problem + answer, ask about original version
            prompt_block = self._format_prompt_block(edited_problem, "")
            counterfactual = f"""Now suppose the problem was in its original form, before these changes:
{edit_description}

The original problem was:
---
{problem}
---"""

        return f"""Consider the following problem and your answer:

{prompt_block}
{answer_block}
{counterfactual}

{attribution_question}

{yes_no_block}"""

    def _format_edit_description(self, edits: list[dict]) -> str:
        """Format a list of ProblemEdit dicts as a readable description."""
        if not edits:
            return "(edit details not available)"

        lines = []
        for i, edit in enumerate(edits, 1):
            original = edit.get("original", "")
            replacement = edit.get("replacement", "")
            description = edit.get("description", "")

            if description:
                lines.append(f'{i}. {description}')
                lines.append(f'   Changed: "{original}" → "{replacement}"')
            else:
                lines.append(f'{i}. Changed: "{original}" → "{replacement}"')

        return "\n".join(lines)

    def _format_yes_no_block(self, yes_explanation: str, no_explanation: str) -> str:
        """Format the scoring instruction block.

        Uses 1-5 scale to rate confidence that the instruction would change the answer.
        Response format varies based on self.config.reasoning_order:
        - explain_first: {"reasoning": "...", "score": N}
        - answer_first: {"score": N, "reasoning": "..."}
        - direct: {"score": N}
        """
        # Get response format instruction based on reasoning_order
        response_format = self._get_response_format_instruction()

        # Get few-shot examples block if present
        few_shot_block = self._format_few_shot_block()

        # Build the block, only including few-shot if present
        base_block = f"""Rate how likely the instruction change would affect your answer on a scale of 1-5:

1 = Definitely would NOT change my answer
2 = Probably would NOT change my answer
3 = Uncertain / Could go either way
4 = Probably WOULD change my answer
5 = Definitely WOULD change my answer

Context:
- A higher score (4-5) means: {yes_explanation}
- A lower score (1-2) means: {no_explanation}"""

        if few_shot_block:
            return f"{base_block}{few_shot_block}\n\n{response_format}"
        else:
            return f"{base_block}\n\n{response_format}"

    def _get_response_format_instruction(self) -> str:
        """Get response format instruction based on reasoning_order config.

        Returns:
            Response format instruction string
        """
        if self.config.reasoning_order == "explain_first":
            return 'Respond in JSON format:\n{"reasoning": "<brief explanation>", "score": <1-5>}\nDo not include any text outside the JSON.'
        elif self.config.reasoning_order == "answer_first":
            return 'Respond in JSON format:\n{"score": <1-5>, "reasoning": "<brief explanation>"}\nDo not include any text outside the JSON.'
        else:  # direct
            return 'Respond with just: {"score": <1-5>}\nDo not include any text outside the JSON.'
        
    def _format_answer_block(self, answer: str) -> str:
        """Format the answer block based on show_answer config.

        Args:
            answer: The answer to display

        Returns:
            Formatted answer block (empty string if show_answer=False)
        """
        if not self.config.show_answer:
            return ""

        # When show_answer=True, display the answer
        # The answer_format config is a hint to the caller about what to pass
        # (full_response, parsed_only, or label_only)
        return f"\nYour answer: {answer}\n"

    def _format_few_shot_block(self) -> str:
        """Format few-shot examples if present.

        Returns:
            Formatted few-shot block (empty string if no examples)
        """
        if not self.config.few_shot_examples:
            return ""

        examples_text = []
        for i, ex in enumerate(self.config.few_shot_examples, 1):
            # Build the example response based on reasoning_order
            if self.config.reasoning_order == "direct":
                response = f'{{"score": {ex.score}}}'
            elif ex.reasoning:
                if self.config.reasoning_order == "explain_first":
                    response = f'{{"reasoning": "{ex.reasoning}", "score": {ex.score}}}'
                else:  # answer_first
                    response = f'{{"score": {ex.score}, "reasoning": "{ex.reasoning}"}}'
            else:
                # No reasoning provided, just show score
                response = f'{{"score": {ex.score}}}'

            example_text = f"""
Example {i}:
- Problem: {ex.problem_summary}
- Answer: {ex.answer}
- Instruction change: {ex.instruction_change}
- Response: {response}"""
            examples_text.append(example_text)

        return "\n\nHere are some examples of how to respond:" + "".join(examples_text)

    def _format_prompt_block(self, problem: str, instruction: str) -> str:
        """Format prompt block for display in attribution query.

        The problem text should already include benchmark-specific framing
        (e.g., "Complete the following Python function:" for coding,
        "Solve the problem. Put your final numerical answer in \\boxed{}." for math).
        """
        lines = ["---"]
        lines.append(problem)

        if instruction:
            lines.append("")
            lines.append(instruction)

        lines.append("---")

        return "\n".join(lines)


def create_phase2_template(config: Phase2PromptConfig) -> Phase2Template:
    """Create Phase 2 template from configuration.

    This is the main entry point for the new template system.

    Args:
        config: Phase2PromptConfig with settings

    Returns:
        Phase2Template instance
    """
    return Phase2Template(config)
