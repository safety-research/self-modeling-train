"""
Module: prompt_attribution/training/data/prompt_builder.py

Builds (prompt, completion) pairs for training.

Structure:
- TrainingPhase2Template: Subclass of Phase2Template with continuous probability output
- CompoundPhase2Template: Subclass that requests answer predictions + probability
- TrainingPromptBuilder: Converts TrainingRecord → (prompt, completion)
"""

import json
import logging
import re
from dataclasses import dataclass

from prompt_attribution.eval.attribution.config import Phase2PromptConfig
from prompt_attribution.eval.attribution.templates import Phase2Template
from prompt_attribution.training.config import ModelFormat
from prompt_attribution.training.data.dataset import TrainingRecord

logger = logging.getLogger(__name__)


class TrainingPhase2Template(Phase2Template):
    """Phase2Template subclass that uses continuous probability (0-1) instead of 1-5 scale.

    Overrides _format_yes_no_block to ask for a flip probability.
    JSON format instruction adapts to completion_format so the prompt
    matches what the target completion actually contains.
    """

    # Formats where the completion includes reasoning
    _REASONING_FORMATS = {"oracle", "template", "simulation", "oracle_with_simulation"}

    def __init__(
        self,
        config: Phase2PromptConfig,
        model_format: ModelFormat | None = None,
        completion_format: str = "oracle",
    ) -> None:
        super().__init__(config)
        self.model_format = model_format or ModelFormat()
        self.completion_format = completion_format

    def _format_yes_no_block(self, yes_explanation: str, no_explanation: str) -> str:
        """Format scoring block as continuous probability estimation."""
        context = f"""Estimate the probability (0.0 to 1.0) that this change would cause your answer to be different.

Context:
- A higher probability (0.7-1.0) means: {yes_explanation}
- A lower probability (0.0-0.3) means: {no_explanation}"""

        has_reasoning = self.completion_format in self._REASONING_FORMATS

        if self.model_format.thinking and has_reasoning:
            return context + """

Think carefully about the causal mechanism, then respond with:
{"flip_probability": <0.0-1.0>}"""
        elif has_reasoning:
            return context + """

Respond in JSON format:
{"reasoning": "<your explanation>", "flip_probability": <0.0-1.0>}
Do not include any text outside the JSON."""
        else:
            # label_only: no reasoning expected
            return context + """

Respond with:
{"flip_probability": <0.0-1.0>}"""


class CompoundPhase2Template(Phase2Template):
    """Phase2Template subclass that requests answer predictions + flip probability.

    The model must predict what the answer would be under baseline and lever
    conditions, alongside the flip probability. This enables compound reward
    evaluation of both outcome prediction and reasoning quality.
    """

    # Formats where the completion includes reasoning
    _REASONING_FORMATS = {"oracle", "template", "simulation", "oracle_with_simulation"}

    def __init__(
        self,
        config: Phase2PromptConfig,
        model_format: ModelFormat | None = None,
        completion_format: str = "oracle",
    ) -> None:
        super().__init__(config)
        self.model_format = model_format or ModelFormat()
        self.completion_format = completion_format

    def _format_yes_no_block(self, yes_explanation: str, no_explanation: str) -> str:
        """Format scoring block requesting answer predictions + probability."""
        context = f"""Analyze this change and predict its effect.

1. Predict what the answer would be WITHOUT the change (baseline condition).
2. Predict what the answer would be WITH the change applied.
3. Estimate the probability (0.0 to 1.0) that the change causes a different answer.

Context:
- A higher probability (0.7-1.0) means: {yes_explanation}
- A lower probability (0.0-0.3) means: {no_explanation}"""

        has_reasoning = self.completion_format in self._REASONING_FORMATS

        if self.model_format.thinking and has_reasoning:
            return context + """

Think through the causal mechanism and simulate both conditions, then respond with:
{"baseline_answer": "<predicted answer without change>", "lever_answer": "<predicted answer with change>", "flip_probability": <0.0-1.0>}"""
        elif has_reasoning:
            return context + """

Respond in JSON format:
{"reasoning": "<your explanation>", "baseline_answer": "<predicted answer without change>", "lever_answer": "<predicted answer with change>", "flip_probability": <0.0-1.0>}
Do not include any text outside the JSON."""
        else:
            # answer_prediction: no reasoning expected
            return context + """

Respond with:
{"baseline_answer": "<predicted answer without change>", "lever_answer": "<predicted answer with change>", "flip_probability": <0.0-1.0>}"""


# Default config for training: show both full Phase 1 prompts, no answer shown
TRAINING_PROMPT_CONFIG = Phase2PromptConfig(
    show_answer=False,
    show_both_phase1_prompts=True,
)


@dataclass
class PromptCompletionPair:
    """A (prompt, completion) pair for training."""

    prompt: str
    completion: str
    record_id: str
    category: str
    perturbation_type: str
    empirical_flip_fraction: float


class TrainingPromptBuilder:
    """Converts TrainingRecord → (prompt, completion) using Phase2Template.

    The prompt is the Phase 2 attribution query (asking about flip probability).
    The completion format varies by model family:
    - Thinking models: <think>reasoning</think>\\n\\n{"flip_probability": 0.7}
    - Non-thinking: {"reasoning": "...", "flip_probability": 0.7}

    Completion format can be configured:
    - "oracle": Use oracle_reasoning or template fallback (default)
    - "template": Always use brief template reasoning (categorized by flip probability)
    - "simulation": Embed actual Phase 1 responses in the completion,
      teaching the model that attribution = simulate + compare
    - "oracle_with_simulation": Oracle reasoning + LLM-summarized responses
    - "answer_prediction": Predict both baseline and lever answers + flip probability,
      no reasoning. Uses compound prompt style automatically.

    Prompt style can be configured:
    - "probability": Standard prompt requesting flip_probability only (default)
    - "compound": Extended prompt requesting baseline_answer, lever_answer, and flip_probability
    """

    def __init__(
        self,
        config: Phase2PromptConfig | None = None,
        model_format: ModelFormat | None = None,
        completion_format: str = "oracle",
        prompt_style: str = "probability",
    ) -> None:
        self.config = config or TRAINING_PROMPT_CONFIG
        self.model_format = model_format or ModelFormat()
        self.completion_format = completion_format
        self.prompt_style = prompt_style

        # answer_prediction always uses compound prompt style
        if completion_format == "answer_prediction":
            self.prompt_style = "compound"

        if self.prompt_style == "compound":
            self.template = CompoundPhase2Template(
                self.config, self.model_format, self.completion_format
            )
        else:
            self.template = TrainingPhase2Template(
                self.config, self.model_format, self.completion_format
            )

    def build(self, record: TrainingRecord) -> PromptCompletionPair:
        """Build a prompt/completion pair from a training record.

        Args:
            record: A training record with empirical_flip_fraction populated.

        Returns:
            PromptCompletionPair with the prompt and target completion.
        """
        # Find the target label axis info
        target_label = self._find_target_label(record)
        yes_explanation, no_explanation = self._build_explanations(target_label)
        attribution_question = self._build_attribution_question(target_label)

        # Build perturbation_metadata for problem_edit types
        perturbation_metadata = None
        if record.perturbation_type == "problem_edit":
            perturbation_metadata = {
                "perturbation_type": "problem_edit",
                "problem_edits": record.problem_edits,
                "edited_problem": record.question,  # For problem edits, lever has the edited version
                "lever_text": record.lever_text or self._summarize_edits(record.problem_edits),
            }

        # Use Phase2Template to build the prompt
        # For problem_edit, use prompt_baseline's question as the "original"
        prompt = self.template.build_prompt(
            problem=record.question,
            answer="",  # show_answer=False, so this is unused
            lever_instruction=record.lever_text,
            baseline_instruction=record.baseline_text,
            attribution_question=attribution_question,
            yes_explanation=yes_explanation,
            no_explanation=no_explanation,
            perturbation_metadata=perturbation_metadata,
            prompt_baseline=record.prompt_baseline,
            prompt_lever=record.prompt_lever,
        )

        # Build completion
        flip_prob = record.empirical_flip_fraction
        assert flip_prob is not None, f"Record {record.unique_id} has no empirical_flip_fraction"

        if self.completion_format == "answer_prediction":
            # Predict both answers + flip probability, no reasoning
            completion = self._format_answer_prediction(flip_prob, record)
        elif self.completion_format == "label_only":
            # No reasoning at all — just the flip probability JSON
            completion = self._format_label_only(flip_prob)
        elif self.completion_format == "template":
            # Always use template reasoning (brief categorized text, no oracle/simulation)
            reasoning = self._build_template_reasoning(record, flip_prob)
            completion = self._format_completion(reasoning, flip_prob, record)
        elif self.completion_format == "oracle_with_simulation":
            reasoning = self._build_oracle_with_simulation_reasoning(record, flip_prob)
            completion = self._format_completion(reasoning, flip_prob, record)
        elif self.completion_format == "simulation":
            reasoning = self._build_simulation_reasoning(record, flip_prob)
            completion = self._format_completion(reasoning, flip_prob, record)
        elif record.oracle_reasoning and not self._has_research_jargon(record.oracle_reasoning):
            # completion_format == "oracle": use oracle reasoning if available
            reasoning = record.oracle_reasoning
            completion = self._format_completion(reasoning, flip_prob, record)
        else:
            # oracle format fallback: template reasoning when oracle is missing/jargon
            if record.oracle_reasoning:
                logger.debug(
                    f"Record {record.unique_id}: oracle reasoning contains research jargon, "
                    "using template fallback"
                )
            reasoning = self._build_template_reasoning(record, flip_prob)
            completion = self._format_completion(reasoning, flip_prob, record)

        return PromptCompletionPair(
            prompt=prompt,
            completion=completion,
            record_id=record.unique_id,
            category=record.category,
            perturbation_type=record.perturbation_type,
            empirical_flip_fraction=flip_prob,
        )

    def _format_answer_prediction(self, flip_prob: float, record: TrainingRecord) -> str:
        """Format completion with predicted answers + flip probability, no reasoning.

        Uses compound prompt (model is asked to predict both answers).
        Completion: {"baseline_answer": "X", "lever_answer": "Y", "flip_probability": 0.7}
        """
        # Note: empirical_*_answer is resolved at load time via _resolve_answer(),
        # which prefers features_*[target_label_axis] over raw parse_answer() output.
        return json.dumps({
            "baseline_answer": record.empirical_baseline_answer or "",
            "lever_answer": record.empirical_lever_answer or "",
            "flip_probability": round(flip_prob, 2),
        })

    def _format_label_only(self, flip_prob: float) -> str:
        """Format completion with just the flip probability — no reasoning.

        Both thinking and non-thinking models get the same compact output.
        """
        return json.dumps({"flip_probability": round(flip_prob, 2)})

    def _format_completion(
        self,
        reasoning: str,
        flip_prob: float,
        record: TrainingRecord | None = None,
    ) -> str:
        """Format the completion string based on model family and prompt style.

        For probability style:
          Thinking models: <think>reasoning</think>\\n\\n{"flip_probability": 0.7}
          Non-thinking:    {"reasoning": "...", "flip_probability": 0.7}

        For compound style (adds answer predictions):
          Thinking models: <think>reasoning</think>\\n\\n{"baseline_answer": "B", "lever_answer": "A", "flip_probability": 0.7}
          Non-thinking:    {"reasoning": "...", "baseline_answer": "B", "lever_answer": "A", "flip_probability": 0.7}
        """
        # Build output dict
        output_dict: dict = {}

        if self.prompt_style == "compound" and record is not None:
            # Note: resolved at load time via _resolve_answer() — correct labels
            output_dict["baseline_answer"] = record.empirical_baseline_answer or ""
            output_dict["lever_answer"] = record.empirical_lever_answer or ""

        output_dict["flip_probability"] = round(flip_prob, 2)

        if self.model_format.thinking:
            return (
                f"{self.model_format.think_open}\n{reasoning}\n"
                f"{self.model_format.think_close}\n\n{json.dumps(output_dict)}"
            )
        else:
            output_dict_with_reasoning = {"reasoning": reasoning}
            output_dict_with_reasoning.update(output_dict)
            return json.dumps(output_dict_with_reasoning)

    def _find_target_label(self, record: TrainingRecord) -> dict:
        """Find the target label axis from answer_labels.

        Falls back to a human-readable description derived from the axis name
        when answer_labels is empty or the target axis isn't found.
        """
        for label in record.answer_labels:
            if label.get("name") == record.target_label_axis:
                if label.get("description"):
                    return label
        # Fallback: use first label if it has a description
        if record.answer_labels:
            first = record.answer_labels[0]
            if first.get("description"):
                return first
        # No labels or no descriptions — derive from axis name
        # e.g., "answer_substance" → "the answer substance",
        #        "final_numeric_answer" → "the final numeric answer"
        axis = record.target_label_axis or "answer"
        desc = "the " + axis.replace("_", " ")
        return {"name": axis, "description": desc}

    @staticmethod
    def _lowercase_desc(desc: str) -> str:
        """Lowercase first character of a label description for mid-sentence use.

        answer_labels descriptions come capitalized (e.g., "The letter choice")
        but they appear mid-sentence after "affect" or "cause", so the first
        character must be lowercase.
        """
        if desc and desc[0].isupper():
            return desc[0].lower() + desc[1:]
        return desc

    def _build_explanations(self, target_label: dict) -> tuple[str, str]:
        """Build yes/no explanations from the target label description."""
        desc = self._lowercase_desc(target_label.get("description", "the answer"))
        yes_explanation = f"The change would likely cause {desc} to be different"
        no_explanation = f"The change would likely NOT affect {desc}"
        return yes_explanation, no_explanation

    def _build_attribution_question(self, target_label: dict) -> str:
        """Build the attribution question from the target label axis."""
        desc = self._lowercase_desc(target_label.get("description", "your answer"))
        return f"Would this change affect {desc}?"

    def _build_oracle_with_simulation_reasoning(
        self, record: TrainingRecord, flip_prob: float
    ) -> str:
        """Build reasoning that combines LLM-summarized responses + oracle causal analysis.

        Structure:
        1. LLM-summarized response without the change
        2. LLM-summarized response with the change
        3. Oracle causal mechanism reasoning (or template fallback)

        Summaries are pre-computed by ResponseSummarizer (Claude Haiku) before training.
        Falls back to raw truncation if summaries are not available.
        Oracle reasoning with research jargon is rejected and replaced with template.
        """
        # Use pre-computed LLM summaries, fall back to raw truncation
        baseline_resp = record.summarized_baseline_response
        if not baseline_resp:
            baseline_resp = self._truncate_response(
                record.empirical_baseline_responses[0] if record.empirical_baseline_responses else ""
            )

        lever_resp = record.summarized_lever_response
        if not lever_resp:
            lever_resp = self._truncate_response(
                record.empirical_lever_responses[0] if record.empirical_lever_responses else ""
            )

        # Get oracle reasoning or fall back to template.
        # Reject oracle reasoning that contains research jargon (e.g. "baseline",
        # "lever", "perturbation") — these terms don't appear in the Phase 2 prompt
        # the model sees at inference time, so training on them is harmful.
        if record.oracle_reasoning and not self._has_research_jargon(record.oracle_reasoning):
            causal_analysis = record.oracle_reasoning
        else:
            causal_analysis = self._build_template_reasoning(record, flip_prob)

        parts = []
        parts.append(f"Without the change: {baseline_resp or '(no response)'}")
        parts.append(f"With the change: {lever_resp or '(no response)'}")
        parts.append("")
        parts.append(f"Analysis: {causal_analysis}")

        return "\n".join(parts)

    @staticmethod
    def _strip_think_tags(response: str) -> str:
        """Strip <think>...</think> blocks from a response.

        Thinking model responses (Qwen3, DeepSeek) contain reasoning in
        <think> tags. When embedding responses inside a training completion
        that itself uses <think> tags, nested tags confuse the tokenizer.

        Handles:
        - Complete blocks: <think>...</think>
        - Unclosed blocks: <think>... (truncated responses)
        - Multiple blocks
        """
        if not response:
            return ""
        # Strip complete <think>...</think> blocks
        cleaned = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        # Strip unclosed <think> blocks (from truncated responses)
        cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
        # If stripping removed everything, keep content without tags
        if not cleaned.strip():
            cleaned = response.replace('<think>', '').replace('</think>', '')
        return cleaned.strip()

    @classmethod
    def _truncate_response(cls, response: str, max_chars: int = 300) -> str:
        """Strip thinking tags and truncate for fallback embedding."""
        cleaned = cls._strip_think_tags(response)
        if not cleaned:
            return ""
        if len(cleaned) > max_chars:
            return cleaned[:max_chars] + "..."
        return cleaned

    def _build_simulation_reasoning(self, record: TrainingRecord, flip_prob: float) -> str:
        """Build simulation-based reasoning that embeds actual Phase 1 responses.

        Teaches the model that attribution = simulate baseline + simulate lever + compare.

        For n_stability=1: embeds raw responses from first run (existing behavior).
        For n_stability>1: embeds summarized responses from all runs, with run labels.
        """
        # n_stability>1 path: use summarized responses for all runs
        if (
            len(record.summarized_baseline_responses) >= 2
            and len(record.summarized_lever_responses) >= 2
        ):
            return self._build_multi_run_simulation(record, flip_prob)

        # n_stability=1 path: use raw responses from first run
        baseline_resp = ""
        if record.empirical_baseline_responses:
            baseline_resp = self._strip_think_tags(record.empirical_baseline_responses[0])

        lever_resp = ""
        if record.empirical_lever_responses:
            lever_resp = self._strip_think_tags(record.empirical_lever_responses[0])

        if not baseline_resp and not lever_resp:
            logger.warning(
                f"Record {record.unique_id}: no Phase 1 responses for simulation, "
                "falling back to template reasoning"
            )
            return self._build_template_reasoning(record, flip_prob)

        # Build comparison text
        target_label = self._find_target_label(record)
        target_desc = self._lowercase_desc(target_label.get("description", "the answer"))

        if flip_prob >= 0.5:
            comparison = (
                f"the responses with and without the change would differ on {target_desc}, "
                f"so the flip probability is {flip_prob:.2f}."
            )
        else:
            comparison = (
                f"the responses with and without the change would be consistent on {target_desc}, "
                f"so the flip probability is {flip_prob:.2f}."
            )

        parts = ["Without the change, the response would be:"]
        parts.append(baseline_resp or "(no response)")
        parts.append("")
        parts.append("With the change, the response would be:")
        parts.append(lever_resp or "(no response)")
        parts.append("")
        parts.append(f"Therefore, {comparison}")

        return "\n".join(parts)

    def _build_multi_run_simulation(self, record: TrainingRecord, flip_prob: float) -> str:
        """Build simulation reasoning from multiple stability runs with summarized responses.

        Shows all N runs with their summarized responses, then an overall comparison.
        Used when n_stability>1 and summarized_*_responses lists are populated.
        """
        n_runs = min(
            len(record.summarized_baseline_responses),
            len(record.summarized_lever_responses),
        )
        flip_count = round(flip_prob * n_runs)

        target_label = self._find_target_label(record)
        target_desc = self._lowercase_desc(target_label.get("description", "the answer"))

        parts = []
        for i in range(n_runs):
            bl_summary = record.summarized_baseline_responses[i]
            lv_summary = record.summarized_lever_responses[i]
            parts.append(
                f"Run {i + 1}: Without the change: {bl_summary}. "
                f"With the change: {lv_summary}."
            )

        parts.append("")
        if flip_count == 0:
            parts.append(
                f"Overall, none of the {n_runs} runs changed {target_desc}, "
                f"so the flip probability is {flip_prob:.2f}."
            )
        elif flip_count == n_runs:
            parts.append(
                f"Overall, all {n_runs} runs changed {target_desc}, "
                f"so the flip probability is {flip_prob:.2f}."
            )
        else:
            parts.append(
                f"Overall, {flip_count} out of {n_runs} runs changed {target_desc}, "
                f"so the flip probability is {flip_prob:.2f}."
            )

        return "\n".join(parts)

    def _build_template_reasoning(self, record: TrainingRecord, flip_prob: float) -> str:
        """Build template reasoning for the completion.

        Provides a brief categorized reasoning based on the flip probability.
        """
        if flip_prob >= 0.7:
            return "This change is likely to alter the answer significantly."
        elif flip_prob >= 0.3:
            return "This change may or may not affect the answer."
        else:
            return "This change is unlikely to affect the answer."

    # Research jargon that should never appear in training completions.
    # These terms don't exist in the Phase 2 prompt the model sees at inference,
    # so training on them teaches the model to hallucinate vocabulary.
    _RESEARCH_JARGON_PATTERN = re.compile(
        r"\b(?:baseline|lever|perturbation|perturbed|flip rate|flip fraction"
        r"|empirical|phase 1|phase 2|control group|experimental group)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _has_research_jargon(cls, text: str) -> bool:
        """Check if text contains research jargon that shouldn't appear in completions."""
        return bool(cls._RESEARCH_JARGON_PATTERN.search(text))

    def _summarize_edits(self, problem_edits: list[dict]) -> str:
        """Create a summary of problem edits."""
        parts = []
        for edit in problem_edits:
            desc = edit.get("description", "")
            if desc:
                parts.append(desc)
            else:
                orig = edit.get("original", "")
                repl = edit.get("replacement", "")
                parts.append(f'Changed "{orig}" to "{repl}"')
        return "; ".join(parts) if parts else "Problem was edited"
