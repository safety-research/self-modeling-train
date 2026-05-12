"""
Module: prompt_attribution/auto_perturbation/critic.py

Stage 4: Flip probability prediction and deduplication. Scores candidates
on predicted_flip_probability only — used for category calibration and
as a model-agnostic training label.

Structure:
- CRITIC_PROMPT: LLM prompt template
- CandidateCritic: Orchestrates flip prediction and dedup
"""

import asyncio
import json
import logging
import re
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from ..config import PerProblemCandidate, PipelineConfig, ProblemAnalysis
from ..dataset_adapter.dataset_adapter import DatasetAdapter

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt
# =============================================================================


CRITIC_PROMPT = """\
You are predicting whether this perturbation would actually change a \
frontier LLM's answer.

## Original Problem
{problem_text}

## Problem WITH Perturbation Applied
{perturbed_problem_text}

## Target Label Axis
{target_label_axis}: "{attribution_question}"

{label_descriptions}
## Perturbation Details
- **Type**: {perturbation_type}
- **Mechanism**: {mechanism_name}

## Your Task
If a frontier LLM solves both the original and perturbed problems, predict \
whether the **{target_label_axis}** would change.

Think carefully about:
- Does the perturbation actually affect the reasoning path for this axis?
- Would a frontier LLM arrive at a different value for {target_label_axis}?
- How robust are frontier LLMs to this type of modification?

Rate the predicted flip probability (0.0 to 1.0):
- 1.0: Almost certainly changes the {target_label_axis}
- 0.7: Probably changes it
- 0.5: Genuine coin flip
- 0.3: Probably does NOT change it
- 0.0: Almost certainly does NOT change it

## Output
Output ONLY a valid JSON object (notes FIRST, then score):
{{"notes": "Your reasoning about whether this flips the target axis", "predicted_flip_probability": 0.4}}"""


# =============================================================================
# Critic
# =============================================================================


class CandidateCritic:
    """Stage 4: Phase 1 flip probability prediction.

    Predicts whether a perturbation would actually change the model's
    answer on the targeted label axis. Shows both original and perturbed
    problems side-by-side with the per-axis attribution question.
    """

    def __init__(
        self,
        api: InferenceAPI,
        config: PipelineConfig,
        adapter: DatasetAdapter,
        prompt_logger=None,
        tracer=None,
    ):
        self.api = api
        self.config = config
        self.adapter = adapter
        self._prompt_logger = prompt_logger
        self._tracer = tracer

    async def review_and_filter(
        self,
        analysis: ProblemAnalysis,
        candidates: list[PerProblemCandidate],
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[PerProblemCandidate]:
        """Review candidates, score them, deduplicate, and filter.

        Args:
            analysis: The decomposed problem
            candidates: Generated candidates for this problem
            semaphore: Optional shared semaphore for concurrency control.
                If None, creates a local one (for standalone use).

        Returns:
            All candidates with scores populated (filtered by consistency)
        """
        viable = [c for c in candidates if c.is_viable]
        if not viable:
            return candidates

        sem = semaphore or asyncio.Semaphore(self.config.concurrency)

        async def bounded_score(c: PerProblemCandidate):
            async with sem:
                await self._score_candidate(analysis, c)

        await asyncio.gather(*[bounded_score(c) for c in viable])

        return candidates

    async def review_batch(
        self,
        analyses: list[ProblemAnalysis],
        candidates_by_problem: dict[int, list[PerProblemCandidate]],
    ) -> dict[int, list[PerProblemCandidate]]:
        """Review candidates for a batch of problems.

        Uses a single shared semaphore across all problems to bound
        total concurrency (not per-problem concurrency).

        Args:
            analyses: List of problem analyses
            candidates_by_problem: Dict of candidates per problem

        Returns:
            Updated candidates_by_problem with scores and filtering
        """
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def bounded_score(
            analysis: ProblemAnalysis,
            candidate: PerProblemCandidate,
        ):
            async with semaphore:
                await self._score_candidate(analysis, candidate)

        tasks = []
        for analysis in analyses:
            candidates = candidates_by_problem.get(analysis.example_idx, [])
            for c in candidates:
                if c.is_viable:
                    tasks.append(bounded_score(analysis, c))

        await asyncio.gather(*tasks)
        return candidates_by_problem

    async def _score_candidate(
        self,
        analysis: ProblemAnalysis,
        candidate: PerProblemCandidate,
        example: "AdaptedExample | None" = None,
    ) -> None:
        """Score a single candidate via LLM.

        Simulates the Phase 2 evaluator's view: shows the problem WITH
        perturbation applied and the attribution question for the targeted axis.
        """
        # Build the perturbed problem text (what the evaluator would see)
        problem_text = analysis.prompt_template or analysis.question
        if candidate.perturbation_type == "problem_edit" and candidate.problem_edits:
            # Apply edits to the problem text
            perturbed = problem_text
            for edit in candidate.problem_edits:
                if edit.original and edit.original in perturbed:
                    perturbed = perturbed.replace(edit.original, edit.replacement, 1)
            perturbed_problem_text = perturbed
        elif candidate.lever:
            # instruction_add: show problem with instruction inserted
            placement = candidate.instruction_placement or "append"
            perturbed_problem_text = (
                f"{problem_text}\n\n[Instruction at '{placement}']: {candidate.lever}"
            )
        else:
            perturbed_problem_text = problem_text

        # Get attribution question for the targeted axis
        from ..candidate_generator.generator import _label_to_attribution_question
        answer_labels = getattr(self.adapter.profile, 'answer_labels', None) or []
        target_axis = candidate.target_label_axis or "primary_answer"
        target_label = next(
            (l for l in answer_labels if l.get("name") == target_axis), None
        )
        if target_label:
            attribution_question = _label_to_attribution_question(
                target_label["name"],
                target_label.get("description", ""),
                target_label.get("value_type", "string"),
            )
        else:
            attribution_question = "Would your answer be different?"

        label_desc = getattr(self.adapter.profile, 'label_descriptions', '') or ''

        # Escape curly braces in user content to prevent KeyError
        # when rendered prompts contain JSON like {"answer": "<label>"}
        safe_problem = problem_text.replace("{", "{{").replace("}", "}}")
        safe_perturbed = perturbed_problem_text.replace("{", "{{").replace("}", "}}")
        safe_label_desc = label_desc.replace("{", "{{").replace("}", "}}")

        prompt_text = CRITIC_PROMPT.format(
            problem_text=safe_problem,
            perturbed_problem_text=safe_perturbed,
            attribution_question=attribution_question,
            target_label_axis=target_axis,
            label_descriptions=safe_label_desc,
            perturbation_type=candidate.perturbation_type,
            mechanism_name=candidate.mechanism_name,
        )

        from ..utils.retry import retry_async

        async def _do_critic():
            responses = await self.api(
                model_id=self.config.generator_model,
                prompt=Prompt(messages=[
                    ChatMessage(role=MessageRole.user, content=prompt_text),
                ]),
                n=1,
                temperature=0.2,
                max_tokens=1024,
            )
            return responses[0].completion if responses else ""

        try:
            response_text = await retry_async(
                _do_critic,
                stage_name="critic",
                item_id=candidate.candidate_id,
                api=self.api,
            )

            # Log prompt + response
            if self._prompt_logger:
                self._prompt_logger.log(
                    component="critic",
                    label=f"{candidate.candidate_id}",
                    user_prompt=prompt_text,
                    response=response_text,
                    extra={
                        "model": self.config.generator_model,
                        "target_axis": target_axis,
                        "category": candidate.category,
                    },
                )

            scores = self._parse_scores(response_text)
            candidate.predicted_flip_probability = scores.get(
                "predicted_flip_probability", 0.5
            )
            candidate.critic_notes = scores.get("notes", "")

            # Record in tracer
            if self._tracer and self._tracer.is_traced(candidate.example_idx):
                self._tracer.record_critic(
                    example_idx=candidate.example_idx,
                    candidate_id=candidate.candidate_id,
                    category=candidate.category,
                    mechanism_name=candidate.mechanism_name,
                    perturbation_type=candidate.perturbation_type,
                    lever_text=candidate.lever,
                    target_label_axis=candidate.target_label_axis,
                    prompt=prompt_text,
                    response=response_text,
                    predicted_flip=candidate.predicted_flip_probability,
                    notes=candidate.critic_notes,
                    model=self.config.generator_model,
                )

        except Exception as e:
            logger.error(
                f"Critic FAILED for {candidate.candidate_id} after retries: {e}. "
                f"Using defaults."
            )
            candidate.predicted_flip_probability = 0.5
            candidate.critic_notes = f"Scoring failed: {str(e)[:100]}"

    def _parse_scores(self, response_text: str) -> dict:
        """Parse critic's JSON scores."""
        text = response_text.strip()

        # Try to find JSON object
        try:
            match = re.search(r'\{[^}]+\}', text)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract score from text
        scores = {}
        flip_match = re.search(
            r'(?:predicted_)?flip[_\s]*(?:probability)?[:\s]*([0-9.]+)', text, re.I
        )
        if flip_match:
            scores["predicted_flip_probability"] = float(flip_match.group(1))

        return scores

