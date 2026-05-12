"""
Module: prompt_attribution/auto_perturbation/empirical_verification/verifier.py

Stage 5: Phase 1 empirical verification. Runs baseline + lever prompts
to get ground-truth flip labels. NOT for filtering — all candidates are kept.

Records full model responses (not just parsed answers) because intermediate
reasoning steps are valuable training data.

Structure:
- EmpiricalVerifier: Runs Phase 1 verification with stability runs
"""

import asyncio
import logging
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from ..config import (
    PerProblemCandidate,
    PipelineConfig,
    VerificationResult,
)
from ..dataset_adapter.dataset_adapter import DatasetAdapter, AdaptedExample

logger = logging.getLogger(__name__)


class EmpiricalVerifier:
    """Stage 5: Phase 1 empirical verification for training data.

    Runs baseline + lever prompts for each (problem, candidate) pair
    with multiple stability runs. Records full model responses.

    NO filtering. All candidates are kept — this is a data enrichment step,
    not a filter.

    Optionally accepts a TargetModelClient for vLLM-based verification
    instead of using the safetytooling InferenceAPI.
    """

    def __init__(
        self,
        api: InferenceAPI,
        config: PipelineConfig,
        adapter: DatasetAdapter,
        target_model_client: "object | None" = None,
        tracer=None,
    ):
        self.api = api
        self.config = config
        self.adapter = adapter
        self._verifier = adapter.create_verifier()
        self._target_client = target_model_client
        self._tracer = tracer
        logger.info(
            f"EmpiricalVerifier: target_client={'SET' if target_model_client else 'NONE'}, "
            f"judge_model={config.judge_model}"
        )

    async def verify(
        self,
        example: AdaptedExample,
        candidate: PerProblemCandidate,
    ) -> VerificationResult:
        """Verify a single (problem, candidate) pair.

        Runs baseline and lever prompts for stability_n_runs iterations.
        Records full responses and computes flip statistics.

        Args:
            example: The problem example
            candidate: The perturbation candidate

        Returns:
            VerificationResult with full responses and flip stats
        """
        n_runs = self.config.stability_n_runs
        baseline_responses = []
        lever_responses = []
        parsed_baselines = []
        parsed_levers = []
        flip_count = 0

        # Build prompts once (they don't change across runs)
        # Use axis-specific preamble + response format when available
        target_axis = candidate.target_label_axis
        baseline_prompt = self.adapter.make_axis_baseline_prompt(
            example, target_axis, candidate.baseline,
        )
        if candidate.perturbation_type == "problem_edit":
            lever_prompt = self.adapter.make_axis_edited_prompt(
                example, target_axis, candidate.problem_edits, candidate.baseline,
            )
        else:
            lever_prompt = self.adapter.make_axis_lever_prompt(
                example, target_axis, candidate.lever, candidate.baseline,
            )

        # Run all baseline+lever calls in parallel, cached by (prompt, run_idx).
        # Each stability run has its own cache slot but is reusable across restarts.
        import asyncio as _asyncio
        baseline_tasks = [
            self._call_model(baseline_prompt, no_cache=False, cache_suffix=f"_run{i}")
            for i in range(n_runs)
        ]
        lever_tasks = [
            self._call_model(lever_prompt, no_cache=False, cache_suffix=f"_run{i}")
            for i in range(n_runs)
        ]
        all_results = await _asyncio.gather(*baseline_tasks, *lever_tasks)
        baseline_resps = list(all_results[:n_runs])
        lever_resps = list(all_results[n_runs:])

        # Phase 1 (sync): Parse all responses
        baseline_responses = list(baseline_resps)
        lever_responses = list(lever_resps)
        for run_idx in range(n_runs):
            baseline_answer = self._verifier.parse_answer(baseline_resps[run_idx])
            parsed_baselines.append(
                str(baseline_answer) if baseline_answer is not None else ""
            )
            lever_answer = self._verifier.parse_answer(lever_resps[run_idx])
            parsed_levers.append(
                str(lever_answer) if lever_answer is not None else ""
            )

        # Phase 2: Determine flip using shared extraction module.
        # Uses same judge config/prompts as GT cache for consistent labels.
        target_axis = candidate.target_label_axis
        answer_labels = getattr(self._verifier, 'answer_labels', None) or []

        target_label = None
        if target_axis and answer_labels:
            target_label = next(
                (l for l in answer_labels if l.get("name") == target_axis),
                None,
            )

        if target_label:
            # Use shared flip computation (label-based extraction + comparison)
            from prompt_attribution.shared.answer_extraction import compute_flip

            flip_result = await compute_flip(
                list(baseline_resps[:n_runs]),
                list(lever_resps[:n_runs]),
                target_label,
                api=self.api,
            )
            flip_count = flip_result.flip_count
            flip_fraction = flip_result.flip_fraction
            flipped = flip_result.flipped
        elif target_axis and not target_label:
            # Target axis specified but not found in labels — fall back to full comparison
            for run_idx in range(n_runs):
                if not self._verifier.answers_match(
                    baseline_resps[run_idx], lever_resps[run_idx]
                ):
                    flip_count += 1
            flip_fraction = flip_count / n_runs if n_runs > 0 else 0.0
            flipped = flip_fraction > 0.5
        else:
            # No target axis or no labels — use parse_answer + answers_match
            # (matches original pipeline behavior for records without answer_labels)
            for run_idx in range(n_runs):
                b_parsed = self._verifier.parse_answer(baseline_resps[run_idx])
                l_parsed = self._verifier.parse_answer(lever_resps[run_idx])
                if b_parsed is not None and l_parsed is not None:
                    if not self._verifier.answers_match(b_parsed, l_parsed):
                        flip_count += 1
            flip_fraction = flip_count / n_runs if n_runs > 0 else 0.0
            flipped = flip_fraction > 0.5

        # Compute edit metrics (for all perturbation types)
        from ..dataset_adapter.edit_utils import compute_edit_metrics
        candidate.edit_distance, candidate.edit_fraction = compute_edit_metrics(
            baseline_prompt, lever_prompt,
        )

        # Extract features from first run (enrichment, not flip detection).
        # LLM judge extraction uses Haiku (cheap, fast, always available via API)
        # regardless of judge_model — the verifier may be a local vLLM
        # model that safetytooling can't route for text analysis.
        # Both extractions run in parallel.
        from prompt_attribution.shared.answer_extraction import JUDGE_MODEL as _JUDGE_MODEL
        features_baseline = {}
        features_lever = {}
        if hasattr(self._verifier, 'extract_features_async') and baseline_responses:
            features_baseline, features_lever = await asyncio.gather(
                self._verifier.extract_features_async(
                    baseline_responses[0], api=self.api,
                    model_id=_JUDGE_MODEL,
                ),
                self._verifier.extract_features_async(
                    lever_responses[0], api=self.api,
                    model_id=_JUDGE_MODEL,
                ),
            )

        # Record in tracer
        if self._tracer and self._tracer.is_traced(example.idx):
            self._tracer.record_verify(
                example_idx=example.idx,
                candidate_id=candidate.candidate_id,
                category=candidate.category,
                mechanism_name=candidate.mechanism_name,
                perturbation_type=candidate.perturbation_type,
                lever_text=candidate.lever,
                target_label_axis=candidate.target_label_axis,
                baseline_prompt=baseline_prompt,
                lever_prompt=lever_prompt,
                baseline_responses=baseline_responses,
                lever_responses=lever_responses,
                flip_fraction=flip_fraction,
                flipped=flipped,
            )

        return VerificationResult(
            flipped=flipped,
            flip_count=flip_count,
            n_runs=n_runs,
            flip_fraction=flip_fraction,
            baseline_answer=parsed_baselines[0] if parsed_baselines else "",
            lever_answer=parsed_levers[0] if parsed_levers else "",
            baseline_responses=baseline_responses,
            lever_responses=lever_responses,
            parsed_baseline_answers=parsed_baselines,
            parsed_lever_answers=parsed_levers,
            features_baseline=features_baseline,
            features_lever=features_lever,
        )

    async def verify_batch(
        self,
        examples: list[AdaptedExample],
        candidates_by_problem: dict[int, list[PerProblemCandidate]],
    ) -> dict[int, list[PerProblemCandidate]]:
        """Verify all candidates for a batch of problems.

        Updates each candidate's verification_results in-place.

        Args:
            examples: List of problem examples
            candidates_by_problem: Dict of candidates per problem

        Returns:
            Updated candidates_by_problem with verification results
        """
        example_map = {ex.idx: ex for ex in examples}
        semaphore = asyncio.Semaphore(self.config.verification_concurrency)
        # Use target model ID when target client is provided (critic == verifier)
        model_id = (
            self.config.target_model_id
            if self._target_client and self.config.target_model_id
            else self.config.judge_model
        )

        async def bounded_verify(
            example: AdaptedExample,
            candidate: PerProblemCandidate,
        ):
            async with semaphore:
                if not candidate.passed_critic:
                    return  # Skip candidates that failed critic

                result = await self.verify(example, candidate)
                candidate.verification_results[model_id] = result.to_dict()

                logger.info(
                    f"Verified {candidate.candidate_id}: "
                    f"flip_fraction={result.flip_fraction:.2f} "
                    f"({result.flip_count}/{result.n_runs})"
                )

        tasks = []
        for example_idx, candidates in candidates_by_problem.items():
            example = example_map.get(example_idx)
            if example is None:
                continue
            for candidate in candidates:
                tasks.append(bounded_verify(example, candidate))

        total = len(tasks)
        logger.info(f"Starting verification: {total} (candidate, problem) pairs")
        await asyncio.gather(*tasks)
        logger.info("Verification complete")

        return candidates_by_problem

    async def _call_model(self, prompt_text: str, no_cache: bool = False, cache_suffix: str = "") -> str:
        """Call the verification model and return response text.

        Routes through TargetModelClient when available (e.g., for vLLM),
        otherwise uses the safetytooling InferenceAPI.

        Args:
            prompt_text: The prompt to send.
            no_cache: If True, bypass caching so identical prompts return
                independent responses (needed for stability runs).
        """
        if self._target_client is not None:
            try:
                result = await self._target_client.call(prompt_text, no_cache=no_cache, cache_suffix=cache_suffix)
                if not result:
                    logger.warning(
                        f"Target client returned empty for prompt "
                        f"({len(prompt_text)} chars)"
                    )
                return result
            except Exception as e:
                logger.warning(f"Target model call failed (will retry): {e}")
                # Retry once for target model
                import asyncio as _asyncio
                await _asyncio.sleep(2)
                try:
                    result = await self._target_client.call(prompt_text, no_cache=no_cache)
                    return result
                except Exception as e2:
                    logger.error(f"Target model call FAILED after retry: {e2}")
                    return ""

        from ..utils.retry import retry_async

        async def _do_judge():
            responses = await self.api(
                model_id=self.config.judge_model,
                prompt=Prompt(messages=[
                    ChatMessage(role=MessageRole.user, content=prompt_text),
                ]),
                n=1,
                temperature=self.config.judge_temperature,
                max_tokens=4096,
            )
            return responses[0].completion if responses else ""

        try:
            return await retry_async(
                _do_judge,
                stage_name="verifier_judge",
                item_id=f"prompt_{len(prompt_text)}chars",
            )
        except Exception as e:
            logger.error(f"Judge call FAILED after retries: {e}")
            return ""
