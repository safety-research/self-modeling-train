"""
Module: prompt_attribution/auto_perturbation/critic_feedback/mini_verifier.py

Lightweight verification for the feedback loop. Runs 1-2 inference calls
against the target model to get actual flip results, without the full
feature extraction of Stage 5.

Compared to EmpiricalVerifier (Stage 5):
- Fewer runs (default 2 vs 5)
- No feature extraction (saves API calls)
- Uses TargetModelClient (vLLM or API) instead of always InferenceAPI
- Returns a simpler result type (MiniVerificationResult)
- Stores results in candidate.verification_results["mini_verify_{model_id}"]

Structure:
- MiniVerificationResult: Lightweight verification result
- MiniVerifier: Runs baseline+lever against target model, detects flips
"""

import asyncio
import logging
from dataclasses import dataclass, asdict
from ..config import (
    PerProblemCandidate,
    PipelineConfig,
)
from ..dataset_adapter.dataset_adapter import DatasetAdapter, AdaptedExample
from prompt_attribution.shared.inference.target_model_client import TargetModelClient

logger = logging.getLogger(__name__)


@dataclass
class MiniVerificationResult:
    """Lightweight verification result from mini-verify in feedback loop.

    Simpler than VerificationResult — no feature extraction, fewer fields.
    Used for feedback loop decisions, not final training data.

    Attributes:
        flipped: Majority-vote flip decision.
        flip_count: Number of runs where the answer flipped.
        n_runs: Total number of runs.
        flip_fraction: flip_count / n_runs (soft label).
        baseline_response: First run baseline response (for feedback messages).
        lever_response: First run lever response (for feedback messages).
        baseline_answer: Parsed baseline answer (first run).
        lever_answer: Parsed lever answer (first run).
    """

    flipped: bool
    flip_count: int
    n_runs: int
    flip_fraction: float
    baseline_response: str = ""
    lever_response: str = ""
    baseline_answer: str = ""
    lever_answer: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MiniVerificationResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MiniVerifier:
    """Lightweight verification for the critic feedback loop.

    Runs baseline + lever prompts against the target model with fewer
    stability runs than full Stage 5 verification. Used to get actual
    flip results for feedback to the generator.
    """

    # LLM judge model for feature extraction (cheap, always available via API)
    _JUDGE_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        target_client: TargetModelClient,
        adapter: DatasetAdapter,
        config: PipelineConfig,
        api=None,
        tracer=None,
    ):
        self.target_client = target_client
        self.adapter = adapter
        self.config = config
        self.api = api
        self._verifier = adapter.create_verifier()
        self._tracer = tracer

    async def _detect_flip(
        self,
        baseline_resp: str,
        lever_resp: str,
        target_axis: str,
        answer_labels: list[dict],
    ) -> bool:
        """Detect flip using feature extraction with LLM judge support.

        Priority:
        1. Programmatic extraction (fast, cheap)
        2. LLM judge extraction via Haiku (for string/semantic features)
        3. Default verifier parse_answer + answers_match
        """
        from ..dataset_adapter.answer_parser import (
            _extract_programmatic,
            extract_features_async,
        )

        if not baseline_resp or not lever_resp:
            return False

        # 1. Try programmatic extraction first (fast path)
        if answer_labels:
            for label in answer_labels:
                if label.get("verification_method") != "programmatic":
                    continue
                b_val = str(_extract_programmatic(baseline_resp, label) or "").strip().lower()
                l_val = str(_extract_programmatic(lever_resp, label) or "").strip().lower()
                if b_val and l_val and b_val != l_val:
                    return True

        # 2. LLM judge extraction via Haiku (for llm_judge and string labels)
        #    Both extractions run in parallel.
        has_judge_labels = any(
            l.get("verification_method") == "llm_judge"
            or l.get("value_type") == "string"
            for l in answer_labels
        )
        if has_judge_labels and self.api:
            try:
                feats_b, feats_l = await asyncio.gather(
                    extract_features_async(
                        baseline_resp, answer_labels, self.api, self._JUDGE_MODEL,
                    ),
                    extract_features_async(
                        lever_resp, answer_labels, self.api, self._JUDGE_MODEL,
                    ),
                )
                for key in feats_b:
                    if key in feats_l:
                        v1 = str(feats_b[key]).strip().lower()
                        v2 = str(feats_l[key]).strip().lower()
                        if v1 and v2 and v1 != v2:
                            return True
            except Exception as e:
                logger.debug(f"LLM judge extraction failed: {e}")

        # 3. Fallback: default verifier
        b_parsed = self._verifier.parse_answer(baseline_resp)
        l_parsed = self._verifier.parse_answer(lever_resp)
        if b_parsed is not None and l_parsed is not None:
            return not self._verifier.answers_match(b_parsed, l_parsed)

        return False

    def _extract_display_answer(
        self,
        response: str,
        target_axis: str,
        answer_labels: list[dict],
    ) -> str:
        """Extract a human-readable answer for display in feedback messages."""
        from ..dataset_adapter.answer_parser import _extract_programmatic

        if not response:
            return ""

        # Try categorical labels first (more readable than numeric)
        for label in answer_labels:
            if label.get("verification_method") != "programmatic":
                continue
            if label.get("value_type") in ("categorical", "string"):
                val = _extract_programmatic(response, label)
                if val:
                    return str(val)

        # Fallback to parse_answer
        parsed = self._verifier.parse_answer(response)
        return str(parsed) if parsed is not None else ""

    async def mini_verify(
        self,
        example: AdaptedExample,
        candidate: PerProblemCandidate,
    ) -> MiniVerificationResult:
        """Verify a single (problem, candidate) pair with minimal runs.

        Args:
            example: The problem example.
            candidate: The perturbation candidate.

        Returns:
            MiniVerificationResult with flip stats and first-run responses.
        """
        n_runs = self.config.stability_n_runs

        # Build prompts with axis-specific preamble + response format
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
        # The target client caches by prompt text — append run_idx to key
        # so each stability run has its own cache slot but is reusable across restarts.
        baseline_tasks = [
            self.target_client.call(baseline_prompt, no_cache=False, cache_suffix=f"_run{i}")
            for i in range(n_runs)
        ]
        lever_tasks = [
            self.target_client.call(lever_prompt, no_cache=False, cache_suffix=f"_run{i}")
            for i in range(n_runs)
        ]
        all_results = await asyncio.gather(*baseline_tasks, *lever_tasks)
        baseline_resps = list(all_results[:n_runs])
        lever_resps = list(all_results[n_runs:])

        # Check flips — use multi-label comparison for robustness.
        # The first programmatic label can be unreliable (e.g., numeric
        # label that maps everything to 0). Check all programmatic labels
        # and use the target axis label if available.
        flip_count = 0
        first_baseline_answer = ""
        first_lever_answer = ""
        answer_labels = getattr(self._verifier, 'answer_labels', None) or []

        # Parallelize _detect_flip across all runs
        flip_results = await asyncio.gather(*[
            self._detect_flip(
                baseline_resps[i], lever_resps[i],
                candidate.target_label_axis, answer_labels,
            )
            for i in range(n_runs)
        ])
        flip_count = sum(1 for f in flip_results if f)

        if baseline_resps:
            first_baseline_answer = self._extract_display_answer(
                baseline_resps[0], candidate.target_label_axis, answer_labels,
            )
            first_lever_answer = self._extract_display_answer(
                lever_resps[0], candidate.target_label_axis, answer_labels,
            )

        flip_fraction = flip_count / n_runs if n_runs > 0 else 0.0
        flipped = flip_fraction > 0.5

        # Record in tracer
        if self._tracer and self._tracer.is_traced(example.idx):
            self._tracer.record_mini_verify(
                example_idx=example.idx,
                candidate_id=candidate.candidate_id,
                category=candidate.category,
                mechanism_name=candidate.mechanism_name,
                perturbation_type=candidate.perturbation_type,
                lever_text=candidate.lever,
                target_label_axis=candidate.target_label_axis,
                baseline_prompt=baseline_prompt,
                lever_prompt=lever_prompt,
                baseline_response=baseline_resps[0] if baseline_resps else "",
                lever_response=lever_resps[0] if lever_resps else "",
                flip_result={"flipped": flipped, "flip_fraction": flip_fraction,
                             "flip_count": flip_count, "n_runs": n_runs},
            )

        return MiniVerificationResult(
            flipped=flipped,
            flip_count=flip_count,
            n_runs=n_runs,
            flip_fraction=flip_fraction,
            baseline_response=baseline_resps[0] if baseline_resps else "",
            lever_response=lever_resps[0] if lever_resps else "",
            baseline_answer=first_baseline_answer,
            lever_answer=first_lever_answer,
        )

    async def mini_verify_batch(
        self,
        examples: list[AdaptedExample],
        candidates_by_problem: dict[int, list[PerProblemCandidate]],
        sample_fraction: float = 1.0,
    ) -> None:
        """Verify a batch of candidates, storing results in-place.

        Args:
            examples: List of problem examples.
            candidates_by_problem: Dict of candidates per problem.
            sample_fraction: Fraction of candidates to verify (0.0-1.0).
                Set <1.0 to save time in the feedback loop.
        """
        import random

        example_map = {ex.idx: ex for ex in examples}
        model_id = self.config.target_model_id
        verify_key = f"mini_verify_{model_id}"

        # Collect all (example, candidate) pairs to verify
        all_pairs: list[tuple[AdaptedExample, PerProblemCandidate]] = []
        for example_idx, candidates in candidates_by_problem.items():
            example = example_map.get(example_idx)
            if example is None:
                continue
            for candidate in candidates:
                if not candidate.is_viable:
                    continue
                # Skip candidates already verified with this model
                if verify_key in candidate.verification_results:
                    continue
                all_pairs.append((example, candidate))

        # Sample if requested
        if sample_fraction < 1.0 and len(all_pairs) > 1:
            n_to_verify = max(1, int(len(all_pairs) * sample_fraction))

            # Prioritize candidates near category boundaries (uncertain)
            def _uncertainty_score(pair: tuple) -> float:
                """Higher score = more uncertain = higher priority."""
                c = pair[1]
                pred = c.predicted_flip_probability
                if pred is None:
                    return 0.5  # default to uncertain
                return 1.0 - abs(pred - 0.5) * 2  # 0.5 → 1.0, 0.0/1.0 → 0.0

            all_pairs.sort(key=_uncertainty_score, reverse=True)
            # Take top uncertain ones + random from the rest
            n_priority = min(n_to_verify // 2, len(all_pairs))
            priority = all_pairs[:n_priority]
            remaining = all_pairs[n_priority:]
            random.shuffle(remaining)
            sampled = priority + remaining[: n_to_verify - n_priority]
            all_pairs = sampled

        if not all_pairs:
            return

        semaphore = asyncio.Semaphore(self.config.verification_concurrency)

        async def bounded_verify(
            example: AdaptedExample,
            candidate: PerProblemCandidate,
        ) -> None:
            async with semaphore:
                try:
                    result = await self.mini_verify(example, candidate)
                    candidate.verification_results[verify_key] = result.to_dict()
                    logger.debug(
                        f"Mini-verified {candidate.candidate_id}: "
                        f"flip={result.flipped} "
                        f"({result.flip_count}/{result.n_runs})"
                    )
                except Exception as e:
                    logger.warning(
                        f"Mini-verify failed for {candidate.candidate_id}: {e}"
                    )

        logger.info(
            f"Mini-verifying {len(all_pairs)} candidates against "
            f"{model_id} ({self.config.stability_n_runs} runs each)"
        )
        await asyncio.gather(
            *[bounded_verify(ex, cand) for ex, cand in all_pairs]
        )
        logger.info("Mini-verification complete")
