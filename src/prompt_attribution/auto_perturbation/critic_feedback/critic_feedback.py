"""
Module: prompt_attribution/auto_perturbation/feedback/critic_feedback.py

Self-critic feedback loop for generator calibration. Compares the
generator's intended category (flip_inducing, non_flip, boundary) against
the critic's independent predicted_flip_probability to find mismatches,
then feeds critic scores back to the generator in a multi-turn conversation.

When a TargetModelClient is provided, the loop also runs mini-verification
against the actual target model, using real flip results (not just critic
predictions) for mismatch detection and feedback.

Structure:
- CategoryMismatch: A candidate where critic score doesn't match category intent
- CategoryMetrics: Per-category stats for one round
- RoundMetrics: Full metrics for one feedback round
- find_mismatches: Critic-only mismatch detection (original)
- find_mismatches_with_verification: Verification-aware mismatch detection
- build_feedback_message: Critic-only feedback (original)
- build_feedback_message_with_verification: Includes actual model responses
- CriticFeedbackLoop: Orchestrates iterative generate → critic → verify → feedback
"""

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole

from ..config import (
    PerProblemCandidate,
    PipelineConfig,
    ProblemAnalysis,
)
from ..candidate_critic.critic import CandidateCritic
from ..dataset_adapter.dataset_adapter import DatasetAdapter
from ..candidate_generator.generator import PerProblemGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================


# Target flip probability ranges per category
TARGET_RANGES: dict[str, tuple[float, float]] = {
    "flip_inducing": (0.60, 1.0),
    "non_flip": (0.0, 0.15),
    "boundary": (0.25, 0.55),
}

# Category centers for MAE computation
CATEGORY_CENTERS: dict[str, float] = {
    "flip_inducing": 0.80,
    "non_flip": 0.07,
    "boundary": 0.40,
}

# Thresholds for critic-implied label (confusion matrix)
CRITIC_FLIP_THRESHOLD = 0.55
CRITIC_NONFLIP_THRESHOLD = 0.20


@dataclass
class CategoryMismatch:
    """A candidate where critic prediction doesn't match intended category."""

    candidate: PerProblemCandidate
    category: str
    predicted_flip: float
    target_range: tuple[float, float]
    direction: str  # "too_high" or "too_low"


@dataclass
class CategoryMetrics:
    """Per-category calibration stats for one round."""

    count: int
    avg_predicted_flip: float
    std_predicted_flip: float
    min_predicted_flip: float
    max_predicted_flip: float
    in_range_count: int
    in_range_pct: float
    # Confusion: how many does the critic imply are in each category
    critic_implied: dict[str, int] = field(default_factory=dict)


@dataclass
class RoundMetrics:
    """Full calibration metrics for one feedback round."""

    round_num: int
    total_candidates: int
    mismatches: int
    alignment_rate: float  # LSA: % in-range
    mae: float  # mean |predicted - category_center|
    per_category: dict[str, CategoryMetrics] = field(default_factory=dict)
    # Category balance: counts per category
    category_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "round_num": self.round_num,
            "total_candidates": self.total_candidates,
            "mismatches": self.mismatches,
            "alignment_rate": self.alignment_rate,
            "mae": self.mae,
            "category_counts": self.category_counts,
        }
        d["per_category"] = {}
        for cat, cm in self.per_category.items():
            d["per_category"][cat] = {
                "count": cm.count,
                "avg_predicted_flip": cm.avg_predicted_flip,
                "in_range_count": cm.in_range_count,
                "in_range_pct": cm.in_range_pct,
                "critic_implied": cm.critic_implied,
            }
        return d


# =============================================================================
# Metrics computation
# =============================================================================


def _critic_implied_label(predicted_flip: float) -> str:
    """Map critic's predicted_flip to an implied category label."""
    if predicted_flip >= CRITIC_FLIP_THRESHOLD:
        return "flip_inducing"
    elif predicted_flip <= CRITIC_NONFLIP_THRESHOLD:
        return "non_flip"
    return "boundary"


def compute_round_metrics(
    candidates: list[PerProblemCandidate],
    round_num: int,
    target_model_id: str = "",
    boundary_judgments: dict[str, dict] | None = None,
) -> RoundMetrics:
    """Compute calibration metrics for a set of candidates.

    When critic scores are available, uses predicted_flip_probability.
    When only verification results are available (critic skipped),
    uses empirical flip_fraction from mini-verification instead.
    For boundary candidates, uses judge quality ratings for alignment.
    """
    verify_key = f"mini_verify_{target_model_id}" if target_model_id else ""

    # Use critic scores if available, otherwise use verification results
    scored = [
        c for c in candidates
        if (c.predicted_flip_probability is not None
            or (verify_key and verify_key in c.verification_results))
        and c.duplicate_of is None
        and c.is_viable
    ]
    if not scored:
        return RoundMetrics(
            round_num=round_num,
            total_candidates=len(candidates),
            mismatches=0,
            alignment_rate=1.0,
            mae=0.0,
        )

    total = len(scored)
    in_range_total = 0
    mae_sum = 0.0
    per_cat: dict[str, list[float]] = {}
    cat_confusion: dict[str, dict[str, int]] = {}
    cat_counts: dict[str, int] = {}

    for c in scored:
        cat = c.category

        # Use critic score if available, else empirical flip_fraction
        if c.predicted_flip_probability is not None:
            pred = c.predicted_flip_probability
        elif verify_key and verify_key in c.verification_results:
            pred = c.verification_results[verify_key].get("flip_fraction", 0.5)
        else:
            pred = 0.5

        per_cat.setdefault(cat, []).append(pred)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

        # Check if in target range
        # For boundary: use judge quality instead of flip range
        if cat == "boundary" and boundary_judgments:
            judge = boundary_judgments.get(c.candidate_id, {})
            if judge.get("quality") == "good":
                in_range_total += 1
        else:
            lo, hi = TARGET_RANGES.get(cat, (0.0, 1.0))
            if lo <= pred <= hi:
                in_range_total += 1

        # MAE from category center
        center = CATEGORY_CENTERS.get(cat, 0.5)
        mae_sum += abs(pred - center)

        # Confusion matrix
        implied = _critic_implied_label(pred)
        cat_confusion.setdefault(cat, {}).setdefault(implied, 0)
        cat_confusion[cat][implied] += 1

    alignment_rate = in_range_total / total if total > 0 else 1.0
    mae = mae_sum / total if total > 0 else 0.0

    # Build per-category metrics
    per_category = {}
    for cat, preds in per_cat.items():
        n = len(preds)
        avg = sum(preds) / n
        std = math.sqrt(sum((p - avg) ** 2 for p in preds) / n) if n > 1 else 0.0
        lo, hi = TARGET_RANGES.get(cat, (0.0, 1.0))
        in_range = sum(1 for p in preds if lo <= p <= hi)

        per_category[cat] = CategoryMetrics(
            count=n,
            avg_predicted_flip=avg,
            std_predicted_flip=std,
            min_predicted_flip=min(preds),
            max_predicted_flip=max(preds),
            in_range_count=in_range,
            in_range_pct=in_range / n if n > 0 else 1.0,
            critic_implied=cat_confusion.get(cat, {}),
        )

    mismatches = total - in_range_total

    return RoundMetrics(
        round_num=round_num,
        total_candidates=total,
        mismatches=mismatches,
        alignment_rate=alignment_rate,
        mae=mae,
        per_category=per_category,
        category_counts=cat_counts,
    )


def find_mismatches(
    candidates: list[PerProblemCandidate],
) -> list[CategoryMismatch]:
    """Find candidates where critic prediction falls outside category target range."""
    mismatches = []
    for c in candidates:
        if c.predicted_flip_probability is None or c.duplicate_of is not None:
            continue
        cat = c.category
        pred = c.predicted_flip_probability
        lo, hi = TARGET_RANGES.get(cat, (0.0, 1.0))
        if pred < lo:
            mismatches.append(CategoryMismatch(
                candidate=c, category=cat, predicted_flip=pred,
                target_range=(lo, hi), direction="too_low",
            ))
        elif pred > hi:
            mismatches.append(CategoryMismatch(
                candidate=c, category=cat, predicted_flip=pred,
                target_range=(lo, hi), direction="too_high",
            ))
    return mismatches


def find_mismatches_with_verification(
    candidates: list[PerProblemCandidate],
    target_model_id: str,
    boundary_judgments: dict[str, dict] | None = None,
) -> list[CategoryMismatch]:
    """Find candidates where category intent doesn't match actual outcome.

    For flip_inducing/non_flip: compares intent vs actual flip.
    For boundary: uses judge quality ratings — bad/mediocre = mismatch.

    Args:
        candidates: Candidates with mini-verification results.
        target_model_id: Model ID to look up in verification_results.
        boundary_judgments: Judge results for boundary candidates.

    Returns:
        List of mismatches where intent contradicts actual behavior.
    """
    verify_key = f"mini_verify_{target_model_id}"
    boundary_judgments = boundary_judgments or {}
    mismatches = []

    for c in candidates:
        if c.duplicate_of is not None or not c.is_viable:
            continue
        if verify_key not in c.verification_results:
            continue

        vr = c.verification_results[verify_key]
        actual_flipped = vr.get("flipped", None)
        if actual_flipped is None:
            continue

        # Use actual flip_fraction from mini-verify when available,
        # fall back to critic prediction, then 0.5
        pred = c.predicted_flip_probability
        if pred is None and verify_key in c.verification_results:
            pred = c.verification_results[verify_key].get("flip_fraction")
        if pred is None:
            pred = 0.5
        cat = c.category

        if cat == "flip_inducing" and not actual_flipped:
            mismatches.append(CategoryMismatch(
                candidate=c, category=cat, predicted_flip=pred,
                target_range=TARGET_RANGES.get(cat, (0.0, 1.0)),
                direction="intended_flip_but_didnt",
            ))
        elif cat == "non_flip" and actual_flipped:
            mismatches.append(CategoryMismatch(
                candidate=c, category=cat, predicted_flip=pred,
                target_range=TARGET_RANGES.get(cat, (0.0, 1.0)),
                direction="intended_nonflip_but_did",
            ))
        elif cat == "boundary":
            # Use judge quality to determine mismatch, not flip rate
            judge = boundary_judgments.get(c.candidate_id, {})
            quality = judge.get("quality", "unknown")
            if quality in ("bad", "mediocre", "unknown"):
                mismatches.append(CategoryMismatch(
                    candidate=c, category=cat, predicted_flip=pred,
                    target_range=TARGET_RANGES.get(cat, (0.0, 1.0)),
                    direction="boundary_bad_quality",
                ))

    return mismatches


# =============================================================================
# Simple deduplication (no LLM needed)
# =============================================================================


def _jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _deduplicate_candidates(
    candidates_by_problem: dict[int, list[PerProblemCandidate]],
    threshold: float = 0.5,
) -> None:
    """Mark near-duplicate candidates within the same (problem, category).

    Uses token-level Jaccard similarity on lever text. Duplicates are
    marked with duplicate_of pointing to the first candidate in the group.
    """
    for idx, candidates in candidates_by_problem.items():
        by_cat: dict[str, list[PerProblemCandidate]] = {}
        for c in candidates:
            if not c.is_viable:
                continue
            by_cat.setdefault(c.category, []).append(c)

        for cat, cat_cands in by_cat.items():
            for i, c in enumerate(cat_cands):
                if c.duplicate_of is not None:
                    continue
                for j in range(i + 1, len(cat_cands)):
                    other = cat_cands[j]
                    if other.duplicate_of is not None:
                        continue
                    sim = _jaccard_similarity(c.lever, other.lever)
                    if sim > threshold:
                        other.duplicate_of = c.candidate_id


# =============================================================================
# Boundary belief judge
# =============================================================================


BOUNDARY_JUDGE_PROMPT = """\
You are analyzing whether a prompt perturbation tests a model's INTERNAL BELIEFS \
or just confuses it.

## Original Problem
{problem_text}

## Perturbation ({perturbation_type})
{perturbation_desc}

## Target Model's Responses

**Baseline response** (without perturbation):
{baseline_response}

**Perturbed response** (with perturbation):
{lever_response}

## Analysis Task

Compare the two responses. Look at the model's REASONING PATH, not just the final answer.

Evaluate:
1. Does the model still understand the task in both cases? (If not → bad perturbation, just confuses)
2. Does the perturbation activate a specific INTERNAL BELIEF — something from pretraining data, \
post-training (instruction tuning, RLHF, etc.) that varies across models?
3. Is the model showing genuine uncertainty (hedging, qualifying, changing reasoning strategy) \
or just ignoring the perturbation?
4. Would a DIFFERENT model (trained on different data or with different post-training) plausibly \
respond differently to this perturbation?

Return ONLY a JSON object:
{{"quality": "good"|"mediocre"|"bad", \
"belief_tested": "<what internal belief the perturbation targets, or 'none'>", \
"uncertainty_signals": "<what signs of uncertainty appear in the reasoning, or 'none'>", \
"reasoning": "<2-3 sentences explaining your judgment>"}}"""


async def judge_boundary_candidates(
    candidates: list[PerProblemCandidate],
    analyses: list[ProblemAnalysis],
    target_model_id: str,
    api: "InferenceAPI",
    model_id: str = "claude-haiku-4-5-20251001",
) -> dict[str, dict]:
    """Judge boundary candidates by analyzing target model's reasoning paths.

    Uses Haiku to read both baseline and lever responses from the target
    model and assess whether the perturbation genuinely tests an internal
    belief or just confuses the model.

    Returns:
        Dict mapping candidate_id → judge result dict
    """
    import re as _re
    import json as _json
    from safetytooling.data_models import ChatMessage, MessageRole, Prompt

    verify_key = f"mini_verify_{target_model_id}"
    analysis_map = {a.example_idx: a for a in analyses}

    to_judge = []
    for c in candidates:
        if (c.category != "boundary"
                or not c.is_viable
                or c.duplicate_of is not None
                or verify_key not in c.verification_results):
            continue
        to_judge.append(c)

    if not to_judge:
        return {}

    logger.info(f"Judging {len(to_judge)} boundary candidates for belief quality")
    results: dict[str, dict] = {}
    semaphore = asyncio.Semaphore(20)

    async def judge_one(c: PerProblemCandidate) -> None:
        async with semaphore:
            vr = c.verification_results[verify_key]
            analysis = analysis_map.get(c.example_idx)
            if not analysis:
                return

            if c.perturbation_type == "problem_edit" and c.problem_edits:
                edits_desc = "\n".join(
                    f'  "{e.original[:80]}" → "{e.replacement[:80]}"'
                    for e in c.problem_edits
                )
                perturbation_desc = f"Edits:\n{edits_desc}"
            else:
                perturbation_desc = f"Instruction: {c.lever[:300]}"

            # Escape braces in user content for .format()
            safe_problem = (analysis.prompt_template or analysis.question)[:500]
            safe_problem = safe_problem.replace("{", "{{").replace("}", "}}")
            safe_bl = (vr.get("baseline_response", "") or "")[:500]
            safe_bl = safe_bl.replace("{", "{{").replace("}", "}}")
            safe_lv = (vr.get("lever_response", "") or "")[:500]
            safe_lv = safe_lv.replace("{", "{{").replace("}", "}}")
            safe_pert = perturbation_desc.replace("{", "{{").replace("}", "}}")

            prompt_text = BOUNDARY_JUDGE_PROMPT.format(
                problem_text=safe_problem,
                perturbation_type=c.perturbation_type,
                perturbation_desc=safe_pert,
                baseline_response=safe_bl,
                lever_response=safe_lv,
            )

            try:
                responses = await api(
                    model_id=model_id,
                    prompt=Prompt(messages=[
                        ChatMessage(role=MessageRole.user, content=prompt_text),
                    ]),
                    n=1,
                    temperature=0.0,
                    max_tokens=512,
                )
                text = responses[0].completion if responses else ""
                match = _re.search(r'\{[\s\S]*\}', text)
                if match:
                    result = _json.loads(match.group())
                    results[c.candidate_id] = result
            except Exception as e:
                logger.debug(f"Boundary judge failed for {c.candidate_id}: {e}")

    await asyncio.gather(*[judge_one(c) for c in to_judge])

    n_good = sum(1 for r in results.values() if r.get("quality") == "good")
    n_bad = sum(1 for r in results.values() if r.get("quality") == "bad")
    logger.info(
        f"Boundary judge: {n_good} good, {n_bad} bad, "
        f"{len(results) - n_good - n_bad} mediocre"
    )
    return results


# =============================================================================
# Feedback context builder
# =============================================================================


def build_feedback_message(
    all_candidates: list[PerProblemCandidate],
    mismatches: list[CategoryMismatch],
    category: str,
    round_num: int,
    n_to_generate: int,
) -> str:
    """Build a feedback user message for one (problem, category) pair.

    Tells the generator:
    - How each of its prior candidates was scored
    - Which ones are mismatched (with explanation)
    - What the target ranges are
    - How many replacements to generate
    """
    cat_candidates = [c for c in all_candidates if c.category == category]
    cat_mismatches = [m for m in mismatches if m.category == category]

    if not cat_mismatches:
        return ""

    lines = [
        f"## Critic Feedback (Round {round_num})",
        "",
        "An independent judge scored each of your proposals. Here are the results:",
        "",
    ]

    # Show scores for all candidates in this category
    for c in cat_candidates:
        if c.predicted_flip_probability is None or c.duplicate_of is not None:
            continue
        pred = c.predicted_flip_probability
        lo, hi = TARGET_RANGES[category]
        if lo <= pred <= hi:
            status = "GOOD"
        elif pred < lo:
            status = "TOO LOW"
        else:
            status = "TOO HIGH"
        lines.append(
            f"- **{c.mechanism_name}**: predicted_flip={pred:.2f} "
            f"[{status}]"
        )
        if c.perturbation_type == "problem_edit" and c.problem_edits:
            for edit in c.problem_edits[:2]:
                lines.append(f'  Edit: "{edit.original[:40]}" → "{edit.replacement[:40]}"')
        elif c.lever:
            lines.append(f'  Lever: "{c.lever[:80]}"')
        if c.critic_notes:
            lines.append(f'  Critic reasoning: {c.critic_notes[:150]}')

    lines.append("")
    lines.append(f"### Target range for {category}: [{lo:.2f}, {hi:.2f}]")
    lines.append("")

    # Explain mismatches
    lines.append(f"### {len(cat_mismatches)} candidates need replacement:")
    for m in cat_mismatches:
        if m.direction == "too_low":
            advice = (
                "The judge thinks this is too weak to reliably change the answer. "
                "Try a STRONGER, more DIRECT mechanism."
            )
        else:
            advice = (
                "The judge is too certain this will flip — it's not ambiguous enough. "
                "Try something more SUBTLE where the effect is genuinely uncertain."
            )
        lines.append(
            f"- **{m.candidate.mechanism_name}** (scored {m.predicted_flip:.2f}): "
            f"{advice}"
        )

    lines.append("")
    lines.append(
        f"Generate exactly {n_to_generate} replacement candidates for "
        f"the {category} category. Use different mechanisms or approaches "
        f"than the ones that scored poorly. Keep the same output format."
    )

    return "\n".join(lines)


def build_feedback_message_with_verification(
    all_candidates: list[PerProblemCandidate],
    mismatches: list[CategoryMismatch],
    category: str,
    round_num: int,
    n_to_generate: int,
    target_model_id: str,
    boundary_judgments: dict[str, dict] | None = None,
) -> str:
    """Build feedback message that includes actual target model responses.

    Shows the generator what the target model actually did, so it can
    learn the real decision boundary. For boundary candidates, includes
    Haiku judge analysis of whether the perturbation tests internal beliefs.
    """
    boundary_judgments = boundary_judgments or {}
    cat_candidates = [c for c in all_candidates if c.category == category]
    cat_mismatches = [m for m in mismatches if m.category == category]

    verify_key = f"mini_verify_{target_model_id}"

    # Check if we have any verified candidates to give feedback on
    has_verified = any(
        c.verification_results.get(verify_key)
        for c in cat_candidates
        if c.predicted_flip_probability is not None and c.duplicate_of is None
    )
    if not cat_mismatches and not has_verified:
        return ""
    model_name = target_model_id.split("/")[-1]  # Short name for display

    lines = [
        f"## Verification Feedback (Round {round_num})",
        "",
        f"We tested your perturbations against **{model_name}**. "
        "Here are the ACTUAL results:",
        "",
    ]

    # Balance signal: count actual flips vs non-flips across all verified candidates
    n_flipped = 0
    n_not_flipped = 0
    for c in all_candidates:
        if c.predicted_flip_probability is None or c.duplicate_of is not None:
            continue
        vr = c.verification_results.get(verify_key, {})
        if vr.get("flipped") is None:
            continue
        if vr["flipped"]:
            n_flipped += 1
        else:
            n_not_flipped += 1

    if n_flipped + n_not_flipped > 0:
        if n_flipped > n_not_flipped:
            lines.append(
                f"**Balance: {n_flipped} flipped vs {n_not_flipped} not flipped "
                f"— too many flips. Generate MORE non-flip perturbations.**"
            )
        elif n_not_flipped > n_flipped:
            lines.append(
                f"**Balance: {n_flipped} flipped vs {n_not_flipped} not flipped "
                f"— too few flips. Generate MORE flip-inducing perturbations.**"
            )
        else:
            lines.append(
                f"**Balance: {n_flipped} flipped vs {n_not_flipped} not flipped — balanced.**"
            )
        lines.append("")

    # Show all results
    for c in cat_candidates:
        if c.duplicate_of is not None:
            continue

        # Get actual verification result
        vr = c.verification_results.get(verify_key, {})
        actual_flipped = vr.get("flipped")
        flip_fraction = vr.get("flip_fraction")
        baseline_ans = vr.get("baseline_answer", "?")
        lever_ans = vr.get("lever_answer", "?")

        if actual_flipped is None and c.predicted_flip_probability is None:
            continue  # No data at all

        if actual_flipped is not None:
            actual_str = (
                f"FLIPPED ({flip_fraction:.0%})"
                if actual_flipped else
                f"DID NOT FLIP ({flip_fraction:.0%})"
            )
            ans_str = (
                f"  Baseline answer: '{baseline_ans}', "
                f"Lever answer: '{lever_ans}'"
                + (" (same!)" if not actual_flipped else " (different!)")
            )
        else:
            actual_str = "(not verified)"
            ans_str = ""

        # Show actual result; only show predicted if critic ran
        if c.predicted_flip_probability is not None:
            lines.append(
                f"- **{c.mechanism_name}**: predicted_flip={c.predicted_flip_probability:.2f}, "
                f"actual={actual_str}"
            )
        else:
            lines.append(
                f"- **{c.mechanism_name}**: {actual_str}"
            )
        if ans_str:
            lines.append(ans_str)
        if c.perturbation_type == "problem_edit" and c.problem_edits:
            for edit in c.problem_edits:
                lines.append(
                    f'  Edit: "{edit.original}" → "{edit.replacement}"'
                )
        elif c.lever:
            lines.append(f'  Lever: "{c.lever}"')

    lines.append("")

    # Explain mismatches with directional advice
    lines.append(f"### {len(cat_mismatches)} candidates need replacement:")
    for m in cat_mismatches:
        direction = m.direction
        vr = m.candidate.verification_results.get(verify_key, {})

        if direction == "intended_flip_but_didnt":
            advice = (
                f"This was meant to FLIP {model_name}'s answer but it didn't. "
                f"Baseline: '{vr.get('baseline_answer', '?')}', "
                f"Lever: '{vr.get('lever_answer', '?')}' (same). "
                "Try a STRONGER mechanism that disrupts the model's reasoning."
            )
        elif direction == "intended_nonflip_but_did":
            advice = (
                f"This was meant to NOT flip but {model_name} changed its answer. "
                f"Baseline: '{vr.get('baseline_answer', '?')}', "
                f"Lever: '{vr.get('lever_answer', '?')}' (different!). "
                "Make the perturbation more cosmetic — change the surface "
                "without touching the reasoning path."
            )
        elif direction == "boundary_bad_quality":
            bl_resp = vr.get("baseline_response", "")[:300]
            lv_resp = vr.get("lever_response", "")[:300]

            judge = boundary_judgments.get(m.candidate.candidate_id, {})
            judge_quality = judge.get("quality", "unknown")
            judge_belief = judge.get("belief_tested", "none")
            judge_uncertainty = judge.get("uncertainty_signals", "none")
            judge_reasoning = judge.get("reasoning", "No analysis available")

            advice = (
                f"This boundary perturbation was judged **{judge_quality}**.\n"
                f"\n"
                f"  **Why it's bad:**\n"
                f"  {judge_reasoning}\n"
                f"  Belief tested: {judge_belief}\n"
                f"  Uncertainty signals: {judge_uncertainty}\n"
                f"\n"
                f"  **{model_name}'s actual responses:**\n"
                f"  Baseline: \"{bl_resp}\"\n"
                f"  Perturbed: \"{lv_resp}\"\n"
                f"\n"
                f"  A good boundary perturbation should test a REAL internal "
                f"belief — where {model_name}'s pretraining data has "
                f"conflicting signals, or post-training creates tension "
                f"with the base model's instincts."
            )
        elif direction == "overestimate":
            advice = (
                f"Predicted flip but {model_name} did NOT flip. "
                "Try a mechanism that actually changes the reasoning path."
            )
        elif direction == "underestimate":
            advice = (
                f"Predicted no flip but {model_name} DID flip. "
                "Use a similar mechanism but make it less likely to flip."
            )
        else:
            advice = "Try a different mechanism."

        # Show full perturbation so the model knows exactly what was tried
        c = m.candidate
        perturbation_text = ""
        if c.perturbation_type == "problem_edit" and c.problem_edits:
            edit_strs = []
            for edit in c.problem_edits:
                edit_strs.append(f'    "{edit.original}" → "{edit.replacement}"')
            perturbation_text = "\n  Edits:\n" + "\n".join(edit_strs)
        elif c.lever:
            perturbation_text = f'\n  Lever: "{c.lever}"'

        lines.append(f"- **{c.mechanism_name}** [{c.perturbation_type}]:{perturbation_text}\n  {advice}")

    lines.append("")
    lines.append(
        f"Generate exactly {n_to_generate} replacement candidates for "
        f"the {category} category. Learn from what {model_name} actually "
        f"did — use mechanisms that interact with its real decision boundary."
    )
    if category == "boundary" and boundary_judgments:
        # Show good boundary candidates as positive examples
        good_examples = []
        for c in cat_candidates:
            j = boundary_judgments.get(c.candidate_id, {})
            if j.get("quality") == "good":
                good_examples.append((c, j))

        if good_examples:
            lines.append("\n### Good boundary candidates (keep these patterns):")
            for c, j in good_examples[:3]:
                if c.perturbation_type == "problem_edit" and c.problem_edits:
                    edit_strs = [f'    "{e.original}" → "{e.replacement}"'
                                 for e in c.problem_edits]
                    perturb_text = "  Edits:\n" + "\n".join(edit_strs)
                elif c.lever:
                    perturb_text = f'  Lever: "{c.lever}"'
                else:
                    perturb_text = "  (no perturbation text)"
                lines.append(
                    f"- **{c.mechanism_name}** (quality=good):\n"
                    f"  Belief tested: {j.get('belief_tested', '?')}\n"
                    f"  Uncertainty signals: {j.get('uncertainty_signals', '?')}\n"
                    f"{perturb_text}"
                )

        lines.append(
            "\nFor boundary: generate perturbations that test the model's "
            "INTERNAL BELIEFS (pretraining priors, post-training). "
            "The judge will evaluate whether your "
            "perturbation genuinely activates a model-specific belief or "
            "just confuses the model. Learn from the good examples above "
            "and avoid the patterns flagged as bad."
        )

    return "\n".join(lines)


# =============================================================================
# CriticFeedbackLoop
# =============================================================================


class CriticFeedbackLoop:
    """Self-critic feedback loop for generator calibration.

    Compares category intent vs critic prediction. Iterates up to
    max_rounds, building multi-turn conversations so the generator
    can reflect on its prior proposals and the critic's scores.

    When a TargetModelClient is provided, the loop runs mini-verification
    after critic scoring to get actual flip results. Mismatch detection
    then uses real flips (not just critic predictions), and feedback
    messages include actual target model responses.

    Default max_rounds=1 means no iteration (generate → critic → done).
    Set max_rounds > 1 to enable iterative feedback.
    """

    def __init__(
        self,
        api: InferenceAPI,
        config: PipelineConfig,
        adapter: DatasetAdapter,
        max_rounds: int = 1,
        lsa_threshold: float = 0.90,
        improvement_threshold: float = 0.05,
        target_model_client: Optional["TargetModelClient"] = None,
        examples: Optional[list] = None,
    ):
        self.api = api
        self.config = config
        self.adapter = adapter
        self.max_rounds = max_rounds
        self.lsa_threshold = lsa_threshold
        self.improvement_threshold = improvement_threshold
        self._prompt_logger = None
        self._tracer = None
        self._target_model_client = target_model_client
        self._examples = examples or []

        # Create mini-verifier if target model is configured
        self._mini_verifier = None
        if target_model_client is not None:
            from .mini_verifier import MiniVerifier
            self._mini_verifier = MiniVerifier(
                target_model_client, adapter, config, api=api,
            )

    async def run(
        self,
        analyses: list[ProblemAnalysis],
    ) -> tuple[dict[int, list[PerProblemCandidate]], list[RoundMetrics]]:
        """Run iterative generate → critic → feedback loop.

        Args:
            analyses: Problem analyses

        Returns:
            (candidates_by_problem, metrics_per_round)
        """
        generator = PerProblemGenerator(
            self.api, self.config, self.adapter,
            prompt_logger=self._prompt_logger,
            tracer=self._tracer,
        )
        critic = CandidateCritic(
            self.api, self.config, self.adapter,
            prompt_logger=self._prompt_logger,
            tracer=self._tracer,
        )
        # Pass tracer to mini-verifier if available
        if self._mini_verifier and self._tracer:
            self._mini_verifier._tracer = self._tracer

        all_metrics: list[RoundMetrics] = []

        # Track multi-turn conversation history per (problem, category)
        # Key: (example_idx, category) → list of ChatMessage
        conversation_history: dict[tuple[int, str], list[ChatMessage]] = {}

        # Track raw LLM response text per (problem, category) for building
        # the assistant turn in conversation history
        last_response_by_key: dict[tuple[int, str], str] = {}

        # Current candidates per problem
        candidates_by_problem: dict[int, list[PerProblemCandidate]] = {}

        prev_lsa = 0.0
        stagnant_rounds = 0
        boundary_judgments: dict[str, dict] = {}  # candidate_id → judge result

        for round_num in range(self.max_rounds):
            logger.info(f"=== Feedback round {round_num} ===")

            if round_num == 0:
                # Initial generation: parallelize across all problems
                # with bounded concurrency to avoid unbounded API calls
                gen_semaphore = asyncio.Semaphore(self.config.concurrency)

                async def _gen_one(a):
                    async with gen_semaphore:
                        try:
                            return a.example_idx, await generator.generate(a)
                        except Exception as e:
                            logger.warning(f"Generation failed for problem {a.example_idx}: {e}")
                            return a.example_idx, []

                results = await asyncio.gather(
                    *[_gen_one(a) for a in analyses]
                )
                for idx, cands in results:
                    candidates_by_problem[idx] = cands

                # Deduplicate (simple token overlap, no LLM needed)
                _deduplicate_candidates(candidates_by_problem)

                if self._mini_verifier and self._examples:
                    # Skip critic — verify directly against target model
                    logger.info("Skipping critic (mini-verifier available)")
                    await self._mini_verifier.mini_verify_batch(
                        self._examples,
                        candidates_by_problem,
                        sample_fraction=self.config.mini_verify_sample_fraction,
                    )

                    # Judge boundary candidates' reasoning paths
                    all_cands_flat = [
                        c for cands in candidates_by_problem.values() for c in cands
                    ]
                    boundary_judgments = await judge_boundary_candidates(
                        all_cands_flat, analyses,
                        self.config.target_model_id, self.api,
                    )
                else:
                    # No target model — use critic for predicted labels
                    await critic.review_batch(analyses, candidates_by_problem)

            else:
                # Find mismatches across all problems
                all_cands = [
                    c for cands in candidates_by_problem.values() for c in cands
                ]

                # Use verification-aware mismatch detection when available
                if self._mini_verifier and self.config.target_model_id:
                    mismatches = find_mismatches_with_verification(
                        all_cands, self.config.target_model_id,
                        boundary_judgments=boundary_judgments,
                    )
                else:
                    mismatches = find_mismatches(all_cands)

                if not mismatches:
                    logger.info("No mismatches found. Stopping.")
                    break

                # Check category balance — if any category is underrepresented,
                # add synthetic mismatches so it gets more candidates regenerated
                cat_counts_mm: dict[str, int] = {}
                for c in all_cands:
                    if c.duplicate_of is None and c.is_viable:
                        cat_counts_mm[c.category] = cat_counts_mm.get(c.category, 0) + 1
                total_viable = sum(cat_counts_mm.values())
                target_quotas = self.config.category_quotas
                for cat, target_frac in target_quotas.items():
                    actual_frac = cat_counts_mm.get(cat, 0) / total_viable if total_viable > 0 else 0
                    if actual_frac < target_frac * 0.5:  # < half of target quota
                        # Add mismatches for the weakest candidates in this category
                        cat_cands = [c for c in all_cands if c.category == cat
                                     and c.duplicate_of is None and c.is_viable]
                        # Mark random ones for regeneration to boost this category
                        n_deficit = max(1, int(target_frac * total_viable) - len(cat_cands))
                        logger.info(
                            f"Category {cat} underrepresented ({actual_frac:.0%} vs "
                            f"{target_frac:.0%} target), adding {n_deficit} regeneration slots"
                        )
                        for c in cat_cands[:n_deficit]:
                            # Use mini-verify flip_fraction when available
                            pred = c.predicted_flip_probability
                            if pred is None:
                                vr = c.verification_results.get(verify_key, {})
                                pred = vr.get("flip_fraction", 0.5)
                            mismatches.append(CategoryMismatch(
                                candidate=c, category=cat,
                                predicted_flip=pred,
                                target_range=TARGET_RANGES.get(cat, (0.0, 1.0)),
                                direction="underrepresented_category",
                            ))

                # Group mismatches by (problem, category)
                mismatches_by_key: dict[tuple[int, str], list[CategoryMismatch]] = {}
                for m in mismatches:
                    key = (m.candidate.example_idx, m.category)
                    mismatches_by_key.setdefault(key, []).append(m)

                # Prepare all regeneration tasks, then run in parallel
                regen_tasks = []
                for (example_idx, category), cat_mismatches in mismatches_by_key.items():
                    analysis = next(
                        (a for a in analyses if a.example_idx == example_idx), None
                    )
                    if analysis is None:
                        continue

                    n_to_replace = len(cat_mismatches)

                    # Build feedback message (verification-aware when available)
                    problem_cands = candidates_by_problem.get(example_idx, [])
                    if self._mini_verifier and self.config.target_model_id:
                        feedback_msg = build_feedback_message_with_verification(
                            all_candidates=problem_cands,
                            mismatches=cat_mismatches,
                            category=category,
                            round_num=round_num,
                            n_to_generate=n_to_replace,
                            target_model_id=self.config.target_model_id,
                            boundary_judgments=boundary_judgments,
                        )
                    else:
                        feedback_msg = build_feedback_message(
                            all_candidates=problem_cands,
                            mismatches=cat_mismatches,
                            category=category,
                            round_num=round_num,
                            n_to_generate=n_to_replace,
                        )

                    if not feedback_msg:
                        continue

                    # Log the feedback message
                    if self._prompt_logger:
                        self._prompt_logger.log(
                            component="feedback_loop",
                            label=f"round_{round_num}_problem_{example_idx}_{category}",
                            user_prompt=feedback_msg,
                            extra={
                                "round": round_num,
                                "category": category,
                                "n_mismatches": len(cat_mismatches),
                                "n_to_replace": n_to_replace,
                            },
                        )

                    # Record feedback in tracer
                    if self._tracer and self._tracer.is_traced(example_idx):
                        self._tracer.record_feedback(
                            example_idx=example_idx,
                            round_num=round_num,
                            category=category,
                            mismatches=[
                                {"candidate_id": m.candidate.candidate_id,
                                 "direction": m.direction,
                                 "predicted_flip": m.predicted_flip}
                                for m in cat_mismatches
                            ],
                            feedback_message=feedback_msg,
                            new_candidates=[],  # filled after regen
                        )

                    # Build multi-turn conversation for this (problem, category)
                    key = (example_idx, category)
                    prior_turns = conversation_history.get(key, [])

                    # Add the prior assistant response + feedback user turn
                    if key in last_response_by_key:
                        prior_turns.append(ChatMessage(
                            role=MessageRole.assistant,
                            content=last_response_by_key[key],
                        ))
                    prior_turns.append(ChatMessage(
                        role=MessageRole.user,
                        content=feedback_msg,
                    ))
                    conversation_history[key] = prior_turns

                    regen_tasks.append((
                        example_idx, category, cat_mismatches,
                        analysis, {category: list(prior_turns)},
                    ))

                # Parallelize all regenerations in this round
                rn = round_num  # capture by value

                async def _regen_one(ex_idx, cat, cat_mm, anal, pbc):
                    try:
                        new_cands = await generator.generate(
                            anal, prior_turns_by_category=pbc,
                        )
                        new_cat = [c for c in new_cands if c.category == cat]
                        for c in new_cat:
                            c.candidate_id = f"r{rn}_{c.candidate_id}"
                        # Skip critic when mini-verifier handles verification
                        if not (self._mini_verifier and self._examples):
                            await critic.review_and_filter(anal, new_cat)
                        return ex_idx, cat, cat_mm, new_cat
                    except Exception as e:
                        logger.warning(
                            f"Regen failed for problem {ex_idx} {cat}: {e}"
                        )
                        return ex_idx, cat, cat_mm, []

                if regen_tasks:
                    regen_results = await asyncio.gather(
                        *[_regen_one(*t) for t in regen_tasks]
                    )
                    for ex_idx, cat, cat_mm, new_cat_cands in regen_results:
                        problem_cands = candidates_by_problem.get(ex_idx, [])
                        mm_ids = {m.candidate.candidate_id for m in cat_mm}
                        kept = [c for c in problem_cands if c.candidate_id not in mm_ids]
                        kept.extend(new_cat_cands)

                        candidates_by_problem[ex_idx] = kept

                # Mini-verify newly regenerated candidates
                if self._mini_verifier and self._examples:
                    await self._mini_verifier.mini_verify_batch(
                        self._examples,
                        candidates_by_problem,
                        sample_fraction=1.0,  # verify all new candidates
                    )

                    # Re-judge boundary candidates
                    all_cands_flat = [
                        c for cands in candidates_by_problem.values() for c in cands
                    ]
                    new_judgments = await judge_boundary_candidates(
                        all_cands_flat, analyses,
                        self.config.target_model_id, self.api,
                    )
                    boundary_judgments.update(new_judgments)

            # Compute metrics for this round
            all_cands = [
                c for cands in candidates_by_problem.values() for c in cands
            ]
            metrics = compute_round_metrics(
                all_cands, round_num, self.config.target_model_id,
                boundary_judgments=boundary_judgments,
            )
            all_metrics.append(metrics)

            logger.info(
                f"Round {round_num}: LSA={metrics.alignment_rate:.1%}, "
                f"MAE={metrics.mae:.3f}, mismatches={metrics.mismatches}"
            )

            prev_lsa = metrics.alignment_rate

            # Store raw response text for conversation history
            # (We need the actual LLM response, but since we don't have it
            # from the generator API, we reconstruct from candidates)
            for example_idx, cands in candidates_by_problem.items():
                for cat in ["flip_inducing", "non_flip", "boundary"]:
                    cat_cands = [c for c in cands if c.category == cat]
                    if cat_cands:
                        # Reconstruct a JSON-like response from candidates
                        response_items = []
                        for c in cat_cands:
                            item = {
                                "perturbation_type": c.perturbation_type,
                                "mechanism_name": c.mechanism_name,
                                "target_element": c.target_element,
                                "mechanism_application": c.mechanism_application,
                                "lever": c.lever,
                                "baseline": c.baseline,
                            }
                            if c.problem_edits:
                                item["problem_edits"] = [
                                    {
                                        "field": e.field,
                                        "original": e.original,
                                        "replacement": e.replacement,
                                        "description": e.description,
                                    }
                                    for e in c.problem_edits
                                ]
                            response_items.append(item)
                        key = (example_idx, cat)
                        last_response_by_key[key] = json.dumps(
                            response_items, indent=2
                        )

            # Convergence checks
            # 1. Empirical balance: flip/non-flip ratio AND category coverage
            if self._mini_verifier and self.config.target_model_id:
                verify_key = f"mini_verify_{self.config.target_model_id}"
                n_flip = 0
                n_noflip = 0
                cat_counts: dict[str, int] = {}
                for cands in candidates_by_problem.values():
                    for c in cands:
                        if c.duplicate_of is not None or not c.is_viable:
                            continue
                        cat_counts[c.category] = cat_counts.get(c.category, 0) + 1
                        vr = c.verification_results.get(verify_key, {})
                        if vr.get("flipped") is None:
                            continue
                        if vr["flipped"]:
                            n_flip += 1
                        else:
                            n_noflip += 1
                total_verified = n_flip + n_noflip
                if total_verified > 0:
                    ratio = min(n_flip, n_noflip) / max(n_flip, n_noflip) if max(n_flip, n_noflip) > 0 else 0
                    # Check category coverage: each category should have ≥15% of total
                    total_cands = sum(cat_counts.values())
                    min_cat_frac = min(
                        cat_counts.get(cat, 0) / total_cands
                        for cat in ["flip_inducing", "non_flip", "boundary"]
                    ) if total_cands > 0 else 0
                    logger.info(
                        f"Round {round_num} balance: "
                        f"{n_flip} flipped / {n_noflip} not flipped "
                        f"(ratio={ratio:.2f}), "
                        f"category counts={cat_counts} "
                        f"(min_frac={min_cat_frac:.2f})"
                    )
                    if ratio >= 0.8 and min_cat_frac >= 0.15:
                        logger.info(
                            f"Converged: empirical balance ratio {ratio:.2f} >= 0.8, "
                            f"all categories >= 15%"
                        )
                        break
                    elif ratio >= 0.8 and min_cat_frac < 0.15:
                        logger.info(
                            f"Flip balance OK but category {min(cat_counts, key=cat_counts.get)} "
                            f"underrepresented ({min_cat_frac:.0%}), continuing..."
                        )

            # 2. LSA threshold
            if metrics.alignment_rate >= self.lsa_threshold:
                logger.info(
                    f"Converged: LSA {metrics.alignment_rate:.1%} "
                    f">= {self.lsa_threshold:.1%}"
                )
                break

            # 3. Plateau: 3 consecutive rounds of stagnation
            if round_num > 0:
                improvement = metrics.alignment_rate - prev_lsa
                if improvement < self.improvement_threshold:
                    stagnant_rounds += 1
                    if stagnant_rounds >= 3:
                        logger.info(
                            f"Plateau: {stagnant_rounds} consecutive rounds "
                            f"with < {self.improvement_threshold:.1%} improvement"
                        )
                        break
                else:
                    stagnant_rounds = 0

            prev_lsa = metrics.alignment_rate

        # Single contrastive pass after loop converges
        if (
            self.config.enable_contrastive_pairs
            and self._mini_verifier
            and self._examples
        ):
            candidates_by_problem = await self._generate_contrastive_pass(
                candidates_by_problem, analyses, generator, critic,
            )

        return candidates_by_problem, all_metrics

    async def _generate_contrastive_pass(
        self,
        candidates_by_problem: dict[int, list[PerProblemCandidate]],
        analyses: list[ProblemAnalysis],
        generator: PerProblemGenerator,
        critic: CandidateCritic,
    ) -> dict[int, list[PerProblemCandidate]]:
        """Single contrastive generation pass after feedback loop converges.

        For each problem, picks balanced anchors (equal flipped/non-flipped)
        and asks the generator for near-miss variants. Verifies them and
        links successful pairs.
        """
        verify_key = f"mini_verify_{self.config.target_model_id}"
        model_name = self.config.target_model_id.split("/")[-1]
        example_map = {ex.idx: ex for ex in self._examples}

        logger.info("Contrastive pass: generating near-miss variants")

        async def _contrastive_one_problem(example_idx, candidates):
            """Generate contrastive pairs for one problem (parallelizable)."""
            # Collect anchors with clear outcomes
            flipped_anchors = []
            nonflip_anchors = []
            for c in candidates:
                if c.contrastive_pair_id or c.duplicate_of:
                    continue
                vr = c.verification_results.get(verify_key, {})
                flip_frac = vr.get("flip_fraction", 0.5)
                if flip_frac > 0.8:
                    flipped_anchors.append(c)
                elif flip_frac < 0.2:
                    nonflip_anchors.append(c)

            # Balance: pick equal from each direction
            n_each = min(2, len(flipped_anchors), len(nonflip_anchors))
            if n_each == 0:
                n_each = min(2, max(len(flipped_anchors), len(nonflip_anchors)))
            selected = flipped_anchors[:n_each] + nonflip_anchors[:n_each]
            if not selected:
                return

            # Build contrastive prompt
            analysis = next(
                (a for a in analyses if a.example_idx == example_idx), None,
            )
            if analysis is None:
                return

            lines = [
                f"Generate contrastive variants for these verified perturbations.",
                f"Each variant should be VERY SIMILAR to the original but achieve "
                f"the OPPOSITE flip outcome on {model_name}.",
                f"Differ in only 1-2 subtle aspects.",
                f"Add `\"contrastive_source\": \"<original_mechanism_name>\"` to each.",
                "",
            ]
            for c in selected:
                vr = c.verification_results.get(verify_key, {})
                flipped = vr.get("flipped", False)
                base_ans = vr.get("baseline_answer", "?")
                lever_ans = vr.get("lever_answer", "?")
                if flipped:
                    lines.append(
                        f"- **{c.mechanism_name}** FLIPPED ('{base_ans}'→'{lever_ans}'). "
                        f"Create a variant that does NOT flip."
                    )
                else:
                    lines.append(
                        f"- **{c.mechanism_name}** did NOT flip ('{base_ans}'→'{lever_ans}'). "
                        f"Create a variant that DOES flip."
                    )

            contrastive_msg = "\n".join(lines)

            # Use generator with contrastive prompt as a feedback turn
            for cat in ["flip_inducing", "non_flip", "boundary"]:
                cat_selected = [c for c in selected if c.category == cat]
                if not cat_selected:
                    continue

                try:
                    prior = [
                        ChatMessage(role=MessageRole.user, content=contrastive_msg),
                    ]
                    new_cands = await generator.generate(
                        analysis,
                        prior_turns_by_category={cat: prior},
                    )
                    new_cat = [c for c in new_cands if c.category == cat]

                    # Verify each contrast, only keep TRUE minimal pairs
                    # (opposite outcome from anchor), max 1 per anchor
                    example = example_map.get(example_idx)
                    if example:
                        for nc in new_cat:
                            # Find anchor: use contrastive_source_id if set,
                            # otherwise match to any unmatched anchor in this category
                            anchor = None
                            if nc.contrastive_source_id:
                                anchor = next(
                                    (c for c in candidates
                                     if c.mechanism_name == nc.contrastive_source_id
                                     and not c.contrastive_pair_id),
                                    None,
                                )
                            if anchor is None:
                                # Fallback: match to first unmatched anchor in category
                                anchor = next(
                                    (c for c in cat_selected
                                     if not c.contrastive_pair_id),
                                    None,
                                )
                            if anchor is None:
                                continue

                            anchor_vr = anchor.verification_results.get(verify_key, {})
                            anchor_flipped = anchor_vr.get("flipped")
                            if anchor_flipped is None:
                                continue

                            # Mini-verify the contrast
                            result = await self._mini_verifier.mini_verify(
                                example, nc,
                            )
                            nc.verification_results[verify_key] = result.to_dict()

                            # Only keep if opposite outcome
                            if result.flipped != anchor_flipped:
                                if nc.contrastive_pair_id:
                                    anchor.contrastive_pair_id = nc.contrastive_pair_id
                                    anchor.contrastive_role = "anchor"
                                candidates.append(nc)
                                logger.info(
                                    f"Contrastive pair: {anchor.mechanism_name}"
                                    f"({'flip' if anchor_flipped else 'no-flip'}) "
                                    f"↔ {nc.mechanism_name}"
                                    f"({'flip' if result.flipped else 'no-flip'})"
                                )
                                break  # max 1 per anchor per category
                except Exception as e:
                    logger.warning(
                        f"Contrastive generation failed for problem "
                        f"{example_idx} {cat}: {e}"
                    )

        # Parallelize contrastive generation across all problems
        await asyncio.gather(*[
            _contrastive_one_problem(idx, cands)
            for idx, cands in candidates_by_problem.items()
        ])

        n_contrastive = sum(
            1 for cands in candidates_by_problem.values()
            for c in cands if c.contrastive_pair_id
        )
        logger.info(f"Contrastive pass complete: {n_contrastive} candidates tagged")
        return candidates_by_problem
