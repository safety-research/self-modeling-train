"""
Module: prompt_attribution/auto_perturbation/training_data.py

Stage 6: Training data export. Converts pipeline outputs into JSONL
training data format with both critic predictions and empirical labels.

Structure:
- TrainingExample: Dataclass for a single training example
- export_training_data: Export function producing JSONL + stats
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from ..config import PerProblemCandidate, VerificationResult
from ..dataset_adapter.dataset_adapter import DatasetAdapter, AdaptedExample

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example for the attribution model.

    Contains the problem, perturbation, both critic-predicted and
    empirical ground-truth labels, and full model responses.
    """

    # Globally unique row identifier (set by corpus orchestrator)
    # Format: {dataset_id}::{example_idx}::{perturbation_id}
    unique_id: str = ""

    # Dataset + problem identification
    dataset_id: str = ""
    example_idx: int = 0
    question: str = ""
    ground_truth_answer: str = ""

    # Prompts
    prompt_baseline: str = ""
    prompt_lever: str = ""

    # Perturbation
    perturbation_id: str = ""
    lever_text: str = ""
    baseline_text: str = ""
    mechanism_name: str = ""
    category: str = ""  # flip_inducing, non_flip, boundary

    # Critic labels (None when critic is skipped)
    predicted_flip_probability: Optional[float] = None
    consistency_score: Optional[float] = None

    # Capability tags (from discovery)
    capability_tags: list[str] = field(default_factory=list)

    # Complexity axes (from discovery, for training data stratification)
    context_source: str = ""  # single_source | multi_source | multimodal_context
    context_length: str = ""  # short | long | multi_document
    interaction_mode: str = ""  # static | tool_use | multi_turn

    # Empirical labels (model-specific, from Phase 1 verification)
    empirical_flipped: Optional[bool] = None
    empirical_flip_fraction: Optional[float] = None
    empirical_flip_count: Optional[int] = None
    empirical_n_runs: Optional[int] = None
    empirical_baseline_answer: str = ""
    empirical_lever_answer: str = ""
    judge_model: str = ""

    # Full responses (stored for training data richness)
    empirical_baseline_responses: list[str] = field(default_factory=list)
    empirical_lever_responses: list[str] = field(default_factory=list)

    # Extracted features from ideated answer_labels (from first verification run)
    features_baseline: dict[str, str] = field(default_factory=dict)
    features_lever: dict[str, str] = field(default_factory=dict)

    # Perturbation type and edit details
    perturbation_type: str = "instruction_add"
    problem_edits: list[dict] = field(default_factory=list)
    edit_distance: Optional[int] = None
    edit_fraction: Optional[float] = None

    # Target label axis — which answer label this perturbation targets for flip detection
    target_label_axis: str = ""

    # How the flip was verified: "programmatic" (exact/numeric match) or "llm_judge" (Haiku)
    # Programmatic labels are more reliable; llm_judge labels may have extraction errors
    verification_method: str = ""

    # Full label definitions from the dataset profile (names, types, verification methods)
    answer_labels: list[dict] = field(default_factory=list)

    # Where the lever instruction was inserted into the prompt
    # Values: "prepend", "after_context", "after_question", "after_choices", "append"
    instruction_placement: str = ""

    # The prompt template with {instruction} placeholder showing insertion position
    prompt_template: str = ""

    # Example-level fields needed for template reconstruction
    context: Optional[str] = None       # Context/passage text (if present)
    choices: Optional[list[str]] = None  # MCQ choices (if MCQ)

    # Profile-level label names (for CLASSIFICATION task type)
    label_names: list[str] = field(default_factory=list)

    # Contrastive pair metadata (links near-identical perturbations with opposite flip outcomes)
    contrastive_pair_id: Optional[str] = None  # shared UUID linking anchor + contrast
    contrastive_role: str = ""  # "anchor" or "contrast"
    contrastive_source_id: Optional[str] = None  # candidate_id of the anchor

    def to_dict(self) -> dict:
        return asdict(self)


def _get_verification_method(
    target_axis: str,
    answer_labels: list[dict],
    feat_baseline: dict,
    feat_lever: dict,
) -> str:
    """Determine verification method and flag judge failures.

    Returns:
        "programmatic" — deterministic comparison (reliable)
        "llm_judge" — LLM-based extraction (less reliable)
        "llm_judge_failed" — LLM judge returned (unknown) for any feature
    """
    if not target_axis or not answer_labels:
        return "programmatic"

    target_label = next((l for l in answer_labels if l.get("name") == target_axis), None)
    if not target_label:
        return "programmatic"

    vtype = target_label.get("value_type", "string")
    method = target_label.get("verification_method", "programmatic")

    if vtype == "string" or method == "llm_judge":
        # Check if judge produced (unknown)
        b_val = str(feat_baseline.get(target_axis, ""))
        l_val = str(feat_lever.get(target_axis, ""))
        if "(unknown)" in b_val or "(unknown)" in l_val:
            return "llm_judge_failed"
        return "llm_judge"

    return "programmatic"


def export_training_data(
    examples: list[AdaptedExample],
    candidates_by_problem: dict[int, list[PerProblemCandidate]],
    adapter: DatasetAdapter,
    judge_model: str,
    output_dir: Path,
    capability_tags: list[str] | None = None,
    context_source: str = "",
    context_length: str = "",
    interaction_mode: str = "",
) -> list[TrainingExample]:
    """Export pipeline outputs as training data.

    Writes:
    - training_data.jsonl: One example per line
    - training_data_stats.json: Summary statistics

    Args:
        examples: List of problem examples
        candidates_by_problem: Dict of candidates per problem (with scores)
        adapter: Dataset adapter for building prompts
        judge_model: Model ID used for verification
        output_dir: Output directory
        capability_tags: Capability domains from discovery (e.g., ["math_reasoning"])
        context_source: Complexity axis — single_source, multi_source, multimodal_context
        context_length: Complexity axis — short, long, multi_document
        interaction_mode: Complexity axis — static, tool_use, multi_turn

    Returns:
        List of TrainingExample objects
    """
    example_map = {ex.idx: ex for ex in examples}
    dataset_id = adapter.profile.dataset_id
    tags = capability_tags or []

    # Extract profile-level metadata for recoverability
    profile_answer_labels = adapter.profile.answer_labels
    profile_prompt_template = adapter.profile.prompt_template
    profile_instruction_placement = adapter.profile.instruction_placement
    profile_label_names = adapter.profile.label_names

    training_examples = []

    for example_idx, candidates in candidates_by_problem.items():
        example = example_map.get(example_idx)
        if example is None:
            continue

        for candidate in candidates:
            if not candidate.passed_critic:
                continue

            # Build prompts — use axis-specific preamble + format when available
            target_axis = candidate.target_label_axis
            prompt_baseline = adapter.make_axis_baseline_prompt(
                example, target_axis, candidate.baseline,
            )
            if candidate.perturbation_type == "problem_edit":
                prompt_lever = adapter.make_axis_edited_prompt(
                    example, target_axis, candidate.problem_edits, candidate.baseline,
                )
            else:
                prompt_lever = adapter.make_axis_lever_prompt(
                    example, target_axis, candidate.lever, candidate.baseline,
                )

            # Resolve instruction placement: per-candidate override or profile default
            effective_placement = (
                candidate.instruction_placement
                if candidate.instruction_placement
                else profile_instruction_placement
            )

            # Extract empirical results if available
            empirical_flipped = None
            empirical_flip_fraction = None
            empirical_flip_count = None
            empirical_n_runs = None
            empirical_baseline_answer = ""
            empirical_lever_answer = ""
            baseline_responses = []
            lever_responses = []
            feat_baseline: dict[str, str] = {}
            feat_lever: dict[str, str] = {}

            if judge_model in candidate.verification_results:
                vr = VerificationResult.from_dict(
                    candidate.verification_results[judge_model]
                )
                empirical_flipped = vr.flipped
                empirical_flip_fraction = vr.flip_fraction
                empirical_flip_count = vr.flip_count
                empirical_n_runs = vr.n_runs
                empirical_baseline_answer = vr.baseline_answer
                empirical_lever_answer = vr.lever_answer
                baseline_responses = vr.baseline_responses
                lever_responses = vr.lever_responses
                feat_baseline = vr.features_baseline
                feat_lever = vr.features_lever

            te = TrainingExample(
                dataset_id=dataset_id,
                example_idx=example_idx,
                question=example.question,
                ground_truth_answer=example.ground_truth_answer,
                capability_tags=tags,
                prompt_baseline=prompt_baseline,
                prompt_lever=prompt_lever,
                perturbation_id=candidate.candidate_id,
                lever_text=candidate.lever,
                baseline_text=candidate.baseline,
                mechanism_name=candidate.mechanism_name,
                category=candidate.category,
                predicted_flip_probability=candidate.predicted_flip_probability,
                consistency_score=candidate.consistency_score,
                empirical_flipped=empirical_flipped,
                empirical_flip_fraction=empirical_flip_fraction,
                empirical_flip_count=empirical_flip_count,
                empirical_n_runs=empirical_n_runs,
                empirical_baseline_answer=empirical_baseline_answer,
                empirical_lever_answer=empirical_lever_answer,
                judge_model=judge_model,
                empirical_baseline_responses=baseline_responses,
                empirical_lever_responses=lever_responses,
                features_baseline=feat_baseline,
                features_lever=feat_lever,
                perturbation_type=candidate.perturbation_type,
                problem_edits=[e.to_dict() for e in candidate.problem_edits],
                edit_distance=candidate.edit_distance,
                edit_fraction=candidate.edit_fraction,
                target_label_axis=candidate.target_label_axis,
                verification_method=_get_verification_method(
                    candidate.target_label_axis, profile_answer_labels,
                    feat_baseline, feat_lever,
                ),
                answer_labels=profile_answer_labels,
                instruction_placement=effective_placement,
                prompt_template=profile_prompt_template,
                context=example.context,
                choices=example.choices,
                label_names=profile_label_names,
                # Complexity axes
                context_source=context_source,
                context_length=context_length,
                interaction_mode=interaction_mode,
                # Contrastive pair metadata
                contrastive_pair_id=candidate.contrastive_pair_id,
                contrastive_role=candidate.contrastive_role or "",
                contrastive_source_id=candidate.contrastive_source_id,
            )
            training_examples.append(te)

    # Drop failed problem_edits where baseline == lever (edit didn't apply)
    before = len(training_examples)
    training_examples = [
        te for te in training_examples
        if not (te.perturbation_type == "problem_edit"
                and te.prompt_baseline == te.prompt_lever)
    ]
    n_dropped = before - len(training_examples)
    if n_dropped:
        logger.warning(
            f"Dropped {n_dropped} problem_edit examples where edit failed "
            f"(baseline == lever)"
        )

    # Drop examples where LLM judge failed to extract features —
    # flip labels are unreliable when judge returns "(unknown)"
    before = len(training_examples)
    training_examples = [
        te for te in training_examples
        if te.verification_method != "llm_judge_failed"
    ]
    n_judge_dropped = before - len(training_examples)
    if n_judge_dropped:
        logger.warning(
            f"Dropped {n_judge_dropped} examples where LLM judge failed "
            f"to extract features (unreliable flip labels)"
        )

    # Write JSONL
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "training_data.jsonl"
    with open(jsonl_path, "w") as f:
        for te in training_examples:
            f.write(json.dumps(te.to_dict()) + "\n")

    # Compute and write stats
    stats = _compute_stats(training_examples, dataset_id, judge_model)
    stats_path = output_dir / "training_data_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"Exported {len(training_examples)} training examples to {jsonl_path}"
    )
    logger.info(f"Stats: {json.dumps(stats, indent=2)}")

    return training_examples


def _filter_contrastive_pairs(
    examples: list[TrainingExample],
) -> list[TrainingExample]:
    """Remove contrastive pairs where both sides have the same flip outcome.

    Contrastive pairs are only useful when anchor and contrast have opposite
    outcomes. The mini-verifier check at generation time may disagree with
    the final verification, so we re-check here.
    """
    # Group by pair_id
    by_pair: dict[str, list[TrainingExample]] = {}
    for te in examples:
        if te.contrastive_pair_id:
            by_pair.setdefault(te.contrastive_pair_id, []).append(te)

    # Find bad pairs (same outcome)
    bad_pair_ids: set[str] = set()
    for pair_id, members in by_pair.items():
        flips = [te.empirical_flipped for te in members if te.empirical_flipped is not None]
        if len(flips) >= 2 and len(set(flips)) == 1:
            # All same outcome — not a real contrastive pair
            bad_pair_ids.add(pair_id)

    if bad_pair_ids:
        logger.info(
            f"Filtered {len(bad_pair_ids)} contrastive pairs with same outcome "
            f"(keeping {len(by_pair) - len(bad_pair_ids)} valid pairs)"
        )

    # Remove bad pairs — clear their contrastive metadata but keep the examples
    result = []
    for te in examples:
        if te.contrastive_pair_id in bad_pair_ids:
            te.contrastive_pair_id = None
            te.contrastive_role = ""
            te.contrastive_source_id = None
        result.append(te)

    return result


def _compute_stats(
    examples: list[TrainingExample],
    dataset_id: str,
    judge_model: str,
) -> dict:
    """Compute summary statistics for the training data."""
    total = len(examples)
    if total == 0:
        return {"total_examples": 0, "dataset_id": dataset_id}

    # By category
    by_category: dict[str, list[TrainingExample]] = {}
    for te in examples:
        by_category.setdefault(te.category, []).append(te)

    category_stats = {}
    for cat, cat_examples in by_category.items():
        # Critic stats (None when critic is skipped)
        predicted_flips = [
            te.predicted_flip_probability for te in cat_examples
            if te.predicted_flip_probability is not None
        ]
        avg_predicted = (
            sum(predicted_flips) / len(predicted_flips)
            if predicted_flips else None
        )

        # Empirical stats (if verification was run)
        empirical_flips = [
            te.empirical_flip_fraction
            for te in cat_examples
            if te.empirical_flip_fraction is not None
        ]
        avg_empirical = (
            sum(empirical_flips) / len(empirical_flips)
            if empirical_flips else None
        )

        category_stats[cat] = {
            "count": len(cat_examples),
            "avg_predicted_flip_probability": (
                round(avg_predicted, 3) if avg_predicted is not None else None
            ),
            "avg_empirical_flip_fraction": (
                round(avg_empirical, 3) if avg_empirical is not None else None
            ),
        }

    # By mechanism
    mechanism_counts: dict[str, int] = {}
    for te in examples:
        mechanism_counts[te.mechanism_name] = (
            mechanism_counts.get(te.mechanism_name, 0) + 1
        )

    # By perturbation type
    by_ptype: dict[str, list[TrainingExample]] = {}
    for te in examples:
        by_ptype.setdefault(te.perturbation_type, []).append(te)

    ptype_stats = {}
    for ptype, ptype_examples in by_ptype.items():
        p_empirical = [
            te.empirical_flip_fraction
            for te in ptype_examples
            if te.empirical_flip_fraction is not None
        ]
        edit_fracs = [
            te.edit_fraction
            for te in ptype_examples
            if te.edit_fraction is not None
        ]
        ptype_stats[ptype] = {
            "count": len(ptype_examples),
            "avg_empirical_flip_fraction": (
                round(sum(p_empirical) / len(p_empirical), 3)
                if p_empirical else None
            ),
            "avg_edit_fraction": (
                round(sum(edit_fracs) / len(edit_fracs), 4)
                if edit_fracs else None
            ),
        }

    # Overall empirical stats
    all_empirical = [
        te.empirical_flip_fraction
        for te in examples
        if te.empirical_flip_fraction is not None
    ]
    overall_empirical_flip_rate = (
        sum(all_empirical) / len(all_empirical)
        if all_empirical else None
    )

    return {
        "total_examples": total,
        "dataset_id": dataset_id,
        "judge_model": judge_model,
        "by_category": category_stats,
        "mechanism_distribution": dict(
            sorted(mechanism_counts.items(), key=lambda x: -x[1])
        ),
        "overall_avg_predicted_flip_probability": (
            round(
                sum(p for p in (te.predicted_flip_probability for te in examples) if p is not None)
                / max(sum(1 for te in examples if te.predicted_flip_probability is not None), 1),
                3,
            ) if any(te.predicted_flip_probability is not None for te in examples) else None
        ),
        "overall_empirical_flip_rate": (
            round(overall_empirical_flip_rate, 3)
            if overall_empirical_flip_rate is not None else None
        ),
        "by_perturbation_type": ptype_stats,
    }
