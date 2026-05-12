"""
Module: prompt_attribution/training/hooks/gt_validation.py

Ground truth validation for training data.
Ensures GT cache is from the correct base model before training starts.

Structure:
- validate_gt_config(): Check GT cache path is configured and exists
- validate_loaded_gt(): Check inference_model field matches base model
"""

import logging
from collections import Counter
from pathlib import Path

from prompt_attribution.training.data.dataset import TrainingRecord

logger = logging.getLogger(__name__)


def validate_gt_config(
    gt_cache_path: str | Path | None,
    base_model: str,
    training_data_path: str | Path,
) -> None:
    """Validate that a base model GT cache is configured before loading data.

    RL training requires ground truth from the base model being trained,
    not from a different model (e.g., Haiku from auto-perturbation pipeline).

    Raises:
        ValueError: If no GT cache path is configured.
        FileNotFoundError: If the cache file doesn't exist.
    """
    if not gt_cache_path:
        raise ValueError(
            f"base_model_gt_cache_path is required. Ground truth must come from "
            f"the base model ({base_model}), not from the training data "
            f"file which may use a different model (e.g., Claude Haiku).\n"
            f"Run Phase 1 GT inference with the base model:\n"
            f"  python -m scripts.training.run_ground_truth \\\n"
            f"    --input {training_data_path} \\\n"
            f"    --output <gt_cache_path>.jsonl \\\n"
            f"    --model {base_model} \\\n"
            f"    --vllm_url <VLLM_URL> --n_runs 1\n"
            f"Then pass --base_model_gt_cache_path <gt_cache_path>.jsonl"
        )

    gt_file = Path(gt_cache_path)
    if not gt_file.exists():
        raise FileNotFoundError(
            f"Base model GT cache not found: {gt_file}. "
            f"Run Phase 1 GT inference with {base_model} first."
        )
    logger.info(f"GT cache configured: {gt_file} (base_model={base_model})")


def validate_loaded_gt(
    records: list[TrainingRecord],
    base_model: str,
    gt_cache_path: str | Path | None = None,
) -> None:
    """Validate that loaded records have GT from the correct base model.

    Checks the inference_model field on records. If any record has
    inference_model set to a value that doesn't match base_model,
    raises ValueError if the GT cache is incompatible.

    Args:
        records: All loaded training records (train + test).
        base_model: Expected base model name.
        gt_cache_path: Path to GT cache.
    """
    model_counts: Counter[str] = Counter()
    for r in records:
        model_counts[r.inference_model or "(empty)"] += 1

    for model, count in model_counts.most_common():
        logger.info(f"  GT inference_model='{model}': {count} records")

    # Check for mismatches (empty is allowed for legacy data)
    mismatched = {
        m: c for m, c in model_counts.items()
        if m != "(empty)" and m != base_model and m != "tinker_lora"
    }
    if mismatched:
        mismatch_lines = "\n".join(
            f"  - '{m}': {c} records" for m, c in mismatched.items()
        )
        raise ValueError(
                f"Ground truth model mismatch! Expected base model '{base_model}' "
                f"but found records with different inference_model:\n{mismatch_lines}\n"
                f"The training data has GT from the wrong model. "
                f"Run Phase 1 GT with the base model:\n"
                f"  python -m scripts.training.run_ground_truth "
                f"--model {base_model} ..."
            )

    # Warn if all records have empty inference_model (legacy data)
    if model_counts.get("(empty)", 0) == len(records):
        logger.warning(
            f"All {len(records)} records have empty inference_model — "
            f"cannot verify GT provenance. Consider regenerating GT cache."
        )
