"""
Module: prompt_attribution/training/hooks/early_stopping.py

Early stopping logic for RL training loops.
Reads full eval results from backfill_evals_loop.sh on disk.

Structure:
- EarlyStopState: Mutable state tracking best C-index and patience counter
- should_stop(): Check if training should stop
- read_full_eval_results(): Read eval results.json for a given checkpoint step
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EarlyStopState:
    """Mutable state for early stopping across training iterations."""

    best_c_index: float = 0.0
    best_checkpoint_step: int = 0
    no_improve_count: int = 0
    last_ckpt_checked: int = 0

    def save(self, run_dir: str | Path) -> None:
        """Persist early stop state to JSON for resume support."""
        path = Path(run_dir) / "early_stop_state.json"
        with open(path, "w") as f:
            json.dump({
                "best_c_index": self.best_c_index,
                "best_checkpoint_step": self.best_checkpoint_step,
                "no_improve_count": self.no_improve_count,
                "last_ckpt_checked": self.last_ckpt_checked,
            }, f, indent=2)

    @classmethod
    def load(cls, run_dir: str | Path) -> "EarlyStopState":
        """Restore early stop state from JSON, or return fresh state if not found."""
        path = Path(run_dir) / "early_stop_state.json"
        if not path.exists():
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
            state = cls(
                best_c_index=data.get("best_c_index", 0.0),
                best_checkpoint_step=data.get("best_checkpoint_step", 0),
                no_improve_count=data.get("no_improve_count", 0),
                last_ckpt_checked=data.get("last_ckpt_checked", 0),
            )
            logger.info(
                f"\033[36m[RESUME]\033[0m Restored early stop state: "
                f"best_c_index={state.best_c_index:.4f}, "
                f"no_improve={state.no_improve_count}, "
                f"last_checked={state.last_ckpt_checked}"
            )
            return state
        except (ValueError, OSError, KeyError) as e:
            logger.warning(f"Failed to load early stop state from {path}: {e}")
            return cls()


def read_full_eval_results(
    output_dir: str | Path,
    run_name: str,
    step: int,
) -> dict | None:
    """Read full eval results from disk for a given checkpoint step.

    Path: {output_dir}/{run_name}/full_eval/eval_step_{step}/results.json
    Written by backfill_evals_loop.sh running alongside training.
    """
    results_path = (
        Path(output_dir) / run_name / "full_eval" / f"eval_step_{step}" / "results.json"
    )
    if not results_path.exists():
        return None
    try:
        with open(results_path) as f:
            return json.load(f)
    except (ValueError, OSError) as e:
        logger.warning(f"Failed to read full eval results at {results_path}: {e}")
        return None


def should_stop(
    state: EarlyStopState,
    step: int,
    *,
    output_dir: str | Path,
    run_name: str,
    checkpoint_interval: int,
    warmup_steps: int = 0,
    c_index_floor: float = 0.0,
    patience: int = 3,
    ml_logger: Any = None,
    run_dir: str | Path | None = None,
) -> bool:
    """Check if training should stop based on full eval results from backfill.

    Reads eval results from disk (written by backfill_evals_loop.sh).
    Two stopping modes:
    - Floor-based: stop if live C-index drops below c_index_floor
    - Patience-based: stop if live C-index hasn't improved for `patience` checkpoints

    If ml_logger is provided, logs full eval metrics to wandb when found.

    Returns True if training should stop.
    """
    if step < warmup_steps:
        return False

    latest_ckpt_step = (step // checkpoint_interval) * checkpoint_interval
    if latest_ckpt_step <= state.last_ckpt_checked:
        return False  # No new checkpoint to check

    full_eval_metrics = read_full_eval_results(output_dir, run_name, latest_ckpt_step)
    if full_eval_metrics is not None:
        state.last_ckpt_checked = latest_ckpt_step
    if full_eval_metrics is None:
        logger.info(
            f"No full eval results for checkpoint step {latest_ckpt_step} yet "
            f"(waiting for backfill). Skipping early stop check."
        )
        return False

    # Support both E3 (c_index) and multi-task (overall_score) metrics
    c_index = full_eval_metrics.get(
        "multitask/overall_score",
        full_eval_metrics.get("live_c_index", full_eval_metrics.get("c_index", 0.0)),
    )
    mse = full_eval_metrics.get(
        "live_mse", full_eval_metrics.get("mse", 0.0)
    )
    logger.info(
        f"Early stop check at step {step} (ckpt {latest_ckpt_step}): "
        f"full eval live_c_index={c_index:.4f} "
        f"(best={state.best_c_index:.4f}, no_improve={state.no_improve_count})"
    )

    # Log full eval metrics to wandb
    if ml_logger is not None:
        wandb_metrics: dict[str, float] = {
            "full_eval/c_index": c_index,
            "full_eval/mse": mse,
            "full_eval/best_c_index": max(state.best_c_index, c_index),
        }
        # Include per-category breakdown if available
        per_cat = full_eval_metrics.get("live_per_category") or full_eval_metrics.get("per_category")
        if per_cat:
            for cat, cat_metrics in per_cat.items():
                if isinstance(cat_metrics, dict):
                    for k, v in cat_metrics.items():
                        if isinstance(v, (int, float)):
                            wandb_metrics[f"full_eval/{cat}/{k}"] = v
        ml_logger.log_metrics(wandb_metrics, step=step)

    # Floor check: catastrophic collapse
    if (
        c_index_floor > 0
        and c_index < c_index_floor
        and step > checkpoint_interval * 2  # Skip first 2 checkpoints
    ):
        logger.warning(
            f"\033[31m[EARLY STOP]\033[0m full eval live C-index={c_index:.4f} "
            f"< floor {c_index_floor} at step {step}"
        )
        return True

    # Patience check: no improvement from best
    if patience > 0:
        if c_index > state.best_c_index:
            state.best_c_index = c_index
            state.best_checkpoint_step = latest_ckpt_step
            state.no_improve_count = 0
        else:
            state.no_improve_count += 1

        if state.no_improve_count >= patience:
            logger.warning(
                f"\033[31m[EARLY STOP]\033[0m No full eval live C-index improvement for "
                f"{state.no_improve_count} checkpoints "
                f"(best={state.best_c_index:.4f}, current={c_index:.4f}) "
                f"at step {step}"
            )
            if run_dir:
                state.save(run_dir)
            return True

    # Persist state after each check for resume support
    if run_dir:
        state.save(run_dir)

    return False
