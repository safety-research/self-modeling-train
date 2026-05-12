"""
CLI: Multi-task GRPO training.

Trains on any combination of the 11 introspection tasks using the
MultitaskRecord data format. Reward dispatches per gt_type.

Usage:
    # All tasks:
    python scripts/training/run_grpo_multitask.py --tasks all

    # Single task (any of the 11):
    python scripts/training/run_grpo_multitask.py --tasks e3

    # Subset:
    python scripts/training/run_grpo_multitask.py --tasks e1,e3,e6

    # Warm-init from a previous checkpoint:
    python scripts/training/run_grpo_multitask.py --tasks all --warm-init-checkpoint TINKER_PATH

    # Smoke test:
    python scripts/training/run_grpo_multitask.py --tasks e3 --n-steps 2 --n-samples 4
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root so TINKER_API_KEY etc. are available.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJ_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJ_DIR / "outputs/training/multitask_balanced"
DEFAULT_OUTPUT_DIR = PROJ_DIR / "outputs/training/multitask_grpo"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-task GRPO training")

    # Task selection
    parser.add_argument("--tasks", type=str, default="all",
                        help="Tasks: 'all', 'e3', 'e1,e3,e6', etc.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Path to multitask balanced data dir")

    # Model
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--warm-init-checkpoint", type=str, default="",
                        help="Tinker path to a previous checkpoint to warm-init from")
    parser.add_argument("--lora-rank", type=int, default=32)

    # Training (defaults match run_grpo_single_task.py — the canonical config)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k-completions", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--generation-temperature", type=float, default=0.7)
    parser.add_argument("--max-generation-tokens", type=int, default=2048)

    # Loss / reward
    parser.add_argument("--loss-fn", type=str, default="importance_sampling",
                        choices=["importance_sampling", "ppo", "cispo"])
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--reward-type", type=str, default="mse",
                        choices=["mse", "binary", "bce", "neg_bce", "compound"],
                        help="Reward type (default: mse, matching canonical config)")

    # Checkpointing / eval (0 = auto, once per epoch)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--eval-n-samples", type=int, default=50)

    # Early stopping
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-warmup", type=int, default=1000)

    # Eval toggles (all disabled by default for fast training)
    parser.add_argument("--enable-mini-eval", action="store_true",
                        help="Enable per-step mini evaluations during training")
    parser.add_argument("--enable-early-stopping", action="store_true",
                        help="Enable early stopping checks")
    parser.add_argument("--enable-final-eval", action="store_true",
                        help="Enable final evaluation after training")

    # Output
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-project", type=str, default="prompt-attribution-multitask-grpo")
    parser.add_argument("--run-name", type=str, default="")

    # Debug
    parser.add_argument("--n-samples", type=int, default=0,
                        help="Limit training data (0=all)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Build config
    sys.path.insert(0, str(PROJ_DIR / "src"))
    from prompt_attribution.training.config import (
        GRPOConfig,
        MultitaskDataConfig,
        TrainingSchedule,
    )

    config = GRPOConfig(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        seed=args.seed,
        load_checkpoint=args.warm_init_checkpoint,
        batch_size=args.batch_size,
        k_completions=args.k_completions,
        generation_temperature=args.generation_temperature,
        max_generation_tokens=args.max_generation_tokens,
        loss_fn=args.loss_fn,
        clip_epsilon=args.clip_epsilon,
        reward_type=args.reward_type,
        schedule=TrainingSchedule(
            base_lr=args.lr,
            lr_warmup_steps=50,
            lr_decay="cosine",
        ),
        n_steps=args.n_steps,
        eval_interval_steps=args.eval_interval,
        checkpoint_interval_steps=args.checkpoint_interval,
        eval_n_samples=args.eval_n_samples,
        early_stop_patience=args.early_stop_patience,
        early_stop_warmup_steps=args.early_stop_warmup,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        run_name=args.run_name or f"mt_{args.tasks.replace(',', '_')}",
        n_samples=args.n_samples,
        disable_mini_eval=not args.enable_mini_eval,
        disable_early_stopping=not args.enable_early_stopping,
        disable_final_eval=not args.enable_final_eval,
        # Multi-task config
        multitask=MultitaskDataConfig(
            multitask_data_dir=args.data_dir,
            tasks=args.tasks,
        ),
    )

    logger.info(f"Multi-task GRPO: tasks={args.tasks}, data={args.data_dir}")
    logger.info(f"Config: batch={args.batch_size}, K={args.k_completions}, "
                f"lr={args.lr}, steps={args.n_steps}")

    from prompt_attribution.training.train_grpo import train_grpo
    asyncio.run(train_grpo(config))


if __name__ == "__main__":
    main()
