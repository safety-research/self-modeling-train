"""
CLI: GRPO single-task training — full dataset, multi-model scale-up.

4 configs (array index 0-3): Llama-8B, Qwen3-8B, Llama-70B, Qwen3-32B.
All from scratch, mse_format reward, lr=2e-5.

Usage:
    # Run config at index N:
    python -m scripts.training.run_grpo_single_task --index 0

    # List all configs:
    python -m scripts.training.run_grpo_single_task --list

    # Smoke test (2 samples, 5 steps):
    python -m scripts.training.run_grpo_single_task --index 0 --n_steps 5 --n_samples 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
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

# ── Paths ────────────────────────────────────────────────────────────────────

PROJ_DIR = Path(__file__).resolve().parents[2]

# Per-model datasets: each model family has its own corpus + GT cache
# (empirical flip rates are model-specific)
LLAMA_CORPUS = PROJ_DIR / "outputs/auto_perturbation/corpus_llama8b"
LLAMA_TRAINING_DATA = str(LLAMA_CORPUS / "training_data.jsonl")
LLAMA_GT_CACHE = ""  # Set this to a precomputed GT cache JSONL if available

QWEN3_8B_CORPUS = PROJ_DIR / "outputs/auto_perturbation/corpus_qwen8b"
QWEN3_8B_TRAINING_DATA = str(QWEN3_8B_CORPUS / "training_data.jsonl")
QWEN3_8B_GT_CACHE = ""

# Larger model corpora
LLAMA_70B_CORPUS = PROJ_DIR / "outputs/auto_perturbation/corpus_llama70b"
LLAMA_70B_TRAINING_DATA = str(LLAMA_70B_CORPUS / "training_data.jsonl")
LLAMA_70B_GT_CACHE = ""

QWEN3_32B_CORPUS = PROJ_DIR / "outputs/auto_perturbation/corpus_qwen32b"
QWEN3_32B_TRAINING_DATA = str(QWEN3_32B_CORPUS / "training_data.jsonl")
QWEN3_32B_GT_CACHE = ""

OUTPUT_DIR = PROJ_DIR / "outputs/sweeps/grpo_single_task"
WANDB_PROJECT = "prompt-attribution-grpo"

# ── Model definitions ────────────────────────────────────────────────────────

MODELS = [
    # 8B models
    {
        "name": "llama8b",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "renderer": "llama3",
        "thinking": False,
        "training_data": LLAMA_TRAINING_DATA,
        "gt_cache": LLAMA_GT_CACHE,
        "train_records_approx": 16_864,  # 19,840 * 0.85
    },
    {
        "name": "qwen8b",
        "base_model": "Qwen/Qwen3-8B",
        "renderer": "qwen3",
        "thinking": True,  # Enable thinking (chain-of-thought)
        "training_data": QWEN3_8B_TRAINING_DATA,
        "gt_cache": QWEN3_8B_GT_CACHE,
        "train_records_approx": 14_882,  # 17,509 * 0.85
    },
    # Larger models
    {
        "name": "llama70b",
        "base_model": "meta-llama/Llama-3.3-70B-Instruct",
        "renderer": "llama3",
        "thinking": False,
        "training_data": LLAMA_70B_TRAINING_DATA,
        "gt_cache": LLAMA_70B_GT_CACHE,
        "train_records_approx": 16_864,
    },
    {
        "name": "qwen32b",
        "base_model": "Qwen/Qwen3-32B",
        "renderer": "qwen3",
        "thinking": True,
        "training_data": QWEN3_32B_TRAINING_DATA,
        "gt_cache": QWEN3_32B_GT_CACHE,
        "train_records_approx": 14_882,
    },
]


# ── Config builder ───────────────────────────────────────────────────────────

def get_configs(
    n_steps: int | None = None,
    early_stop_warmup_steps: int | None = None,
) -> list[dict]:
    """Return 4 configs, one per model.

    Layout:
        0: llama8b_scratch_mse   — Llama-3.1-8B
        1: qwen8b_scratch_mse    — Qwen3-8B (thinking)
        2: llama70b_scratch_mse  — Llama-3.3-70B
        3: qwen32b_scratch_mse   — Qwen3-32B (thinking)
    """
    batch_size = 64
    configs = []
    for model in MODELS:
        # Compute per-model steps from dataset size
        batches_per_epoch = max(1, model["train_records_approx"] // batch_size)
        model_n_steps = n_steps or (batches_per_epoch * 10)   # 10 epochs
        model_warmup = early_stop_warmup_steps or (batches_per_epoch * 6)  # 6 epochs

        configs.append({
            "base_model": model["base_model"],
            "model_name": model["name"],
            "renderer": model["renderer"],
            "thinking": model["thinking"],
            "load_checkpoint": "",  # Train from scratch (no warm-init)
            "batch_size": batch_size,
            "k_completions": 16,
            "generation_temperature": 0.7,
            "max_generation_tokens": 2048,
            "reward_type": "mse_format",
            "loss_fn": "importance_sampling",
            "lr": 2e-5,
            "lr_warmup_steps": 50,
            "lr_decay": "cosine",
            "lora_rank": 32,
            "filter_zero_std_groups": False,
            "n_steps": model_n_steps,
            "eval_interval_steps": 0,         # auto = once per epoch
            "checkpoint_interval_steps": 0,    # auto = once per epoch
            "early_stop_patience": 3,
            "early_stop_c_index": 0.54,
            "early_stop_warmup_steps": model_warmup,
            "gt_cache": model["gt_cache"],
            "dataset": model["training_data"],
            "output_dir": str(OUTPUT_DIR),
            "wandb_project": WANDB_PROJECT,
            "run_name": f"{model['name']}_scratch_mse",
        })
    return configs


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRPO single-task training — full dataset, multi-model scale-up"
    )
    parser.add_argument("--index", type=int, default=None,
                        help="Config index 0-3 (see --list)")
    parser.add_argument("--n_steps", type=int, default=None,
                        help="Override total training steps (default: 10 epochs)")
    parser.add_argument("--early_stop_warmup", type=int, default=None,
                        help="Override early stop warmup steps (default: 6 epochs)")
    parser.add_argument("--n_samples", type=int, default=0,
                        help="Limit dataset size (0=all, use 2 for smoke test)")
    parser.add_argument("--training-data", type=str, default=None,
                        help="Override the config's training_data JSONL path")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override OUTPUT_DIR (default: outputs/sweeps/grpo_single_task)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override per-step batch size (default 64; use small for smoke)")
    parser.add_argument("--k_completions", type=int, default=None,
                        help="Override completions per prompt (default 16; use small for smoke)")
    parser.add_argument("--list", action="store_true",
                        help="List all configs and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = get_configs(
        n_steps=args.n_steps,
        early_stop_warmup_steps=args.early_stop_warmup,
    )

    if args.list:
        print(f"\nGRPO — {len(configs)} configs:\n")
        for i, cfg in enumerate(configs):
            print(
                f"  [{i}] {cfg['run_name']:30s}  "
                f"model={cfg['base_model']:45s}  "
                f"bs={cfg['batch_size']}  K={cfg['k_completions']}  "
                f"steps={cfg['n_steps']}  warmup={cfg['early_stop_warmup_steps']}"
            )
        print()
        return

    if args.index is None:
        print("Error: --index required (0-3). Use --list to see configs.")
        sys.exit(1)

    if args.index < 0 or args.index >= len(configs):
        print(f"Error: --index must be 0-{len(configs)-1}")
        sys.exit(1)

    cfg = configs[args.index]

    # CLI overrides
    if args.training_data:
        cfg["dataset"] = args.training_data
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    cfg["output_dir"] = str(output_dir)
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.k_completions is not None:
        cfg["k_completions"] = args.k_completions

    # Save all configs for reference
    configs_path = output_dir / "configs.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(configs_path, "w") as f:
        json.dump(configs, f, indent=2)

    logger.info(f"\033[36m[STEP]\033[0m Running config [{args.index}]: {cfg['run_name']}")
    logger.info(f"  model={cfg['base_model']}")
    logger.info(f"  BS={cfg['batch_size']}, K={cfg['k_completions']}, "
                f"reward={cfg['reward_type']}, lr={cfg['lr']}")
    logger.info(f"  n_steps={cfg['n_steps']}, warmup={cfg['early_stop_warmup_steps']}, "
                f"patience={cfg['early_stop_patience']}")
    logger.info(f"  dataset={cfg['dataset']}")

    from prompt_attribution.training.config import (
        DataConfig,
        GRPOConfig,
        ModelFormat,
        TrainingSchedule,
    )
    from prompt_attribution.training.train_grpo import train_grpo

    # Build model format: enable thinking for Qwen3 (chain-of-thought)
    if cfg["thinking"]:
        model_format = ModelFormat(
            thinking=True,
            enable_thinking=True,
            thinking_budget=1024,
        )
    else:
        model_format = ModelFormat(thinking=False, enable_thinking=False)

    grpo_config = GRPOConfig(
        base_model=cfg["base_model"],
        load_checkpoint=cfg["load_checkpoint"],
        lora_rank=cfg["lora_rank"],
        batch_size=cfg["batch_size"],
        k_completions=cfg["k_completions"],
        generation_temperature=cfg["generation_temperature"],
        max_generation_tokens=cfg["max_generation_tokens"],
        reward_type=cfg["reward_type"],
        loss_fn=cfg["loss_fn"],
        filter_zero_std_groups=cfg["filter_zero_std_groups"],
        n_steps=cfg["n_steps"],
        eval_interval_steps=cfg["eval_interval_steps"],
        checkpoint_interval_steps=cfg["checkpoint_interval_steps"],
        early_stop_patience=cfg["early_stop_patience"],
        early_stop_warmup_steps=cfg["early_stop_warmup_steps"],
        early_stop_c_index=cfg["early_stop_c_index"],
        model_format=model_format,
        schedule=TrainingSchedule(
            base_lr=cfg["lr"],
            lr_warmup_steps=cfg["lr_warmup_steps"],
            lr_decay=cfg["lr_decay"],
        ),
        data=DataConfig(
            training_data_path=Path(cfg["dataset"]),
            base_model_gt_cache_path=Path(cfg["gt_cache"]) if cfg["gt_cache"] else None,
        ),
        output_dir=Path(cfg["output_dir"]),
        wandb_project=cfg["wandb_project"],
        run_name=cfg["run_name"],
        n_samples=args.n_samples,
    )

    asyncio.run(train_grpo(grpo_config))


if __name__ == "__main__":
    main()
