"""
Module: prompt_attribution/training/train_grpo.py

Thin GRPO outer loop using Tinker cookbook building blocks.
Mirrors cookbook's do_sync_training() but inserts our hooks:
LR scheduling, early stopping, GT refresh, checkpoint post-processing.

The bug-prone gradient path (datum building, advantage computation, training step)
is ENTIRELY cookbook code — we never rewrite it.

Usage:
    asyncio.run(train_grpo(config))
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.rl.data_processing import remove_constant_reward_groups
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.rl.train import (
    gather_with_progress,
    prepare_minibatch,
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.rollouts import do_group_rollout_and_filter_constant_reward
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed

from prompt_attribution.training.config import GRPOConfig, compute_lr
from prompt_attribution.training.data.dataset import TrainingDataset
from prompt_attribution.training.data.prompt_builder import TrainingPromptBuilder
from prompt_attribution.training.rl_datasets.attribution_rl_dataset import build_rl_datasets
from prompt_attribution.training.validation.attribution_evaluator import (
    AttributionEvaluator,
)
from prompt_attribution.training.hooks.early_stopping import (
    EarlyStopState,
    should_stop,
)
from prompt_attribution.training.hooks.gt_validation import (
    validate_gt_config,
    validate_loaded_gt,
)

logger = logging.getLogger(__name__)


def _get_renderer_name(base_model: str) -> str:
    """Map base model name to cookbook renderer name."""
    model_lower = base_model.lower()
    if "gpt-oss" in model_lower or "gpt_oss" in model_lower:
        return "gpt_oss"
    elif "llama" in model_lower:
        return "llama3"
    elif "qwen3" in model_lower:
        return "qwen3"
    elif "qwen" in model_lower:
        return "llama3"  # Qwen2.5 uses similar chat template
    elif "mistral" in model_lower:
        return "llama3"
    else:
        return "llama3"  # Safe default


def _find_resumable_dir(config: GRPOConfig) -> Path | None:
    """Find an existing run directory with checkpoints to resume from.

    Searches output_dir for directories matching the run_name prefix
    (without timestamp) that contain checkpoints.jsonl.
    Returns the most recent match, or None if no resumable run found.
    """
    output_dir = Path(config.output_dir)
    if not config.run_name:
        return None

    # Strip timestamp from run_name if present (format: name_YYYYMMDD_HHMMSS)
    run_name = config.run_name
    parts = run_name.rsplit("_", 2)
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        run_prefix = parts[0]
    else:
        run_prefix = run_name

    candidates = sorted(output_dir.glob(f"{run_prefix}_*"), reverse=True)

    # First pass: prefer directories with checkpoints (true resume)
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        ckpt_file = candidate / "checkpoints.jsonl"
        if ckpt_file.exists() and ckpt_file.stat().st_size > 0:
            logger.info(
                f"\033[36m[RESUME]\033[0m Found resumable run with checkpoints: {candidate}"
            )
            return candidate

    # Second pass: reuse directory with wandb run (same wandb ID, fresh training)
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        wandb_dirs = list(candidate.glob("wandb/run-*"))
        if wandb_dirs:
            logger.info(
                f"\033[36m[RESUME]\033[0m Found existing run (no checkpoint, "
                f"will restart with same wandb): {candidate}"
            )
            return candidate

    return None


def _setup_output_dir(config: GRPOConfig) -> Path:
    """Create or reuse output directory.

    If a previous run with the same name prefix has checkpoints,
    reuses that directory for seamless resume.
    """
    # Try to find existing run to resume
    existing = _find_resumable_dir(config)
    if existing:
        logger.info(f"Reusing existing run directory for resume: {existing}")
        (existing / "eval_transcripts").mkdir(exist_ok=True)
        return existing

    # Create new directory
    output_dir = Path(config.output_dir)
    if config.run_name:
        output_dir = output_dir / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_transcripts").mkdir(exist_ok=True)

    config_path = output_dir / "config.json"
    config_dict = asdict(config)
    for k, v in config_dict.items():
        if isinstance(v, Path):
            config_dict[k] = str(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, Path):
                    v[kk] = str(vv)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info(f"Config saved to {config_path}")
    return output_dir


def _patch_dict_mean_for_strings() -> None:
    """Patch tinker_cookbook's dict_mean to skip non-numeric values.

    This allows putting string metrics (like gt_label='Yes') in
    transition.metrics without crashing the aggregation.

    Must patch both the module attribute AND the already-imported reference
    in metric_util, since Python's `from X import Y` binds Y locally.
    """
    import numpy as np
    from tinker_cookbook.rl import metric_util
    from tinker_cookbook.utils import misc_utils

    def _safe_dict_mean(list_of_dicts: list) -> dict:
        key2values: dict = {}
        for d in list_of_dicts:
            for k, v in d.items():
                try:
                    key2values.setdefault(k, []).append(float(v))
                except (ValueError, TypeError):
                    pass  # Skip non-numeric (strings, None, etc.)
        return {k: float(np.mean(values)) for k, values in key2values.items()}

    misc_utils.dict_mean = _safe_dict_mean
    metric_util.dict_mean = _safe_dict_mean


def _install_trajectory_colorizer() -> None:
    """Install a logging filter that colorizes Tinker trajectory logs.

    Colors: reward line (green/yellow/red by value), question (cyan),
    response (dim), metrics (magenta), boundaries (yellow).
    """
    import re as _re

    _C = "\033[36m"   # cyan
    _G = "\033[32m"   # green
    _Y = "\033[33m"   # yellow
    _R = "\033[31m"   # red
    _M = "\033[35m"   # magenta
    _D = "\033[2m"    # dim
    _RST = "\033[0m"

    class _TrajectoryColorizer(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if "Trajectory Group" not in msg and "trajectory idx=" not in msg:
                return True
            # Entire trajectory group is one big message — colorize inline
            lines = msg.split("\n")
            colored = []
            for line in lines:
                if "====== Trajectory Group" in line or "====== End Trajectory" in line:
                    colored.append(f"{_Y}{line}{_RST}")
                elif "trajectory idx=" in line:
                    # Color reward by value
                    m = _re.search(r'reward=([\d.-]+)', line)
                    if m:
                        r = float(m.group(1))
                        rc = _G if r >= 1.0 else (_Y if r >= 0.5 else _R)
                        line = line.replace(f"reward={m.group(1)}", f"reward={rc}{m.group(1)}{_RST}")
                    colored.append(f"{_Y}{line}{_RST}")
                elif "---- datum ----" in line:
                    colored.append(f"{_D}{line}{_RST}")
                elif line.strip().startswith(("format:", "reward:", "correct:", "mse:", "gt:", "predicted_value:", "true_value:", "flip_acc:", "edit_dist:")):
                    colored.append(f"{_M}{line}{_RST}")
                elif "Per-step metrics:" in line or "Trajectory metrics:" in line:
                    colored.append(f"{_M}{line}{_RST}")
                elif "<|start_header_id|>user<|end_header_id|>" in line:
                    colored.append(f"{_C}{line}{_RST}")
                elif "<|start_header_id|>assistant<|end_header_id|>" in line:
                    colored.append(f"{_D}{line}{_RST}")
                else:
                    colored.append(line)
            record.msg = "\n".join(colored)
            record.args = None
            return True

    logging.getLogger("tinker_cookbook.rl.train").addFilter(_TrajectoryColorizer())


def _maybe_resume_wandb(run_dir: Path, wandb_project: str | None) -> None:
    """Pre-initialize wandb with resume if an existing run is found.

    Reads the wandb run ID from the run directory's wandb/ subdirectory.
    If found, calls wandb.init(resume="must") so the cookbook's setup_logging
    attaches to the existing run instead of creating a new one.
    """
    if not wandb_project:
        return

    wandb_dirs = sorted(run_dir.glob("wandb/run-*"))
    if not wandb_dirs:
        return

    # Extract run ID from the FIRST (original) directory, not the latest
    # Failed resume attempts create new dirs — we want the original run
    wandb_dir_name = wandb_dirs[0].name
    parts = wandb_dir_name.split("-")
    if len(parts) < 3:
        return
    run_id = parts[-1]

    try:
        import wandb
        logger.info(f"\033[36m[RESUME]\033[0m Resuming wandb run: {run_id}")
        wandb.init(
            project=wandb_project,
            id=run_id,
            resume="must",
            dir=str(run_dir),
            allow_val_change=True,
        )
    except Exception as e:
        logger.warning(f"Failed to resume wandb run {run_id}: {e}. Will create new run.")


async def train_grpo(config: GRPOConfig) -> None:
    """Run GRPO training using cookbook building blocks + our hooks.

    This is the thin outer loop that replaces GRPOTrainer.train().
    The gradient path (trajectory_to_data, compute_advantages, train_step)
    is entirely Tinker cookbook code — guaranteed correct alignment.
    """
    # === Setup ===
    run_output_dir = _setup_output_dir(config)
    log_path = str(run_output_dir)

    # If resuming, override run_name to match old directory (prevents wandb config conflict)
    if run_output_dir.name != config.run_name:
        logger.info(f"Overriding run_name: {config.run_name} → {run_output_dir.name}")
        object.__setattr__(config, "run_name", run_output_dir.name)

    # Validate GT
    if config.data.base_model_gt_cache_path:
        validate_gt_config(
            config.data.base_model_gt_cache_path,
            config.base_model,
            config.data.training_data_path,
        )

    # Initialize logging (resume wandb run if reusing existing directory)
    _maybe_resume_wandb(run_output_dir, config.wandb_project)
    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.run_name,
        config=config,
    )

    # Colorize trajectory logs for readability in tmux
    _patch_dict_mean_for_strings()
    _install_trajectory_colorizer()

    # Create Tinker training client
    service_client = tinker.ServiceClient()
    renderer_name = _get_renderer_name(config.base_model)
    resume_info = checkpoint_utils.get_last_checkpoint(log_path)

    if resume_info:
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info.state_path,
            )
        )
        start_batch = resume_info.batch or 0
        logger.info(f"Resumed training from {resume_info.state_path}")
    elif hasattr(config, "load_checkpoint") and config.load_checkpoint:
        training_client = await service_client.create_training_client_from_state_async(
            config.load_checkpoint,
        )
        start_batch = 0
        logger.info(f"Loaded weights from {config.load_checkpoint}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.base_model,
            rank=config.lora_rank,
            seed=config.seed,
        )
        start_batch = 0

    tokenizer = training_client.get_tokenizer()
    if renderer_name == "gpt_oss":
        from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
        renderer = GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="high")
    else:
        renderer = renderers.get_renderer(renderer_name, tokenizer)

    # ===================================================================
    # Branch: multi-task vs E3-only data loading
    # ===================================================================
    if config.multitask:
        # --- Multi-task path ---
        from prompt_attribution.training.rl_datasets.multitask_dataset import MultitaskDataset
        from prompt_attribution.training.rl_datasets.multitask_rl_dataset import build_multitask_rl_datasets
        from prompt_attribution.training.envs.multitask_env import FlipJudgeHolder, SamplingClientHolder

        mt_dataset = MultitaskDataset(
            data_dir=config.multitask.multitask_data_dir,
            tasks=config.multitask.tasks,
        )
        mt_dataset.load()

        # Sampling client holder for E8 online reward
        sampling_holder = SamplingClientHolder()

        # LLM judge for E8 open-ended flip detection (uses Anthropic API)
        flip_judge = FlipJudgeHolder()
        logger.info("\033[36m[JUDGE]\033[0m E8 flip judge enabled (Haiku, 300 concurrent)")

        train_ds, test_ds = build_multitask_rl_datasets(
            dataset=mt_dataset,
            renderer=renderer,
            batch_size=config.batch_size,
            k_completions=config.k_completions,
            sampling_client_holder=sampling_holder,
            flip_judge=flip_judge,
            seed=config.seed,
        )

        # Build evaluators
        evaluators = []
        if test_ds is not None:
            evaluators.append(
                RLTestSetEvaluator(
                    test_ds,
                    max_tokens=config.max_generation_tokens,
                    name="multitask_val",
                )
            )

    else:
        # --- E3-only path (backward compat) ---
        sampling_holder = None  # Not used in E3 path

        dataset = TrainingDataset(config.data)
        dataset.load(n_samples=config.n_samples if hasattr(config, "n_samples") else 0)
        if config.data.base_model_gt_cache_path:
            validate_loaded_gt(
                dataset.train_records + dataset.test_records,
                config.base_model,
                config.data.base_model_gt_cache_path,
            )

        prompt_builder = TrainingPromptBuilder(
            model_format=config.model_format,
            completion_format=getattr(config.data, "completion_format", "oracle"),
            prompt_style=getattr(config, "prompt_style", "probability"),
        )

        train_ds, test_ds = build_rl_datasets(
            dataset=dataset,
            prompt_builder=prompt_builder,
            renderer=renderer,
            batch_size=config.batch_size,
            k_completions=config.k_completions,
            reward_type=config.reward_type,
            format_penalty_reward=getattr(config, "format_penalty_reward", -1.0),
            seed=config.seed,
        )

        # Build evaluators
        evaluators = []

        attr_evaluator = AttributionEvaluator(
            test_records=dataset.test_records,
            prompt_builder=prompt_builder,
            tokenizer=tokenizer,
            max_tokens=max(config.max_generation_tokens, 2048),
            n_samples=getattr(config, "eval_n_samples", 50),
            transcript_dir=str(run_output_dir / "eval_transcripts"),
        )
        evaluators.append(attr_evaluator)

        if test_ds is not None:
            evaluators.append(
                RLTestSetEvaluator(
                    test_ds,
                    max_tokens=config.max_generation_tokens,
                    name="test",
                )
            )

    # KL reference client
    kl_reference_client = None
    if config.kl_penalty_coef > 0:
        kl_reference_client = service_client.create_sampling_client(
            base_model=config.base_model,
        )

    # Loss function config
    loss_fn_config = None
    if config.loss_fn == "ppo":
        loss_fn_config = {
            "clip_low_threshold": 1.0 - config.clip_epsilon,
            "clip_high_threshold": 1.0 + config.clip_epsilon,
        }

    max_tokens = config.max_generation_tokens
    temperature = config.generation_temperature
    # Resolve 0 = once per epoch (using actual dataset size after filtering)
    batches_per_epoch = len(train_ds)
    save_every = config.checkpoint_interval_steps if config.checkpoint_interval_steps > 0 else batches_per_epoch
    eval_every = config.eval_interval_steps if config.eval_interval_steps > 0 else batches_per_epoch
    do_remove_constant = getattr(config, "filter_zero_std_groups", False)

    # Early stopping state (restore from disk if resuming)
    if resume_info:
        early_stop_state = EarlyStopState.load(run_output_dir)
    else:
        early_stop_state = EarlyStopState()

    # === Training Loop ===
    n_batches = len(train_ds)
    end_step = config.n_steps if config.n_steps else n_batches
    i_batch = start_batch - 1  # Sentinel: updated each iteration; guards against empty range

    logger.info(
        f"\033[36m[STEP]\033[0m GRPO training: {end_step} steps "
        f"({n_batches} batches/epoch, ~{end_step // max(1, n_batches)} epochs), "
        f"batch_size={config.batch_size}, K={config.k_completions}, "
        f"LR={config.schedule.base_lr}, loss={config.loss_fn}, "
        f"reward={config.reward_type}, "
        f"eval_every={eval_every}, ckpt_every={save_every}"
    )

    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, log_path,
        save_every, start_batch,
    )
    # Update E8 sampling holder (multi-task only)
    if sampling_holder is not None:
        sampling_holder.client = sampling_client

    for i_batch in range(start_batch, end_step):
        t_start = time.time()
        metrics: dict[str, Any] = {
            "progress/batch": i_batch,
            "progress/done_frac": (i_batch + 1) / end_step,
        }

        # Compute LR (our scheduling, not cookbook's flat LR)
        lr = compute_lr(i_batch, config.schedule, end_step)
        metrics["optim/lr"] = lr

        # Run evaluations (optional)
        if not config.disable_mini_eval and eval_every > 0 and i_batch % eval_every == 0:
            eval_sc = await training_client.save_weights_and_get_sampling_client_async(
                f"eval_{i_batch}"
            )
            for ev in evaluators:
                ev_metrics = await ev(eval_sc)
                metrics.update(ev_metrics)

        # Get batch and sample trajectories (cycle through dataset)
        batch_idx = i_batch % n_batches
        env_group_builders_P = train_ds.get_batch(batch_idx)

        # Retry sampling on transient Tinker errors (400s from concurrency limits)
        max_retries = 10
        for attempt in range(max_retries):
            try:
                trajectory_groups_P = await gather_with_progress(
                    (
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            do_remove_constant_reward_groups=False,
                            enable_logging=i < 4,
                        )
                        for i, builder in enumerate(env_group_builders_P)
                    ),
                    desc=f"Sampling batch {i_batch}",
                )
                break
            except (ValueError, Exception) as e:
                if attempt < max_retries - 1 and "400" in str(e):
                    logger.warning(
                        f"\033[33m[WARNING]\033[0m Sampling batch {i_batch} failed "
                        f"(attempt {attempt + 1}/{max_retries}): {e}. Retrying in 30s..."
                    )
                    import asyncio
                    await asyncio.sleep(30)
                else:
                    logger.error(
                        f"\033[31m[ERROR]\033[0m Sampling batch {i_batch} failed "
                        f"after {max_retries} attempts: {e}. Skipping batch."
                    )
                    continue  # Skip to next i_batch
        else:
            continue  # All retries failed, skip batch

        if do_remove_constant:
            n_before = len(trajectory_groups_P)
            trajectory_groups_P = remove_constant_reward_groups(trajectory_groups_P)
            n_after = len(trajectory_groups_P)
            metrics["train/groups_before_filter"] = n_before
            metrics["train/groups_after_filter"] = n_after
            metrics["train/filter_kept_frac"] = n_after / max(1, n_before)

        # Prepare minibatch (cookbook: compute_advantages + assemble_training_data + KL)
        data_D, prep_metrics = await prepare_minibatch(
            env_group_builders_P,
            trajectory_groups_P,
            tokenizer,
            kl_reference_client,
            kl_penalty_coef=config.kl_penalty_coef,
            kl_discount_factor=0.0,
        )
        metrics.update(prep_metrics)

        # Train step (cookbook: pipelined forward_backward + optim_step)
        with timed("train", metrics):
            training_logprobs = await train_step(
                data_D=data_D,
                training_client=training_client,
                learning_rate=lr,
                num_substeps=1,
                loss_fn=config.loss_fn,
                loss_fn_config=loss_fn_config,
                metrics=metrics,
            )

        # Save checkpoint and get new sampling client
        sampling_client, ckpt_metrics = await save_checkpoint_and_get_sampling_client(
            training_client, i_batch + 1, log_path, save_every,
        )
        if sampling_holder is not None:
            sampling_holder.client = sampling_client
        metrics.update(ckpt_metrics)

        metrics["time/total"] = time.time() - t_start

        # Log
        ml_logger.log_metrics(metrics, step=i_batch)

        if i_batch % max(1, getattr(config, "log_interval_steps", 10)) == 0:
            reward = metrics.get("env/all/reward/total/mean", "?")
            reward_str = f"{reward:.3f}" if isinstance(reward, float) else str(reward)
            logger.info(
                f"\033[32m[INFO]\033[0m Step {i_batch}/{end_step} | "
                f"reward={reward_str} | "
                f"lr={lr:.2e} | "
                f"{time.time()-t_start:.1f}s"
            )

        # Early stopping (optional)
        if not config.disable_early_stopping and should_stop(
            early_stop_state,
            i_batch,
            output_dir=config.output_dir,
            run_name=config.run_name,
            checkpoint_interval=save_every,
            warmup_steps=config.early_stop_warmup_steps,
            c_index_floor=config.early_stop_c_index,
            patience=config.early_stop_patience,
            ml_logger=ml_logger,
            run_dir=run_output_dir,
        ):
            logger.warning(
                f"\033[31m[EARLY STOP]\033[0m at step {i_batch} | "
                f"best checkpoint: step {early_stop_state.best_checkpoint_step} "
                f"(C-index={early_stop_state.best_c_index:.4f})"
            )
            break

    # === Final checkpoint ===
    final_step = i_batch + 1  # Actual stopping step (early stop or full run)
    logger.info("Saving final checkpoint...")
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=log_path,
        kind="both",
        loop_state={"batch": end_step},
        ttl_seconds=None,
    )

    # Final eval (optional)
    if not config.disable_final_eval and evaluators:
        final_sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            "final_eval"
        )
        if sampling_holder is not None:
            sampling_holder.client = final_sampling_client
        final_metrics = await evaluators[0](final_sampling_client)
        final_metrics_prefixed = {f"final/{k}": v for k, v in final_metrics.items()}
        ml_logger.log_metrics(final_metrics_prefixed, step=final_step)
        logger.info(
            f"\033[35m[EVAL]\033[0m Final: "
            + " | ".join(f"{k}={v:.4f}" for k, v in final_metrics.items() if isinstance(v, float))
        )

    ml_logger.close()
    logger.info("\033[32m[INFO]\033[0m GRPO training complete!")
