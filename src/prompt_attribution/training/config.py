"""
Module: prompt_attribution/training/config.py

Configuration dataclasses for GRPO training.

Structure:
- ModelFormat: Model-specific output format (thinking vs non-thinking)
- TrainingSchedule: LR schedule and optimizer hyperparams
- DataConfig: Data loading and splitting options
- CompoundRewardConfig: Per-component weights and matchers for compound reward
- COMPOUND_PRESETS: Named preset configurations for compound reward
- MultitaskDataConfig: Multi-task data loading config
- GRPOConfig: Top-level GRPO config
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass
class ModelFormat:
    """Controls completion format and output parsing per model family.

    Thinking models (Qwen3, DeepSeek-R1) use <think>...</think> tags.
    Non-thinking models (Llama, Mistral) put reasoning in the JSON field.

    Set enable_thinking=False on a thinking-capable model (e.g., Qwen3) to
    force non-thinking mode for both training and inference, avoiding
    distribution shift between train and eval.
    """

    thinking: bool = False
    enable_thinking: bool | None = None  # None = auto (True for thinking models)
    think_open: str = "<think>"
    think_close: str = "</think>"
    thinking_budget: int = 0  # 0 = no thinking; >0 = cap thinking tokens
    reasoning_parser: str = ""  # vLLM --reasoning-parser flag (e.g., "qwen3", "kimi_k2")

    def get_thinking_extra_body(self) -> dict:
        """Build extra_body dict for vLLM thinking budget.

        Returns empty dict for non-thinking models (no extra_body needed).
        For thinking-capable models, explicitly sets budget (on or off).
        Used by GT cache, evaluator, and refresh to ensure consistent
        thinking budget across all inference paths.

        NOTE: This is for vLLM only (requires vLLM >= 0.19.0 for enforcement).
        For cloud API providers, use get_api_thinking_kwargs(provider) instead.
        """
        if not self.thinking:
            return {}
        if self.enable_thinking is False:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        if self.thinking_budget == 0:
            # Thinking via system prompt, not extra_body (e.g., GPT-OSS)
            return {}
        # thinking_token_budget: vLLM >= 0.19 hard-caps thinking tokens
        return {"thinking_token_budget": self.thinking_budget}

    def get_api_thinking_kwargs(self, provider: str, model_name: str = "") -> dict:
        """Build provider-specific kwargs for cloud API thinking.

        Each provider has different thinking/reasoning API:
        - anthropic: thinking={"type":"enabled","budget_tokens":N}, temperature=1
        - together: reasoning={"enabled":True} + optional thinking_budget
        - openai: reasoning_effort="medium" for reasoning models (o-series, gpt-5+)
        - google/gemini: thinking_budget=N (handled by our safetytooling patch)

        Returns empty dict for non-thinking models.
        """
        if not self.thinking or self.enable_thinking is False:
            return {}

        if provider == "anthropic":
            budget = max(self.thinking_budget, 1024)  # Anthropic min is 1024
            return {
                "thinking": {"type": "enabled", "budget_tokens": budget},
                "temperature": 1,  # Required when thinking is enabled
            }
        elif provider == "together":
            # Together supports both reasoning toggle and thinking_budget.
            # reasoning={"enabled": True} → populates message.reasoning field.
            # thinking_budget → caps thinking tokens (shares max_tokens).
            result: dict = {"reasoning": {"enabled": True}}
            if self.thinking_budget > 0:
                result["extra_body"] = {"thinking_budget": self.thinking_budget}
            return result
        elif provider == "openai":
            # OpenAI reasoning models (o-series, gpt-5+) support reasoning_effort.
            # Non-reasoning models (gpt-4o) don't — skip for them.
            # Reasoning models also require temperature=1.
            name_lower = model_name.lower()
            is_reasoning = any(k in name_lower for k in ("o1-", "o3", "o4", "gpt-5"))
            if is_reasoning:
                return {"reasoning_effort": "high", "temperature": 1}
            return {}  # Non-reasoning OpenAI model (gpt-4o etc)
        elif provider in ("google", "gemini"):
            # Gemini: thinking_budget kwarg (handled by our safetytooling patch)
            if self.thinking_budget > 0:
                return {"thinking_budget": self.thinking_budget}
            return {}
        else:
            return {}

    @classmethod
    def from_model_name(cls, model_name: str, max_tokens: int = 2048) -> "ModelFormat":
        """Auto-detect format from model name.

        Thinking budget is half of max_tokens — thinking and content share
        the same vLLM token budget, so reserving half for each prevents
        thinking from starving content.
        GPT-OSS uses system prompt instead of extra_body (handled separately).
        """
        name_lower = model_name.lower()
        thinking_budget = max_tokens // 2
        if "qwen3" in name_lower:
            return cls(
                thinking=True, enable_thinking=True,
                thinking_budget=thinking_budget, reasoning_parser="qwen3",
            )
        if "deepseek" in name_lower:
            return cls(
                thinking=True, enable_thinking=True,
                thinking_budget=thinking_budget, reasoning_parser="deepseek_r1",
            )
        if "kimi" in name_lower:
            return cls(
                thinking=True, enable_thinking=True,
                thinking_budget=thinking_budget, reasoning_parser="kimi_k2",
            )
        # GPT-OSS: thinking via "Reasoning: high" system prompt, not extra_body.
        # thinking_budget=0 → get_thinking_extra_body() returns {} (no extra_body sent).
        # Reasoning tokens share the max_tokens budget with content, so no separate cap needed.
        if "gpt-oss" in name_lower:
            return cls(
                thinking=True, enable_thinking=True,
                thinking_budget=0, reasoning_parser="openai_gptoss",
            )
        return cls(thinking=False, enable_thinking=False)


@dataclass
class TrainingSchedule:
    """Learning rate schedule and optimizer hyperparameters."""

    base_lr: float = 2e-5
    lr_warmup_steps: int = 50
    lr_decay: Literal["cosine", "linear", "constant"] = "cosine"
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.01
    max_grad_norm: float = 0.0  # 0.0 = no clipping (matches Tinker cookbook)


def compute_lr(step: int, schedule: "TrainingSchedule", total_steps: int) -> float:
    """Compute learning rate for a given step using the configured schedule.

    Args:
        step: Current global step (0-indexed).
        schedule: TrainingSchedule config.
        total_steps: Total number of training steps.

    Returns:
        Learning rate for this step.
    """
    base_lr = schedule.base_lr
    warmup_steps = schedule.lr_warmup_steps

    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    if schedule.lr_decay == "constant":
        return base_lr

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)

    if schedule.lr_decay == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    elif schedule.lr_decay == "linear":
        return base_lr * (1.0 - progress)
    else:
        return base_lr


@dataclass
class DataConfig:
    """Data loading and splitting configuration."""

    training_data_path: Path = field(default_factory=lambda: Path("training_data.jsonl"))
    eval_data_path: Path | None = None  # Separate eval data file. If set, test_records come from this file (with its own test_split). Useful when training_data_path is a subset (e.g., contrastive pairs) but eval should be on the full corpus.
    test_split: float = 0.15
    seed: int = 42
    balance_by_category: bool = False
    completion_format: Literal["oracle", "template", "simulation", "label_only", "oracle_with_simulation", "answer_prediction"] = "oracle"
    balance_by_flip_label: bool = False  # Weight loss by inverse flip-label frequency
    class_weight_in_loss: bool = False  # Deprecated: kept for config compat, ignored (use balance_by_flip_label)
    base_model_gt_cache_path: Path | None = None  # Cached base model GT for balancing


@dataclass
class CompoundRewardConfig:
    """Configuration for compound reward computation.

    Controls per-component weights and matching strategies for the compound
    reward type in GRPO training. Each component computes a score in [0, 1],
    and the final reward is the weighted sum.
    """

    # Component weights (should sum to ~1.0)
    weights: dict[str, float] = field(default_factory=lambda: {
        "answer_baseline": 0.15,
        "answer_lever": 0.15,
        "flip_direction": 0.15,
        "flip_probability": 0.25,
        "simulation": 0.15,
        "mechanism": 0.15,
    })

    # Matching strategies for reasoning trace analysis
    simulation_matcher: Literal["regex", "ngram", "hybrid"] = "hybrid"
    mechanism_matcher: Literal["keyword", "ngram", "hybrid"] = "hybrid"

    # Thresholds
    mechanism_keyword_threshold: float = 0.5  # Fraction of keywords needed for match
    ngram_ns: tuple[int, ...] = (2, 3)  # N-gram sizes for overlap scoring

    # Numeric answer matching tolerance
    numeric_atol: float = 1e-6
    numeric_rtol: float = 1e-3


COMPOUND_PRESETS: dict[str, CompoundRewardConfig] = {
    # Balanced across all signals
    "balanced": CompoundRewardConfig(),

    # Heavy on answer prediction (tests simulation ability)
    "answer_heavy": CompoundRewardConfig(weights={
        "answer_baseline": 0.25,
        "answer_lever": 0.25,
        "flip_direction": 0.20,
        "flip_probability": 0.10,
        "simulation": 0.10,
        "mechanism": 0.10,
    }),

    # Heavy on reasoning quality
    "reasoning_heavy": CompoundRewardConfig(weights={
        "answer_baseline": 0.10,
        "answer_lever": 0.10,
        "flip_direction": 0.10,
        "flip_probability": 0.15,
        "simulation": 0.25,
        "mechanism": 0.30,
    }),

    # Answer prediction only (no reasoning trace rewards)
    "answers_only": CompoundRewardConfig(weights={
        "answer_baseline": 0.25,
        "answer_lever": 0.25,
        "flip_direction": 0.25,
        "flip_probability": 0.25,
        "simulation": 0.0,
        "mechanism": 0.0,
    }),

    # Probability-focused with answer bridge
    "probability_plus": CompoundRewardConfig(weights={
        "answer_baseline": 0.15,
        "answer_lever": 0.15,
        "flip_direction": 0.20,
        "flip_probability": 0.50,
        "simulation": 0.0,
        "mechanism": 0.0,
    }),
}



@dataclass
class MultitaskDataConfig:
    """Configuration for multi-task training data loading.

    Points to the output of generate_multitask_data.py which contains
    pre-split train/val JSONL files with MultitaskRecord schema.
    """

    multitask_data_dir: Path = field(
        default_factory=lambda: Path("outputs/training/multitask_balanced")
    )
    tasks: str = "all"  # "all", "e3", "e1,e3,e6", etc.


@dataclass
class GRPOConfig:
    """Configuration for GRPO (Group Relative Policy Optimization) training.

    Uses Tinker SamplingClient + clipped surrogate loss (PPO-style).
    Supports warm-starting from a previous checkpoint via load_checkpoint.
    Optional KL penalty against a reference model (warm init or base).
    """

    # Model
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    seed: int = 42
    load_checkpoint: str = ""  # Tinker path or run name to warm-start from (fresh optimizer)
    resume_from: str = ""  # Tinker state path to resume interrupted training (preserves optimizer)

    # Training
    batch_size: int = 4  # Prompts per step (each generates k_completions)
    max_seq_length: int = 4096

    # GRPO-specific
    k_completions: int = 8  # Completions generated per prompt
    generation_temperature: float = 0.7
    max_generation_tokens: int = 1024
    reward_type: Literal[
        "mse", "mse_format", "binary", "binary_format",
        "bce", "neg_bce", "compound", "sim_aware", "sim_only",
    ] = "binary"
    format_penalty: bool = False  # If True, parse failures get format_penalty_reward instead of default pred=0.5
    format_penalty_reward: float = 0.0  # Reward assigned to parse failures when format_penalty=True
    advantage_clip: float = 5.0
    advantage_eps: float = 1e-4  # Epsilon for group std normalization (TRL uses 1e-4)
    advantage_normalization: Literal["group", "none"] = "group"  # "group"=standard GRPO, "none"=Dr. GRPO
    filter_zero_std_groups: bool = False  # DAPO-style: skip groups where all K completions got same reward
    ngrpo_virtual_reward: bool = False  # NGRPO: inject virtual max-reward sample for mean/std computation

    # Loss function (clipped surrogate)
    loss_fn: Literal["importance_sampling", "ppo", "cispo"] = "ppo"  # ppo = true GRPO clipped surrogate
    clip_epsilon: float = 0.2  # [1-eps, 1+eps] clip range (DeepSeek-R1 used 0.28)

    # KL penalty against reference model
    kl_penalty_coef: float = 0.0  # 0=disabled, DeepSeek-R1 used 0.001
    kl_reference: Literal["warm_init", "base_model"] = "warm_init"

    # Compound reward config (used when reward_type="compound")
    compound_preset: str = "balanced"  # Key into COMPOUND_PRESETS
    compound_reward_weights: dict[str, float] | None = None  # Override preset weights
    simulation_matcher: Literal["regex", "ngram", "hybrid"] = "hybrid"
    mechanism_matcher: Literal["keyword", "ngram", "hybrid"] = "hybrid"

    # Prompt style: "probability" = current (flip_probability only),
    # "compound" = predict answers + probability (auto-set when reward_type="compound")
    prompt_style: Literal["probability", "compound"] = "probability"

    # Sub-configs
    schedule: TrainingSchedule = field(
        default_factory=lambda: TrainingSchedule(base_lr=1e-5)
    )
    data: DataConfig = field(default_factory=DataConfig)
    model_format: ModelFormat = field(default_factory=ModelFormat)

    # Logging and checkpointing
    n_steps: int = 7500  # Total training steps (~3 epochs with batch=8)
    eval_interval_steps: int = 500  # Tinker eval (tiny, every 500 steps)
    checkpoint_interval_steps: int = 1000  # Download + optional vLLM eval
    log_interval_steps: int = 50  # Console logging (reward logged every step to wandb)

    # Output
    output_dir: Path = field(default_factory=lambda: Path("outputs/training"))
    wandb_project: str = "prompt-attribution-grpo"
    run_name: str = ""

    # Evaluation (eval_max_tokens is derived from model_format in __post_init__)
    eval_max_tokens: int = 0  # 0 = auto from model_format
    eval_temperature: float = 0.0
    eval_n_samples: int = 50  # Tinker eval sample size (fallback when no vLLM)
    eval_vllm_url: str = ""  # Optional: vLLM URL for fast periodic eval (full test set)
    eval_concurrency: int = 200  # Concurrent requests for vLLM eval

    # Auto-eval: sbatch a full vLLM eval job at each checkpoint save
    auto_eval_on_checkpoint: bool = True
    eval_slurm_partition: str = "general"
    eval_slurm_qos: str = "high"
    eval_slurm_gpu_mem: str = "48G"
    eval_slurm_exclude: str = ""  # Comma-separated nodes to exclude (e.g., "HOST1,HOST2")
    eval_job_timeout_minutes: int = 60

    # Sample logging (debugging observability)
    sample_log_interval_steps: int = 10  # Print GRPO samples every N steps (0 = disabled)
    sample_log_count: int = 2  # Number of prompts to show per interval

    # Early stopping on C-index collapse
    early_stop_c_index: float = 0.54  # Kill if C-index below this after warmup
    early_stop_pred_var: float = 0.001  # Kill if prediction variance below this
    early_stop_warmup_steps: int = 1000  # Don't check early stopping before this step
    early_stop_patience: int = 3  # Consecutive evals below threshold before stopping

    # Multi-task mode (None = E3-only backward compat, set to enable multi-task)
    multitask: MultitaskDataConfig | None = None

    # Optional eval toggles (default: all disabled for fast training)
    disable_mini_eval: bool = True  # Skip per-step mini evaluations during training
    disable_early_stopping: bool = True  # Skip early stopping checks (full eval from backfill)
    disable_final_eval: bool = True  # Skip final evaluation after training

    # Debug
    n_samples: int = 0
    dry_run: bool = False

    def __post_init__(self) -> None:
        """Append timestamp to run_name and auto-detect model format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.run_name:
            self.run_name = f"{self.run_name}_{timestamp}"
        else:
            self.run_name = f"grpo_{timestamp}"

        if not self.model_format.thinking:
            self.model_format = ModelFormat.from_model_name(self.base_model)

        # If user explicitly set enable_thinking=False on a thinking model,
        # override to non-thinking format
        if self.model_format.enable_thinking is False and self.model_format.thinking:
            self.model_format.thinking = False

        # eval_max_tokens: only used by Tinker mini-eval (50 samples, speed over accuracy).
        # vLLM full eval omits max_tokens entirely (server decides via max_model_len).
        if self.eval_max_tokens == 0:
            self.eval_max_tokens = 256

        # Auto-set prompt_style when reward_type is compound
        if self.reward_type == "compound" and self.prompt_style == "probability":
            self.prompt_style = "compound"

    def get_loss_fn_config(self) -> dict | None:
        """Build loss_fn_config dict for Tinker forward_backward().

        Returns clip thresholds for ppo/cispo, None for importance_sampling.
        PPO/CISPO use [1-eps, 1+eps] clip range on the probability ratio.
        """
        if self.loss_fn == "importance_sampling":
            return None
        return {
            "clip_low_threshold": 1.0 - self.clip_epsilon,
            "clip_high_threshold": 1.0 + self.clip_epsilon,
        }

    def get_compound_reward_config(self) -> CompoundRewardConfig:
        """Build CompoundRewardConfig from preset + overrides."""
        preset_name = self.compound_preset
        if preset_name not in COMPOUND_PRESETS:
            raise ValueError(
                f"Unknown compound_preset: {preset_name}. "
                f"Available: {list(COMPOUND_PRESETS.keys())}"
            )
        config = CompoundRewardConfig(
            weights=dict(COMPOUND_PRESETS[preset_name].weights),
            simulation_matcher=self.simulation_matcher,
            mechanism_matcher=self.mechanism_matcher,
            ngram_ns=COMPOUND_PRESETS[preset_name].ngram_ns,
            mechanism_keyword_threshold=COMPOUND_PRESETS[preset_name].mechanism_keyword_threshold,
            numeric_atol=COMPOUND_PRESETS[preset_name].numeric_atol,
            numeric_rtol=COMPOUND_PRESETS[preset_name].numeric_rtol,
        )
        # Override weights if specified
        if self.compound_reward_weights:
            config.weights.update(self.compound_reward_weights)
        return config
