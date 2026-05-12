"""
Module: prompt_attribution/training/envs/attribution_env.py

ProblemEnv subclass for attribution flip probability prediction.
The env presents a Phase 2 attribution prompt and rewards the model
for accurately predicting the flip probability.

Structure:
- AttributionEnv: Single-turn env that parses flip_probability and computes reward
- AttributionEnvGroupBuilder: Creates K identical envs for GRPO group sampling
"""

import logging
import math
from dataclasses import dataclass
from typing import Callable

import tinker
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    Metrics,
    StepResult,
)
from tinker_cookbook import renderers
from tinker_cookbook.utils import logtree

from prompt_attribution.training.data.dataset import TrainingRecord
from prompt_attribution.training.data.prompt_builder import TrainingPromptBuilder
from prompt_attribution.training.validation.attribution_evaluator import (
    parse_flip_probability,
)

logger = logging.getLogger(__name__)


class AttributionEnv(ProblemEnv):
    """Single-turn attribution environment.

    Presents a Phase 2 attribution prompt asking the model to predict
    how likely a lever instruction is to change the model's answer.
    Rewards are based on how accurately the model predicts the flip probability.

    Overrides step() to use our custom reward computation instead of the
    default format_coef*(format-1) + answer.
    """

    def __init__(
        self,
        record: TrainingRecord,
        prompt_builder: TrainingPromptBuilder,
        renderer: renderers.Renderer,
        reward_type: str = "binary",
        format_penalty_reward: float = -1.0,
    ) -> None:
        super().__init__(renderer=renderer, format_coef=0.0)
        self._record = record
        self._prompt_builder = prompt_builder
        self._reward_type = reward_type
        self._format_penalty_reward = format_penalty_reward
        self._pair = prompt_builder.build(record)

    def get_question(self) -> str:
        """Return the Phase 2 attribution prompt."""
        return self._pair.prompt

    def check_format(self, sample_str: str) -> bool:
        """Check if model output contains a parseable flip_probability."""
        prob = parse_flip_probability(sample_str)
        # Default parse returns 0.5, which is ambiguous.
        # Check if we can find an explicit value.
        return prob != 0.5 or "flip_probability" in sample_str.lower()

    def check_answer(self, sample_str: str) -> bool:
        """Check if predicted flip direction matches ground truth."""
        prob = parse_flip_probability(sample_str)
        gt = self._record.empirical_flip_fraction or 0.0
        return (prob >= 0.5) == (gt >= 0.5)

    def get_reference_answer(self) -> str:
        """Return ground truth flip fraction for logging."""
        return f"flip_probability={self._record.empirical_flip_fraction}"

    async def step(self, action: Action) -> StepResult:
        """Parse model output and compute custom attribution reward.

        Reward design (all bounded to prevent collapse):

        For 'binary_format' and 'mse_format' reward types:
          - Parse fail: 0.0
          - Parse OK: 0.5 (format bonus) + base_reward
          - binary_format base: 1.0 if correct direction, 0.0 if wrong → total [0.5, 1.5]
          - mse_format base: 1.0 - (pred-gt)² → total [0.5, 1.5]

        For 'hard_binary_format':
          - Parse fail: 0.0
          - Parse OK: 0.5 + 1.0 if |pred-gt| < 1e-9 else 0.0 → total {0.5, 1.5}

        For 'neg_bce_format':
          - Parse fail: 0.0
          - Parse OK: 0.5 + negBCE(pred, gt), floor ≈ -16.1 → total [-15.6, 0.5]

        For legacy types (mse, binary, bce, neg_bce):
          Uses old reward computation with format_penalty_reward.
        """
        # Decode action tokens to text
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        # Parse flip probability
        predicted_prob = parse_flip_probability(content)
        gt_prob = self._record.empirical_flip_fraction or 0.0
        format_ok = parse_success and self.check_format(content)

        # Compute reward
        reward = self._compute_reward(predicted_prob, gt_prob, format_ok)

        # Metrics for logging/aggregation
        direction_correct = (predicted_prob >= 0.5) == (gt_prob >= 0.5)
        metrics: Metrics = {
            "format": float(format_ok),
            "correct": float(direction_correct) if format_ok else 0.0,
            "predicted_prob": predicted_prob,
            "true_prob": gt_prob,
            "reward": reward,
        }

        # Log for logtree inspection
        with logtree.scope_header("Attribution Reward"):
            logtree.table_from_dict(
                {
                    "predicted_prob": f"{predicted_prob:.3f}",
                    "true_prob": f"{gt_prob:.3f}",
                    "reward_type": self._reward_type,
                    "format_ok": format_ok,
                    "reward": f"{reward:.3f}",
                    "category": self._record.category,
                },
                caption="Attribution reward components",
            )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )

    def _compute_reward(
        self, predicted: float, ground_truth: float, format_ok: bool
    ) -> float:
        """Compute bounded reward for a single prediction.

        Format-aware reward types:
          - Parse fail → 0.0
          - Parse OK → 0.5 (format bonus) + base_reward

        Reward type ranges:
          binary_format:       {0.5, 1.5}
          mse_format:          [0.5, 1.5]
          hard_binary_format:  {0.5, 1.5}
          neg_bce_format:      [-15.6, 0.5]
        """
        FORMAT_BONUS = 0.5

        if self._reward_type == "binary_format":
            if not format_ok:
                return 0.0
            correct = float((predicted >= 0.5) == (ground_truth >= 0.5))
            return FORMAT_BONUS + correct

        elif self._reward_type == "mse_format":
            if not format_ok:
                return 0.0
            mse_reward = 1.0 - (predicted - ground_truth) ** 2
            return FORMAT_BONUS + mse_reward

        elif self._reward_type == "hard_binary_format":
            if not format_ok:
                return 0.0
            exact_match = float(abs(predicted - ground_truth) < 1e-9)
            return FORMAT_BONUS + exact_match

        elif self._reward_type == "neg_bce_format":
            eps = 1e-7
            if not format_ok:
                # Assign maximally wrong prediction: pred≈0 when gt=1, pred≈1 when gt=0
                wrong_pred = eps if ground_truth >= 0.5 else (1 - eps)
                neg_bce = ground_truth * math.log(wrong_pred) + (1 - ground_truth) * math.log(1 - wrong_pred)
                return neg_bce  # No format bonus; worst possible score ≈ −16.1
            p = max(eps, min(1 - eps, predicted))
            neg_bce = ground_truth * math.log(p) + (1 - ground_truth) * math.log(1 - p)
            return FORMAT_BONUS + neg_bce

        # Legacy reward types (unbounded, kept for backward compat)
        if not format_ok:
            return self._format_penalty_reward
        from prompt_attribution.training.trainers.grpo_reward import GRPOReward

        reward_fn = GRPOReward(reward_type=self._reward_type)
        rewards = reward_fn.compute_rewards([predicted], [ground_truth])
        return rewards[0].item()


@dataclass(frozen=True)
class AttributionEnvGroupBuilder(ProblemGroupBuilder):
    """Creates K identical AttributionEnv instances for one training record.

    Each env in the group gets the same prompt (same record), producing
    K diverse completions for GRPO advantage centering.
    """

    # Override parent fields
    env_thunk: Callable[[], ProblemEnv]
    num_envs: int
    dataset_name: str = "attribution"
    # Our additional metadata for per-category logging
    category: str = ""
    perturbation_type: str = ""

    def logging_tags(self) -> list[str]:
        """Return tags for per-category metric aggregation."""
        tags = [self.dataset_name]
        if self.category:
            tags.append(self.category)
        return tags


def make_env_group_builder(
    record: TrainingRecord,
    prompt_builder: TrainingPromptBuilder,
    renderer: renderers.Renderer,
    k_completions: int,
    reward_type: str = "binary",
    format_penalty_reward: float = -1.0,
) -> AttributionEnvGroupBuilder:
    """Factory to create an AttributionEnvGroupBuilder for a single record."""
    def env_thunk() -> AttributionEnv:
        return AttributionEnv(
            record=record,
            prompt_builder=prompt_builder,
            renderer=renderer,
            reward_type=reward_type,
            format_penalty_reward=format_penalty_reward,
        )

    return AttributionEnvGroupBuilder(
        env_thunk=env_thunk,
        num_envs=k_completions,
        dataset_name="attribution",
        category=record.category or "",
        perturbation_type=record.perturbation_type or "",
    )
