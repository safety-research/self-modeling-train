"""
Module: prompt_attribution/training/rl_datasets/attribution_rl_dataset.py

RLDataset wrapper for attribution GRPO training.
Wraps TrainingDataset to conform to the Tinker cookbook's RLDataset interface.

Structure:
- AttributionRLDataset: get_batch() returns list[EnvGroupBuilder] for GRPO sampling
- build_rl_datasets(): Factory that loads data and returns train/test datasets
"""

import logging
import random
from math import ceil
from typing import Sequence

from tinker_cookbook import renderers
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset

from prompt_attribution.training.data.dataset import TrainingDataset, TrainingRecord
from prompt_attribution.training.data.prompt_builder import TrainingPromptBuilder
from prompt_attribution.training.envs.attribution_env import make_env_group_builder

logger = logging.getLogger(__name__)


class AttributionRLDataset(RLDataset):
    """RLDataset adapter for attribution GRPO training.

    Each batch returns a list of EnvGroupBuilder, one per training record.
    Each builder creates K identical environments for group sampling.
    """

    def __init__(
        self,
        records: list[TrainingRecord],
        prompt_builder: TrainingPromptBuilder,
        renderer: renderers.Renderer,
        batch_size: int,
        k_completions: int,
        reward_type: str = "binary",
        format_penalty_reward: float = -1.0,
        seed: int = 42,
    ) -> None:
        self._records = records
        self._prompt_builder = prompt_builder
        self._renderer = renderer
        self._batch_size = batch_size
        self._k_completions = k_completions
        self._reward_type = reward_type
        self._format_penalty_reward = format_penalty_reward
        self._indices: list[int] = list(range(len(records)))
        # Shuffle on init
        rng = random.Random(seed)
        rng.shuffle(self._indices)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Return a batch of EnvGroupBuilders, one per training record."""
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._indices))
        batch_indices = self._indices[start:end]

        builders: list[EnvGroupBuilder] = []
        for idx in batch_indices:
            record = self._records[idx]
            builder = make_env_group_builder(
                record=record,
                prompt_builder=self._prompt_builder,
                renderer=self._renderer,
                k_completions=self._k_completions,
                reward_type=self._reward_type,
                format_penalty_reward=self._format_penalty_reward,
            )
            builders.append(builder)

        return builders

    def __len__(self) -> int:
        return ceil(len(self._records) / self._batch_size)


def _filter_long_prompts(
    records: list[TrainingRecord],
    prompt_builder: TrainingPromptBuilder,
    max_prompt_chars: int = 80000,
) -> list[TrainingRecord]:
    """Filter out records whose prompts would exceed the model's context window.

    Uses character count as a fast proxy (~4 chars per token for English).
    80K chars ≈ 20K tokens, leaving room for max_tokens within 32K context.
    """
    filtered = []
    n_skipped = 0
    for r in records:
        pair = prompt_builder.build(r)
        if len(pair.prompt) > max_prompt_chars:
            n_skipped += 1
        else:
            filtered.append(r)
    if n_skipped > 0:
        logger.warning(
            f"Filtered {n_skipped}/{len(records)} records with prompts > {max_prompt_chars} chars "
            f"(would exceed model context window)"
        )
    return filtered


def build_rl_datasets(
    dataset: TrainingDataset,
    prompt_builder: TrainingPromptBuilder,
    renderer: renderers.Renderer,
    batch_size: int,
    k_completions: int,
    reward_type: str = "binary",
    format_penalty_reward: float = -1.0,
    seed: int = 42,
    max_prompt_chars: int = 80000,
) -> tuple[AttributionRLDataset, AttributionRLDataset | None]:
    """Build train and optional test RL datasets.

    Args:
        dataset: Already-loaded TrainingDataset.
        prompt_builder: Builds attribution prompts from records.
        renderer: Cookbook renderer for tokenization.
        batch_size: Number of problems per training batch.
        k_completions: Number of completions per problem (group size).
        reward_type: Reward function type (binary, bce, mse, neg_bce, compound).
        format_penalty_reward: Reward for format parse failures.
        seed: Random seed for shuffling.
        max_prompt_chars: Filter records with prompts longer than this (0=no filter).

    Returns:
        (train_dataset, test_dataset or None)
    """
    train_records = _filter_long_prompts(dataset.train_records, prompt_builder, max_prompt_chars)
    train_ds = AttributionRLDataset(
        records=train_records,
        prompt_builder=prompt_builder,
        renderer=renderer,
        batch_size=batch_size,
        k_completions=k_completions,
        reward_type=reward_type,
        format_penalty_reward=format_penalty_reward,
        seed=seed,
    )

    test_ds = None
    if dataset.test_records:
        test_records = _filter_long_prompts(dataset.test_records, prompt_builder, max_prompt_chars)
        test_ds = AttributionRLDataset(
            records=test_records,
            prompt_builder=prompt_builder,
            renderer=renderer,
            batch_size=batch_size,
            k_completions=k_completions,
            reward_type=reward_type,
            format_penalty_reward=format_penalty_reward,
            seed=seed + 1,
        )

    logger.info(
        f"RL datasets: {len(train_records)} train records "
        f"({len(train_ds)} batches, K={k_completions}), "
        f"{len(dataset.test_records)} test records"
    )
    return train_ds, test_ds
