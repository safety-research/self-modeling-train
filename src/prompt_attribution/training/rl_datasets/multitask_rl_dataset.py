"""
Module: prompt_attribution/training/rl_datasets/multitask_rl_dataset.py

RLDataset wrapper for multi-task GRPO training.
Wraps MultitaskDataset to conform to the Tinker cookbook's RLDataset interface.

Structure:
- MultitaskRLDataset: get_batch() returns list[EnvGroupBuilder] for GRPO sampling
- build_multitask_rl_datasets(): Factory that returns train/val datasets
"""

import logging
import random
from math import ceil
from typing import Optional, Sequence

from tinker_cookbook import renderers
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset

from prompt_attribution.training.data.multitask.schema import MultitaskRecord
from prompt_attribution.training.rl_datasets.multitask_dataset import MultitaskDataset
from prompt_attribution.training.envs.multitask_env import (
    FlipJudgeHolder,
    SamplingClientHolder,
    make_multitask_env_group_builder,
)

logger = logging.getLogger(__name__)


class MultitaskRLDataset(RLDataset):
    """RLDataset adapter for multi-task GRPO training.

    Each batch returns a list of EnvGroupBuilder, one per training record.
    Records may be from different tasks — the env handles reward dispatch.
    """

    def __init__(
        self,
        records: list[MultitaskRecord],
        renderer: renderers.Renderer,
        batch_size: int,
        k_completions: int,
        sampling_client_holder: Optional[SamplingClientHolder] = None,
        flip_judge: Optional[FlipJudgeHolder] = None,
        seed: int = 42,
    ) -> None:
        self._records = records
        self._renderer = renderer
        self._batch_size = batch_size
        self._k_completions = k_completions
        self._sampling_client_holder = sampling_client_holder
        self._flip_judge = flip_judge
        self._indices: list[int] = list(range(len(records)))
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
            builder = make_multitask_env_group_builder(
                record=record,
                renderer=self._renderer,
                k_completions=self._k_completions,
                sampling_client_holder=self._sampling_client_holder,
                flip_judge=self._flip_judge,
            )
            builders.append(builder)

        return builders

    def __len__(self) -> int:
        return ceil(len(self._records) / self._batch_size)


def _filter_long_prompts(
    records: list[MultitaskRecord],
    max_prompt_chars: int = 64000,
) -> list[MultitaskRecord]:
    """Filter records whose task_prompt exceeds max chars."""
    filtered = []
    n_skipped = 0
    for r in records:
        if len(r.task_prompt) > max_prompt_chars:
            n_skipped += 1
        else:
            filtered.append(r)
    if n_skipped > 0:
        logger.warning(
            f"Filtered {n_skipped}/{len(records)} records with prompts > "
            f"{max_prompt_chars} chars"
        )
    return filtered


def build_multitask_rl_datasets(
    dataset: MultitaskDataset,
    renderer: renderers.Renderer,
    batch_size: int,
    k_completions: int,
    sampling_client_holder: Optional[SamplingClientHolder] = None,
    flip_judge: Optional[FlipJudgeHolder] = None,
    seed: int = 42,
    max_prompt_chars: int = 64000,
) -> tuple[MultitaskRLDataset, MultitaskRLDataset | None]:
    """Build train and optional val RL datasets.

    Args:
        dataset: Already-loaded MultitaskDataset.
        renderer: Cookbook renderer for tokenization.
        batch_size: Number of problems per training batch.
        k_completions: Number of completions per problem (group size).
        sampling_client_holder: Mutable holder for E8 online reward.
        flip_judge: Optional LLM judge for E8 open-ended flip detection.
        seed: Random seed for shuffling.
        max_prompt_chars: Filter records with prompts longer than this.

    Returns:
        (train_dataset, val_dataset or None)
    """
    train_records = _filter_long_prompts(dataset.train_records, max_prompt_chars)
    train_ds = MultitaskRLDataset(
        records=train_records,
        renderer=renderer,
        batch_size=batch_size,
        k_completions=k_completions,
        sampling_client_holder=sampling_client_holder,
        flip_judge=flip_judge,
        seed=seed,
    )

    val_ds = None
    if dataset.val_records:
        val_records = _filter_long_prompts(dataset.val_records, max_prompt_chars)
        val_ds = MultitaskRLDataset(
            records=val_records,
            renderer=renderer,
            batch_size=batch_size,
            k_completions=k_completions,
            sampling_client_holder=sampling_client_holder,
            flip_judge=flip_judge,
            seed=seed + 1,
        )

    logger.info(
        f"RL datasets: {len(train_records)} train records "
        f"({len(train_ds)} batches, K={k_completions}), "
        f"{len(dataset.val_records)} val records"
    )
    return train_ds, val_ds
