"""
Module: prompt_attribution/training/rl_datasets/multitask_dataset.py

Simple loader for pre-split MultitaskRecord JSONL files.
No GT cache, no balancing, no splitting — data is pre-processed.

Structure:
- MultitaskDataset: loads train/val records, filters by task
"""

import json
import logging
from collections import Counter
from pathlib import Path

from prompt_attribution.training.data.multitask.schema import MultitaskRecord

logger = logging.getLogger(__name__)

# ANSI colors
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"

# Task type strings matching TaskType enum values
VALID_TASK_TYPES = {
    "e1_flip_prediction",
    "e2_output_prediction",
    "e3_flip_probability",
    "e4_correctness_probability",
    "e5_confidence_calibration",
    "e6_perturbation_ranking",
    "e7_component_ablation",
    "e8_propose_flip",
    "e9_feature_presence",
    "e10a_margin",
    "e10b_second",
}

# Short aliases for CLI convenience
TASK_ALIASES = {
    "e1": "e1_flip_prediction",
    "e2": "e2_output_prediction",
    "e3": "e3_flip_probability",
    "e4": "e4_correctness_probability",
    "e5": "e5_confidence_calibration",
    "e6": "e6_perturbation_ranking",
    "e7": "e7_component_ablation",
    "e8": "e8_propose_flip",
    "e9": "e9_feature_presence",
    "e10a": "e10a_margin",
    "e10b": "e10b_second",
}

# Task type → per-task filename (without .jsonl)
TASK_TO_FILENAME = {
    "e1_flip_prediction": "e01_flip_prediction",
    "e2_output_prediction": "e02_output_prediction",
    "e3_flip_probability": "e03_flip_probability",
    "e4_correctness_probability": "e04_correctness_probability",
    "e5_confidence_calibration": "e05_confidence_calibration",
    "e6_perturbation_ranking": "e06_perturbation_ranking",
    "e7_component_ablation": "e07_component_ablation",
    "e8_propose_flip": "e08_propose_flip",
    "e9_feature_presence": "e09_feature_presence",
    "e10a_margin": "e10a_margin",
    "e10b_second": "e10b_second",
}


def _resolve_tasks(tasks_str: str) -> list[str]:
    """Resolve task string to list of full task type names.

    Args:
        tasks_str: "all", "e3", "e1,e3,e6", etc.

    Returns:
        List of full task type strings.
    """
    if tasks_str.strip().lower() == "all":
        return sorted(VALID_TASK_TYPES)

    task_types = []
    for t in tasks_str.split(","):
        t = t.strip().lower()
        if t in TASK_ALIASES:
            task_types.append(TASK_ALIASES[t])
        elif t in VALID_TASK_TYPES:
            task_types.append(t)
        else:
            raise ValueError(
                f"Unknown task '{t}'. Valid: {sorted(TASK_ALIASES.keys())} or 'all'"
            )
    return task_types


class MultitaskDataset:
    """Loads pre-split MultitaskRecord JSONL for multi-task RL training.

    Data is pre-processed (balanced, split) by generate_multitask_data.py.
    This class just loads and optionally filters by task type.
    """

    def __init__(self, data_dir: Path, tasks: str = "all"):
        self.data_dir = Path(data_dir)
        self.task_types = _resolve_tasks(tasks)
        self.train_records: list[MultitaskRecord] = []
        self.val_records: list[MultitaskRecord] = []

    def load(self) -> None:
        """Load train and val records from JSONL files."""
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"

        if not train_dir.exists():
            raise FileNotFoundError(f"Train dir not found: {train_dir}")

        # Load from per-task files if specific tasks, combined if all
        if len(self.task_types) == len(VALID_TASK_TYPES):
            # All tasks — load combined file
            self.train_records = self._load_jsonl(train_dir / "combined_all_tasks.jsonl")
            if (val_dir / "combined_all_tasks.jsonl").exists():
                self.val_records = self._load_jsonl(val_dir / "combined_all_tasks.jsonl")
        else:
            # Specific tasks — load per-task files
            for task_type in self.task_types:
                filename = TASK_TO_FILENAME.get(task_type, task_type) + ".jsonl"
                train_path = train_dir / filename
                if train_path.exists():
                    self.train_records.extend(self._load_jsonl(train_path))
                else:
                    logger.warning(f"Train file not found: {train_path}")

                val_path = val_dir / filename
                if val_path.exists():
                    self.val_records.extend(self._load_jsonl(val_path))

        self._log_stats()

    @staticmethod
    def _load_jsonl(path: Path) -> list[MultitaskRecord]:
        """Load MultitaskRecord objects from a JSONL file."""
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(MultitaskRecord.from_dict(json.loads(line)))
        return records

    def _log_stats(self) -> None:
        """Log dataset statistics."""
        train_by_task = Counter(r.task_type for r in self.train_records)
        val_by_task = Counter(r.task_type for r in self.val_records)
        train_by_gt = Counter(r.gt_type for r in self.train_records)

        logger.info(
            f"{GREEN}[LOADED]{RESET} {len(self.train_records)} train + "
            f"{len(self.val_records)} val records from {self.data_dir.name}"
        )
        logger.info(f"  Tasks: {sorted(self.task_types)}")
        for task_type in sorted(set(list(train_by_task.keys()) + list(val_by_task.keys()))):
            logger.info(
                f"  {task_type}: {train_by_task.get(task_type, 0)} train + "
                f"{val_by_task.get(task_type, 0)} val"
            )
        logger.info(f"  GT types: {dict(train_by_gt)}")
