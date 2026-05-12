"""
Generate multi-task introspection training data from auto_perturbation corpus.

Tier 1 tasks (no inference needed): e1, e2, e3, e6, e9
Tier 2 tasks (need vLLM server):   e4, e5, e7, e8, e10a, e10b

Usage:
    # Tier 1 only (balanced 1.5K/task, 85/15 train/val):
    uv run python scripts/data_gen/hf/generate_multitask_data.py \
        --corpus-dir outputs/auto_perturbation/corpus_<MODEL>_<DATE>

    # All tasks:
    uv run python scripts/data_gen/hf/generate_multitask_data.py \
        --corpus-dir outputs/auto_perturbation/corpus_<MODEL>_<DATE> \
        --tasks e1 e2 e3 e4 e5 e6 e7 e8 e9 e10a e10b \
        --vllm-url http://HOST:PORT/v1 \
        --model-id meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from prompt_attribution.training.data.multitask.schema import MultitaskRecord, TaskType
from prompt_attribution.training.data.multitask.tier1_converter import Tier1Converter, unique_base_filter, slim_select
from prompt_attribution.training.data.multitask.html_viewer import generate_html

# Colored logging
RESET = "\033[0m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s {CYAN}[%(levelname)s]{RESET} %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logs from openai/anthropic SDKs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)


TIER1_TASKS = {"e1", "e2", "e3", "e6", "e8", "e9"}
TIER2_TASKS = {"e4", "e5", "e7", "e10a", "e10b"}
VALID_TASKS = TIER1_TASKS | TIER2_TASKS


def load_corpus(corpus_dir: Path) -> list[dict]:
    """Load training_data.jsonl from corpus directory."""
    jsonl_path = corpus_dir / "training_data.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"No training_data.jsonl in {corpus_dir}")

    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    config_path = corpus_dir / "corpus_config.json"
    corpus_config = {}
    if config_path.exists():
        with open(config_path) as f:
            corpus_config = json.load(f)

    logger.info(
        f"{GREEN}[LOADED]{RESET} {len(rows)} rows from {corpus_dir.name} "
        f"(target: {corpus_config.get('target_model_id', 'unknown')})"
    )
    return rows


def write_jsonl(records: list[MultitaskRecord], path: Path) -> None:
    """Write records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"{GREEN}[SAVED]{RESET} {len(records)} records -> {path}")


# ---------------------------------------------------------------------------
# Per-task cache for resume support
# ---------------------------------------------------------------------------

def _cache_dir(output_dir: Path) -> Path:
    return output_dir / ".cache"


def _tier2_config_path(output_dir: Path) -> Path:
    return _cache_dir(output_dir) / "_tier2_config.json"


def save_tier2_config(output_dir: Path, config: dict) -> None:
    """Save Tier 2 generation config for resume validation."""
    path = _tier2_config_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_tier2_config(output_dir: Path) -> dict | None:
    """Load saved Tier 2 config, or None if not found."""
    path = _tier2_config_path(output_dir)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_task_cache(
    output_dir: Path, task_type: str, records: list[MultitaskRecord],
) -> None:
    """Save raw (pre-balance) records for a task to cache."""
    cache = _cache_dir(output_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{task_type}.jsonl"
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"{GREEN}[CACHE]{RESET} Saved {len(records)} raw records -> {path}")


def load_task_cache(
    output_dir: Path, task_type: str,
) -> list[MultitaskRecord] | None:
    """Load cached raw records for a task, or None if not cached."""
    path = _cache_dir(output_dir) / f"{task_type}.jsonl"
    if not path.exists():
        return None
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(MultitaskRecord.from_dict(json.loads(line)))
    logger.info(
        f"{CYAN}[RESUME]{RESET} Loaded {len(records)} cached records for {task_type}"
    )
    return records


# ---------------------------------------------------------------------------
# Problem-level train/val split + balanced selection
# ---------------------------------------------------------------------------


def split_problems(
    all_rows: list[dict],
    val_split: float,
    seed: int,
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    """Split unique problems into train/val sets.

    Returns (train_problems, val_problems) as sets of (dataset_id, example_idx).
    The same split is used across ALL tasks for consistency.
    """
    rng = random.Random(seed)
    problems = sorted(set(
        (r.get("dataset_id", ""), r.get("example_idx", 0)) for r in all_rows
    ))
    rng.shuffle(problems)
    n_val = int(len(problems) * val_split)
    val_problems = set(map(tuple, problems[:n_val]))
    train_problems = set(map(tuple, problems[n_val:]))
    logger.info(
        f"{CYAN}[SPLIT]{RESET} {len(problems)} unique problems -> "
        f"{len(train_problems)} train + {len(val_problems)} val "
        f"({val_split:.0%} held out)"
    )
    return train_problems, val_problems  # type: ignore[return-value]


def select_prioritizing_unique(
    records: list[MultitaskRecord],
    target: int,
    seed: int,
) -> list[MultitaskRecord]:
    """Select up to `target` records, prioritizing unique entries.

    Uses unique_id for deduplication (not dataset_id/example_idx) so that
    lever-negative variants (e.g., E9 _lever records with different gt)
    are treated as distinct entries.

    1. One record per unique_id — shuffled
    2. If < target, fill with remaining records
    """
    rng = random.Random(seed)

    seen: set[str] = set()
    unique = []
    fill = []
    for r in records:
        key = r.unique_id
        if key not in seen:
            seen.add(key)
            unique.append(r)
        else:
            fill.append(r)

    rng.shuffle(unique)
    rng.shuffle(fill)

    if len(unique) >= target:
        return unique[:target]
    return unique + fill[:target - len(unique)]


def _stratify_key(r: MultitaskRecord) -> str:
    """Assign a stratum key for GT-aware balancing."""
    if r.gt_type == "binary":
        return r.gt_label or "unknown"
    elif r.gt_type == "continuous" and r.gt_value is not None:
        # Bin into low/high for balanced sampling
        return "high" if r.gt_value >= 0.5 else "low"
    elif r.gt_type == "mcq":
        return r.gt_label or "unknown"
    return "default"


def select_stratified(
    records: list[MultitaskRecord],
    target: int,
    seed: int,
) -> list[MultitaskRecord]:
    """Select up to `target` records with GT-aware stratified sampling.

    Balances across GT strata (e.g., Yes/No for binary, low/high for
    continuous), then fills each stratum prioritizing unique entries.
    """
    rng = random.Random(seed)

    # Group by stratum
    strata: dict[str, list[MultitaskRecord]] = {}
    for r in records:
        key = _stratify_key(r)
        strata.setdefault(key, []).append(r)

    # If only one stratum or "text" type, fall back to simple selection
    if len(strata) <= 1:
        return select_prioritizing_unique(records, target, seed)

    # Equal allocation per stratum, then redistribute remainder
    n_strata = len(strata)
    per_stratum = target // n_strata
    remainder = target % n_strata

    selected = []
    # Sort keys for determinism, give remainder to smallest strata first
    sorted_keys = sorted(strata.keys(), key=lambda k: len(strata[k]))
    for i, key in enumerate(sorted_keys):
        alloc = per_stratum + (1 if i < remainder else 0)
        # Cap at available records
        alloc = min(alloc, len(strata[key]))
        selected.extend(select_prioritizing_unique(strata[key], alloc, seed))

    # If some strata were too small, redistribute to larger strata
    if len(selected) < target:
        used_ids = {r.unique_id for r in selected}
        remaining = [r for r in records if r.unique_id not in used_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[:target - len(selected)])

    return selected


def balance_and_split(
    records: list[MultitaskRecord],
    target: int,
    val_split: float,
    train_problems: set[tuple[str, int]],
    val_problems: set[tuple[str, int]],
    seed: int,
) -> tuple[list[MultitaskRecord], list[MultitaskRecord]]:
    """Partition records by problem-level split, then select to target size.

    Uses GT-aware stratified sampling for balanced label distribution.
    Returns (train_records, val_records).
    """
    train_pool = [r for r in records if (r.dataset_id, r.example_idx) in train_problems]
    val_pool = [r for r in records if (r.dataset_id, r.example_idx) in val_problems]

    target_train = int(target * (1 - val_split))
    target_val = target - target_train

    train_selected = select_stratified(train_pool, target_train, seed)
    val_selected = select_stratified(val_pool, target_val, seed)

    return train_selected, val_selected


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def write_stats(
    train_by_task: dict[str, list[MultitaskRecord]],
    val_by_task: dict[str, list[MultitaskRecord]],
    output_dir: Path,
    corpus_dir: str,
    elapsed: float,
    target_per_task: int,
    val_split: float,
) -> None:
    """Write summary stats JSON."""
    stats: dict = {
        "corpus_dir": corpus_dir,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "target_per_task": target_per_task,
        "val_split": val_split,
        "tasks": {},
    }

    all_tasks = sorted(set(list(train_by_task.keys()) + list(val_by_task.keys())))
    for task_type in all_tasks:
        train_recs = train_by_task.get(task_type, [])
        val_recs = val_by_task.get(task_type, [])
        all_recs = train_recs + val_recs

        task_stats: dict = {
            "train_count": len(train_recs),
            "val_count": len(val_recs),
            "total_count": len(all_recs),
            "gt_type": all_recs[0].gt_type if all_recs else "?",
        }

        if all_recs and all_recs[0].gt_type in ("binary", "mcq"):
            task_stats["gt_distribution"] = dict(Counter(r.gt_label for r in all_recs))
        elif all_recs and all_recs[0].gt_type == "continuous":
            values = [r.gt_value for r in all_recs if r.gt_value is not None]
            task_stats["gt_mean"] = round(sum(values) / max(len(values), 1), 4)

        task_stats["datasets"] = dict(
            Counter(r.dataset_id for r in all_recs).most_common(10)
        )
        stats["tasks"][task_type] = task_stats

    stats["total_train"] = sum(len(r) for r in train_by_task.values())
    stats["total_val"] = sum(len(r) for r in val_by_task.values())

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"{GREEN}[SAVED]{RESET} stats -> {stats_path}")


# ---------------------------------------------------------------------------
# Task → filename mapping
# ---------------------------------------------------------------------------

TASK_TO_FILENAME = {
    TaskType.E1_FLIP_PREDICTION.value: "e01_flip_prediction",
    TaskType.E2_OUTPUT_PREDICTION.value: "e02_output_prediction",
    TaskType.E3_FLIP_PROBABILITY.value: "e03_flip_probability",
    TaskType.E4_CORRECTNESS_PROBABILITY.value: "e04_correctness_probability",
    TaskType.E5_CONFIDENCE_CALIBRATION.value: "e05_confidence_calibration",
    TaskType.E6_PERTURBATION_RANKING.value: "e06_perturbation_ranking",
    TaskType.E7_COMPONENT_ABLATION.value: "e07_component_ablation",
    TaskType.E8_PROPOSE_FLIP.value: "e08_propose_flip",
    TaskType.E9_FEATURE_PRESENCE.value: "e09_feature_presence",
    TaskType.E10A_MARGIN.value: "e10a_margin",
    TaskType.E10B_SECOND.value: "e10b_second",
}


# ---------------------------------------------------------------------------
# Tier 2 runner
# ---------------------------------------------------------------------------


async def run_tier2(
    tasks: set[str],
    rows: list[dict],
    records_by_task: dict[str, list[MultitaskRecord]],
    vllm_url: str,
    model_id: str,
    corpus_dir: str,
    n_resample: int,
    max_concurrent: int,
    max_rows: int,
    seed: int,
    output_dir: Path | None = None,
    anthropic_model: str | None = None,
    anthropic_api_key: str | None = None,
    enable_judge: bool = False,
    judge_haiku_model: str = "claude-haiku-4-5-20251001",
    judge_opus_model: str = "claude-opus-4-5-20251101",
) -> None:
    """Run Tier 2 tasks that require model inference.

    Each task's results are cached to disk as soon as it completes,
    so that --resume can skip finished tasks after a crash.
    """
    from prompt_attribution.training.data.multitask.tier2_collector import (
        HaikuModelClient,
        JudgeClient,
        ModelClient,
        ModelClientConfig,
        Tier2Collector,
    )

    if anthropic_model:
        client = HaikuModelClient(
            model_id=anthropic_model,
            api_key=anthropic_api_key,
            max_concurrent=max_concurrent,
            temperature=0.7,
            max_tokens=2048,
        )
        logger.info(f"{CYAN}[TIER2]{RESET} Using Anthropic API: {anthropic_model}")
        if "e10a" in tasks or "e10b" in tasks:
            logger.warning(
                f"{YELLOW}[WARN]{RESET} Skipping E10 (margin + second choice) for "
                f"closed-source API model '{anthropic_model}'. E10 requires real "
                f"token logprobs — use vLLM with an open-weight model instead."
            )
            tasks = tasks - {"e10a", "e10b"}
    else:
        # Enable thinking for GPT-OSS models
        sys_prompt = None
        if model_id and "gpt-oss" in model_id.lower():
            from prompt_attribution.training.data.multitask.tier2_collector import GPT_OSS_SYSTEM_PROMPT
            sys_prompt = GPT_OSS_SYSTEM_PROMPT
            logger.info(f"{CYAN}[TIER2]{RESET} GPT-OSS detected — enabling thinking (Reasoning: high)")
        config = ModelClientConfig(
            vllm_url=vllm_url,
            model_id=model_id,
            temperature=0.7,
            max_tokens=2048,
            max_concurrent=max_concurrent,
            system_prompt=sys_prompt,
        )
        client = ModelClient(config)

    # Create judge client for LLM-based evaluation (E4 correctness, E7 flip detection + decomposition)
    judge_client = None
    if enable_judge:
        judge_client = JudgeClient(
            haiku_model=judge_haiku_model,
            opus_model=judge_opus_model,
        )
        logger.info(
            f"{CYAN}[JUDGE]{RESET} Enabled — flip: {judge_haiku_model}, "
            f"correctness: {judge_opus_model}, decomposer: {judge_haiku_model}"
        )

    collector = Tier2Collector(
        client=client,
        corpus_dir=corpus_dir,
        n_resample=n_resample,
        seed=seed,
        max_rows=max_rows,
        cache_dir=_cache_dir(output_dir) if output_dir else None,
        judge_client=judge_client,
    )

    # --- Helper to run a task and cache result immediately ---
    async def _run_and_cache(name: str, coro):
        result = await coro
        # Save to cache as soon as this task completes
        if output_dir is not None:
            if name == "e4":
                save_task_cache(output_dir, TaskType.E4_CORRECTNESS_PROBABILITY.value, result)
            elif name == "e5":
                save_task_cache(output_dir, TaskType.E5_CONFIDENCE_CALIBRATION.value, result)
            elif name == "e7":
                save_task_cache(output_dir, TaskType.E7_COMPONENT_ABLATION.value, result)
            elif name == "e10":
                margin_records, second_records = result
                save_task_cache(output_dir, TaskType.E10A_MARGIN.value, margin_records)
                save_task_cache(output_dir, TaskType.E10B_SECOND.value, second_records)
        return result

    # Run all Tier 2 tasks concurrently to keep vLLM fully saturated
    async_tasks = []
    task_names = []

    if "e4" in tasks:
        logger.info(f"{MAGENTA}[E4]{RESET} Collecting correctness probability...")
        async_tasks.append(_run_and_cache("e4", collector.collect_e4(rows)))
        task_names.append("e4")

    if "e5" in tasks:
        logger.info(f"{MAGENTA}[E5]{RESET} Collecting confidence calibration...")
        async_tasks.append(_run_and_cache("e5", collector.collect_e5(rows)))
        task_names.append("e5")

    if "e7" in tasks:
        logger.info(
            f"{MAGENTA}[E7]{RESET} Collecting component ablation "
            f"(target=1500, early stopping)..."
        )
        async_tasks.append(_run_and_cache("e7", collector.collect_e7(rows, target_records=1500)))
        task_names.append("e7")

    if "e10a" in tasks or "e10b" in tasks:
        logger.info(f"{MAGENTA}[E10]{RESET} Collecting margin & second choice...")
        async_tasks.append(_run_and_cache("e10", collector.collect_e10(rows)))
        task_names.append("e10")

    # Gather all concurrently
    results = await asyncio.gather(*async_tasks)

    # Map results back
    for name, result in zip(task_names, results):
        if name == "e4":
            records_by_task[TaskType.E4_CORRECTNESS_PROBABILITY.value] = result
        elif name == "e5":
            records_by_task[TaskType.E5_CONFIDENCE_CALIBRATION.value] = result
        elif name == "e7":
            records_by_task[TaskType.E7_COMPONENT_ABLATION.value] = result
        elif name == "e10":
            margin_records, second_records = result
            if "e10a" in tasks:
                records_by_task[TaskType.E10A_MARGIN.value] = margin_records
            if "e10b" in tasks:
                records_by_task[TaskType.E10B_SECOND.value] = second_records

    logger.info(
        f"{GREEN}[TIER2]{RESET} Total model calls: {client._call_count}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-task introspection training data from corpus."
    )
    parser.add_argument(
        "--corpus-dir", type=Path, required=True,
        help="Path to auto_perturbation corpus directory",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["e1", "e2", "e3", "e6", "e8", "e9"],
        choices=sorted(VALID_TASKS),
        help="Tasks to generate (default: Tier 1 only)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--target-per-task", type=int, default=1500,
        help="Target number of examples per task (default: 1500). "
        "Set to 0 to disable balancing and keep all records.",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15,
        help="Fraction of data to hold out for validation (default: 0.15). "
        "Split is at the problem level for consistency across tasks.",
    )
    parser.add_argument(
        "--max-viewer-examples", type=int, default=200,
        help="Max examples per task in HTML viewer",
    )
    # Tier 2 args
    parser.add_argument(
        "--vllm-url", type=str, default=None,
        help="vLLM server URL (required for Tier 2 tasks with local model)",
    )
    parser.add_argument(
        "--model-id", type=str, default=None,
        help="Model ID for vLLM (required for Tier 2 tasks with local model)",
    )
    parser.add_argument(
        "--anthropic-model", type=str, default=None,
        help="Use Anthropic API instead of vLLM for Tier 2 tasks "
             "(e.g. claude-haiku-4-5-20251001). Uses ANTHROPIC_API_KEY env var.",
    )
    parser.add_argument(
        "--n-resample", type=int, default=1,
        help="Number of resamples for Tier 2 ground truth (default: 1)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=512,
        help="Max concurrent model calls (default: 512)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=2000,
        help="Max rows per Tier 2 task before balance (default: 2000). "
        "Set higher than --target-per-task to allow for val split overhead.",
    )
    parser.add_argument(
        "--enable-judge", action="store_true",
        help="Enable LLM judge for E4 correctness (Opus), E7 flip detection (Haiku), "
             "and E7 decomposition (Haiku). Requires ANTHROPIC_API_KEY env var. "
             "Does not affect E8 (E8 reward is computed online during GRPO training).",
    )
    parser.add_argument(
        "--judge-haiku-model", type=str, default="claude-haiku-4-5-20251001",
        help="Haiku model for flip judge + decomposer (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--judge-opus-model", type=str, default="claude-opus-4-5-20251101",
        help="Opus model for E4 correctness judge (default: claude-opus-4-5-20251101)",
    )
    parser.add_argument(
        "--unique-base", action="store_true",
        help="Apply Dataset B sampling: one perturbation per unique problem (~1.5K rows)",
    )
    parser.add_argument(
        "--slim", action="store_true",
        help="Slim mode: select 3 perturbations per problem (1 flip, 1 non_flip, 1 boundary).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from cached per-task results in output_dir/.cache/. "
             "Skips Tier 1 and any Tier 2 tasks that already completed.",
    )
    args = parser.parse_args()

    tasks = set(args.tasks)

    # Validate corpus
    if not args.corpus_dir.exists():
        logger.error(f"{RED}[ERROR]{RESET} Corpus dir not found: {args.corpus_dir}")
        sys.exit(1)

    # Validate Tier 2 requirements
    tier2_requested = tasks & TIER2_TASKS
    if tier2_requested and not args.vllm_url and not args.anthropic_model:
        logger.error(
            f"{RED}[ERROR]{RESET} Tier 2 tasks {tier2_requested} require either "
            f"--vllm-url/--model-id (local vLLM) or --anthropic-model (Anthropic API)"
        )
        sys.exit(1)

    # Output dir
    if args.output_dir is None:
        corpus_name = args.corpus_dir.name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(f"outputs/training/multitask_{corpus_name}_{timestamp}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{CYAN}[START]{RESET} Output: {args.output_dir}")
    logger.info(
        f"  Tasks: {sorted(tasks)} | "
        f"Target: {args.target_per_task}/task | "
        f"Val: {args.val_split:.0%}"
    )

    # Save command for reproducibility
    cmd_path = args.output_dir / "command.txt"
    cmd_path.write_text(" ".join(sys.argv))

    start = time.time()

    # Load corpus
    all_rows = load_corpus(args.corpus_dir)

    # --- Problem-level train/val split ---
    train_problems, val_problems = split_problems(
        all_rows, val_split=args.val_split, seed=args.seed,
    )

    # --- Row selection mode ---
    if args.slim:
        slim_data = slim_select(all_rows, seed=args.seed)
        rows = slim_data["flip"] + slim_data["non_flip"] + slim_data["boundary"]
        logger.info(
            f"{CYAN}[SLIM]{RESET} Selected {len(rows)} rows "
            f"({len(slim_data['flip'])} flip + {len(slim_data['non_flip'])} non_flip "
            f"+ {len(slim_data['boundary'])} boundary)"
        )
    elif args.unique_base:
        rows = unique_base_filter(all_rows, seed=args.seed)
    else:
        rows = all_rows

    slim_data: dict[str, list] = {}
    records_by_task: dict[str, list[MultitaskRecord]] = {}

    # --- Resume: load cached tasks ---
    if args.resume:
        logger.info(f"{CYAN}[RESUME]{RESET} Checking cache in {args.output_dir}/.cache/")

    # Map task short names to TaskType values for cache lookup
    _TASK_MAP = {
        "e1": TaskType.E1_FLIP_PREDICTION.value,
        "e2": TaskType.E2_OUTPUT_PREDICTION.value,
        "e3": TaskType.E3_FLIP_PROBABILITY.value,
        "e6": TaskType.E6_PERTURBATION_RANKING.value,
        "e8": TaskType.E8_PROPOSE_FLIP.value,
        "e9": TaskType.E9_FEATURE_PRESENCE.value,
        "e4": TaskType.E4_CORRECTNESS_PROBABILITY.value,
        "e5": TaskType.E5_CONFIDENCE_CALIBRATION.value,
        "e7": TaskType.E7_COMPONENT_ABLATION.value,
        "e10a": TaskType.E10A_MARGIN.value,
        "e10b": TaskType.E10B_SECOND.value,
    }

    # --- Tier 1 (synchronous, no inference) ---
    # Always regenerate Tier 1 — it's fast (<10s) and avoids stale caches
    # when converter code changes (e.g., E6 multi-ranking, E9 lever negatives).
    tier1_requested = tasks & TIER1_TASKS
    if tier1_requested:
        if args.resume:
            logger.info(f"{CYAN}[RESUME]{RESET} Regenerating Tier 1 (always fresh, <10s)...")

        _force_noshow = False
        _corpus_config_path = args.corpus_dir / "corpus_config.json"
        if _corpus_config_path.exists():
            import json as _json
            _cc = _json.load(open(_corpus_config_path))
            _target = _cc.get("target_model_id", "")
            _force_noshow = any(
                _target.startswith(p) for p in ("claude-", "gpt-", "o1-", "o3-", "o4-")
            )
            if _force_noshow:
                logger.info(
                    f"{YELLOW}[INFO]{RESET} API target model detected ({_target}): "
                    f"using force_noshow=True for E1/E2 to avoid embedding cross-model answers"
                )

        converter = Tier1Converter(
            corpus_dir=str(args.corpus_dir),
            seed=args.seed,
            force_noshow=_force_noshow,
        )

        if "e1" in tasks:
            logger.info(f"{MAGENTA}[E1]{RESET} Converting flip prediction...")
            if args.slim:
                e1_rows = slim_data["flip"] + slim_data["non_flip"]
                records_by_task[TaskType.E1_FLIP_PREDICTION.value] = converter.convert_e1(e1_rows)
            else:
                records_by_task[TaskType.E1_FLIP_PREDICTION.value] = converter.convert_e1(rows)

        if "e2" in tasks:
            logger.info(f"{MAGENTA}[E2]{RESET} Converting output prediction...")
            records_by_task[TaskType.E2_OUTPUT_PREDICTION.value] = converter.convert_e2(rows)

        if "e3" in tasks:
            logger.info(f"{MAGENTA}[E3]{RESET} Converting flip probability...")
            if args.slim:
                records_by_task[TaskType.E3_FLIP_PROBABILITY.value] = converter.convert_e3(
                    slim_data["boundary"]
                )
            else:
                records_by_task[TaskType.E3_FLIP_PROBABILITY.value] = converter.convert_e3(rows)

        if "e6" in tasks:
            logger.info(f"{MAGENTA}[E6]{RESET} Converting perturbation ranking...")
            records_by_task[TaskType.E6_PERTURBATION_RANKING.value] = converter.convert_e6(all_rows)

        if "e8" in tasks:
            logger.info(f"{MAGENTA}[E8]{RESET} Converting propose-flip (prompt only, online reward)...")
            records_by_task[TaskType.E8_PROPOSE_FLIP.value] = converter.convert_e8(rows)

        if "e9" in tasks:
            logger.info(f"{MAGENTA}[E9]{RESET} Converting feature presence...")
            records_by_task[TaskType.E9_FEATURE_PRESENCE.value] = converter.convert_e9(rows)

        # Cache Tier 1 results
        for t in tier1_requested:
            tt = _TASK_MAP[t]
            if tt in records_by_task:
                save_task_cache(args.output_dir, tt, records_by_task[tt])

    # --- Tier 2 (async, requires vLLM or Anthropic API) ---
    # On resume, validate config before loading cached Tier 2 tasks.
    # Config mismatch (e.g., changed max_rows) invalidates stale caches.
    current_tier2_config = {
        "model": args.anthropic_model or args.model_id or "default",
        "n_resample": args.n_resample,
        "max_rows": args.max_rows,
        "enable_judge": args.enable_judge,
        "judge_haiku_model": args.judge_haiku_model,
        "judge_opus_model": args.judge_opus_model,
    }

    tier2_remaining = set()
    if tier2_requested:
        # Check if cached config matches current config
        cached_config = load_tier2_config(args.output_dir) if args.resume else None
        config_match = cached_config == current_tier2_config if cached_config else False

        if args.resume and not config_match and cached_config is not None:
            logger.info(
                f"{YELLOW}[RESUME]{RESET} Tier 2 config changed — invalidating "
                f"stale caches. Old: {cached_config}, New: {current_tier2_config}"
            )

        for t in tier2_requested:
            tt = _TASK_MAP.get(t, t)
            if args.resume and config_match:
                cached = load_task_cache(args.output_dir, tt)
                if cached is not None:
                    records_by_task[tt] = cached
                    continue
            tier2_remaining.add(t)

        # E10 is special: both e10a and e10b come from one collector call
        # If either is missing, we need to run the e10 collector
        if ("e10a" in tier2_requested or "e10b" in tier2_requested) and \
           ("e10a" in tier2_remaining or "e10b" in tier2_remaining):
            tier2_remaining.discard("e10a")
            tier2_remaining.discard("e10b")
            if "e10a" in tier2_requested:
                tier2_remaining.add("e10a")
            if "e10b" in tier2_requested:
                tier2_remaining.add("e10b")

    if tier2_remaining:
        _model_label = args.anthropic_model or args.model_id or "default"
        logger.info(
            f"\n{CYAN}[TIER2]{RESET} Starting async collection for {sorted(tier2_remaining)} "
            f"(model={_model_label}, n_resample={args.n_resample}, max_rows={args.max_rows})"
        )
        asyncio.run(
            run_tier2(
                tasks=tier2_remaining,
                rows=rows,
                records_by_task=records_by_task,
                vllm_url=args.vllm_url or "",
                model_id=args.model_id or "default",
                corpus_dir=str(args.corpus_dir),
                n_resample=args.n_resample,
                max_concurrent=args.max_concurrent,
                max_rows=args.max_rows,
                seed=args.seed,
                output_dir=args.output_dir,
                anthropic_model=args.anthropic_model,
                enable_judge=args.enable_judge,
                judge_haiku_model=args.judge_haiku_model,
                judge_opus_model=args.judge_opus_model,
            )
        )
        # Save config after successful Tier 2 completion
        save_tier2_config(args.output_dir, current_tier2_config)
    elif tier2_requested:
        logger.info(f"{GREEN}[RESUME]{RESET} All Tier 2 tasks loaded from cache!")

    # --- Balance and split ---
    train_by_task: dict[str, list[MultitaskRecord]] = {}
    val_by_task: dict[str, list[MultitaskRecord]] = {}

    for task_type, records in records_by_task.items():
        if args.target_per_task > 0:
            train_recs, val_recs = balance_and_split(
                records=records,
                target=args.target_per_task,
                val_split=args.val_split,
                train_problems=train_problems,
                val_problems=val_problems,
                seed=args.seed,
            )
        else:
            # No balancing — just split by problem
            train_recs = [r for r in records if (r.dataset_id, r.example_idx) in train_problems]
            val_recs = [r for r in records if (r.dataset_id, r.example_idx) in val_problems]

        train_by_task[task_type] = train_recs
        val_by_task[task_type] = val_recs

        logger.info(
            f"  {task_type}: {len(records)} raw -> "
            f"{len(train_recs)} train + {len(val_recs)} val"
        )

    # --- Write outputs ---
    train_dir = args.output_dir / "train"
    val_dir = args.output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for task_type in sorted(set(list(train_by_task.keys()) + list(val_by_task.keys()))):
        filename = TASK_TO_FILENAME.get(task_type, task_type)
        if train_by_task.get(task_type):
            write_jsonl(train_by_task[task_type], train_dir / f"{filename}.jsonl")
        if val_by_task.get(task_type):
            write_jsonl(val_by_task[task_type], val_dir / f"{filename}.jsonl")

    # Write combined files
    all_train = []
    for recs in train_by_task.values():
        all_train.extend(recs)
    if all_train:
        write_jsonl(all_train, train_dir / "combined_all_tasks.jsonl")

    all_val = []
    for recs in val_by_task.values():
        all_val.extend(recs)
    if all_val:
        write_jsonl(all_val, val_dir / "combined_all_tasks.jsonl")

    # Write stats
    elapsed = time.time() - start
    write_stats(
        train_by_task, val_by_task, args.output_dir,
        str(args.corpus_dir), elapsed, args.target_per_task, args.val_split,
    )

    # Generate HTML viewer (from train set for inspection)
    all_records_for_viewer = {}
    for task_type in train_by_task:
        all_records_for_viewer[task_type] = (
            train_by_task.get(task_type, []) + val_by_task.get(task_type, [])
        )

    if all_records_for_viewer:
        logger.info(f"{CYAN}[HTML]{RESET} Generating debug viewer...")
        generate_html(
            records_by_task=all_records_for_viewer,
            output_path=args.output_dir / "debug_viewer.html",
            corpus_dir=str(args.corpus_dir),
            max_per_task=args.max_viewer_examples,
        )

    # Summary
    logger.info(f"\n{GREEN}[DONE]{RESET} Generated in {elapsed:.1f}s:")
    for task_type in sorted(set(list(train_by_task.keys()) + list(val_by_task.keys()))):
        n_train = len(train_by_task.get(task_type, []))
        n_val = len(val_by_task.get(task_type, []))
        logger.info(f"  {task_type}: {n_train} train + {n_val} val = {n_train + n_val}")
    total_train = sum(len(r) for r in train_by_task.values())
    total_val = sum(len(r) for r in val_by_task.values())
    logger.info(f"  Total: {total_train} train + {total_val} val = {total_train + total_val}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"    train/ — {len(train_by_task)} task files + combined")
    logger.info(f"    val/   — {len(val_by_task)} task files + combined")


if __name__ == "__main__":
    main()
