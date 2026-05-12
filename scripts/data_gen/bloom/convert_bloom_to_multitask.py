"""
Convert BLOOM fork data into MultitaskRecord format for GRPO training.

Default `--template ungrounded`: scenario + change only; the model
predicts the judge score (1-10) without seeing the rollout transcript.

Usage:
    # Default (ungrounded):
    uv run python scripts/data_gen/bloom/convert_bloom_to_multitask.py \
        --input-dir outputs/training/bloom_fork_<MODEL>_<DATE> \
        --output-dir outputs/training/bloom_multitask_<MODEL>_<DATE>

    # Dry run:
    uv run python scripts/data_gen/bloom/convert_bloom_to_multitask.py \
        --input-dir outputs/training/bloom_fork_<MODEL>_<DATE> --dry-run
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

# Regex to strip empty **Tools:** sections from scenario descriptions.
# Matches: **Tools:** followed by 1-3 empty ```xml ``` blocks (optionally with whitespace).
_EMPTY_TOOLS_RE = re.compile(
    r"\s*\*\*Tools:\*\*\s*(?:```xml\s*```\s*)+",
    re.DOTALL,
)


def _strip_empty_tools(text: str) -> tuple[str, bool]:
    """Remove empty **Tools:** ```xml ``` blocks from scenario text.

    Returns (cleaned_text, was_stripped).
    """
    cleaned, n = _EMPTY_TOOLS_RE.subn("", text)
    return cleaned.rstrip(), n > 0


def convert_record(raw: dict, idx: int) -> dict:
    """Convert a fork record to MultitaskRecord format (ungrounded judge score).

    Uses `task_prompt_ungrounded` (scenario + change only, no conversation
    or baseline score). GT is `forked_score / 10.0`, normalized to 0–1 for
    the MSE reward. The `template_variant` contains "score" so the trainer
    applies the matching /10 normalization.
    """
    forked_score = raw.get("forked_score") or raw.get("gt_score") or 1
    prompt, _ = _strip_empty_tools(raw["task_prompt_ungrounded"])
    return {
        "task_type": "e3_flip_probability",
        "template_variant": f"bloom_fork_score_{raw.get('fork_type', 'unknown')}",
        "task_prompt": prompt,
        "gt_value": forked_score / 10.0,  # Normalized to 0-1 for MSE reward
        "gt_type": "continuous",
        "unique_id": f"bloom_fork_{raw['behavior']}_s{raw['scenario_idx']}_{idx}",
        "corpus_dir": "bloom_fork",
        "dataset_id": f"bloom_fork:{raw['behavior']}",
        "example_idx": raw["scenario_idx"],
        "question": raw.get("scenario_description", "")[:500],
        "perturbation_type": raw.get("fork_type", ""),
        "category": raw.get("fork_category", ""),
        "empirical_flip_fraction": raw.get("gt_value", 0.0),
        "capability_tags": [
            "behavioral",
            raw["behavior"],
            "simenv" if raw.get("is_simenv") else "conversation",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BLOOM fork output → MultitaskRecord")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--use-combined", action="store_true", default=True,
                        help="Use balanced combined file (bloom_fork_combined.jsonl) instead of per-behavior files")
    parser.add_argument("--use-raw", action="store_true",
                        help="Use all per-behavior files (raw, unbalanced) instead of balanced combined")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.use_raw:
        args.use_combined = False

    convert_fn = convert_record
    prompt_field = "task_prompt_ungrounded"

    input_dir = args.input_dir

    if args.use_combined:
        combined_path = input_dir / "bloom_fork_combined.jsonl"
        if not combined_path.exists():
            print(f"{RED}[ERROR]{RESET} Balanced file not found: {combined_path}")
            return
        jsonl_files = [combined_path]
        print(f"{CYAN}[INFO]{RESET} Using balanced combined file ({combined_path.name})")
    else:
        jsonl_files = sorted(input_dir.glob("bloom_fork_*.jsonl"))
        jsonl_files = [f for f in jsonl_files
                       if f.name not in ("bloom_fork_combined.jsonl", "bloom_fork_combined_all.jsonl")]
        if not jsonl_files:
            print(f"{RED}[ERROR]{RESET} No bloom_fork_*.jsonl files in {input_dir}")
            return
        print(f"{CYAN}[INFO]{RESET} Using {len(jsonl_files)} per-behavior files (raw)")

    print(f"{CYAN}[INFO]{RESET} Template: ungrounded (judge score 1-10)")

    # Load and convert
    all_records = []
    stats = Counter()
    score_dist: Counter = Counter()

    for fpath in jsonl_files:
        for line in open(fpath):
            raw = json.loads(line.strip())
            stats["total"] += 1
            stats[f"behavior_{raw['behavior']}"] += 1

            if not raw.get(prompt_field):
                stats["skipped_no_prompt"] += 1
                continue

            rec = convert_fn(raw, len(all_records))
            all_records.append(rec)

            if raw.get("flipped"):
                stats["flipped"] += 1
            else:
                stats["not_flipped"] += 1
            if raw.get("is_simenv"):
                stats["simenv"] += 1
            else:
                stats["conversation"] += 1
            stats[f"fork_{raw.get('fork_type', 'unknown')}"] += 1
            score_dist[raw.get("forked_score", 0)] += 1
            # Check if empty tools were stripped
            _, was_stripped = _strip_empty_tools(raw.get("task_prompt_ungrounded", ""))
            if was_stripped:
                stats["stripped_empty_tools"] += 1

    # Print stats
    print(f"{CYAN}[INFO]{RESET} Loaded {stats['total']} raw records from {len(jsonl_files)} files")
    print(f"{CYAN}[INFO]{RESET} Converted: {len(all_records)} records")
    print(f"{CYAN}[INFO]{RESET} Skipped: {stats.get('skipped_no_prompt', 0)} (no prompt)")
    print(f"{CYAN}[INFO]{RESET} Mode: {stats.get('simenv', 0)} SimEnv, "
          f"{stats.get('conversation', 0)} conversation")
    print(f"{CYAN}[INFO]{RESET} Flip balance: {stats['flipped']} flipped, "
          f"{stats['not_flipped']} not flipped "
          f"({stats['flipped'] / max(stats['flipped'] + stats['not_flipped'], 1):.1%})")
    print(f"{CYAN}[INFO]{RESET} Fork types: "
          + ", ".join(f"{k.replace('fork_', '')}={v}"
                      for k, v in sorted(stats.items()) if k.startswith("fork_")))
    if score_dist:
        print(f"{CYAN}[INFO]{RESET} Score distribution (forked_score): "
              + ", ".join(f"{s}:{c}" for s, c in sorted(score_dist.items())))
    if stats.get("stripped_empty_tools"):
        print(f"{CYAN}[INFO]{RESET} Stripped empty **Tools:** sections: {stats['stripped_empty_tools']} records")

    if args.dry_run:
        print(f"\n{YELLOW}[DRY RUN]{RESET} Would write {len(all_records)} records. Exiting.")
        return

    # Split train/val
    rng = random.Random(args.seed)
    rng.shuffle(all_records)
    n_val = int(len(all_records) * args.val_split)
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]

    # Output
    if args.output_dir is None:
        suffix = input_dir.name.replace("bloom_fork_", "")
        args.output_dir = input_dir.parent / f"bloom_fork_multitask_{suffix}"

    output_dir = args.output_dir
    for split_name, split_records in [("train", train_records), ("val", val_records)]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        e3_path = split_dir / "e03_flip_probability.jsonl"
        with open(e3_path, "w") as f:
            for r in split_records:
                f.write(json.dumps(r) + "\n")

        combined_path = split_dir / "combined_all_tasks.jsonl"
        with open(combined_path, "w") as f:
            for r in split_records:
                f.write(json.dumps(r) + "\n")

    # Stats file
    stats_dict = {
        "total_raw": stats["total"],
        "total_records": len(all_records),
        "train": len(train_records),
        "val": len(val_records),
        "simenv": stats.get("simenv", 0),
        "conversation": stats.get("conversation", 0),
        "flip_rate": stats["flipped"] / max(stats["flipped"] + stats["not_flipped"], 1),
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats_dict, f, indent=2)

    print(f"\n{GREEN}[DONE]{RESET} Wrote {len(train_records)} train, {len(val_records)} val")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
