"""
Module: prompt_attribution/auto_perturbation/run_full_corpus.py

Multi-dataset orchestrator for full corpus generation. Runs the pipeline
sequentially across all cached datasets, appending results to a shared
training_data.jsonl and training_data_review.csv for incremental monitoring.

Usage:
    python -m prompt_attribution.auto_perturbation.run_full_corpus \
        --datasets_cache outputs/auto_perturbation/.discovery_cache/datasets.jsonl \
        --n_samples 10 --concurrency 50

    # Resume after crash (skip already-completed datasets):
    python -m prompt_attribution.auto_perturbation.run_full_corpus \
        --datasets_cache outputs/auto_perturbation/.discovery_cache/datasets.jsonl \
        --output_dir outputs/auto_perturbation/corpus_<MODEL>_<DATE> \
        --skip_completed

Structure:
- load_datasets: Read cached datasets.jsonl
- sanitize_dataset_id: Convert HF IDs to safe directory names
- append_to_shared_outputs: Merge per-dataset JSONL into shared files
- run_corpus: Main orchestration loop
"""

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

from safetytooling.utils.utils import setup_environment

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full corpus generation across all cached datasets"
    )
    parser.add_argument(
        "--datasets_cache", type=str,
        default="outputs/auto_perturbation/.discovery_cache/datasets.jsonl",
        help="Path to cached datasets.jsonl from discovery stage",
    )
    parser.add_argument(
        "--n_samples", type=int, default=20,
        help="Number of samples per dataset (default: 20)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=100,
        help="Max concurrent generator/critic API calls — Claude API (default: 100)",
    )
    parser.add_argument(
        "--verification_concurrency", type=int, default=16,
        help="Max concurrent target model calls (default: 16, suitable for vLLM; "
             "set higher for cloud API)",
    )
    parser.add_argument(
        "--generator_model", type=str, default="claude-opus-4-5-20251101",
        help="Model for generation/decomposition/critic",
    )
    parser.add_argument(
        "--judge_model", type=str, default="claude-haiku-4-5-20251001",
        help="Model for Phase 1 verification",
    )
    parser.add_argument(
        "--stability_n_runs", type=int, default=1,
        help="Stability runs per candidate (default: 1)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--skip_completed", action="store_true",
        help="Skip datasets already completed (reads progress.json)",
    )
    parser.add_argument(
        "--dump_prompts", action="store_true",
        help="Dump full prompts per dataset (warning: large output)",
    )
    parser.add_argument(
        "--max_datasets", type=int, default=None,
        help="Limit number of datasets to process (for testing)",
    )
    parser.add_argument(
        "--no_feedback_loop", action="store_true",
        help="Disable critic→generator feedback loop (enabled by default)",
    )
    parser.add_argument(
        "--feedback_max_rounds", type=int, default=5,
        help="Max feedback loop iterations (default: 5)",
    )
    parser.add_argument(
        "--skip_verify", action="store_true",
        help="Skip empirical verification (faster, critic-only labels). "
             "Re-run later with --stage verify to add empirical labels.",
    )
    parser.add_argument(
        "--parallel_datasets", type=int, default=1,
        help="Number of datasets to process in parallel (default: 1)",
    )
    parser.add_argument(
        "--verify_only", action="store_true",
        help="Run only verification + re-export on an existing corpus dir. "
             "Requires --output_dir pointing to a completed corpus.",
    )

    # Target model for verification-aware feedback loop
    parser.add_argument(
        "--target_model_id", type=str, default="",
        help="Target model for verification (e.g., meta-llama/Llama-3.1-8B-Instruct). "
             "If set without --target_model_url, auto-launches vLLM on Slurm.",
    )
    parser.add_argument(
        "--target_model_url", type=str, default="",
        help="vLLM URL for target model (e.g., http://localhost:8240/v1). "
             "If empty and --target_model_id is set, auto-launches on Slurm.",
    )
    parser.add_argument(
        "--vllm_port", type=int, default=8240,
        help="Port for auto-launched vLLM server (default: 8240)",
    )
    parser.add_argument(
        "--vllm_gpu", type=int, default=1,
        help="Number of GPUs for auto-launched vLLM (default: 1)",
    )
    parser.add_argument(
        "--vllm_max_model_len", type=int, default=8192,
        help="Max model context length for vLLM (default: 8192)",
    )

    # Interaction mode filter
    parser.add_argument(
        "--interaction_modes", type=str, default="static",
        help="Comma-separated interaction modes to process (default: static). "
             "Use 'all' to process all modes, or 'multi_turn,tool_use' etc.",
    )

    # Contrastive pairs
    parser.add_argument(
        "--no_contrastive_pairs", action="store_true",
        help="Disable contrastive pair generation (enabled by default when --target_model_id is set)",
    )

    # Slim mode
    parser.add_argument(
        "--slim", action="store_true",
        help="Slim corpus: fewer candidates/category, verify with n_runs>=3, "
             "no feedback loop. Cuts generation time significantly.",
    )

    # Debug / tracing
    parser.add_argument(
        "--trace_examples", type=int, default=2,
        help="Trace N random examples per dataset through all stages (HTML debug report, default: 2)",
    )
    parser.add_argument(
        "--no_discovery_trace", action="store_true",
        help="Disable HTML discovery trace output (enabled by default)",
    )

    # HuggingFace upload
    parser.add_argument(
        "--hf_repo_id", type=str, default="",
        help="HuggingFace repo ID to upload corpus to (e.g., your-org/prompt-attribution-data). "
             "If set, uploads training_data.jsonl + corpus_stats.json after completion.",
    )

    return parser.parse_args()


# =============================================================================
# vLLM Slurm lifecycle
# =============================================================================


def _launch_vllm_on_slurm(
    model_id: str,
    port: int = 8240,
    n_gpu: int = 1,
    max_model_len: int = 8192,
) -> tuple[int, str]:
    """Launch a vLLM server on Slurm and wait for it to be ready.

    Cluster-specific Slurm flags (partition, qos) are read from environment
    variables so this works against any cluster:

      SLURM_PARTITION  default: "default"
      SLURM_QOS        default: "normal"

    Returns:
        (slurm_job_id, vllm_url)
    """
    import os
    import subprocess

    sanitized = model_id.replace("/", "_").replace("-", "_")[:30]
    job_name = f"vllm_{sanitized}_corpus"
    partition = os.environ.get("SLURM_PARTITION", "default")
    qos = os.environ.get("SLURM_QOS", "normal")

    cmd = [
        "srun",
        f"--partition={partition}",
        f"--qos={qos}",
        f"--gres=gpu:{n_gpu}",
        "--cpus-per-task=8",
        "--mem=32G",
        "--time=12:00:00",
        f"--job-name={job_name}",
        "--",
        "vllm", "serve", model_id,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--disable-log-requests",
    ]
    if n_gpu > 1:
        cmd.extend(["--tensor-parallel-size", str(n_gpu)])

    logger.info(f"Launching vLLM on Slurm: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for the server to be ready by reading output lines
    # srun allocates a node, then vLLM starts. We need the node hostname.
    node_hostname = None
    import time as _time
    start = _time.time()
    timeout = 1200  # 20 min for Slurm allocation + vLLM startup (TP=4 is slow)

    # Read srun output until we get the job allocation, then stop
    # reading stdout so we can poll health instead.
    while _time.time() - start < 120:  # 2 min for allocation
        if proc.poll() is not None:
            remaining = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(
                f"vLLM srun exited with code {proc.returncode}. "
                f"Output: {remaining[-500:]}"
            )

        line = proc.stdout.readline() if proc.stdout else ""
        if not line:
            _time.sleep(1)
            continue

        line = line.strip()
        if line:
            logger.info(f"  [vllm] {line}")

        # Once allocated, stop reading stdout — vLLM will log a lot
        if "has been allocated resources" in line:
            break

    # Find the Slurm job ID
    result = subprocess.run(
        ["squeue", "-u", subprocess.check_output(["whoami"]).decode().strip(),
         "--name", job_name, "--noheader", "-o", "%i %N"],
        capture_output=True, text=True,
    )
    lines = result.stdout.strip().split("\n")
    if not lines or not lines[0].strip():
        raise RuntimeError(
            f"Could not find Slurm job '{job_name}'. "
            f"Check squeue manually."
        )

    parts = lines[0].strip().split()
    job_id = int(parts[0])
    node = parts[1] if len(parts) > 1 else "localhost"
    vllm_url = f"http://{node}:{port}/v1"

    logger.info(f"vLLM job {job_id} on {node}, URL: {vllm_url}")

    # Poll health endpoint until ready
    import httpx
    health_url = f"http://{node}:{port}/health"
    while _time.time() - start < timeout:
        try:
            resp = httpx.get(health_url, timeout=5)
            if resp.status_code == 200:
                logger.info(f"vLLM server ready at {vllm_url}")
                return job_id, vllm_url
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        _time.sleep(5)

    raise TimeoutError(
        f"vLLM server not ready after {timeout}s. "
        f"Job {job_id} on {node}, port {port}"
    )


def _kill_slurm_job(job_id: int) -> None:
    """Cancel a Slurm job."""
    import subprocess
    logger.info(f"Cancelling Slurm job {job_id}")
    subprocess.run(["scancel", str(job_id)], check=False)


# =============================================================================
# HuggingFace upload
# =============================================================================


def _upload_to_hf(corpus_dir: Path, repo_id: str) -> None:
    """Upload corpus outputs to HuggingFace Hub.

    Uploads training_data.jsonl, corpus_stats.json, corpus_config.json,
    and progress.json to the specified dataset repo.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        logger.info(f"Uploading corpus to HuggingFace: {repo_id}")

        # Create repo if it doesn't exist
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

        # Upload key files
        files_to_upload = [
            "training_data.jsonl",
            "corpus_stats.json",
            "corpus_config.json",
            "progress.json",
            "training_data_review.csv",
        ]
        for fname in files_to_upload:
            fpath = corpus_dir / fname
            if fpath.exists() and fpath.stat().st_size > 0:
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fname,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                logger.info(f"  Uploaded {fname} ({fpath.stat().st_size:,} bytes)")

        logger.info(f"Upload complete: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        logger.error(f"HuggingFace upload failed: {e}")
        logger.error("Corpus is saved locally — upload manually later.")


# =============================================================================
# Dataset filters
# =============================================================================


# Datasets requiring non-text input (images, audio) — filter these out.
_IMAGE_PATTERNS = [
    "mathvista", "mathvision", "mathverse", "zerobench", "figureqa", "vikhyatk",
    "clinical-narrative-image", "clinical-cross-modal", "unidoc",
    "macbench", "gqa", "vqa", "visual", "diagram", "chart",
    "captioning", "ocr",
    # Confirmed image-dependent via question sampling
    "mmmu", "mmstar", "weatherqa", "internscience/sfe",
    "fineweb-edu", "eai-taxonomy-code",  # web crawl with embedded image refs
    "clarusc64/clinical",  # niche clinical datasets with nonsensical labels
    "mmlu-medical-cot",  # CoT distillation data, choices stripped out
    "browsecomp-plus-corpus",  # retrieval corpus with placeholder stubs, no real content
    "eleutherai/race", "ehovy/race",  # multi-question-per-row nested JSON, can't extract
    "jam-alt",  # raw lyrics corpus, no question or ground truth
]
_AUDIO_PATTERNS = ["speecheval", "voiceassistant", "asr", "tts", "spoken"]

_FILTER_PATTERNS = _IMAGE_PATTERNS + _AUDIO_PATTERNS

# Max dataset size — datasets above this try to download too much data
# and hang. We only need 10 samples, but load_dataset downloads everything.
_MAX_DATASET_SIZE = 200_000


def load_datasets(cache_path: str, n_samples: int = 10) -> list[dict]:
    """Load dataset entries from cached datasets.jsonl.

    Filters out image/audio-dependent datasets. For tiny datasets
    (n_total < n_samples), caps n_samples to n_total.
    """
    datasets = []
    skipped = 0
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if not entry.get("suitable", True):
                continue

            did_lower = entry["dataset_id"].lower()
            tags = [t.lower() for t in entry.get("capability_tags", [])]
            n_total = entry.get("profile", {}).get("n_total", 0)

            # Filter: image/audio modality, or too large to download
            if (any(p in did_lower for p in _FILTER_PATTERNS)
                    or "multimodal" in tags
                    or n_total > _MAX_DATASET_SIZE):
                skipped += 1
                continue

            datasets.append(entry)

    if skipped:
        logger.info(
            f"Filtered out {skipped} image/audio datasets. "
            f"Keeping {len(datasets)}."
        )

    # Interleave by capability tag for diversity from the start
    # (so training_data.jsonl isn't all-math then all-safety, etc.)
    datasets = _interleave_by_category(datasets)

    return datasets


def _interleave_by_category(datasets: list[dict]) -> list[dict]:
    """Reorder datasets round-robin across capability categories.

    E.g., [math, knowledge, safety, code, math, knowledge, safety, ...]
    so the incremental JSONL has diversity from the first rows.
    """
    from collections import defaultdict

    buckets: dict[str, list[dict]] = defaultdict(list)
    for d in datasets:
        tags = d.get("capability_tags", ["other"])
        primary = tags[0] if tags else "other"
        buckets[primary].append(d)

    # Round-robin across buckets
    result = []
    bucket_iters = {k: iter(v) for k, v in buckets.items()}
    while bucket_iters:
        exhausted = []
        for key in list(bucket_iters.keys()):
            try:
                result.append(next(bucket_iters[key]))
            except StopIteration:
                exhausted.append(key)
        for key in exhausted:
            del bucket_iters[key]

    return result


def sanitize_dataset_id(dataset_id: str) -> str:
    """Convert HF dataset ID to filesystem-safe directory name.

    Examples:
        your-user/dataset-name → your-user__dataset-name
        Idavidrein/gpqa:gpqa_diamond → Idavidrein__gpqa__gpqa_diamond
    """
    return dataset_id.replace("/", "__").replace(":", "__")


def make_unique_id(dataset_id: str, example_idx: int, perturbation_id: str) -> str:
    """Create globally unique row ID as a short hash.

    Hash is deterministic: same inputs always produce the same ID,
    so rows can be reliably backtracked to their source.
    """
    raw = f"{dataset_id}::{example_idx}::{perturbation_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def append_to_shared_outputs(
    per_dataset_jsonl: Path,
    shared_jsonl: Path,
    shared_csv: Path,
    dataset_id: str,
) -> int:
    """Read per-dataset training_data.jsonl, add unique_id, append to shared files.

    Returns number of rows appended.
    """
    if not per_dataset_jsonl.exists():
        return 0

    rows_appended = 0
    rows_skipped_bl_eq_lv = 0
    with (
        open(per_dataset_jsonl) as f_in,
        open(shared_jsonl, "a") as f_jsonl,
        open(shared_csv, "a", newline="") as f_csv,
    ):
        csv_writer = csv.writer(f_csv, quoting=csv.QUOTE_ALL)
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            # Skip rows where edit failed to apply (baseline == lever)
            if row.get("prompt_baseline") == row.get("prompt_lever"):
                rows_skipped_bl_eq_lv += 1
                continue

            # Add unique_id
            uid = make_unique_id(
                dataset_id,
                row.get("example_idx", 0),
                row.get("perturbation_id", ""),
            )
            row["unique_id"] = uid

            # Append to shared JSONL
            f_jsonl.write(json.dumps(row) + "\n")

            # Append to shared CSV (3 columns)
            csv_writer.writerow([
                uid,
                row.get("prompt_baseline", ""),
                row.get("prompt_lever", ""),
            ])

            rows_appended += 1

    if rows_skipped_bl_eq_lv:
        logger.info(
            f"  Skipped {rows_skipped_bl_eq_lv} rows where baseline==lever "
            f"(edit failed to apply)"
        )

    return rows_appended


def update_progress(
    progress_path: Path,
    dataset_id: str,
    status: str,
    n_examples: int = 0,
    error: str = "",
    duration_s: float = 0.0,
    **kwargs,
):
    """Update progress.json with dataset status.

    Extra kwargs (interaction_mode, context_source) are stored for
    resumability — when re-running later with different interaction
    modes, we can skip datasets already processed.
    """
    progress = {}
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)

    progress[dataset_id] = {
        "status": status,
        "n_examples": n_examples,
        "error": error[:500] if error else "",
        "duration_s": round(duration_s, 1),
        "timestamp": datetime.now().isoformat(),
        "interaction_mode": kwargs.get("interaction_mode", "static"),
        "context_source": kwargs.get("context_source", "single_source"),
    }

    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def get_completed_datasets(progress_path: Path) -> set[str]:
    """Get set of dataset IDs that completed successfully."""
    if not progress_path.exists():
        return set()
    with open(progress_path) as f:
        progress = json.load(f)
    return {
        did for did, info in progress.items()
        if info.get("status") == "success"
    }


async def run_verify_only(args):
    """Run verification on an existing corpus (no generation)."""
    from .config import PipelineConfig
    from .pipeline import TrainingDataPipeline

    setup_environment()

    corpus_dir = Path(args.output_dir)
    if not corpus_dir.exists():
        raise ValueError(f"Corpus dir not found: {corpus_dir}")

    from prompt_attribution.shared.inference.api_factory import get_regular_api
    api = get_regular_api(concurrency=args.concurrency)

    progress_path = corpus_dir / "progress.json"
    shared_jsonl = corpus_dir / "training_data.jsonl"
    shared_csv = corpus_dir / "training_data_review.csv"

    # Find all dataset subdirs with critic_output.json (completed generation)
    dataset_dirs = sorted([
        d for d in corpus_dir.iterdir()
        if d.is_dir() and (d / "critic_output.json").exists()
    ])
    logger.info(f"Found {len(dataset_dirs)} dataset dirs to verify")

    # Rebuild shared JSONL/CSV from scratch (with verification results)
    if shared_jsonl.exists():
        shared_jsonl.rename(shared_jsonl.with_suffix(".jsonl.bak"))
    if shared_csv.exists():
        shared_csv.rename(shared_csv.with_suffix(".csv.bak"))

    with open(shared_csv, "w", newline="") as f:
        csv.writer(f, quoting=csv.QUOTE_ALL).writerow([
            "unique_id", "prompt_baseline", "prompt_lever",
        ])

    n_success = 0
    n_failed = 0
    total_examples = 0

    async def verify_one(i: int, run_dir: Path):
        nonlocal n_success, n_failed, total_examples

        # Load config to reconstruct pipeline
        config_path = run_dir / "config.json"
        if not config_path.exists():
            logger.warning(f"No config.json in {run_dir}, skipping")
            return

        with open(config_path) as f:
            config_dict = json.load(f)

        dataset_id = config_dict.get("dataset_id", run_dir.name)
        logger.info(f"[{i+1}/{len(dataset_dirs)}] Verifying: {dataset_id}")

        start = time.time()
        try:
            config_kwargs = {
                k: v for k, v in config_dict.items()
                if k in PipelineConfig.__dataclass_fields__
            }
            # Fix types that don't survive JSON round-trip
            if "output_dir" in config_kwargs:
                config_kwargs["output_dir"] = Path(config_kwargs["output_dir"])
            config = PipelineConfig(**config_kwargs)
            config.judge_model = args.judge_model
            config.stability_n_runs = args.stability_n_runs
            config.concurrency = args.concurrency
            config.verification_concurrency = args.verification_concurrency

            pipeline = TrainingDataPipeline(config, api)
            # Run full pipeline — it will load cached intermediates
            # (examples, decompose, generation, critic) from run_dir
            # and only actually execute verify + export
            result = await pipeline.run(run_dir=run_dir)

            # Append to shared outputs
            per_dataset_jsonl = run_dir / "training_data.jsonl"
            n_rows = append_to_shared_outputs(
                per_dataset_jsonl, shared_jsonl, shared_csv, dataset_id,
            )

            duration = time.time() - start
            total_examples += n_rows
            n_success += 1
            logger.info(
                f"[{i+1}/{len(dataset_dirs)}] VERIFIED: {dataset_id} — "
                f"{n_rows} examples in {duration:.1f}s"
            )

        except Exception as e:
            duration = time.time() - start
            n_failed += 1
            logger.error(
                f"[{i+1}/{len(dataset_dirs)}] FAILED: {dataset_id} — "
                f"{type(e).__name__}: {e}"
            )

    # Run in parallel batches
    par = args.parallel_datasets
    for batch_start in range(0, len(dataset_dirs), par):
        batch = list(enumerate(dataset_dirs))[batch_start:batch_start + par]
        await asyncio.gather(*[verify_one(i, d) for i, d in batch])

    logger.info(
        f"\nVERIFICATION COMPLETE: {n_success} verified, {n_failed} failed, "
        f"{total_examples} total examples"
    )


async def run_corpus(args):
    """Main corpus generation loop."""
    from .config import PipelineConfig
    from .pipeline import TrainingDataPipeline

    setup_environment()

    if args.verify_only:
        return await run_verify_only(args)

    # Load datasets
    datasets = load_datasets(args.datasets_cache)
    logger.info(f"Loaded {len(datasets)} datasets from cache")

    # Filter by complexity tags — matches profiling_trace.html rules:
    #   skip if context_source == "multimodal_context"
    #   skip if interaction_mode not in allowed modes (default: static only)
    #   skip if context_length == "long"
    allowed_modes = set(args.interaction_modes.split(","))
    if "all" not in allowed_modes:
        before = len(datasets)
        datasets = [
            d for d in datasets
            if (d.get("complexity", {}).get("interaction_mode", "static") in allowed_modes
                and d.get("complexity", {}).get("context_source", "single_source") != "multimodal_context"
                and d.get("complexity", {}).get("context_length", "short") != "long")
        ]
        n_filtered = before - len(datasets)
        if n_filtered:
            logger.info(
                f"Filtered out {n_filtered} datasets (non-static, multimodal, "
                f"or long context). Keeping {len(datasets)}."
            )

    if args.max_datasets:
        datasets = datasets[:args.max_datasets]
        logger.info(f"Limited to {len(datasets)} datasets (--max_datasets)")

    # Setup output directory
    if args.output_dir:
        corpus_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        corpus_dir = Path("outputs/auto_perturbation") / f"corpus_{timestamp}"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Clear adapter cache to force fresh LLM detection for all datasets
    # (old heuristic profiles may have wrong field mappings)
    adapter_cache = corpus_dir / ".adapter_cache"
    if adapter_cache.exists() and not args.skip_completed:
        import shutil
        shutil.rmtree(adapter_cache)
        logger.info("Cleared adapter cache for fresh LLM detection")

    # Shared output paths
    shared_jsonl = corpus_dir / "training_data.jsonl"
    shared_csv = corpus_dir / "training_data_review.csv"
    progress_path = corpus_dir / "progress.json"
    errors_log = corpus_dir / "errors.log"

    # Initialize CSV header if new file
    if not shared_csv.exists() or shared_csv.stat().st_size == 0:
        with open(shared_csv, "w", newline="") as f:
            csv.writer(f, quoting=csv.QUOTE_ALL).writerow([
                "unique_id", "prompt_baseline", "prompt_lever",
            ])

    # Set up discovery tracer (enabled by default)
    discovery_tracer = None
    if not args.no_discovery_trace:
        from .discovery_tracer import DiscoveryTracer
        discovery_tracer = DiscoveryTracer()
        discovery_tracer.record_profiling_config(
            datasets_cache=args.datasets_cache,
            n_datasets=len(datasets),
            n_samples=args.n_samples,
        )

    # Save corpus config
    corpus_config = {
        "datasets_cache": args.datasets_cache,
        "n_datasets": len(datasets),
        "n_samples": args.n_samples,
        "concurrency": args.concurrency,
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "target_model_id": args.target_model_id,
        "stability_n_runs": args.stability_n_runs,
        "started_at": datetime.now().isoformat(),
    }
    with open(corpus_dir / "corpus_config.json", "w") as f:
        json.dump(corpus_config, f, indent=2)

    # Check for already-completed datasets
    completed = set()
    if args.skip_completed:
        completed = get_completed_datasets(progress_path)
        if completed:
            logger.info(f"Skipping {len(completed)} already-completed datasets")

    # Create shared API instance (reused across all datasets).
    # Caching is enabled at the API level for non-stability calls (decompose,
    # generate, critic). Stability verification uses per-call no_cache=True
    # on TargetModelClient to get independent responses for identical prompts.
    from prompt_attribution.shared.inference.api_factory import get_regular_api
    api = get_regular_api(concurrency=args.concurrency)

    # Auto-launch vLLM on Slurm if target_model_id is set but no URL provided.
    # The server persists for the entire corpus run and is killed in the finally block.
    vllm_slurm_job_id = None
    is_api_model = any(
        args.target_model_id.startswith(p)
        for p in ("claude-", "gpt-", "o1-", "o3-", "o4-")
    )
    if args.target_model_id and not args.target_model_url and not is_api_model:
        logger.info(
            f"Auto-launching vLLM for {args.target_model_id} on Slurm "
            f"(port {args.vllm_port}, {args.vllm_gpu} GPU(s))"
        )
        vllm_slurm_job_id, vllm_url = _launch_vllm_on_slurm(
            model_id=args.target_model_id,
            port=args.vllm_port,
            n_gpu=args.vllm_gpu,
            max_model_len=args.vllm_max_model_len,
        )
        args.target_model_url = vllm_url
        logger.info(f"vLLM auto-launched: job {vllm_slurm_job_id}, URL {vllm_url}")
    elif is_api_model:
        logger.info(
            f"Target model {args.target_model_id} is an API model — "
            f"skipping vLLM launch, using safetytooling InferenceAPI"
        )

    # LLM capability classification for all datasets
    # Keyword-based tagging is unreliable — LLM reads samples + description
    # for accurate classification. One cheap Haiku call per dataset.
    # Results are saved back to the cache so next run skips re-classification.
    from .dataset_discovery.profile_datasets import DatasetDiscovery
    discovery = DatasetDiscovery(tracer=discovery_tracer)

    # Check if all datasets already have LLM-classified tags (from previous run)
    n_needs_classify = sum(
        1 for d in datasets
        if not d.get("_llm_classified")
    )
    if n_needs_classify > 0:
        logger.info(
            f"Running LLM capability classification on {n_needs_classify} "
            f"datasets ({len(datasets) - n_needs_classify} already classified)"
        )
        datasets = await discovery.refine_tags_with_llm(
            datasets, api, force_all=True,
        )
        # Mark as classified and persist back to cache
        for d in datasets:
            d["_llm_classified"] = True
        cache_path = Path(args.datasets_cache)
        if cache_path.exists():
            try:
                with open(cache_path, "w") as f:
                    for d in datasets:
                        f.write(json.dumps(d) + "\n")
                logger.info(f"Saved updated capability tags to {cache_path}")
            except PermissionError:
                logger.info(f"Skipping writeback to {cache_path} (read-only, protected)")
    else:
        logger.info("All datasets already have LLM-classified capability tags, skipping")

    skip_verify = args.skip_verify

    # Corpus-level counters (use a lock for thread-safe updates)
    import threading
    _lock = threading.Lock()
    total_examples = 0
    n_success = 0
    n_failed = 0
    corpus_start = time.time()

    async def process_one_dataset(i: int, entry: dict):
        """Process a single dataset through the pipeline."""
        nonlocal total_examples, n_success, n_failed

        dataset_id = entry["dataset_id"]
        capability_tags = entry.get("capability_tags", [])
        profile = entry.get("profile")

        # Skip if already completed
        if dataset_id in completed:
            logger.info(f"[{i+1}/{len(datasets)}] SKIP (completed): {dataset_id}")
            return

        logger.info(
            f"\n{'='*60}\n"
            f"[{i+1}/{len(datasets)}] Starting: {dataset_id}\n"
            f"  task_type={entry.get('task_type')}, "
            f"tags={capability_tags}\n"
            f"{'='*60}"
        )

        update_progress(progress_path, dataset_id, "running")

        dataset_start = time.time()
        run_dir = corpus_dir / sanitize_dataset_id(dataset_id)

        try:
            n_total_ds = (profile or {}).get("n_total", 9999)
            effective_n_samples = min(args.n_samples, n_total_ds)
            if effective_n_samples <= 0:
                logger.warning(f"Skipping {dataset_id}: n_total={n_total_ds}, no samples")
                update_progress(
                    progress_path, dataset_id, "skipped",
                    error=f"n_total={n_total_ds}", duration_s=0,
                )
                return

            # Slim mode: 1 candidate per category, n_runs=3, no feedback
            slim = getattr(args, "slim", False)
            if slim:
                from .config import CategoryConfig
                cat_configs = {
                    "flip_inducing": CategoryConfig(n_to_generate=1, temperature=1.0),
                    "non_flip": CategoryConfig(n_to_generate=1, temperature=1.0),
                    "boundary": CategoryConfig(n_to_generate=1, temperature=1.0),
                }
                stability = max(args.stability_n_runs, 3)
            else:
                cat_configs = None
                stability = args.stability_n_runs

            config_kwargs: dict = dict(
                generator_model=args.generator_model,
                judge_model=args.judge_model,
                n_samples=effective_n_samples,
                stability_n_runs=stability,
                concurrency=args.concurrency,
                verification_concurrency=args.verification_concurrency,
                dataset_id=dataset_id,
                dataset_profile=profile,
                capability_tags=capability_tags,
                difficulty_tier="saturated",
                enable_feedback_loop=not args.no_feedback_loop if not slim else False,
                feedback_max_rounds=args.feedback_max_rounds,
                target_model_id=args.target_model_id,
                target_model_url=args.target_model_url,
                enable_contrastive_pairs=not args.no_contrastive_pairs if not slim else False,
                output_dir=corpus_dir,
                dump_prompts=args.dump_prompts,
                trace_examples=args.trace_examples,
                slim_mode=slim,
            )
            if cat_configs:
                config_kwargs["category_configs"] = cat_configs

            config = PipelineConfig(**config_kwargs)

            pipeline = TrainingDataPipeline(config, api, discovery_tracer=discovery_tracer)
            result = await pipeline.run(
                run_dir=run_dir,
                skip_verify=skip_verify,
            )

            # Skip if LLM detected non-text modality
            if result.get("skipped"):
                duration = time.time() - dataset_start
                update_progress(
                    progress_path, dataset_id, "skipped",
                    error=result.get("reason", ""), duration_s=duration,
                )
                logger.info(
                    f"[{i+1}/{len(datasets)}] SKIPPED: {dataset_id} — "
                    f"{result.get('reason', 'unknown')}"
                )
                return

            # Append to shared outputs
            per_dataset_jsonl = run_dir / "training_data.jsonl"
            n_rows = append_to_shared_outputs(
                per_dataset_jsonl, shared_jsonl, shared_csv, dataset_id,
            )

            duration = time.time() - dataset_start
            with _lock:
                total_examples += n_rows
                n_success += 1

            update_progress(
                progress_path, dataset_id, "success",
                n_examples=n_rows, duration_s=duration,
            )
            logger.info(
                f"[{i+1}/{len(datasets)}] SUCCESS: {dataset_id} — "
                f"{n_rows} examples in {duration:.1f}s "
                f"(total: {total_examples})"
            )

        except Exception as e:
            duration = time.time() - dataset_start
            with _lock:
                n_failed += 1
            error_msg = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()

            update_progress(
                progress_path, dataset_id, "failed",
                error=error_msg, duration_s=duration,
            )

            with open(errors_log, "a") as f:
                f.write(
                    f"\n{'='*60}\n"
                    f"Dataset: {dataset_id}\n"
                    f"Time: {datetime.now().isoformat()}\n"
                    f"Duration: {duration:.1f}s\n"
                    f"{'='*60}\n"
                    f"{tb}\n"
                )

            logger.error(
                f"[{i+1}/{len(datasets)}] FAILED: {dataset_id} — "
                f"{error_msg} (see errors.log)"
            )

    # Run datasets — parallel or sequential.
    # Wrapped in try/finally to ensure auto-launched vLLM server is killed.
    try:
        pending = [
            (i, entry) for i, entry in enumerate(datasets)
            if entry["dataset_id"] not in completed
        ]

        if args.parallel_datasets > 1:
            # Use semaphore so new datasets start as soon as a slot frees up
            # (unlike batch-and-wait which blocks on the slowest dataset)
            sem = asyncio.Semaphore(args.parallel_datasets)
            async def _limited(i, entry):
                async with sem:
                    await process_one_dataset(i, entry)
            await asyncio.gather(
                *[_limited(i, entry) for i, entry in pending]
            )
        else:
            for i, entry in pending:
                await process_one_dataset(i, entry)

        # Write final corpus stats
        corpus_duration = time.time() - corpus_start
        corpus_stats = {
            "total_examples": total_examples,
            "n_datasets_attempted": len(datasets) - len(completed),
            "n_success": n_success,
            "n_failed": n_failed,
            "n_skipped": len(completed),
            "duration_s": round(corpus_duration, 1),
            "duration_human": f"{corpus_duration/3600:.1f} hours",
            "generator_model": args.generator_model,
            "judge_model": args.judge_model,
            "n_samples_per_dataset": args.n_samples,
            "completed_at": datetime.now().isoformat(),
        }
        with open(corpus_dir / "corpus_stats.json", "w") as f:
            json.dump(corpus_stats, f, indent=2)

        # Render discovery trace
        if discovery_tracer:
            if discovery_tracer._model_traces:
                trace_path = corpus_dir / "research_trace.html"
                discovery_tracer.render_research_html(trace_path)
                logger.info(f"Research trace: {trace_path}")
            if discovery_tracer._dataset_traces:
                trace_path = corpus_dir / "profiling_trace.html"
                discovery_tracer.render_profiling_html(trace_path)
                logger.info(f"Profiling trace: {trace_path}")

        logger.info(
            f"\n{'='*60}\n"
            f"CORPUS GENERATION COMPLETE\n"
            f"  Total examples: {total_examples}\n"
            f"  Datasets: {n_success} success, {n_failed} failed\n"
            f"  Duration: {corpus_duration/3600:.1f} hours\n"
            f"  Output: {corpus_dir}\n"
            f"{'='*60}"
        )

        # Upload to HuggingFace if repo ID is set
        if args.hf_repo_id:
            _upload_to_hf(corpus_dir, args.hf_repo_id)

    finally:
        # Kill auto-launched vLLM server
        if vllm_slurm_job_id is not None:
            logger.info(
                f"Corpus run finished — killing auto-launched vLLM "
                f"(Slurm job {vllm_slurm_job_id})"
            )
            _kill_slurm_job(vllm_slurm_job_id)


def main():
    args = parse_args()

    # Colored logging: green=INFO, yellow=WARNING, red=ERROR
    class ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: "\033[90m",     # gray
            logging.INFO: "\033[32m",      # green
            logging.WARNING: "\033[33m",   # yellow
            logging.ERROR: "\033[31m",     # red
            logging.CRITICAL: "\033[1;31m",  # bold red
        }
        RESET = "\033[0m"

        def format(self, record):
            color = self.COLORS.get(record.levelno, "")
            record.levelname = f"{color}{record.levelname}{self.RESET}"
            record.msg = f"{color}{record.msg}{self.RESET}"
            return super().format(record)

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)

    asyncio.run(run_corpus(args))


if __name__ == "__main__":
    main()
