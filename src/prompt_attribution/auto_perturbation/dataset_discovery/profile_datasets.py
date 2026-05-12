"""
Module: prompt_attribution/auto_perturbation/discovery/profile_datasets.py

Profiles and validates HuggingFace datasets for the perturbation pipeline.
Takes seed dataset IDs (from research_agent or known_benchmarks) and checks
each one for suitability: field detection, task type, text-only, etc.

Usage:
    python -m prompt_attribution.auto_perturbation.dataset_discovery.profile_datasets \\
        --frontier_seeds frontier_seeds.jsonl --output datasets.jsonl

    python -m prompt_attribution.auto_perturbation.dataset_discovery.profile_datasets \\
        --output datasets.jsonl --refresh

Structure:
- CapabilityDomain: Enum of capability categories for diversity tracking
- SuitabilityResult: Result of suitability check for a dataset
- DatasetDiscovery: Orchestrates search + profiling + filtering
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from huggingface_hub import HfApi

from ..dataset_adapter.dataset_adapter import DatasetDetector, DatasetProfile, TaskType
from .known_benchmarks import KNOWN_BENCHMARK_MAP

if TYPE_CHECKING:
    from ..discovery_tracer import DiscoveryTracer

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Domains for Diversity
# =============================================================================


class CapabilityDomain(str, Enum):
    """Capability categories for ensuring dataset diversity."""

    MATH_REASONING = "math_reasoning"
    LOGICAL_REASONING = "logical_reasoning"
    KNOWLEDGE_QA = "knowledge_qa"
    READING_COMPREHENSION = "reading_comprehension"
    CODE = "code"
    SAFETY_ETHICS = "safety_ethics"
    CLASSIFICATION = "classification"
    COMMONSENSE = "commonsense"
    SCIENCE = "science"
    LANGUAGE_UNDERSTANDING = "language_understanding"
    TEXT_GENERATION = "text_generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    DIALOG = "dialog"
    INSTRUCTION_FOLLOWING = "instruction_following"
    DOMAIN_SPECIFIC = "domain_specific"


# Keywords to detect capability domain from dataset metadata
_CAPABILITY_KEYWORDS = {
    CapabilityDomain.MATH_REASONING: [
        "math", "arithmetic", "algebra", "geometry", "calculus",
        "gsm", "grade school", "numerical", "computation",
    ],
    CapabilityDomain.LOGICAL_REASONING: [
        "logic", "reasoning", "deduction", "inference", "entailment",
        "nli", "natural language inference", "syllogism",
    ],
    CapabilityDomain.KNOWLEDGE_QA: [
        "trivia", "knowledge", "factual", "encyclopedia", "quiz",
        "history", "geography", "general knowledge",
    ],
    CapabilityDomain.READING_COMPREHENSION: [
        "reading comprehension", "passage", "context", "extractive",
        "squad", "comprehension", "document",
    ],
    CapabilityDomain.CODE: [
        "code", "programming", "python", "java", "coding",
        "software", "algorithm", "function", "humaneval",
    ],
    CapabilityDomain.SAFETY_ETHICS: [
        "safety", "toxic", "toxigen", "harmful", "bias", "fairness",
        "ethics", "jailbreak", "refusal", "adversarial", "beaver",
        "harmbench", "do-not-answer", "xstest", "civil_comment",
        "hate", "offensive", "abuse", "danger", "alignment",
    ],
    CapabilityDomain.CLASSIFICATION: [
        "sentiment", "classification", "topic", "categoriz",
        "detect", "spam", "hate speech",
    ],
    CapabilityDomain.COMMONSENSE: [
        "commonsense", "common sense", "winograd", "physical",
        "social", "everyday", "intuition",
    ],
    CapabilityDomain.SCIENCE: [
        "science", "physics", "chemistry", "biology", "medical",
        "scientific", "experiment",
    ],
    CapabilityDomain.LANGUAGE_UNDERSTANDING: [
        "grammar", "linguistic", "paraphrase", "semantic",
        "coreference", "word sense", "disambiguation",
        "nli", "mnli", "rte", "cola", "glue", "super_glue",
        "boolq", "wic", "paws", "multirc", "entailment",
    ],
    CapabilityDomain.TEXT_GENERATION: [
        "generation", "story", "creative", "writing", "completion",
        "language model", "text generation", "tinystories",
    ],
    CapabilityDomain.SUMMARIZATION: [
        "summary", "summarization", "abstractive", "extractive",
        "tldr", "headline", "cnn_dailymail", "xsum",
    ],
    CapabilityDomain.TRANSLATION: [
        "translation", "translate", "nmt", "parallel corpus",
        "bilingual", "wmt", "opus",
    ],
    CapabilityDomain.DIALOG: [
        "dialog", "dialogue", "conversation", "chat", "counseling",
        "customer service", "persona",
    ],
    CapabilityDomain.INSTRUCTION_FOLLOWING: [
        "instruction following", "ifeval", "constraint",
        "format following",
    ],
    CapabilityDomain.DOMAIN_SPECIFIC: [
        "medical", "clinical", "legal", "finance", "financial",
        "pharmaceutical", "biomedical", "pubmed",
    ],
}

def _get_curated_seeds(min_tier: str = "moderate") -> list[str]:
    """Build curated seed list from KNOWN_BENCHMARK_MAP, filtered by difficulty.

    Single source of truth: all seeds come from known_benchmarks.py.
    By default skips saturated benchmarks (declared in _EVAL_RESERVED below)
    since frontier LLMs score >95% on them — perturbations won't flip.

    Args:
        min_tier: Minimum difficulty tier to include.
            "frontier" = only hardest, "moderate" = include moderate (default),
            "saturated" = include everything.

    Returns:
        List of HuggingFace dataset IDs (deduplicated).
    """
    tier_order = {"frontier": 2, "moderate": 1, "saturated": 0}
    min_level = tier_order.get(min_tier, 1)

    seeds = []
    seen_hf_ids = set()
    for _name, info in KNOWN_BENCHMARK_MAP.items():
        level = tier_order.get(info.get("tier", "moderate"), 1)
        if level >= min_level:
            hf_id = info["hf_id"]
            if hf_id not in seen_hf_ids:
                seeds.append(hf_id)
                seen_hf_ids.add(hf_id)

    return seeds


# =============================================================================
# Suitability Check
# =============================================================================


@dataclass
class SuitabilityResult:
    """Result of checking whether a dataset is suitable for perturbation generation.

    Attributes:
        suitable: Whether the dataset passes all checks
        reasons: Why the dataset was rejected (if not suitable)
        capability_tags: Detected capability domains
        profile: The auto-detected dataset profile (if detection succeeded)
    """

    suitable: bool
    reasons: list[str] = field(default_factory=list)
    capability_tags: list[str] = field(default_factory=list)
    profile: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Dataset Discovery
# =============================================================================


class DatasetDiscovery:
    """Orchestrates HuggingFace dataset search, profiling, and filtering.

    Attributes:
        detector: DatasetDetector for auto-profiling
        cache_dir: Directory for caching discovery results
        cache_ttl_days: How long cached results are valid
    """

    def __init__(
        self,
        cache_dir: Path = Path("outputs/auto_perturbation/.discovery_cache"),
        cache_ttl_days: int = 7,
        tracer: Optional["DiscoveryTracer"] = None,
    ):
        self.detector = DatasetDetector(cache_dir=cache_dir / "profiles")
        self.cache_dir = cache_dir
        self.cache_ttl_days = cache_ttl_days
        self._tracer = tracer

    def discover(
        self,
        min_downloads: int = 1000,
        max_datasets: int = 200,
        include_curated: bool = True,
        refresh: bool = False,
        extra_seeds: list[str] | None = None,
        search_hub: bool = False,
        seed_categories: dict[str, str] | None = None,
    ) -> list[dict]:
        """Discover suitable datasets from HuggingFace Hub.

        Args:
            min_downloads: Minimum download count threshold
            max_datasets: Maximum datasets to return
            include_curated: Whether to include curated seed datasets
            refresh: Force refresh, ignoring cache
            extra_seeds: Additional dataset IDs to include (e.g., from
                research agent). Format: "dataset_id" or "dataset_id:config".
            search_hub: Whether to search HF Hub for additional datasets.
                Off by default — the research agent is more effective.
                Enable with --search_hub for finding niche datasets.
            seed_categories: LLM-generated category mapping from research agent.
                Maps base HF ID → category (e.g., "safety", "code", "reasoning").
                Used by _detect_capabilities for higher-quality tagging.

        Returns:
            List of dataset entries with profile + suitability info
        """
        # Check cache
        cache_path = self.cache_dir / "datasets.jsonl"

        # Load existing results for resume (skip already-profiled datasets)
        already_profiled: set[str] = set()
        existing_results: list[dict] = []
        if not refresh and cache_path.exists():
            existing_results = self._load_cache(cache_path)
            already_profiled = {r["dataset_id"] for r in existing_results}
            logger.info(
                f"Loaded {len(existing_results)} already-profiled datasets from cache "
                f"(will skip these, profile only new ones)"
            )

        # Clear adapter profile cache on refresh (stale profiles cause wrong detections)
        if refresh:
            profiles_dir = self.cache_dir / "profiles"
            if profiles_dir.exists():
                import shutil
                shutil.rmtree(profiles_dir)
                logger.info("Cleared adapter profile cache for fresh detection")

        logger.info("Starting dataset discovery from HuggingFace Hub")

        if self._tracer:
            self._tracer.record_profiling_config(
                cache_dir=str(self.cache_dir),
                cache_ttl_days=self.cache_ttl_days,
                include_curated=include_curated,
                search_hub=search_hub,
                max_datasets=max_datasets,
            )

        # Step 1: Collect candidate dataset IDs
        candidate_ids = set()

        # Curated seeds from KNOWN_BENCHMARK_MAP (skips saturated by default)
        curated_seeds = []
        if include_curated:
            curated_seeds = _get_curated_seeds(min_tier="moderate")
            candidate_ids.update(curated_seeds)
            logger.info(f"Added {len(curated_seeds)} curated seeds (>= moderate tier)")

        # Extra seeds (e.g., from research agent)
        extra_seed_list = list(extra_seeds) if extra_seeds else []
        if extra_seed_list:
            candidate_ids.update(extra_seed_list)
            logger.info(f"Added {len(extra_seed_list)} extra seed datasets")

        # Optionally search HuggingFace Hub (off by default — slow + rate limited)
        hub_datasets_found = set()
        if search_hub:
            hub_datasets_found = self._search_hub(min_downloads)
            candidate_ids.update(hub_datasets_found)
            logger.info(f"Added {len(hub_datasets_found)} from HF Hub search")

        # Store seed categories for capability detection
        self._seed_categories = seed_categories or {}

        # Remove None entries (unmapped benchmarks from research agent)
        candidate_ids.discard(None)
        logger.info(f"Total candidates: {len(candidate_ids)}")

        # Record seed sources in tracer
        if self._tracer:
            self._tracer.record_profiling_config(
                seed_sources=f"curated={len(curated_seeds)}, "
                             f"research_agent={len(extra_seed_list)}, "
                             f"hub_search={len(hub_datasets_found)}",
                total_candidates=len(candidate_ids),
            )

        # Step 2: Profile and filter each dataset
        # Process curated seeds first (higher priority), then hub results
        curated_set = set(_get_curated_seeds(min_tier="saturated"))  # all known IDs for ordering
        curated = sorted(candidate_ids & curated_set)
        hub_only = sorted(candidate_ids - curated_set)
        ordered_ids = curated + hub_only

        # Build seed source lookup for tracer
        _extra_seed_set = set(extra_seed_list)
        _hub_set = hub_datasets_found

        results = list(existing_results)  # Start with already-profiled results
        gated_datasets = []  # Track gated datasets for user action

        # Write results incrementally to a temp file, then atomic-rename
        # on completion. Prevents null-byte corruption from mid-write crashes.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".jsonl.tmp")
        cache_file = open(tmp_path, "w")
        # Write existing results first (resume mode)
        for r in results:
            cache_file.write(json.dumps(r) + "\n")
        cache_file.flush()

        n_skipped = 0
        n_profiled = 0
        for i, dataset_id in enumerate(ordered_ids):
            if len(results) >= max_datasets:
                break

            # Skip already-profiled datasets (resume after 429 / crash)
            if dataset_id in already_profiled:
                n_skipped += 1
                continue

            # Brief delay every 20 actually-profiled datasets to avoid HF rate limits
            n_profiled += 1
            if n_profiled > 1 and n_profiled % 20 == 0:
                time.sleep(2.0)

            logger.info(f"[{i+1}/{len(ordered_ids)}] Profiling {dataset_id}")
            result = self._check_suitability(dataset_id)

            # Tag seed source on tracer
            if self._tracer and self._tracer._dataset_traces:
                ds_trace = self._tracer._dataset_traces[-1]  # just added by _check_suitability
                base_id = dataset_id.split(":")[0]
                if dataset_id in _extra_seed_set or base_id in _extra_seed_set:
                    ds_trace.seed_source = "research_agent"
                elif dataset_id in _hub_set:
                    ds_trace.seed_source = "hub_search"
                    ds_trace.seed_detail = getattr(self, '_hub_search_source', {}).get(dataset_id, "")
                elif dataset_id in curated_set:
                    ds_trace.seed_source = "curated"

            # Track gated datasets
            if any("Gated dataset" in r for r in result.get("reasons", [])):
                gated_datasets.append(dataset_id)

            if result["suitable"]:
                results.append(result)
                # Write incrementally
                cache_file.write(json.dumps(result) + "\n")
                cache_file.flush()
                logger.info(
                    f"  ✓ Suitable: {result['task_type']}, "
                    f"capabilities: {result['capability_tags']}"
                )
            else:
                logger.info(f"  ✗ Rejected: {result['reasons']}")

        cache_file.close()

        if n_skipped:
            logger.info(f"Skipped {n_skipped} already-profiled datasets (resume mode)")

        # Step 3: Log diversity stats
        self._log_diversity(results)

        # Atomic rename: write final clean version, then rename to avoid corruption
        final_tmp = cache_path.with_suffix(".jsonl.final")
        with open(final_tmp, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        import shutil
        shutil.move(str(final_tmp), str(cache_path))
        logger.info(f"Saved {len(results)} datasets to {cache_path}")
        # Clean up incremental temp file
        if tmp_path.exists():
            tmp_path.unlink()

        # Report gated datasets that need user action
        if gated_datasets:
            logger.warning(
                f"\n{'='*60}\n"
                f"GATED DATASETS ({len(gated_datasets)}) — accept terms to include:\n"
                + "\n".join(
                    f"  https://huggingface.co/datasets/{ds}"
                    for ds in gated_datasets
                )
                + f"\n{'='*60}"
            )

        return results

    def _search_hub(self, min_downloads: int) -> set[str]:
        """Search HuggingFace Hub for eval-relevant datasets.

        Pre-filters out gated, private, and disabled datasets to avoid
        wasting time on datasets that will fail during profiling.
        """
        api = HfApi()
        dataset_ids = set()
        # Track which query found each dataset (for tracer)
        self._hub_search_source: dict[str, str] = {}

        def _add_if_accessible(ds, source: str) -> None:
            """Add dataset ID only if it's publicly accessible."""
            if ds.downloads and ds.downloads >= min_downloads:
                if getattr(ds, "gated", False):
                    return
                if getattr(ds, "private", False):
                    return
                if getattr(ds, "disabled", False):
                    return
                if ds.id not in dataset_ids:
                    dl = ds.downloads
                    dl_str = f"{dl/1_000_000:.1f}M" if dl >= 1_000_000 else f"{dl/1_000:.0f}K" if dl >= 1_000 else str(dl)
                    self._hub_search_source[ds.id] = f"{source}, {dl_str} downloads"
                dataset_ids.add(ds.id)

        # Search by relevant task categories
        task_categories = [
            "question-answering",
            "multiple-choice",
            "text-classification",
            "text-generation",
            "text2text-generation",
            "conversational",
            "dialogue",
        ]

        for task in task_categories:
            try:
                datasets = api.list_datasets(
                    task_categories=task,
                    sort="downloads",
                    direction=-1,
                    limit=50,
                )
                for ds in datasets:
                    _add_if_accessible(ds, f"task={task}")
            except Exception as e:
                logger.warning(f"Error searching for {task}: {e}")

        # Also search by popular eval-related tags
        eval_tags = ["benchmark", "evaluation"]
        for tag in eval_tags:
            try:
                datasets = api.list_datasets(
                    filter=tag,
                    sort="downloads",
                    direction=-1,
                    limit=50,
                )
                for ds in datasets:
                    _add_if_accessible(ds, f"tag={tag}")
            except Exception as e:
                logger.warning(f"Error searching for tag {tag}: {e}")

        return dataset_ids

    # Datasets reserved for eval — exclude from training data generation.
    # Matches both exact HF IDs and base dataset names (after '/').
    _EVAL_RESERVED = {
        "openai/gsm8k", "allenai/wildguardmix", "openai/openai_humaneval",
        "heegyu/bbq", "allenai/wildguardtest", "strongreject",
        "EleutherAI/hendrycks_math", "evalplus/humanevalplus",
    }
    # Also match by base name so alternate HF IDs (e.g., HiTZ/bbq) are caught
    _EVAL_RESERVED_NAMES = {
        "gsm8k", "wildguardmix", "humaneval", "humanevalplus",
        "bbq", "wildguardtest", "strongreject", "hendrycks_math",
    }

    def _check_suitability(self, dataset_id: str) -> dict:
        """Check if a dataset is suitable for perturbation training data.

        Checks:
        1. Detection succeeded (dataset exists, accessible, loadable)
        2. Has a text-based question field (no image/audio-only)
        3. Not reserved for eval (gsm8k, wildguardmix, humaneval, bbq)
        4. Has enough examples (>= 10)
        5. Question text is long enough to perturb (avg >= 20 chars)

        Returns:
            Dict with dataset_id, suitable, reasons, capability_tags, profile
        """
        result = {
            "dataset_id": dataset_id,
            "suitable": False,
            "reasons": [],
            "capability_tags": [],
            "task_type": None,
            "profile": None,
        }

        # Start tracer for this dataset
        ds_trace = None
        if self._tracer:
            ds_trace = self._tracer.start_dataset_profiling(dataset_id)

        # Parse config name if present (e.g., "cais/mmlu:high_school_math")
        config_name = None
        if ":" in dataset_id:
            dataset_id, config_name = dataset_id.rsplit(":", 1)
            result["dataset_id"] = f"{dataset_id}:{config_name}"

        # Check 1: Not reserved for eval (exact ID or base name match)
        base_name = dataset_id.split("/")[-1].split(":")[0].lower()
        if dataset_id in self._EVAL_RESERVED or base_name in self._EVAL_RESERVED_NAMES:
            result["reasons"].append(f"Reserved for eval: {dataset_id}")
            if ds_trace:
                ds_trace.checks.append(("Not reserved for eval", False, f"Reserved: {dataset_id}"))
                ds_trace.rejection_reasons = list(result["reasons"])
            return result
        if ds_trace:
            ds_trace.checks.append(("Not reserved for eval", True, ""))

        # Check 2: Detection succeeds
        try:
            profile = self.detector.detect(dataset_id, config_name)
            if ds_trace:
                ds_trace.checks.append(("Detection succeeded", True, ""))
        except Exception as e:
            err_msg = str(e)
            if "gated" in err_msg.lower():
                # Gated datasets — skip them during discovery rather than crashing
                logger.warning(f"Skipping gated dataset '{dataset_id}' (terms not accepted)")
                result["reasons"].append(f"Gated dataset (requires terms acceptance)")
            else:
                result["reasons"].append(f"Detection failed: {err_msg[:200]}")
            if ds_trace:
                ds_trace.checks.append(("Detection succeeded", False, err_msg[:200]))
                ds_trace.rejection_reasons = list(result["reasons"])
            return result

        result["profile"] = profile.to_dict()
        result["task_type"] = profile.task_type

        # Record heuristic detection results in tracer
        if ds_trace:
            ds_trace.available_columns = getattr(profile, '_column_names', [])
            ds_trace.detection_notes = getattr(profile, '_detection_notes', [])
            ds_trace.adapter_heuristic_results = {
                "task_type": profile.task_type,
                "question_field": profile.question_field,
                "answer_field": profile.answer_field or "(none)",
                "choices_field": profile.choices_field or "(none)",
                "context_field": profile.context_field or "(none)",
                "n_total": str(profile.n_total),
            }

        # Check 3: Has a question field (text-based)
        if not profile.question_field:
            result["reasons"].append("No question field detected")
            if ds_trace:
                ds_trace.checks.append(("Has question field", False, "No question field detected"))
                ds_trace.rejection_reasons = list(result["reasons"])
            return result
        if ds_trace:
            ds_trace.checks.append(("Has question field", True, profile.question_field))

        # Check 4: Not image/audio-only
        if self._is_non_text_dataset(profile):
            result["reasons"].append("Non-text dataset (image/audio)")
            if ds_trace:
                ds_trace.checks.append(("Text-only", False, "Non-text dataset"))
                ds_trace.rejection_reasons = list(result["reasons"])
            return result
        if ds_trace:
            ds_trace.checks.append(("Text-only", True, ""))

        # Check 5: Enough examples
        if profile.n_total < 10:
            result["reasons"].append(f"Too few examples: {profile.n_total}")
            if ds_trace:
                ds_trace.checks.append(("Enough examples", False, f"{profile.n_total} < 10"))
                ds_trace.rejection_reasons = list(result["reasons"])
            return result
        if ds_trace:
            ds_trace.checks.append(("Enough examples", True, f"{profile.n_total}"))

        # Detect capability tags
        result["capability_tags"] = self._detect_capabilities(
            dataset_id, profile, config_name
        )

        # Detect complexity tags
        complexity = self._detect_complexity_tags(dataset_id, profile, config_name)
        result["complexity"] = complexity

        result["suitable"] = True

        # Record detection results in tracer
        if ds_trace:
            ds_trace.suitable = True
            ds_trace.task_type = profile.task_type
            ds_trace.question_field = profile.question_field
            ds_trace.answer_field = profile.answer_field or ""
            ds_trace.choices_field = profile.choices_field or ""
            ds_trace.context_field = profile.context_field or ""
            ds_trace.n_total = profile.n_total
            ds_trace.prompt_template = profile.prompt_template
            ds_trace.capability_tags = list(result["capability_tags"])
            ds_trace.complexity = complexity

            # Record how capabilities were determined
            base_id = dataset_id.split(":")[0] if ":" in dataset_id else dataset_id
            seed_cat = getattr(self, '_seed_categories', {}).get(base_id)
            if seed_cat:
                ds_trace.capability_method = "research_agent"
            else:
                # Check if it came from known benchmarks
                from .known_benchmarks import KNOWN_BENCHMARK_MAP
                known = any(
                    (info.get("hf_id") or "").split(":")[0] == base_id
                    for info in KNOWN_BENCHMARK_MAP.values()
                )
                ds_trace.capability_method = "known_benchmark" if known else "keyword"

        return result

    def _is_non_text_dataset(self, profile: DatasetProfile) -> bool:
        """Check if a dataset is image/audio-only (not text-based).

        Uses HF features metadata (no download) for type/column checks,
        then inspects question field content from cached samples.
        """
        non_text_col_names = [
            "audio", "image", "video", "img", "photo", "picture",
            "sound", "wav", "mp3", "pixel",
        ]

        # 1. Check dataset ID for known multimodal dataset names
        did_lower = profile.dataset_id.lower()
        multimodal_names = [
            "mmmu", "mmstar", "mathvista", "mathvision", "figureqa",
            "vqa", "gqa", "weatherqa", "zerobench", "unidoc",
            "captioning", "ocr",
        ]
        if any(name in did_lower for name in multimodal_names):
            return True

        # 2. Fast metadata check: inspect HF features for Image/Audio types
        #    and column names — uses cached features from detect(), no extra API call
        features = getattr(profile, '_features', None)
        if features:
            for feat_name, feat_type in features.items():
                type_str = type(feat_type).__name__.lower()
                if type_str in ("image", "audio"):
                    return True
                feat_str = str(feat_type).lower()
                if "image(" in feat_str or "audio(" in feat_str:
                    return True
                if any(nt in feat_name.lower() for nt in non_text_col_names):
                    return True

        # 3. Content check using cached samples from detect() (no re-download)
        samples = getattr(profile, '_cached_samples', None)
        if samples:
            q_values = [str(s.get(profile.question_field, "")) for s in samples]

            # Very short questions — likely labels or file paths
            avg_len = sum(len(q) for q in q_values) / max(len(q_values), 1)
            if avg_len < 20:
                return True

            # File extensions or URLs
            file_indicators = [".png", ".jpg", ".jpeg", ".wav", ".mp3", ".mp4"]
            url_count = sum(1 for q in q_values if q.startswith(("http://", "https://", "s3://")))
            ext_count = sum(1 for q in q_values if any(q.lower().endswith(ext) for ext in file_indicators))
            if url_count >= len(q_values) * 0.5 or ext_count >= len(q_values) * 0.5:
                return True

            # Questions referencing non-text modalities
            modality_ref_phrases = [
                "<image", "the image", "the figure", "the diagram",
                "the chart", "the graph", "the picture", "the photo",
                "the screenshot", "the illustration", "the drawing",
                "shown above", "shown below", "in the picture",
                "in the figure", "in the image", "in the diagram",
                "refer to the image", "see the image", "look at the",
                "the following image", "attached image",
                "the audio", "the recording", "the sound", "listen to",
                "the speech", "the voice", "the clip",
                "the video", "watch the", "the clip shows",
                "in the video", "the footage",
                "the table above", "the table below", "refer to the table",
                "see the table", "the following table",
            ]
            modality_ref_count = sum(
                1 for q in q_values
                if any(phrase in q.lower() for phrase in modality_ref_phrases)
            )
            if modality_ref_count >= len(q_values) * 0.5:
                return True

        return False

    # Map category strings → CapabilityDomain.
    #
    # Handles both KNOWN_BENCHMARK_MAP categories ("code", "knowledge", ...)
    # and registered benchmark domains ("coding", "fairness") so either
    # vocabulary resolves to the same CapabilityDomain.
    _CATEGORY_TO_DOMAIN = {
        # Discovery categories (KNOWN_BENCHMARK_MAP)
        "math": CapabilityDomain.MATH_REASONING.value,
        "code": CapabilityDomain.CODE.value,
        "knowledge": CapabilityDomain.KNOWLEDGE_QA.value,
        "reasoning": CapabilityDomain.LOGICAL_REASONING.value,
        "safety": CapabilityDomain.SAFETY_ETHICS.value,
        "instruction": CapabilityDomain.KNOWLEDGE_QA.value,
        "language": CapabilityDomain.LANGUAGE_UNDERSTANDING.value,
        "multilingual": CapabilityDomain.LANGUAGE_UNDERSTANDING.value,
        "classification": CapabilityDomain.CLASSIFICATION.value,
        # Registered benchmark domain aliases
        "coding": CapabilityDomain.CODE.value,  # legacy name, same as "code"
        "fairness": CapabilityDomain.SAFETY_ETHICS.value,
        # New category mappings
        "text_generation": CapabilityDomain.TEXT_GENERATION.value,
        "summarization": CapabilityDomain.SUMMARIZATION.value,
        "translation": CapabilityDomain.TRANSLATION.value,
        "dialog": CapabilityDomain.DIALOG.value,
        "instruction_following": CapabilityDomain.INSTRUCTION_FOLLOWING.value,
        "domain_specific": CapabilityDomain.DOMAIN_SPECIFIC.value,
        "medical": CapabilityDomain.DOMAIN_SPECIFIC.value,
        "legal": CapabilityDomain.DOMAIN_SPECIFIC.value,
        "finance": CapabilityDomain.DOMAIN_SPECIFIC.value,
    }

    def _detect_capabilities(
        self,
        dataset_id: str,
        profile: DatasetProfile,
        config_name: Optional[str],
    ) -> list[str]:
        """Detect capability domains from dataset metadata.

        Priority:
        1. LLM-generated category from research agent (highest quality)
        2. Known benchmark category from KNOWN_BENCHMARK_MAP
        3. Keyword matching on dataset ID and task type
        4. Task type heuristics

        Categories not in _CATEGORY_TO_DOMAIN are passed through as-is
        (e.g., "instruction_following", "multimodal" from research agent).
        """
        tags = []

        # Check research agent's LLM-generated category first (best quality)
        base_id = dataset_id.split(":")[0] if ":" in dataset_id else dataset_id
        seed_cat = getattr(self, '_seed_categories', {}).get(base_id)
        if seed_cat:
            domain = self._CATEGORY_TO_DOMAIN.get(seed_cat)
            if domain:
                tags.append(domain)
            else:
                # New category from research agent — pass through as-is
                tags.append(seed_cat)

        # Check known benchmark category
        if not tags:
            for _name, info in KNOWN_BENCHMARK_MAP.items():
                known_base = (info.get("hf_id") or "").split(":")[0]
                if known_base == base_id:
                    category = info.get("category", "")
                    domain = self._CATEGORY_TO_DOMAIN.get(category)
                    if domain and domain not in tags:
                        tags.append(domain)
                    break

        # Keyword matching
        search_text = (
            f"{dataset_id} {config_name or ''} {profile.task_type}"
        ).lower()

        for domain, keywords in _CAPABILITY_KEYWORDS.items():
            if any(kw in search_text for kw in keywords):
                if domain.value not in tags:
                    tags.append(domain.value)

        # Task type heuristics
        # Legacy sub-type heuristics (for cached profiles with old task types)
        if profile.task_type in (TaskType.OPEN_NUMERIC.value, "open_numeric") and not tags:
            tags.append(CapabilityDomain.MATH_REASONING.value)
        if profile.task_type in (TaskType.SAFETY_REFUSAL.value, "safety_refusal") and not tags:
            tags.append(CapabilityDomain.SAFETY_ETHICS.value)
        if profile.task_type in (TaskType.CLASSIFICATION.value, "classification") and not tags:
            tags.append(CapabilityDomain.CLASSIFICATION.value)
        if profile.context_field and not tags:
            tags.append(CapabilityDomain.READING_COMPREHENSION.value)

        return tags if tags else [CapabilityDomain.KNOWLEDGE_QA.value]

    async def classify_capability_with_llm(
        self,
        dataset_id: str,
        profile: DatasetProfile,
        api: "InferenceAPI",
        model_id: str = "claude-haiku-4-5-20251001",
        _ds_trace: Optional["DatasetProfilingTrace"] = None,
    ) -> list[str]:
        """LLM-based capability domain classification.

        Sends the LLM the dataset schema, HF description, and sample
        questions, then asks it to classify into CapabilityDomain categories.

        Used as a fallback when keyword matching and known benchmark lookup
        both fail to assign a meaningful capability tag.

        Args:
            dataset_id: HuggingFace dataset ID
            profile: The auto-detected DatasetProfile
            api: InferenceAPI for making LLM calls
            model_id: Model to use (default: Haiku for cost efficiency)

        Returns:
            List of capability domain strings (1-2 tags)
        """
        import re
        from safetytooling.data_models import ChatMessage, MessageRole, Prompt

        # Load sample questions
        sample_qs = []
        try:
            config_name = None
            ds_id = dataset_id
            if ":" in dataset_id:
                ds_id, config_name = dataset_id.rsplit(":", 1)

            samples, _ = self.detector.retry_hf_call(
                lambda: self.detector._load_samples(
                    ds_id, config_name, profile.split, n=3,
                ),
                timeout=15.0,
            )
            q_field = profile.question_field
            for s in (samples or []):
                q = str(s.get(q_field, ""))[:200]
                if q:
                    sample_qs.append(q)
        except Exception:
            pass

        # Get HF dataset card description (disk-cached)
        base_id = dataset_id.split(":")[0]
        description = (self._get_dataset_card(base_id) or "")[:500]

        # Build valid domains list with descriptions
        domain_descriptions = {
            "math_reasoning": "Math word problems, arithmetic, algebra, geometry, calculus",
            "logical_reasoning": "Logic puzzles, deduction, inference, syllogisms, NLI",
            "knowledge_qa": "Trivia, factual questions, encyclopedic knowledge, quizzes",
            "reading_comprehension": "Passage-based QA, extractive QA, document QA",
            "code": "Programming, software engineering, algorithm design",
            "safety_ethics": "Toxicity, bias, harmful content, jailbreak detection, alignment",
            "classification": "Sentiment analysis, topic classification, intent detection",
            "commonsense": "Physical/social commonsense, everyday reasoning",
            "science": "Physics, chemistry, biology, scientific reasoning",
            "language_understanding": "Grammar, paraphrase, entailment, coreference, GLUE tasks",
            "text_generation": "Creative writing, story generation, text completion, language modeling",
            "summarization": "Text summarization (abstractive/extractive)",
            "translation": "Machine translation, bilingual tasks",
            "dialog": "Dialog systems, conversational AI, counseling",
            "instruction_following": "Instruction compliance, format following, constraint satisfaction",
            "domain_specific": "Medical, legal, finance, chemistry (specialized domain knowledge)",
        }
        domains_str = "\n".join(
            f"- {name}: {desc}" for name, desc in domain_descriptions.items()
        )

        samples_str = ""
        for i, q in enumerate(sample_qs[:3], 1):
            samples_str += f"{i}. {q}\n"
        if not samples_str:
            samples_str = "(no samples available)\n"

        prompt_text = f"""Classify this HuggingFace dataset into capability domains.

## Dataset: {dataset_id}
## Task Type: {profile.task_type}
{f"## Description: {description[:500]}" if description else ""}
## Sample Questions:
{samples_str}
## Valid Capability Domains:
{domains_str}

Choose the MOST specific domain that fits. Do NOT default to knowledge_qa unless
the dataset is genuinely about factual/trivia knowledge.

Return ONLY a JSON object:
{{"primary_tag": "<domain>", "secondary_tag": "<domain or null>", "reasoning": "<1 sentence>"}}"""

        try:
            responses = await api(
                model_id=model_id,
                prompt=Prompt(messages=[
                    ChatMessage(role=MessageRole.user, content=prompt_text),
                ]),
                n=1,
                temperature=0.0,
                max_tokens=256,
            )
            text = responses[0].completion if responses else ""

            # Record LLM classification in tracer
            if _ds_trace:
                from ..discovery_tracer import DiscoveryLLMCall
                _ds_trace.llm_calls.append(DiscoveryLLMCall(
                    stage="classify_capability",
                    prompt=prompt_text, response=text,
                    model=model_id, temperature=0.0,
                    label=dataset_id,
                ))

            json_match = re.search(r'\{[\s\S]*\}', text)
            if not json_match:
                return [CapabilityDomain.KNOWLEDGE_QA.value]

            result = json.loads(json_match.group())
            valid_domains = {d.value for d in CapabilityDomain}

            tags = []
            primary = result.get("primary_tag", "")
            if primary in valid_domains:
                tags.append(primary)
            secondary = result.get("secondary_tag")
            if secondary and secondary in valid_domains and secondary != primary:
                tags.append(secondary)

            return tags if tags else [CapabilityDomain.KNOWLEDGE_QA.value]

        except Exception as e:
            logger.warning(
                f"LLM capability classification failed for {dataset_id}: {e}"
            )
            return [CapabilityDomain.KNOWLEDGE_QA.value]

    async def refine_tags_with_llm(
        self,
        datasets: list[dict],
        api: "InferenceAPI",
        model_id: str = "claude-haiku-4-5-20251001",
        force_all: bool = False,
        concurrency: int = 20,
    ) -> list[dict]:
        """Refine capability tags using LLM for datasets with default tags.

        Only processes datasets where the heuristic assigned the fallback
        knowledge_qa tag (unless force_all=True). Other datasets keep their
        existing tags unchanged.

        Args:
            datasets: List of dataset entries (from datasets.jsonl)
            api: InferenceAPI for making LLM calls
            model_id: Model to use
            force_all: If True, re-classify all datasets (not just defaults)

        Returns:
            The same list with updated capability_tags (modified in-place)
        """
        import asyncio

        to_classify = []
        for entry in datasets:
            tags = entry.get("capability_tags", [])
            if force_all or tags == [CapabilityDomain.KNOWLEDGE_QA.value]:
                to_classify.append(entry)

        if not to_classify:
            logger.info("No datasets need LLM capability classification")
            return datasets

        logger.info(
            f"Running LLM capability classification on {len(to_classify)} datasets"
        )

        # Build dataset_id → trace lookup for recording LLM calls
        _trace_map: dict[str, "DatasetProfilingTrace"] = {}
        if self._tracer:
            for dt in self._tracer._dataset_traces:
                _trace_map[dt.dataset_id] = dt

        async def classify_one(entry: dict) -> None:
            profile_dict = entry.get("profile", {})
            if not profile_dict:
                return

            profile = DatasetProfile.from_dict(profile_dict)
            old_tags = entry.get("capability_tags", [])
            ds_trace = _trace_map.get(entry["dataset_id"])
            new_tags = await self.classify_capability_with_llm(
                entry["dataset_id"], profile, api, model_id,
                _ds_trace=ds_trace,
            )
            if new_tags != old_tags:
                logger.info(
                    f"  {entry['dataset_id']}: {old_tags} → {new_tags}"
                )
                entry["capability_tags"] = new_tags

        # Run concurrently in batches
        batch_size = concurrency
        for i in range(0, len(to_classify), batch_size):
            batch = to_classify[i:i + batch_size]
            await asyncio.gather(*[classify_one(e) for e in batch])

        # Log updated distribution
        from collections import Counter
        tag_counts = Counter()
        for entry in datasets:
            for tag in entry.get("capability_tags", []):
                tag_counts[tag] += 1
        logger.info("Updated tag distribution:")
        for tag, count in tag_counts.most_common():
            logger.info(f"  {tag}: {count}")

        return datasets

    def _detect_complexity_tags(
        self,
        dataset_id: str,
        profile: DatasetProfile,
        config_name: Optional[str],
    ) -> dict[str, str]:
        """Detect context_source, context_length, interaction_mode.

        Priority:
        1. Known benchmark tags from KNOWN_BENCHMARK_MAP
        2. HF dataset card description keywords
        3. Statistical heuristics (avg token count, field structure)
        4. Default: single_source / short / static

        Returns:
            Dict with keys: context_source, context_length, interaction_mode.
        """
        from .known_benchmarks import (
            KNOWN_BENCHMARK_MAP,
            DEFAULT_CONTEXT_SOURCE,
            DEFAULT_CONTEXT_LENGTH,
            DEFAULT_INTERACTION_MODE,
        )

        result = {
            "context_source": DEFAULT_CONTEXT_SOURCE,
            "context_length": DEFAULT_CONTEXT_LENGTH,
            "interaction_mode": DEFAULT_INTERACTION_MODE,
        }

        # 1. Check known benchmarks first
        base_id = dataset_id.split(":")[0] if ":" in dataset_id else dataset_id
        for _name, info in KNOWN_BENCHMARK_MAP.items():
            hf_id = info.get("hf_id") or ""
            known_base = hf_id.split(":")[0] if ":" in hf_id else hf_id
            if known_base == base_id:
                if "context_source" in info:
                    result["context_source"] = info["context_source"]
                if "context_length" in info:
                    result["context_length"] = info["context_length"]
                if "interaction_mode" in info:
                    result["interaction_mode"] = info["interaction_mode"]
                return result

        # 2. Description-based keyword detection
        # Start with short description from profile metadata
        description = getattr(profile, 'description', '') or ''
        search_text = (
            f"{dataset_id} {config_name or ''} {description}"
        ).lower()

        # If short description is sparse, try fetching the full dataset card
        # README from HuggingFace Hub for richer keyword matching
        if len(description) < 50:
            card_text = self._get_dataset_card(base_id)
            if card_text:
                search_text = f"{search_text} {card_text}".lower()

        # Context source keywords
        multi_source_kw = [
            "multi-hop", "multihop", "multi hop", "multi_hop",
            "compositional", "supporting facts", "multiple passages",
            "cross-document", "multi-document",
        ]
        multimodal_kw = [
            "agent", "tool use", "tool_use", "function calling",
            "environment", "web browsing", "code execution",
            "interactive", "sandbox",
        ]
        if any(kw in search_text for kw in multimodal_kw):
            result["context_source"] = "multimodal_context"
        elif any(kw in search_text for kw in multi_source_kw):
            result["context_source"] = "multi_source"

        # Interaction mode keywords
        multi_turn_kw = [
            "multi-turn", "multiturn", "multi turn", "conversation",
            "dialogue", "dialog", "chat", "conversational",
        ]
        tool_use_kw = [
            "tool", "function calling", "api call", "tool_use",
            "agent", "agentic",
        ]
        if any(kw in search_text for kw in tool_use_kw):
            result["interaction_mode"] = "tool_use"
        elif any(kw in search_text for kw in multi_turn_kw):
            result["interaction_mode"] = "multi_turn"

        # Check field structure for multi-turn detection (before context length,
        # so we can account for multi-turn serialization inflation)
        features = getattr(profile, 'features', None) or {}
        if isinstance(features, dict):
            field_names = set(features.keys()) if features else set()
            mt_fields = {"messages", "conversations", "turns", "dialogue"}
            if field_names & mt_fields:
                result["interaction_mode"] = "multi_turn"
            tool_fields = {"tools", "functions", "api_calls", "actions"}
            if field_names & tool_fields:
                result["interaction_mode"] = "tool_use"
                result["context_source"] = "multimodal_context"

        # Statistical heuristics for context length
        # Skip for multi-turn datasets — their serialized question length is
        # inflated by conversation structure (list of role/content dicts), not
        # by genuinely long context like documents or passages.
        if result["interaction_mode"] != "multi_turn":
            if hasattr(profile, '_sample_lengths') and profile._sample_lengths:
                avg_len = sum(profile._sample_lengths) / len(profile._sample_lengths)
                if avg_len > 5000:
                    result["context_length"] = "multi_document"
                elif avg_len > 500:
                    result["context_length"] = "long"
            elif "long" in search_text or "document" in search_text:
                result["context_length"] = "long"

        return result

    def _get_dataset_card(self, dataset_id: str) -> str:
        """Fetch dataset card text with in-memory + disk caching."""
        if not hasattr(self, '_card_cache'):
            self._card_cache: dict[str, str] = {}

        if dataset_id in self._card_cache:
            return self._card_cache[dataset_id]

        # Check disk cache
        if self.cache_dir:
            import hashlib
            card_cache_path = self.cache_dir / f"card_{hashlib.md5(dataset_id.encode()).hexdigest()}.txt"
            if card_cache_path.exists():
                card_text = card_cache_path.read_text()
                self._card_cache[dataset_id] = card_text
                return card_text

        # Fetch from HF Hub
        card_text = ""
        try:
            from huggingface_hub import DatasetCard
            card = DatasetCard.load(dataset_id)
            card_text = card.text[:5000] if card.text else ""
        except Exception:
            pass

        self._card_cache[dataset_id] = card_text

        # Persist to disk
        if self.cache_dir:
            try:
                import hashlib
                card_cache_path = self.cache_dir / f"card_{hashlib.md5(dataset_id.encode()).hexdigest()}.txt"
                card_cache_path.write_text(card_text)
            except Exception:
                pass

        return card_text

    def _log_diversity(self, results: list[dict]) -> None:
        """Log diversity statistics across discovered datasets."""
        capability_counts: dict[str, int] = {}
        task_type_counts: dict[str, int] = {}

        for r in results:
            for cap in r.get("capability_tags", []):
                capability_counts[cap] = capability_counts.get(cap, 0) + 1
            tt = r.get("task_type", "unknown")
            task_type_counts[tt] = task_type_counts.get(tt, 0) + 1

        logger.info("=== Diversity Report ===")
        logger.info(f"Total suitable datasets: {len(results)}")
        logger.info("By capability domain:")
        for cap, count in sorted(capability_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {cap}: {count}")
        logger.info("By task type:")
        for tt, count in sorted(task_type_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {tt}: {count}")

        # Flag under-represented capabilities
        all_caps = set(c.value for c in CapabilityDomain)
        found_caps = set(capability_counts.keys())
        missing = all_caps - found_caps
        if missing:
            logger.warning(f"Under-represented capabilities: {missing}")

    def _save_cache(self, cache_path: Path, results: list[dict]) -> None:
        """Save discovery results to cache."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

    def _load_cache(self, cache_path: Path) -> list[dict]:
        """Load discovery results from cache."""
        results = []
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Discover HuggingFace datasets suitable for perturbation training data"
    )
    parser.add_argument(
        "--min_downloads", type=int, default=50,
        help="Minimum download count threshold (default: 50)",
    )
    parser.add_argument(
        "--max_datasets", type=int, default=250,
        help="Maximum datasets to discover (default: 250)",
    )
    parser.add_argument(
        "--output", type=str, default="datasets.jsonl",
        help="Output file path (default: datasets.jsonl)",
    )
    parser.add_argument(
        "--cache_dir", type=str,
        default="outputs/auto_perturbation/.discovery_cache",
        help="Cache directory for discovery results",
    )
    parser.add_argument(
        "--cache_ttl_days", type=int, default=7,
        help="Cache TTL in days (default: 7)",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force refresh, ignoring cache",
    )
    parser.add_argument(
        "--no_curated", action="store_true",
        help="Skip curated seed datasets",
    )
    parser.add_argument(
        "--frontier_seeds", type=str, default=None,
        help="Path to frontier_seeds.jsonl from research agent (extra seeds)",
    )
    parser.add_argument(
        "--search_hub", action="store_true",
        help="Also search HF Hub for datasets (off by default — slow + rate limited)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=20,
        help="Max concurrent LLM calls for tag classification (default: 20)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load .env for HF_TOKEN and other secrets
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not installed, rely on env vars

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load extra seeds from research agent output
    # Preserves LLM-generated categories for better capability tagging
    extra_seeds = None
    seed_categories: dict[str, str] = {}
    if args.frontier_seeds:
        extra_seeds = []
        with open(args.frontier_seeds) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    hf_id = entry.get("hf_id")
                    if hf_id:
                        extra_seeds.append(hf_id)
                        # Preserve LLM-generated category from research agent
                        cat = entry.get("category")
                        if cat:
                            seed_categories[hf_id.split(":")[0]] = cat
        logger.info(f"Loaded {len(extra_seeds)} frontier seeds from {args.frontier_seeds}")

    # Always enable discovery tracer for HTML trace output
    from ..discovery_tracer import DiscoveryTracer
    tracer = DiscoveryTracer()

    discovery = DatasetDiscovery(
        cache_dir=Path(args.cache_dir),
        cache_ttl_days=args.cache_ttl_days,
        tracer=tracer,
    )

    results = discovery.discover(
        min_downloads=args.min_downloads,
        max_datasets=args.max_datasets,
        include_curated=not args.no_curated,
        refresh=args.refresh,
        extra_seeds=extra_seeds,
        search_hub=args.search_hub,
        seed_categories=seed_categories,
    )

    # Write to output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Render HTML trace (timestamped to preserve history)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = output_path.parent / f"profiling_trace_{ts}.html"
    tracer.render_profiling_html(trace_path)
    # Also write a "latest" copy for convenience
    import shutil
    latest_path = output_path.parent / "profiling_trace.html"
    shutil.copy2(trace_path, latest_path)
    logger.info(f"HTML trace: {trace_path}")

    logger.info(f"Wrote {len(results)} datasets to {output_path}")


if __name__ == "__main__":
    main()
