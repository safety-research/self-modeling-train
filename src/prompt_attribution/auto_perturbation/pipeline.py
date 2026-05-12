"""
Module: prompt_attribution/auto_perturbation/pipeline.py

Full pipeline orchestration for training data generation.

Stages: adapt → decompose → generate → critic → verify → export

Each stage reads/writes JSON intermediates so stages can run independently.
After each stage, a human-readable summary is logged and accumulated into
pipeline_summary.txt for easy inspection.

Structure:
- _StageSummary: Helper for building formatted summary blocks
- TrainingDataPipeline: Main pipeline class with per-stage summaries
"""

import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from safetytooling.apis import InferenceAPI

from .config import (
    PerProblemCandidate,
    PipelineConfig,
    ProblemAnalysis,
    VerificationResult,
)
from .candidate_critic.critic import CandidateCritic
from .dataset_adapter.dataset_adapter import AdaptedExample, DatasetAdapter, DatasetDetector
from .problem_decomposer.decomposer import ProblemDecomposer
from .candidate_generator.generator import PerProblemGenerator
from .training_export.training_data import export_training_data
from .empirical_verification.verifier import EmpiricalVerifier

logger = logging.getLogger(__name__)

W = 72  # Summary block width


class _StageSummary:
    """Builds a human-readable summary block for one pipeline stage."""

    def __init__(self, stage_name: str, stage_number: str):
        self.lines: list[str] = []
        self.lines.append("")
        self.lines.append("=" * W)
        header = f"  Stage {stage_number}: {stage_name}"
        self.lines.append(header)
        self.lines.append("=" * W)

    def kv(self, key: str, value: object) -> None:
        self.lines.append(f"  {key}: {value}")

    def blank(self) -> None:
        self.lines.append("")

    def section(self, title: str) -> None:
        self.lines.append(f"\n  --- {title} ---")

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        if not rows:
            return
        widths = []
        for i, h in enumerate(headers):
            col_max = max((len(str(r[i])) for r in rows), default=0)
            widths.append(max(len(h), col_max))
        fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
        self.lines.append(fmt.format(*headers))
        self.lines.append("  " + "  ".join("-" * w for w in widths))
        for row in rows:
            self.lines.append(fmt.format(*[str(c) for c in row]))

    def sample(self, label: str, text: str, max_len: int = 120) -> None:
        preview = text.replace("\n", " ")[:max_len]
        if len(text) > max_len:
            preview += "..."
        self.lines.append(f"    [{label}] {preview}")

    def warning(self, msg: str) -> None:
        self.lines.append(f"  ⚠ {msg}")

    def render(self) -> str:
        self.lines.append("=" * W)
        return "\n".join(self.lines)


class TrainingDataPipeline:
    """Pipeline for domain-agnostic training data generation.

    Stages:
    1. Auto-Adapt: Detect dataset characteristics
    2. Decompose: Generic structural analysis
    3. Generate: 3-category perturbation generation (includes mechanism discovery)
    4. Critic: Quality filtering + flip prediction
    5. Verify: Phase 1 empirical verification (full responses)
    6. Export: Training data output (JSONL)
    """

    def __init__(self, config: PipelineConfig, api: InferenceAPI, discovery_tracer=None):
        self.config = config
        self.api = api
        self._adapter: Optional[DatasetAdapter] = None
        self._prompt_logger = None  # Created in run() when dump_prompts is True
        self._tracer = None  # Created in run() when trace_examples > 0
        self._discovery_tracer = discovery_tracer  # Optional DiscoveryTracer for pre-pipeline traces
        self._ds_trace = None  # Current dataset's DatasetProfilingTrace

    @property
    def adapter(self) -> DatasetAdapter:
        if self._adapter is None:
            self._adapter = self._build_adapter()
        return self._adapter

    def _build_adapter(self) -> DatasetAdapter:
        """Build adapter from registered benchmark, pre-computed profile, or auto-detect."""
        from .dataset_adapter.dataset_adapter import DatasetProfile

        dataset_id = self.config.dataset_id

        # Parse config name if present
        config_name = None

        # Use pre-computed profile from discovery if available
        if self.config.dataset_profile:
            profile = DatasetProfile.from_dict(self.config.dataset_profile)
            logger.info(
                f"Using pre-computed profile: {dataset_id}, "
                f"task_type={profile.task_type}, "
                f"question_field={profile.question_field}"
            )
            return DatasetAdapter(profile)

        # Fall back to auto-detection
        cache_dir = Path("outputs/auto_perturbation/.discovery_cache/profiles")
        detector = DatasetDetector(cache_dir=cache_dir)
        profile = detector.detect(dataset_id, config_name)
        logger.info(
            f"Auto-detected dataset: {dataset_id}, "
            f"task_type={profile.task_type}, "
            f"question_field={profile.question_field}"
        )
        return DatasetAdapter(profile)

    async def _run_label_ideation_if_needed(
        self,
        examples: list | None = None,
    ) -> None:
        """Run LLM-based label ideation for all datasets.

        For DatasetAdapter: uses HF samples directly.
        (legacy BenchmarkAdapter path removed)
        """
        adapter = self.adapter
        profile = adapter.profile

        # Skip if already has rich labels (more than just defaults)
        if profile.answer_labels and len(profile.answer_labels) > 1:
            return

        if isinstance(adapter, DatasetAdapter):
            logger.info(f"Running label ideation for {profile.task_type} dataset")
            detector = DatasetDetector(
                cache_dir=Path("outputs/auto_perturbation/.discovery_cache/profiles")
            )
            adapter.profile = await detector.ideate_answer_labels(
                profile, self.api, self.config.generator_model,
                _ds_trace=self._ds_trace,
            )
        elif examples:
            
            from .dataset_adapter.label_ideation import LabelIdeator
            logger.info(
                f"Running label ideation for registered benchmark "
                f"{profile.dataset_id}"
            )
            samples = [
                {"question": ex.question, "answer": ex.ground_truth_answer}
                for ex in examples[:5]
            ]
            ideator = LabelIdeator(
                self.api, self.config.generator_model,
                prompt_logger=self._prompt_logger,
            )
            labels, placement = await ideator.ideate_labels(
                dataset_id=profile.dataset_id,
                task_type=profile.task_type,
                samples=samples,
                question_field="question",
                answer_field="answer",
                label_names=getattr(profile, 'label_names', None),
                _ds_trace=self._ds_trace,
            )
            profile.answer_labels = [l.to_dict() for l in labels]
            logger.info(
                f"Ideated {len(profile.answer_labels)} answer labels: "
                f"{[l.get('name') for l in profile.answer_labels]}"
            )
        else:
            return

        # Build a human-readable label description string that downstream
        # stages (decomposer, generator) can include in their prompts
        label_desc_parts = []
        for label in profile.answer_labels:
            name = label.get("name", "")
            desc = label.get("description", "")
            if name and desc:
                label_desc_parts.append(f"- {name}: {desc}")
        if label_desc_parts:
            profile.label_descriptions = (
                "Answer label axes (what defines a 'flip'):\n"
                + "\n".join(label_desc_parts)
            )
        else:
            profile.label_descriptions = ""

        # Note: label_descriptions (answer label axes) are internal pipeline
        # metadata for the generator/decomposer/critic. They are NOT added
        # to the prompt template — the model answering the question shouldn't
        # see internal flip-detection axes like "contains_negative_indicator".

        logger.info(
            f"Label axes: {[l.get('name') for l in profile.answer_labels]}"
        )

    async def run(
        self,
        stage: Optional[str] = None,
        run_dir: Optional[Path] = None,
        skip_verify: bool = False,
    ) -> dict:
        """Run the pipeline (all stages or a specific stage).

        Args:
            stage: If set, run only this stage. One of:
                "adapt", "decompose", "ideate", "generate", "critic",
                "verify", "export", or None for all.
            run_dir: If set, use this existing run directory (for resuming
                     individual stages). If None, creates a new timestamped dir.
            skip_verify: If True, skip empirical verification and go straight
                to export with critic-only labels. The run_dir preserves all
                intermediates so verify can be re-run later.

        Returns:
            Dict with pipeline results and output paths
        """
        # Setup output directory
        if run_dir is not None:
            run_dir = Path(run_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self.config.output_dir / f"run_v11_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        summaries: list[str] = []

        # Create prompt logger if dump_prompts is enabled
        if self.config.dump_prompts:
            from .prompt_logger import PromptLogger
            self._prompt_logger = PromptLogger(run_dir / "prompts.md")

        # Save config
        with open(run_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Pipeline output: {run_dir}")
        logger.info(f"Dataset: {self.config.dataset_id}")

        # Check if this benchmark type is supported for perturbation generation
        cs = self.config.context_source
        im = self.config.interaction_mode
        if cs == "multimodal_context" or im in ("tool_use", "multi_turn"):
            logger.warning(
                f"Skipping {self.config.dataset_id} ({cs}/{im}) — "
                f"perturbation generation only supports static "
                f"single/multi-source benchmarks"
            )
            if self._ds_trace:
                self._ds_trace.suitable = False
                self._ds_trace.rejection_reasons.append(
                    f"Skipped: unsupported complexity ({cs}/{im}) — "
                    f"pipeline only supports static single/multi-source"
                )
            return {
                "run_dir": str(run_dir),
                "skipped": True,
                "reason": f"unsupported_complexity:{cs}/{im}",
            }

        # Stage 1: Load examples
        adapter = self.adapter

        # Start discovery trace for this dataset (if tracer is active)
        if self._discovery_tracer:
            self._ds_trace = self._discovery_tracer.start_dataset_profiling(
                self.config.dataset_id
            )
            # Record heuristic detection results
            profile = adapter.profile
            self._ds_trace.adapter_heuristic_results = {
                "task_type": profile.task_type,
                "question_field": profile.question_field,
                "answer_field": profile.answer_field or "(none)",
                "choices_field": profile.choices_field or "(none)",
                "context_field": profile.context_field or "(none)",
            }

        # LLM-verify field mapping for DatasetAdapter (fixes MCQ choices,
        # pair-input detection, solution-vs-context confusion, etc.)
        if isinstance(adapter, DatasetAdapter):
            cache_dir = Path("outputs/auto_perturbation/.discovery_cache/profiles")
            detector = DatasetDetector(cache_dir=cache_dir)
            old_profile = adapter.profile.to_dict()
            adapter.profile = await detector.detect_with_llm(
                adapter.profile, self.api, self.config.generator_model,
                _ds_trace=self._ds_trace,
            )
            # Record which fields changed
            if self._ds_trace:
                new_profile = adapter.profile.to_dict()
                changed = [
                    k for k in ["question_field", "answer_field", "choices_field",
                                "context_field", "task_type", "prompt_template"]
                    if old_profile.get(k) != new_profile.get(k)
                ]
                self._ds_trace.adapter_fields_changed = changed
                self._ds_trace.final_prompt_template = adapter.profile.prompt_template

            if adapter.profile.task_type == "non_text":
                logger.warning(
                    f"Dataset {self.config.dataset_id} requires non-text modality, skipping"
                )
                return {"run_dir": str(run_dir), "skipped": True, "reason": "non_text"}

        examples = adapter.load_examples(
            self.config.n_samples, self.config.random_seed,
        )
        logger.info(f"Loaded {len(examples)} examples")

        # Run label ideation to get answer feature axes (for all dataset types)
        await self._run_label_ideation_if_needed(examples)

        # Save examples
        with open(run_dir / "examples.json", "w") as f:
            json.dump([e.to_dict() for e in examples], f, indent=2)

        # Create tracer if enabled (after examples + labels are ready)
        if self.config.trace_examples > 0:
            from .pipeline_tracer import PipelineTracer
            self._tracer = PipelineTracer(
                self.config.trace_examples, self.config.random_seed,
            )
            # Resolve effective verification model (target model overrides default)
            effective_verif = (
                self.config.target_model_id
                if self.config.target_model_id
                else self.config.judge_model
            )
            self._tracer.set_config_info({
                "dataset_id": self.config.dataset_id,
                "generator_model (perturbation design)": self.config.generator_model,
                "judge_model (answers questions)": effective_verif,
                "judge_model (feature extraction)": "claude-haiku-4-5-20251001",
                "n_samples": self.config.n_samples,
                "stability_n_runs": self.config.stability_n_runs,
                "enable_feedback_loop": self.config.enable_feedback_loop,
                "feedback_max_rounds": self.config.feedback_max_rounds,
            })
            self._tracer.select_examples([ex.idx for ex in examples])
            # Record adapt stage for traced examples
            profile = adapter.profile
            profile_info = {
                "task_type": getattr(profile, "task_type", ""),
                "capability_tags": self.config.capability_tags,
                "instruction_placement": getattr(profile, "instruction_placement", ""),
                "answer_labels": getattr(profile, "answer_labels", []),
                "prompt_template": getattr(profile, "prompt_template", ""),
                "label_names": getattr(profile, "label_names", []),
            }
            for ex in examples:
                if self._tracer.is_traced(ex.idx):
                    self._tracer.record_adapt(
                        example_idx=ex.idx,
                        question=ex.question,
                        ground_truth_answer=ex.ground_truth_answer,
                        context=ex.context or "",
                        choices=ex.choices,
                        dataset_id=self.config.dataset_id,
                        profile_info=profile_info,
                    )

        summaries.append(self._summarize_adapt(examples))

        if stage == "adapt":
            self._write_summary(summaries, run_dir)
            return {"run_dir": str(run_dir), "n_examples": len(examples)}

        # Stage 2: Decompose
        analyses = await self._run_decompose(examples, run_dir, stage)
        summaries.append(self._summarize_decompose(analyses))
        if stage == "decompose":
            self._write_summary(summaries, run_dir)
            return {"run_dir": str(run_dir), "n_analyses": len(analyses)}

        # Create target model client if configured (shared across all stages)
        target_client = None
        if self.config.target_model_id:
            from prompt_attribution.shared.inference.target_model_client import (
                TargetModelClient, TargetModelConfig,
            )
            # Detect API-based target models (Claude, GPT, etc.) — skip vLLM
            model_id = self.config.target_model_id
            is_api_model = any(
                model_id.startswith(p)
                for p in ("claude-", "gpt-", "o1-", "o3-", "o4-")
            )
            target_cfg = TargetModelConfig(
                model_id=model_id,
                vllm_url=self.config.target_model_url,
                auto_launch_vllm=(
                    not self.config.target_model_url and not is_api_model
                ),
                temperature=self.config.target_model_temperature,
                max_tokens=self.config.target_model_max_tokens,
            )
            target_client = TargetModelClient(target_cfg, self.api)
            await target_client.start()
            logger.info(
                f"Target model client ready: {self.config.target_model_id}"
            )

        try:
            return await self._run_stages_3_to_6(
                examples, analyses, summaries, run_dir, stage,
                skip_verify, target_client,
            )
        finally:
            if target_client is not None:
                await target_client.shutdown()

    async def _run_stages_3_to_6(
        self,
        examples: list[AdaptedExample],
        analyses: list[ProblemAnalysis],
        summaries: list[str],
        run_dir: Path,
        stage: Optional[str],
        skip_verify: bool,
        target_client: Optional[object],
    ) -> dict:
        """Run stages 3-6 with shared target model client."""
        adapter = self.adapter

        # Stage 3+4: Generate + Critic (with optional feedback loop)
        if self.config.enable_feedback_loop:
            # Feedback loop: iterative generate → critic → verify → feedback
            from .critic_feedback.critic_feedback import CriticFeedbackLoop
            logger.info("Stage 3+4: Generate + Critic with feedback loop")
            max_rounds = getattr(self.config, 'feedback_max_rounds', 10)

            feedback_loop = CriticFeedbackLoop(
                api=self.api,
                config=self.config,
                adapter=self.adapter,
                max_rounds=max_rounds,
                target_model_client=target_client,
                examples=examples,
            )
            feedback_loop._prompt_logger = self._prompt_logger
            feedback_loop._tracer = self._tracer
            candidates_by_problem, round_metrics = await feedback_loop.run(
                analyses,
            )

            # Save feedback metrics
            with open(run_dir / "feedback_metrics.json", "w") as f:
                json.dump(
                    [m.to_dict() for m in round_metrics],
                    f, indent=2,
                )

            # Save generation + critic output
            gen_serializable = {
                "_metadata": self._build_generation_metadata(),
                **{
                    str(k): [c.to_dict() for c in v]
                    for k, v in candidates_by_problem.items()
                },
            }
            with open(run_dir / "generation_output.json", "w") as f:
                json.dump(gen_serializable, f, indent=2)
            with open(run_dir / "critic_output.json", "w") as f:
                json.dump(gen_serializable, f, indent=2)

            total = sum(len(v) for v in candidates_by_problem.values())
            logger.info(
                f"Feedback loop complete: {total} candidates, "
                f"{len(round_metrics)} rounds"
            )
            summaries.append(self._summarize_generate(candidates_by_problem))
            last = round_metrics[-1] if round_metrics else None
            summaries.append(
                "=" * 72 + "\n"
                "  Stage 4: Critic — SKIPPED (feedback loop handles scoring)\n"
                "=" * 72 + "\n"
                f"  Feedback loop ran {len(round_metrics)} rounds.\n"
                f"  Final LSA: {last.alignment_rate:.1%}\n"
                f"  Final MAE: {last.mae:.3f}\n"
                if last else
                "=" * 72 + "\n"
                "  Stage 4: Critic — SKIPPED (feedback loop handles scoring)\n"
                "=" * 72 + "\n"
            )
        else:
            # Standard: generate then critic (no feedback)
            candidates_by_problem = await self._run_generate(
                analyses, run_dir, stage,
            )
            summaries.append(self._summarize_generate(candidates_by_problem))
            if stage == "generate":
                self._write_summary(summaries, run_dir)
                total_candidates = sum(
                    len(c) for c in candidates_by_problem.values()
                )
                return {
                    "run_dir": str(run_dir),
                    "n_candidates": total_candidates,
                }

            # Stage 4: Critic review
            candidates_by_problem = await self._run_critic(
                analyses, candidates_by_problem, run_dir, stage,
            )
            summaries.append(self._summarize_critic(candidates_by_problem))
        if stage == "critic":
            self._write_summary(summaries, run_dir)
            passing = sum(
                sum(1 for c in cands if c.passed_critic)
                for cands in candidates_by_problem.values()
            )
            return {
                "run_dir": str(run_dir),
                "n_passing": passing,
            }

        # Stage 5: Empirical verification (skippable)
        # Verifies ALL candidates including contrastive pairs generated
        # during the feedback loop. When target_model_id is set, uses
        # the same target model so critic == verifier (same decision boundary).
        if skip_verify:
            logger.info("Skipping verification (--skip_verify). "
                        "Re-run with --stage verify to add empirical labels.")
        else:
            candidates_by_problem = await self._run_verify(
                examples, candidates_by_problem, run_dir, stage,
                target_model_client=target_client,
            )
            summaries.append(self._summarize_verify(candidates_by_problem))

            # Slim mode: retry based on VERIFIER thresholds (not critic)
            if self.config.slim_mode:
                candidates_by_problem = await self._slim_retry_loop(
                    examples, analyses, candidates_by_problem, run_dir,
                    target_model_client=target_client,
                    max_retries=2,
                )

            if stage == "verify":
                self._write_summary(summaries, run_dir)
                return {"run_dir": str(run_dir)}

        # Stage 6: Export
        # Use target_model_id as judge_model when set (critic == verifier)
        effective_judge_model = (
            self.config.target_model_id
            if self.config.target_model_id
            else self.config.judge_model
        )
        training_examples = export_training_data(
            examples=examples,
            candidates_by_problem=candidates_by_problem,
            adapter=adapter,
            judge_model=effective_judge_model,
            output_dir=run_dir,
            capability_tags=self.config.capability_tags,
            context_source=self.config.context_source,
            context_length=self.config.context_length,
            interaction_mode=self.config.interaction_mode,
        )

        # Record export data in tracer
        if self._tracer:
            for te in training_examples:
                te_dict = te.to_dict()
                self._tracer.record_export(
                    te.example_idx, te.perturbation_id, te_dict,
                )

        summaries.append(self._summarize_export(run_dir))
        summaries.append(self._summarize_output_index(run_dir))
        self._write_summary(summaries, run_dir)

        # Render tracer HTML
        if self._tracer:
            trace_path = run_dir / "trace.html"
            self._tracer.render_html(trace_path)
            logger.info(f"Pipeline trace: {trace_path}")

        return {
            "run_dir": str(run_dir),
            "n_training_examples": len(training_examples),
        }

    # =========================================================================
    # Stage execution methods
    # =========================================================================

    async def _run_decompose(
        self,
        examples: list[AdaptedExample],
        run_dir: Path,
        stage: Optional[str],
    ) -> list[ProblemAnalysis]:
        """Stage 2: Decompose problems."""
        decompose_path = run_dir / "decompose_output.json"

        # Load cached output if available (skip re-running completed stages)
        if stage != "decompose" and decompose_path.exists():
            logger.info("Loading cached decomposition")
            with open(decompose_path) as f:
                data = json.load(f)
            return [ProblemAnalysis.from_dict(d) for d in data]

        logger.info("Stage 2: Decomposing problems")
        decomposer = ProblemDecomposer(
            self.api, self.config, self.adapter,
            prompt_logger=self._prompt_logger,
            tracer=self._tracer,
        )
        analyses = await decomposer.decompose_batch(examples)

        with open(decompose_path, "w") as f:
            json.dump([a.to_dict() for a in analyses], f, indent=2)

        logger.info(f"Decomposed {len(analyses)} problems")
        return analyses

    def _build_generation_metadata(self) -> dict:
        """Build metadata dict for generation_output.json."""
        from .candidate_generator.generator import _label_to_attribution_question
        answer_labels = getattr(self.adapter.profile, 'answer_labels', None) or []
        label_questions = {}
        for label in answer_labels:
            name = label.get("name", "unknown")
            desc = label.get("description", "")
            vtype = label.get("value_type", "string")
            label_questions[name] = _label_to_attribution_question(name, desc, vtype)

        return {
            "label_axes": {
                name: {
                    "attribution_question": label_questions.get(name, ""),
                    "description": label.get("description", ""),
                    "value_type": label.get("value_type", "string"),
                }
                for label in answer_labels
                for name in [label.get("name", "unknown")]
            },
            "difficulty_tier": self.config.difficulty_tier,
        }

    async def _run_generate(
        self,
        analyses: list[ProblemAnalysis],
        run_dir: Path,
        stage: Optional[str],
    ) -> dict[int, list[PerProblemCandidate]]:
        """Stage 3: Generate candidates (includes mechanism discovery).

        The generator discovers attack mechanisms directly from decomposed
        analysis — no separate ideation stage needed.
        """
        generate_path = run_dir / "generation_output.json"

        if stage != "generate" and generate_path.exists():
            logger.info("Loading cached generation")
            with open(generate_path) as f:
                data = json.load(f)
            return {
                int(k): [PerProblemCandidate.from_dict(c) for c in v]
                for k, v in data.items()
                if k != "_metadata"
            }

        logger.info("Stage 3: Generating candidates (with integrated mechanism discovery)")
        generator = PerProblemGenerator(
            self.api, self.config, self.adapter,
            prompt_logger=self._prompt_logger,
            tracer=self._tracer,
        )
        candidates_by_problem = await generator.generate_batch(analyses)

        # Save with metadata
        serializable = {
            "_metadata": self._build_generation_metadata(),
            **{
                str(k): [c.to_dict() for c in v]
                for k, v in candidates_by_problem.items()
            },
        }
        with open(generate_path, "w") as f:
            json.dump(serializable, f, indent=2)

        total = sum(len(v) for v in candidates_by_problem.values())
        logger.info(f"Generated {total} candidates")
        return candidates_by_problem

    async def _run_critic(
        self,
        analyses: list[ProblemAnalysis],
        candidates_by_problem: dict[int, list[PerProblemCandidate]],
        run_dir: Path,
        stage: Optional[str],
    ) -> dict[int, list[PerProblemCandidate]]:
        """Stage 4: Critic review."""
        critic_path = run_dir / "critic_output.json"

        if stage != "critic" and critic_path.exists():
            logger.info("Loading cached critic output")
            with open(critic_path) as f:
                data = json.load(f)
            return {
                int(k): [PerProblemCandidate.from_dict(c) for c in v]
                for k, v in data.items()
            }

        logger.info("Stage 4: Critic review")
        critic = CandidateCritic(
            self.api, self.config, self.adapter,
            prompt_logger=self._prompt_logger,
            tracer=self._tracer,
        )
        candidates_by_problem = await critic.review_batch(
            analyses, candidates_by_problem,
        )

        # Save
        serializable = {
            str(k): [c.to_dict() for c in v]
            for k, v in candidates_by_problem.items()
        }
        with open(critic_path, "w") as f:
            json.dump(serializable, f, indent=2)

        passing = sum(
            sum(1 for c in cands if c.passed_critic)
            for cands in candidates_by_problem.values()
        )
        total = sum(len(v) for v in candidates_by_problem.values())
        logger.info(f"Critic: {passing}/{total} passed")
        return candidates_by_problem

    async def _run_verify(
        self,
        examples: list[AdaptedExample],
        candidates_by_problem: dict[int, list[PerProblemCandidate]],
        run_dir: Path,
        stage: Optional[str],
        target_model_client: Optional[object] = None,
    ) -> dict[int, list[PerProblemCandidate]]:
        """Stage 5: Empirical verification.

        When target_model_client is provided, uses it for inference
        instead of the default judge_model via safetytooling API.
        This ensures critic == verifier (same decision boundary).
        """
        verify_path = run_dir / "verification_output.json"

        if stage != "verify" and verify_path.exists():
            logger.info("Loading cached verification")
            with open(verify_path) as f:
                data = json.load(f)
            return {
                int(k): [PerProblemCandidate.from_dict(c) for c in v]
                for k, v in data.items()
            }

        logger.info("Stage 5: Empirical verification")
        verifier = EmpiricalVerifier(
            self.api, self.config, self.adapter,
            target_model_client=target_model_client,
            tracer=self._tracer,
        )
        candidates_by_problem = await verifier.verify_batch(
            examples, candidates_by_problem,
        )

        # Save
        serializable = {
            str(k): [c.to_dict() for c in v]
            for k, v in candidates_by_problem.items()
        }
        with open(verify_path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info("Verification complete")
        return candidates_by_problem

    # =========================================================================
    # Per-stage summaries
    # =========================================================================

    def _summarize_adapt(self, examples: list[AdaptedExample]) -> str:
        s = _StageSummary("Adapt (load examples)", "1")
        profile = self.adapter.profile
        s.kv("Dataset", profile.dataset_id)
        s.kv("Task type", profile.task_type)
        s.kv("Question field", profile.question_field)
        s.kv("Answer field", profile.answer_field or "(none)")
        s.kv("Answer extraction", profile.answer_extraction)
        s.kv("Instruction placement", profile.instruction_placement)
        if profile.prompt_template:
            tmpl = profile.prompt_template.replace("\n", " ")[:100]
            s.kv("Prompt template", tmpl + ("..." if len(profile.prompt_template) > 100 else ""))
        if profile.answer_labels:
            names = [l.get("name", "?") for l in profile.answer_labels]
            s.kv("Answer labels", f"{len(names)} — {names}")
        s.blank()
        s.kv("Examples loaded", f"{len(examples)} / {profile.n_total or '?'} total")
        s.section("Sample questions")
        for ex in examples[:3]:
            s.sample(f"idx={ex.idx}, answer={ex.ground_truth_answer}", ex.question)
        rendered = s.render()
        logger.info(rendered)
        return rendered

    def _summarize_decompose(self, analyses: list[ProblemAnalysis]) -> str:
        s = _StageSummary("Decompose (structural analysis)", "2")
        s.kv("Problems decomposed", len(analyses))

        # Fallback detection
        fallback_count = sum(
            1 for a in analyses
            if "(fallback" in (a.solution_sketch or "").lower()
        )
        if fallback_count:
            s.warning(f"{fallback_count} problems fell back to minimal decomposition")

        # Per-problem table
        rows = []
        for a in analyses:
            types = Counter(e.element_type for e in a.elements)
            types_str = ", ".join(f"{t}:{c}" for t, c in types.most_common())
            rows.append([str(a.example_idx), str(len(a.elements)), types_str[:50]])
        s.blank()
        s.table(["Problem", "#Elems", "Element types"], rows)

        # Sample elements from first problem
        if analyses and analyses[0].elements:
            s.section(f"Sample elements (problem {analyses[0].example_idx})")
            for elem in analyses[0].elements[:3]:
                desc_text = elem.description or ""
                desc = desc_text[:80] + ("..." if len(desc_text) > 80 else "")
                span_text = elem.text_span or ""
                span = span_text[:60] + ("..." if len(span_text) > 60 else "")
                s.sample(elem.element_type, f"{desc} | \"{span}\"")

        # Sample solution sketch
        if analyses:
            s.section("Sample solution sketch")
            sketch = (analyses[0].solution_sketch or "(none)")[:150]
            s.sample(f"problem {analyses[0].example_idx}", sketch)

        rendered = s.render()
        logger.info(rendered)
        return rendered

    def _summarize_generate(
        self,
        candidates_by_problem: dict[int, list[PerProblemCandidate]],
    ) -> str:
        s = _StageSummary("Generate (perturbation candidates)", "3")
        all_cands = [c for cs in candidates_by_problem.values() for c in cs]
        total = len(all_cands)
        skipped = sum(1 for c in all_cands if c.skip_reason is not None)
        s.kv("Total candidates", total)
        s.kv("Skipped (non-viable)", skipped)

        # By category
        cat_counts = Counter(c.category for c in all_cands if c.is_viable)
        s.blank()
        cat_rows = [[cat, str(cnt)] for cat, cnt in cat_counts.most_common()]
        s.table(["Category", "Count"], cat_rows)

        # By perturbation type
        ptype_counts = Counter(c.perturbation_type for c in all_cands if c.is_viable)
        s.blank()
        ptype_rows = [[pt, str(cnt)] for pt, cnt in ptype_counts.most_common()]
        s.table(["Perturbation type", "Count"], ptype_rows)

        # Sample per category
        shown_cats = set()
        s.section("Sample candidates")
        for c in all_cands:
            if not c.is_viable or c.category in shown_cats:
                continue
            shown_cats.add(c.category)
            lever = c.lever[:100] + ("..." if len(c.lever) > 100 else "")
            s.sample(
                f"{c.category}",
                f"{c.candidate_id} | {c.mechanism_name} | \"{lever}\"",
            )
            if len(shown_cats) >= 3:
                break

        if not all_cands:
            s.warning("No candidates generated!")

        rendered = s.render()
        logger.info(rendered)
        return rendered

    def _summarize_critic(
        self,
        candidates_by_problem: dict[int, list[PerProblemCandidate]],
    ) -> str:
        s = _StageSummary("Critic (quality review)", "4")
        all_cands = [c for cs in candidates_by_problem.values() for c in cs]
        viable = [c for c in all_cands if c.is_viable]
        passed = [c for c in all_cands if c.passed_critic]
        duplicates = sum(1 for c in viable if c.duplicate_of is not None)

        s.kv("Total candidates", len(all_cands))
        s.kv("Viable (not skipped)", len(viable))
        s.kv("Passed critic", f"{len(passed)}/{len(viable)}")
        s.kv("Duplicates", duplicates)

        # Score distributions
        cons_scores = [c.consistency_score for c in viable if c.consistency_score is not None]
        flip_scores = [c.predicted_flip_probability for c in viable if c.predicted_flip_probability is not None]
        if cons_scores:
            s.blank()
            s.kv("Consistency", f"mean={sum(cons_scores)/len(cons_scores):.2f}, "
                 f"min={min(cons_scores):.2f}, max={max(cons_scores):.2f}")
        if flip_scores:
            s.kv("Predicted flip", f"mean={sum(flip_scores)/len(flip_scores):.2f}, "
                 f"min={min(flip_scores):.2f}, max={max(flip_scores):.2f}")

        # Per-category pass rate
        cat_stats: dict[str, dict] = {}
        for c in viable:
            cat_stats.setdefault(c.category, {"total": 0, "passed": 0})
            cat_stats[c.category]["total"] += 1
            if c.passed_critic:
                cat_stats[c.category]["passed"] += 1
        s.blank()
        cat_rows = []
        for cat in sorted(cat_stats.keys()):
            st = cat_stats[cat]
            rate = f"{100 * st['passed'] / st['total']:.0f}%" if st["total"] else "N/A"
            cat_rows.append([cat, str(st["total"]), str(st["passed"]), rate])
        s.table(["Category", "Total", "Passed", "Rate"], cat_rows)

        if len(passed) == 0:
            s.warning("No candidates passed critic!")

        rendered = s.render()
        logger.info(rendered)
        return rendered

    def _summarize_verify(
        self,
        candidates_by_problem: dict[int, list[PerProblemCandidate]],
    ) -> str:
        s = _StageSummary("Verify (empirical flip testing)", "5")
        model_id = self.config.judge_model
        s.kv("Verification model", model_id)
        s.kv("Stability runs", self.config.stability_n_runs)

        all_cands = [c for cs in candidates_by_problem.values() for c in cs]
        verified = [c for c in all_cands if model_id in c.verification_results]
        skipped_critic = [c for c in all_cands if not c.passed_critic]

        s.kv("Verified", len(verified))
        s.kv("Skipped (failed critic)", len(skipped_critic))

        # Flip rate by category
        cat_stats: dict[str, dict] = {}
        for c in verified:
            vr = VerificationResult.from_dict(c.verification_results[model_id])
            cat_stats.setdefault(c.category, {"count": 0, "flipped": 0, "fractions": []})
            cat_stats[c.category]["count"] += 1
            if vr.flipped:
                cat_stats[c.category]["flipped"] += 1
            cat_stats[c.category]["fractions"].append(vr.flip_fraction)

        s.blank()
        cat_rows = []
        for cat in sorted(cat_stats.keys()):
            st = cat_stats[cat]
            avg_frac = sum(st["fractions"]) / len(st["fractions"]) if st["fractions"] else 0
            cat_rows.append([
                cat, str(st["count"]), str(st["flipped"]),
                f"{avg_frac:.2f}",
            ])
        s.table(["Category", "Verified", "Flipped", "Avg Fraction"], cat_rows)

        # Warnings
        for cat, st in cat_stats.items():
            avg = sum(st["fractions"]) / len(st["fractions"]) if st["fractions"] else 0
            if cat == "flip_inducing" and avg < 0.4:
                s.warning(f"flip_inducing avg flip fraction ({avg:.2f}) below target 0.6-0.8")
            if cat == "non_flip" and avg > 0.2:
                s.warning(f"non_flip avg flip fraction ({avg:.2f}) above target 0.0-0.1")

        # Sample results
        s.section("Sample results")
        sample_rows = []
        for c in verified[:6]:
            vr = VerificationResult.from_dict(c.verification_results[model_id])
            sample_rows.append([
                c.candidate_id[:30],
                c.category[:12],
                f"{vr.flip_fraction:.2f}",
                str(vr.baseline_answer)[:8],
                str(vr.lever_answer)[:8],
            ])
        s.table(["Candidate", "Category", "Flip", "Base", "Lever"], sample_rows)

        rendered = s.render()
        logger.info(rendered)
        return rendered

    # -----------------------------------------------------------------
    # Slim mode: retry categories that don't meet empirical thresholds
    # -----------------------------------------------------------------

    SLIM_FLIP_THRESHOLD = 2 / 3      # flip_inducing must be > 2/3
    SLIM_NONFLIP_THRESHOLD = 1 / 3   # non_flip must be < 1/3
    # boundary: between 1/3 and 2/3

    def _slim_check_threshold(self, category: str, flip_fraction: float) -> bool:
        """Check if a candidate's empirical flip_fraction meets its category threshold."""
        if category == "flip_inducing":
            return flip_fraction > self.SLIM_FLIP_THRESHOLD
        elif category == "non_flip":
            return flip_fraction < self.SLIM_NONFLIP_THRESHOLD
        elif category == "boundary":
            return self.SLIM_NONFLIP_THRESHOLD <= flip_fraction <= self.SLIM_FLIP_THRESHOLD
        return True  # Unknown category: accept

    async def _slim_retry_loop(
        self,
        examples: list[AdaptedExample],
        analyses: list,
        candidates_by_problem: dict[int, list[PerProblemCandidate]],
        run_dir: Path,
        target_model_client: Optional[object] = None,
        max_retries: int = 2,
    ) -> dict[int, list[PerProblemCandidate]]:
        """Retry categories that don't meet empirical thresholds.

        For each problem, check each candidate against its category threshold.
        If it fails, regenerate just that category, re-critic, re-verify.
        Up to max_retries per (problem, category) pair.
        """
        model_id = (
            self.config.target_model_id
            if self.config.target_model_id
            else self.config.judge_model
        )

        # Build analyses lookup (list index = example_idx)
        analyses_dict = {i: a for i, a in enumerate(analyses)} if analyses else {}

        # Count initial pass/fail
        n_pass = 0
        n_fail = 0
        failures: list[tuple[int, str]] = []  # (example_idx, category)

        for idx, candidates in candidates_by_problem.items():
            for c in candidates:
                if model_id not in c.verification_results:
                    n_fail += 1
                    failures.append((idx, c.category))
                    continue
                vr = VerificationResult.from_dict(c.verification_results[model_id])
                if self._slim_check_threshold(c.category, vr.flip_fraction):
                    n_pass += 1
                else:
                    n_fail += 1
                    failures.append((idx, c.category))

        total = n_pass + n_fail
        logger.info(
            f"\033[36m[SLIM]\033[0m Threshold check: "
            f"{n_pass}/{total} pass, {n_fail}/{total} fail. "
            f"Retrying up to {max_retries} times..."
        )

        if n_fail == 0:
            return candidates_by_problem

        # Build components for retry
        generator = PerProblemGenerator(
            self.api, self.config, self.adapter,
            prompt_logger=self._prompt_logger,
        )
        verifier = EmpiricalVerifier(
            self.api, self.config, self.adapter,
            target_model_client=target_model_client,
        )

        # Accumulate multi-turn history per (problem, category)
        from safetytooling.data_models import ChatMessage, MessageRole
        history: dict[tuple[int, str], list[ChatMessage]] = {}

        # Retry loop
        for retry_round in range(max_retries):
            if not failures:
                break

            logger.info(
                f"\033[36m[SLIM]\033[0m Retry round {retry_round + 1}: "
                f"{len(failures)} candidates to regenerate"
            )

            # Group failures by example_idx
            failures_by_idx: dict[int, list[str]] = {}
            for idx, cat in failures:
                failures_by_idx.setdefault(idx, []).append(cat)

            new_failures: list[tuple[int, str]] = []

            for idx, failed_cats in failures_by_idx.items():
                example = examples[idx]
                analysis = analyses_dict.get(idx)

                for cat in failed_cats:
                    if analysis is None:
                        new_failures.append((idx, cat))
                        continue

                    # Build feedback from the latest failed candidate
                    old_cands = [
                        c for c in candidates_by_problem[idx]
                        if c.category == cat
                    ]
                    if old_cands:
                        old = old_cands[-1]
                        old_ff = "unknown"
                        if model_id in old.verification_results:
                            old_vr = VerificationResult.from_dict(
                                old.verification_results[model_id]
                            )
                            old_ff = f"{old_vr.flip_fraction:.2f}"
                        threshold_desc = {
                            "flip_inducing": f"flip_fraction > {self.SLIM_FLIP_THRESHOLD:.2f}",
                            "non_flip": f"flip_fraction < {self.SLIM_NONFLIP_THRESHOLD:.2f}",
                            "boundary": f"{self.SLIM_NONFLIP_THRESHOLD:.2f} <= flip_fraction <= {self.SLIM_FLIP_THRESHOLD:.2f}",
                        }.get(cat, "")

                        # Append to accumulated history (multi-turn)
                        key = (idx, cat)
                        history.setdefault(key, []).extend([
                            ChatMessage(
                                role=MessageRole.assistant,
                                content=f"Here is my perturbation: {old.lever or old.baseline}",
                            ),
                            ChatMessage(
                                role=MessageRole.user,
                                content=(
                                    f"When tested on the target model, that perturbation "
                                    f"achieved flip_fraction={old_ff}, but we need "
                                    f"{threshold_desc}. Try a different approach — "
                                    f"{'make the perturbation more disruptive to the answer' if cat == 'flip_inducing' else 'make the perturbation more superficial so it does not change the answer' if cat == 'non_flip' else 'aim for a perturbation with uncertain/moderate effect on the answer'}."
                                ),
                            ),
                        ])

                    prior_messages = history.get((idx, cat))

                    try:
                        new_cands = await generator._generate_for_category(
                            analysis, cat,
                            n_to_generate=1,
                            temperature=1.0,
                            prior_messages=prior_messages,
                        )
                    except Exception as e:
                        logger.warning(
                            f"  Retry gen failed for problem {idx} {cat}: {e}"
                        )
                        new_failures.append((idx, cat))
                        continue

                    if not new_cands:
                        new_failures.append((idx, cat))
                        continue

                    new_cand = new_cands[0]

                    # Remove old failed candidate for this category
                    candidates_by_problem[idx] = [
                        c for c in candidates_by_problem[idx]
                        if c.category != cat
                    ]

                    # Skip critic on retry (just verify directly)
                    try:
                        verified = await verifier.verify_batch(
                            [example], {idx: [new_cand]},
                        )
                        new_cand = verified[idx][0]
                    except Exception as e:
                        logger.warning(
                            f"  Retry verify failed for problem {idx} {cat}: {e}"
                        )
                        new_failures.append((idx, cat))
                        continue

                    # Check threshold — keep candidate either way
                    candidates_by_problem[idx].append(new_cand)
                    if model_id in new_cand.verification_results:
                        vr = VerificationResult.from_dict(
                            new_cand.verification_results[model_id]
                        )
                        if self._slim_check_threshold(cat, vr.flip_fraction):
                            logger.info(
                                f"  \033[32mRetry OK\033[0m: problem {idx} {cat} "
                                f"flip={vr.flip_fraction:.2f}"
                            )
                        else:
                            new_failures.append((idx, cat))
                            logger.info(
                                f"  \033[33mRetry still fails\033[0m: problem {idx} {cat} "
                                f"flip={vr.flip_fraction:.2f}"
                            )
                    else:
                        new_failures.append((idx, cat))

            failures = new_failures

        # Final stats
        final_pass = 0
        final_fail = 0
        for idx, candidates in candidates_by_problem.items():
            for c in candidates:
                if model_id not in c.verification_results:
                    final_fail += 1
                    continue
                vr = VerificationResult.from_dict(c.verification_results[model_id])
                if self._slim_check_threshold(c.category, vr.flip_fraction):
                    final_pass += 1
                else:
                    final_fail += 1

        logger.info(
            f"\033[36m[SLIM]\033[0m After retries: "
            f"{final_pass}/{final_pass + final_fail} pass "
            f"(was {n_pass}/{total})"
        )

        return candidates_by_problem

    def _summarize_export(self, run_dir: Path) -> str:
        s = _StageSummary("Export (training data)", "6")
        stats_path = run_dir / "training_data_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            s.kv("Total training examples", stats.get("total_examples", 0))
            s.kv("Overall predicted flip prob", stats.get("overall_avg_predicted_flip_probability"))
            s.kv("Overall empirical flip rate", stats.get("overall_empirical_flip_rate"))

            # By category
            s.blank()
            cat_rows = []
            for cat, cs in stats.get("by_category", {}).items():
                cat_rows.append([
                    cat,
                    str(cs.get("count", 0)),
                    f"{cs.get('avg_predicted_flip_probability') or 0:.2f}",
                    f"{cs.get('avg_empirical_flip_fraction') or 'N/A'}",
                ])
            s.table(["Category", "Count", "Pred Flip", "Emp Flip"], cat_rows)

            # Mechanism distribution
            mechs = stats.get("mechanism_distribution", {})
            if mechs:
                s.section(f"Top mechanisms ({len(mechs)} total)")
                for name, cnt in list(mechs.items())[:5]:
                    s.kv(f"  {name}", cnt)

            # By perturbation type
            ptypes = stats.get("by_perturbation_type", {})
            if ptypes:
                s.blank()
                pt_rows = []
                for pt, ps in ptypes.items():
                    pt_rows.append([
                        pt,
                        str(ps.get("count", 0)),
                        f"{ps.get('avg_empirical_flip_fraction', 'N/A')}",
                        f"{ps.get('avg_edit_fraction', 'N/A')}",
                    ])
                s.table(["Pert Type", "Count", "Emp Flip", "Edit Frac"], pt_rows)
        else:
            s.warning("training_data_stats.json not found")
        rendered = s.render()
        logger.info(rendered)
        return rendered

    def _summarize_output_index(self, run_dir: Path) -> str:
        """List all output files with descriptions and sizes."""
        descriptions = {
            "config.json": "Pipeline configuration",
            "examples.json": "Loaded examples",
            "decompose_output.json": "Structural analysis",
            "generation_output.json": "Generated candidates",
            "critic_output.json": "Critic scores",
            "verification_output.json": "Empirical results",
            "training_data.jsonl": "Final training data",
            "training_data_stats.json": "Summary statistics",
            "pipeline_summary.txt": "This summary",
        }
        lines = [
            "",
            "=" * W,
            "  Output Directory",
            "=" * W,
            f"  {run_dir}",
            "",
        ]
        for fname, desc in descriptions.items():
            fpath = run_dir / fname
            if fpath.exists():
                size = os.path.getsize(fpath)
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                lines.append(f"  {fname:<30} {desc:<25} {size_str}")
            else:
                lines.append(f"  {fname:<30} (not yet created)")
        lines.append("=" * W)
        rendered = "\n".join(lines)
        logger.info(rendered)
        return rendered

    def _write_summary(self, summaries: list[str], run_dir: Path) -> None:
        """Write all accumulated summaries to pipeline_summary.txt."""
        full_text = "\n\n".join(summaries)
        summary_path = run_dir / "pipeline_summary.txt"
        with open(summary_path, "w") as f:
            f.write(full_text + "\n")
        logger.info(f"Pipeline summary: {summary_path}")
