"""
Module: prompt_attribution/auto_perturbation/config.py

Data structures and configuration for the domain-agnostic training data pipeline.

Structure:
- StructuralElement: A structural element identified in a problem
- ProblemEdit: A find-and-replace edit to a specific problem field
- ProblemAnalysis: Complete structural analysis of a problem (Stage 2 output)
- PerProblemCandidate: A perturbation candidate for a specific problem (Stage 3 output)
- VerificationResult: Empirical Phase 1 verification result (Stage 5 output)
- CategoryConfig: Per-category generation settings
- PipelineConfig: Full pipeline configuration
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# =============================================================================
# Categories
# =============================================================================


class TrainingCategory(str, Enum):
    """Training data categories for balanced perturbation generation."""

    FLIP_INDUCING = "flip_inducing"  # Target: 60-80% flip rate
    NON_FLIP = "non_flip"  # Target: 0-10% flip rate
    BOUNDARY = "boundary"  # Target: 30-50% flip rate


# =============================================================================
# Stage 2: Problem Decomposition
# =============================================================================


@dataclass
class StructuralElement:
    """A structural element identified in a problem.

    Uses generic element types that apply across any domain.

    Attributes:
        element_type: Generic category — one of: content, format, context,
                      assumption, constraint, implicit_premise
        description: Human-readable description of what this element is
        text_span: The relevant text from the problem
        lever_axes: Ways this element could be meaningfully modified
    """

    element_type: str
    description: str
    text_span: str
    lever_axes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StructuralElement":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProblemEdit:
    """A single find-and-replace edit to a specific problem field.

    Used by problem_edit perturbations to modify the original problem content
    rather than adding an instruction. Edits must be minimal and targeted.

    Attributes:
        field: Which part of the input to edit. Values:
               - "question": The question text
               - "context": The context/passage text
               - "choices": MCQ answer choices
               - "full_prompt": Arbitrary find-replace anywhere in the rendered prompt
               - "system_prompt": Edit the system-level instruction
        original: Exact substring to find in the field
        replacement: Text to replace it with
        description: Human-readable description (e.g., "Replace Perth with Brisbane")
    """

    field: str
    original: str
    replacement: str
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProblemEdit":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProblemAnalysis:
    """Complete structural analysis of a problem (Stage 2 output).

    Attributes:
        example_idx: Index of the problem in the dataset
        question: The original problem text
        ground_truth_answer: The correct answer (if available)
        elements: Structural elements identified in the problem
        solution_sketch: Brief description of the solution path
        prompt_template: Full prompt as sent to model (without perturbation instruction)
        task_type: Task type from DatasetProfile (mcq, open_numeric, etc.)
        dataset_id: HuggingFace dataset ID or registered benchmark name
    """

    example_idx: int
    question: str
    ground_truth_answer: str = ""
    elements: list[StructuralElement] = field(default_factory=list)
    solution_sketch: str = ""
    prompt_template: str = ""
    task_type: str = ""
    dataset_id: str = ""

    def to_dict(self) -> dict:
        d = {
            "example_idx": self.example_idx,
            "question": self.question,
            "ground_truth_answer": self.ground_truth_answer,
            "solution_sketch": self.solution_sketch,
            "prompt_template": self.prompt_template,
            "task_type": self.task_type,
            "dataset_id": self.dataset_id,
            "elements": [e.to_dict() for e in self.elements],
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProblemAnalysis":
        elements = [StructuralElement.from_dict(e) for e in d.get("elements", [])]
        return cls(
            example_idx=d["example_idx"],
            question=d["question"],
            ground_truth_answer=d.get("ground_truth_answer", ""),
            solution_sketch=d.get("solution_sketch", ""),
            prompt_template=d.get("prompt_template", ""),
            task_type=d.get("task_type", ""),
            dataset_id=d.get("dataset_id", ""),
            elements=elements,
        )


# =============================================================================
# Stage 3: Generation
# =============================================================================


@dataclass
class PerProblemCandidate:
    """A perturbation candidate generated for a specific problem (Stage 3 output).

    Supports two perturbation types:
    - instruction_add/replace: lever/baseline are instruction texts added to the prompt
    - problem_edit: the original problem content is modified via find-and-replace edits

    Attributes:
        candidate_id: Unique identifier (e.g., "p3_add_constraint_0")
        example_idx: Which problem this is for
        mechanism_name: From ideated mechanism (or static taxonomy family name)
        target_element: Description of the targeted structural element
        mechanism_application: How the mechanism applies to this specific element
        lever: Instruction text (instruction_add) or edit summary (problem_edit)
        baseline: Baseline instruction (usually "" for add-type perturbations)
        category: Training category (flip_inducing, non_flip, boundary)
        skip_reason: If set, this candidate was deemed non-viable
        perturbation_type: One of instruction_add, instruction_replace, problem_edit
        problem_edits: List of ProblemEdit operations (for problem_edit type)
        edit_distance: Char-level distance between baseline and lever prompts
        edit_fraction: edit_distance / len(baseline_prompt)
    """

    candidate_id: str
    example_idx: int
    mechanism_name: str
    target_element: str
    mechanism_application: str
    lever: str
    baseline: str = ""
    category: str = ""
    skip_reason: Optional[str] = None

    # Perturbation type: instruction_add | instruction_replace | problem_edit
    # For instruction_add/replace: lever/baseline contain instruction text
    # For problem_edit: lever contains human-readable edit summary,
    #   problem_edits contains the actual find-and-replace operations
    perturbation_type: str = "instruction_add"
    problem_edits: list[ProblemEdit] = field(default_factory=list)

    # Which label axis this perturbation targets (for per-axis C-index metrics)
    target_label_axis: str = ""

    # Per-candidate instruction placement (for instruction_add type)
    # If set, overrides the adapter's default placement for this candidate
    instruction_placement: str = ""

    # Edit metrics (computed at verification/export time)
    # Measures how much the prompt changed — useful as training signal
    # and for filtering overly-large edits
    edit_distance: Optional[int] = None
    edit_fraction: Optional[float] = None

    # Critic review results (populated in Stage 4)
    consistency_score: Optional[float] = None
    predicted_flip_probability: Optional[float] = None
    critic_notes: str = ""
    duplicate_of: Optional[str] = None

    # Verification results (populated in Stage 5)
    verification_results: dict[str, Any] = field(default_factory=dict)

    # Contrastive pair metadata (populated in Stage 5.5)
    contrastive_pair_id: Optional[str] = None  # shared UUID linking two candidates
    contrastive_role: str = ""  # "anchor" or "contrast"
    contrastive_source_id: Optional[str] = None  # candidate_id of the anchor

    @property
    def is_viable(self) -> bool:
        """Whether this candidate was generated (not skipped)."""
        return self.skip_reason is None

    @property
    def passed_critic(self) -> bool:
        """Whether this candidate passed critic review.

        No filtering — all viable candidates pass. The critic's
        predicted_flip_probability is used for calibration only.
        """
        return self.is_viable

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PerProblemCandidate":
        d2 = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "problem_edits" in d2 and d2["problem_edits"]:
            d2["problem_edits"] = [
                ProblemEdit.from_dict(e) if isinstance(e, dict) else e
                for e in d2["problem_edits"]
            ]
        return cls(**d2)


# =============================================================================
# Stage 5: Verification
# =============================================================================


@dataclass
class VerificationResult:
    """Empirical Phase 1 verification result for one (problem, candidate, model) triple.

    NOT for filtering — all candidates are kept. This provides empirical
    ground truth alongside the critic's predicted labels.

    Attributes:
        flipped: Majority-vote flip decision
        flip_count: Number of runs where the answer flipped
        n_runs: Total number of stability runs
        flip_fraction: flip_count / n_runs (soft label)
        baseline_answer: Parsed baseline answer (from first run)
        lever_answer: Parsed lever answer (from first run)
        baseline_responses: Full model responses for all baseline runs
        lever_responses: Full model responses for all lever runs
        parsed_baseline_answers: Parsed answers for all baseline runs
        parsed_lever_answers: Parsed answers for all lever runs
    """

    flipped: bool
    flip_count: int
    n_runs: int
    flip_fraction: float
    baseline_answer: str = ""
    lever_answer: str = ""
    baseline_responses: list[str] = field(default_factory=list)
    lever_responses: list[str] = field(default_factory=list)
    parsed_baseline_answers: list[str] = field(default_factory=list)
    parsed_lever_answers: list[str] = field(default_factory=list)
    # Extracted features from ideated answer_labels (from first run)
    features_baseline: dict[str, str] = field(default_factory=dict)
    features_lever: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VerificationResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Pipeline Configuration
# =============================================================================


@dataclass
class CategoryConfig:
    """Per-category generation settings.

    Attributes:
        n_to_generate: How many candidates to request from the LLM per problem
        temperature: LLM temperature for this category
    """

    n_to_generate: int
    temperature: float


@dataclass
class PipelineConfig:
    """Full pipeline configuration for training data generation.

    Attributes:
        generator_model: Model for decomposition, generation, and critic
        judge_model: LLM judge for answer parsing/comparison (e.g., Haiku)
        n_samples: Number of problems to process from the dataset
        stability_n_runs: Number of stability runs per (problem, candidate) pair
        concurrency: Max concurrent API calls
        random_seed: For reproducible example sampling
        dataset_id: HuggingFace dataset ID or registered benchmark name
        max_candidates_per_problem: Cap on candidates after critic filtering
        enable_feedback_loop: Whether to enable critic→generator feedback
        output_dir: Base output directory
        dump_prompts: Debug: dump full prompts and responses
        category_configs: Per-category generation settings
        category_quotas: Target fraction for each category when capping
    """

    # Generation model (used for decompose, generate, critic)
    generator_model: str = "claude-opus-4-5-20251101"
    generator_temperature: float = 1.0

    # Judge model — LLM for answer parsing/comparison (always Claude API)
    # Temperature=0.0 for deterministic extraction (aligned with shared/answer_extraction.py)
    judge_model: str = "claude-haiku-4-5-20251001"
    judge_temperature: float = 0.0

    # Experiment settings
    n_samples: int = 10
    stability_n_runs: int = 5
    concurrency: int = 100  # Generator model concurrency (Claude API)
    verification_concurrency: int = 16  # Target model concurrency (vLLM default; set higher for cloud API)
    random_seed: int = 42

    # Generation settings
    max_candidates_per_problem: int = 12

    # Optional self-improvement loop
    enable_feedback_loop: bool = False
    feedback_max_rounds: int = 10

    # Slim mode: generate 2 per category, verify with n_runs=3, filter to
    # 1 per category meeting thresholds (flip > 2/3, non_flip < 1/3, boundary in between).
    # Cuts corpus generation time by ~75% (10h → 2.5h).
    slim_mode: bool = False

    # Per-category generation: how many candidates + temperature
    category_configs: dict[str, CategoryConfig] = field(default_factory=lambda: {
        "flip_inducing": CategoryConfig(n_to_generate=5, temperature=0.9),
        "non_flip": CategoryConfig(n_to_generate=3, temperature=0.7),
        "boundary": CategoryConfig(n_to_generate=4, temperature=1.0),
    })

    # Category quotas: target fraction for each category when capping
    category_quotas: dict[str, float] = field(default_factory=lambda: {
        "flip_inducing": 0.40,
        "non_flip": 0.25,
        "boundary": 0.35,
    })

    # Dataset (HF ID or registered benchmark name)
    dataset_id: str = ""  # set explicitly via --dataset

    # Pre-computed dataset profile from discovery (avoids re-detection)
    dataset_profile: Optional[dict] = None

    # Capability tags from discovery (for training data metadata)
    capability_tags: list[str] = field(default_factory=list)

    # Difficulty tier from discovery (frontier, moderate, saturated)
    difficulty_tier: str = "saturated"

    # Complexity axes from discovery (for training data stratification)
    context_source: str = "single_source"  # single_source | multi_source | multimodal_context
    context_length: str = "short"  # short | long | multi_document
    interaction_mode: str = "static"  # static | tool_use | multi_turn

    # Target model for verification-aware feedback loop
    # When set, the feedback loop runs mini-verification against this model
    target_model_id: str = ""  # e.g. "meta-llama/Llama-3.1-8B-Instruct"
    target_model_url: str = ""  # vLLM URL, or empty for auto-launch
    target_model_temperature: float = 1.0
    target_model_max_tokens: int = 2048  # needs room for reasoning chains on harder benchmarks
    mini_verify_sample_fraction: float = 0.3  # fraction of candidates to spot-check per round

    # Contrastive pair generation (requires target_model_id)
    enable_contrastive_pairs: bool = False
    contrastive_n_attempts: int = 3

    # Output
    output_dir: Path = field(default_factory=lambda: Path("outputs/auto_perturbation"))

    # Debug
    dump_prompts: bool = False
    trace_examples: int = 0  # Trace N random examples through all stages (HTML report)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        d = d.copy()
        if "output_dir" in d:
            d["output_dir"] = Path(d["output_dir"])
        if "category_configs" in d:
            d["category_configs"] = {
                k: CategoryConfig(**v) if isinstance(v, dict) else v
                for k, v in d["category_configs"].items()
            }
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
