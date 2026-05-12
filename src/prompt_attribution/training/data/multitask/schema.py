"""
Module: prompt_attribution/training/data/multitask/schema.py

Schema for multi-task introspection training data.

Structure:
- MultitaskRecord: Extends corpus TrainingExample with task-specific fields
- TaskType: Enum of supported training tasks
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskType(str, Enum):
    """Supported introspection training tasks."""

    E1_FLIP_PREDICTION = "e1_flip_prediction"
    E2_OUTPUT_PREDICTION = "e2_output_prediction"
    E3_FLIP_PROBABILITY = "e3_flip_probability"
    E4_CORRECTNESS_PROBABILITY = "e4_correctness_probability"
    E5_CONFIDENCE_CALIBRATION = "e5_confidence_calibration"
    E6_PERTURBATION_RANKING = "e6_perturbation_ranking"
    E7_COMPONENT_ABLATION = "e7_component_ablation"
    E8_PROPOSE_FLIP = "e8_propose_flip"
    E9_FEATURE_PRESENCE = "e9_feature_presence"
    E10A_MARGIN = "e10a_margin"
    E10B_SECOND = "e10b_second"


class TemplateVariant(str, Enum):
    """Template variants per task."""

    # E1
    E1_SHOW = "e1_show"
    E1_NOSHOW = "e1_noshow"
    # E2
    E2_A_SHOW = "e2_a_show"
    E2_A_NOSHOW = "e2_a_noshow"
    E2_B = "e2_b"
    E2_C = "e2_c"
    # E4, E5
    E4_CORRECTNESS = "e4_correctness"
    E5_BASELINE = "e5_baseline"
    E5_LEVER = "e5_lever"
    # E6, E7, E8, E9 — single template
    E6_RANKING = "e6_ranking"
    E7_ABLATION = "e7_ablation"
    E8_BASE = "e8_base"
    E8_PERT = "e8_pert"
    E9_FEATURE = "e9_feature"
    # E10
    E10_MARGIN = "e10_margin"
    E10_SECOND = "e10_second"


@dataclass
class MultitaskRecord:
    """A single multi-task training example.

    Contains the task-specific prompt and GT label, plus all metadata
    from the source corpus row for traceability.
    """

    # Task identification
    task_type: str = ""  # TaskType value
    template_variant: str = ""  # TemplateVariant value

    # The Phase 2 introspection prompt (what the model sees at training time)
    task_prompt: str = ""

    # Ground truth
    gt_value: Optional[float] = None  # For continuous tasks (E3, E9)
    gt_label: str = ""  # For binary/MCQ tasks (E1: Yes/No, E6: A/B/C)
    gt_labels: list[str] = field(default_factory=list)  # All valid answers when ties exist
    gt_type: str = ""  # "continuous", "binary", "mcq", "text"

    # For E2: the target response text
    gt_text: str = ""

    # Source identification (trace back to corpus)
    unique_id: str = ""  # Original corpus row unique_id
    corpus_dir: str = ""
    dataset_id: str = ""
    example_idx: int = 0

    # Source metadata (carried from corpus for context)
    question: str = ""
    ground_truth_answer: str = ""
    lever_text: str = ""
    perturbation_type: str = ""
    category: str = ""  # flip_inducing, non_flip, boundary
    empirical_flip_fraction: Optional[float] = None
    capability_tags: list[str] = field(default_factory=list)
    target_label_axis: str = ""

    # Prompts (for reference / downstream use)
    prompt_baseline: str = ""
    prompt_lever: str = ""

    # E6-specific: the 3 perturbation options
    e6_options: list[dict] = field(default_factory=list)
    # Format: [{"letter": "A", "lever_text": "...", "flip_fraction": 0.8}, ...]

    # E7-specific: the 3 components
    e7_components: list[dict] = field(default_factory=list)
    # Format: [{"letter": "A", "description": "...", "ablation_flip_rate": 0.8}, ...]

    # E8-specific
    e8_proposed_edit: str = ""
    e8_flip_success: bool = False
    e8_edit_distance: float = 0.0

    # E9-specific
    e9_feature_name: str = ""
    e9_feature_description: str = ""

    # E10-specific
    e10_choice_probs: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        d = {}
        for k, v in self.__dict__.items():
            if v is None or v == "" or v == [] or v == {}:
                continue
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                d[k] = v
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MultitaskRecord":
        """Deserialize from dict, ignoring unknown fields."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)
