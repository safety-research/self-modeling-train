"""
Module: prompt_attribution/auto_perturbation/dataset_adapter.py

Zero-config dataset onboarding. Given a HuggingFace dataset ID, auto-detects
field mappings, task type, prompt template, and answer verification strategy.

Structure:
- TaskType: Enum of supported task types
- DatasetProfile: Auto-detected dataset characteristics
- DatasetDetector: Programmatic heuristics for field/type detection
- AdaptedExample: Generic example from any adapted dataset
- DatasetAdapter: Dynamic adapter implementing prompt-building interface
"""

import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from datasets import load_dataset, load_dataset_builder

if TYPE_CHECKING:
    from ..discovery_tracer import DatasetProfilingTrace

logger = logging.getLogger(__name__)


# =============================================================================
# Task Types
# =============================================================================


class TaskType(str, Enum):
    """Supported task types for auto-adapted datasets.

    Two primary types:
    - MCQ: has choices → prompt includes choice list, extract letter answer
    - OPEN: no choices → free-form response, label ideation provides extraction

    Legacy aliases (open_numeric, classification, yes_no, safety_refusal) are
    kept for backward compatibility with cached profiles and registered
    benchmarks, but all resolve to OPEN behavior. The actual parsing strategy
    comes from label ideation (extraction_pattern / judge_prompt), not the
    task type.
    """

    MCQ = "mcq"  # Multiple choice (A/B/C/D...)
    OPEN = "open"  # Open-ended (all non-MCQ types)

    # Legacy aliases — kept for cached profiles and benchmark_adapter mappings.
    # Detection code now maps these to OPEN; answer_parser still recognizes
    # them for fallback parsing when labels are absent.
    OPEN_NUMERIC = "open_numeric"
    OPEN_TEXT = "open_text"
    CLASSIFICATION = "classification"
    YES_NO = "yes_no"
    SAFETY_REFUSAL = "safety_refusal"


# =============================================================================
# Dataset Profile
# =============================================================================


@dataclass
class DatasetProfile:
    """Auto-detected dataset characteristics.

    Attributes:
        dataset_id: HuggingFace dataset ID (e.g., "user/dataset")
        config_name: Dataset config/subset name (e.g., "main", "high_school_math")
        split: Dataset split to use (e.g., "test")
        task_type: Detected task type
        question_field: Column name for question/prompt
        answer_field: Column name for ground truth answer (may be None)
        choices_field: Column name for MCQ choices (if MCQ)
        context_field: Column name for context/passage (if present)
        label_names: List of label names (for CLASSIFICATION)
        n_choices: Number of choices (for MCQ)
        answer_extraction: How to extract answer from model output
        prompt_template: Auto-generated prompt template string
        instruction_placement: Where lever instruction goes ("append" or "prepend")
        n_total: Total examples in the selected split
    """

    dataset_id: str
    config_name: Optional[str] = None
    split: str = "test"
    task_type: str = TaskType.OPEN_TEXT.value
    question_field: str = "question"
    answer_field: Optional[str] = "answer"
    choices_field: Optional[str] = None
    context_field: Optional[str] = None
    label_names: list[str] = field(default_factory=list)
    n_choices: int = 4
    answer_extraction: str = "exact_match"  # boxed, json_letter, last_number, exact_match, feature_based
    prompt_template: str = ""
    instruction_placement: str = "append"
    n_total: int = 0
    # LLM-ideated answer labels for open_text datasets (from label_ideation.py)
    # Each entry describes a feature to extract from model responses for flip detection
    answer_labels: list[dict] = field(default_factory=list)
    # Human-readable label descriptions (populated after label ideation)
    # Included in decomposer and generator prompts for context
    label_descriptions: str = ""
    # Dot-path extraction for nested fields (e.g., "0.question" for list-of-dicts)
    # Set by LLM detection when question_field contains structured data
    question_extraction: Optional[str] = None
    choices_extraction: Optional[str] = None
    # Whether detect_with_llm() has already been applied to this profile.
    # Prevents redundant LLM calls when loading cached profiles.
    llm_detected: bool = False
    # If True, generator only produces problem_edit (no instruction_add).
    # Used for BLOOM scenarios where instruction_add produces meta-instructions
    # that don't read like natural scenario modifications.
    only_problem_edit: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def cache_key(self) -> str:
        """Generate a cache key for this profile."""
        key = f"{self.dataset_id}:{self.config_name or 'default'}:{self.split}"
        return hashlib.md5(key.encode()).hexdigest()


# =============================================================================
# Adapted Example
# =============================================================================


@dataclass
class AdaptedExample:
    """Generic example from any adapted dataset.

    Attributes:
        idx: Example index
        question: The question/problem text
        ground_truth_answer: The correct answer (if available)
        choices: MCQ choices list (if MCQ)
        context: Context/passage text (if present)
        metadata: Any additional dataset-specific fields
    """

    idx: int
    question: str
    ground_truth_answer: str = ""
    choices: Optional[list[str]] = None
    context: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "idx": self.idx,
            "question": self.question,
            "ground_truth_answer": self.ground_truth_answer,
        }
        if self.choices is not None:
            d["choices"] = self.choices
        if self.context is not None:
            d["context"] = self.context
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# =============================================================================
# Dataset Detector
# =============================================================================


# Priority-ordered field name candidates for each role
_QUESTION_FIELDS = [
    "question", "prompt", "input", "problem", "query", "text",
    "sentence", "premise", "hypothesis", "instruction",
]
_ANSWER_FIELDS = [
    "answer", "target", "output", "solution", "response",
    "gold", "expected",
]
_CHOICES_FIELDS = [
    "choices", "options", "answers", "candidates",
]
_CONTEXT_FIELDS = [
    "context", "passage", "paragraph", "document", "article",
    "support", "evidence",
]
_LABEL_FIELDS = [
    "label", "labels", "class", "category",
]
_SAFETY_KEYWORDS = [
    "safety", "harmful", "jailbreak", "toxic", "refusal",
    "adversarial", "attack", "malicious", "dangerous",
]


class DatasetDetector:
    """Programmatic heuristics to auto-detect dataset characteristics.

    Strategy: Use HF dataset metadata (features schema, description) plus
    sample inspection. LLM fallback only for truly ambiguous cases.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        self._hf_token = self._get_hf_token()
        self._detection_log: list[str] = []  # Accumulated reasoning for tracer
        self._sample_cache: dict[str, tuple[list[dict], int]] = {}  # key → (samples, n_total)

    @staticmethod
    def _get_hf_token() -> Optional[str]:
        """Get HuggingFace token from environment or cached login.

        Checks (in order):
        1. HF_TOKEN environment variable
        2. HUGGING_FACE_HUB_TOKEN environment variable (legacy)
        3. huggingface-cli login cache (~/.huggingface/token or ~/.cache/huggingface/token)
        """
        import os
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            return token

        # Check cached login
        for path in [
            Path.home() / ".huggingface" / "token",
            Path.home() / ".cache" / "huggingface" / "token",
        ]:
            if path.exists():
                token = path.read_text().strip()
                if token:
                    return token

        return None

    def detect(self, dataset_id: str, config_name: Optional[str] = None) -> DatasetProfile:
        """Detect dataset characteristics and return a DatasetProfile.

        Args:
            dataset_id: HuggingFace dataset ID (e.g., "user/dataset")
            config_name: Optional config/subset name

        Returns:
            DatasetProfile with detected characteristics
        """
        # Check cache first
        cache_key = hashlib.md5(
            f"{dataset_id}:{config_name or 'default'}".encode()
        ).hexdigest()
        if self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}.json"
            if cache_path.exists():
                logger.info(f"Loading cached profile for {dataset_id}")
                with open(cache_path) as f:
                    return DatasetProfile.from_dict(json.load(f))

        logger.info(f"Auto-detecting profile for {dataset_id}")
        self._detection_log = []  # Reset for this detection run

        # Step 1: Load metadata (single _load_builder call, reused for splits)
        features = None
        description = ""
        builder = None

        try:
            builder = self.retry_hf_call(
                lambda: self._load_builder(dataset_id, config_name)
            )
            features = builder.info.features
            description = builder.info.description or ""

            # Auto-detect config if required
            if config_name is None and features is None:
                configs = builder.builder_configs
                if configs:
                    config_name = list(configs.keys())[0]
                    logger.info(f"Auto-selected config: {config_name}")
                    builder = self.retry_hf_call(
                        lambda: self._load_builder(dataset_id, config_name)
                    )
                    features = builder.info.features
                    description = builder.info.description or ""
        except Exception as e:
            # Check for "config name is missing" error and try to extract configs
            err_msg = str(e)
            if "Config name is missing" in err_msg or "config" in err_msg.lower():
                config_name = self._extract_first_config(err_msg)
                if config_name:
                    logger.info(f"Auto-detected config from error: {config_name}")
                    try:
                        builder = self.retry_hf_call(
                            lambda: self._load_builder(dataset_id, config_name)
                        )
                        features = builder.info.features
                        description = builder.info.description or ""
                    except Exception:
                        pass

        # Determine split (reuse builder to avoid extra API call)
        split = self._detect_split_from_builder(builder, dataset_id, config_name)

        # Get n_total from builder (already loaded, no extra API call)
        n_total = 0
        if builder and builder.info and builder.info.splits:
            split_info = builder.info.splits.get(split)
            if split_info:
                n_total = split_info.num_examples

        # Load a small sample for inspection (streaming to avoid full download)
        try:
            samples, stream_n_total = self.retry_hf_call(
                lambda: self._load_samples(dataset_id, config_name, split, n=20)
            )
            if n_total == 0:
                n_total = stream_n_total
            if features is None and samples:
                # Infer features from sample keys
                features = None  # Will use column_names from samples directly
        except Exception as e:
            raise ValueError(
                f"Cannot load dataset {dataset_id} (config={config_name}, "
                f"split={split}): {e}"
            )

        column_names = list(features.keys()) if features else list(samples[0].keys())

        # Step 2: Detect field roles
        question_field = self._detect_field(
            column_names, _QUESTION_FIELDS, samples, role="question"
        )
        answer_field = self._detect_field(
            column_names, _ANSWER_FIELDS + _LABEL_FIELDS, samples, role="answer",
            required=False,
        )
        choices_field = self._detect_choices_field(column_names, features, samples)
        context_field = self._detect_field(
            column_names, _CONTEXT_FIELDS, samples, role="context", required=False
        )

        # Step 3: Detect task type
        task_type, label_names, n_choices = self._detect_task_type(
            features, samples, answer_field, choices_field, description,
        )

        # Step 3b: Pair-input fallback for open with label_names — look for
        # a second text field (NLI, paraphrase, entailment tasks).
        # Only for open type with label vocabulary (classification-like).
        _solution_names = {
            "solution", "explanation", "rationale", "reasoning", "cot",
            "chain_of_thought", "steps", "url", "link", "source", "id",
        }
        if (context_field is None
                and task_type == TaskType.OPEN.value and label_names
                and question_field and samples):
            assigned = {question_field, answer_field, choices_field}
            assigned.discard(None)
            unassigned_text = []
            for col in column_names:
                if col in assigned or col.lower() in _solution_names:
                    continue
                vals = [s.get(col) for s in samples[:5] if isinstance(s.get(col), str)]
                if vals:
                    avg_len = sum(len(v) for v in vals) / len(vals)
                    if avg_len > 20:
                        unassigned_text.append((col, avg_len))
            if unassigned_text:
                unassigned_text.sort(key=lambda x: x[1], reverse=True)
                context_field = unassigned_text[0][0]
                logger.info(
                    f"Pair-input detected: using '{context_field}' as context "
                    f"(avg {unassigned_text[0][1]:.0f} chars)"
                )

        # Step 4: Determine answer extraction strategy
        answer_extraction = self._detect_answer_extraction(task_type, samples, answer_field)

        # Step 5: Determine instruction placement (heuristic, may be updated by LLM ideation later)
        instruction_placement = self._detect_instruction_placement(task_type, description)

        # Step 6: Generate prompt template with instruction at detected position
        prompt_template = self._generate_prompt_template(
            task_type, context_field, choices_field, n_choices, label_names,
            instruction_placement=instruction_placement,
        )

        # Step 7: For open type, switch to feature_based extraction
        # (actual label ideation happens async via ideate_answer_labels())
        if task_type == TaskType.OPEN.value:
            answer_extraction = "feature_based"

        profile = DatasetProfile(
            dataset_id=dataset_id,
            config_name=config_name,
            split=split,
            task_type=task_type,
            question_field=question_field,
            answer_field=answer_field,
            choices_field=choices_field,
            context_field=context_field,
            label_names=label_names,
            n_choices=n_choices,
            answer_extraction=answer_extraction,
            prompt_template=prompt_template,
            instruction_placement=instruction_placement,
            n_total=n_total,
        )

        # Store detection metadata (not serialized, used by tracer/discovery)
        profile._column_names = column_names
        profile._detection_notes = list(self._detection_log)
        profile._cached_samples = samples[:5] if samples else []
        profile._features = features  # HF Features object for non-text detection

        # Cache profile
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_dir / f"{cache_key}.json", "w") as f:
                json.dump(profile.to_dict(), f, indent=2)

        logger.info(
            f"Detected: task_type={task_type}, question={question_field}, "
            f"answer={answer_field}, choices={choices_field}, "
            f"context={context_field}, n_total={n_total}"
        )
        return profile

    async def detect_with_llm(
        self,
        profile: DatasetProfile,
        api: "InferenceAPI",
        model_id: str = "claude-haiku-4-5-20251001",
        _ds_trace: Optional["DatasetProfilingTrace"] = None,
    ) -> DatasetProfile:
        """LLM-first dataset detection: the LLM reads the raw schema and
        writes the complete prompt template.

        This is the primary detection method. The LLM sees column names +
        sample rows and outputs:
        1. Field mapping (question, answer, choices, context)
        2. Task type and metadata
        3. The FULL prompt template as a natural, grammatically correct string
        4. A rendered sample_prompt for validation

        The LLM writes the template itself — no heuristic _generate_prompt_template().
        This handles all edge cases: spread-column MCQ, pair-input, coreference
        framing, instruction-following, etc.

        Args:
            profile: Heuristic-detected profile (from detect()) as starting point
            api: InferenceAPI for LLM calls
            model_id: Model to use

        Returns:
            Updated DatasetProfile with LLM-generated template
        """
        # Skip if already LLM-detected (cached profile from a previous run)
        if profile.llm_detected:
            logger.info(
                f"Skipping LLM detection for {profile.dataset_id} (already done)"
            )
            return profile

        from safetytooling.data_models import ChatMessage, MessageRole, Prompt

        # Load samples — prefer cached from detect(), fallback to re-stream
        samples = getattr(profile, '_cached_samples', None)
        if not samples:
            try:
                samples, _ = DatasetDetector.retry_hf_call(
                    lambda: self._load_samples(
                        profile.dataset_id, profile.config_name,
                        profile.split, n=3,
                    ),
                    timeout=30.0,
                )
            except Exception:
                logger.warning(
                    f"Cannot load samples for LLM detection, keeping heuristic profile"
                )
                return profile

        if not samples:
            return profile

        # Build schema with 2 sample rows for the LLM
        column_names = list(samples[0].keys())
        schema_lines = []
        for col in column_names:
            row_previews = []
            for i in range(min(2, len(samples))):
                val = samples[i].get(col)
                if isinstance(val, str):
                    row_previews.append(f'"{val[:150]}"')
                elif isinstance(val, list):
                    row_previews.append(str(val)[:150])
                else:
                    row_previews.append(str(val)[:80])
            schema_lines.append(
                f'  "{col}" ({type(samples[0].get(col)).__name__}):\n'
                f'    row0: {row_previews[0]}\n'
                + (f'    row1: {row_previews[1]}' if len(row_previews) > 1 else '')
            )

        # Get dataset description from HF metadata
        description = ""
        try:
            builder = self.retry_hf_call(
                lambda: self._load_builder(profile.dataset_id, profile.config_name),
                timeout=10.0,
            )
            raw_desc = builder.info.description or ""
            description = raw_desc[:300]
            if len(raw_desc) > 300:
                description += " [truncated]"
        except Exception:
            pass

        prompt_text = f"""You are building a prompt template for an LLM evaluation pipeline. Given a HuggingFace dataset, you must:
1. Identify which columns map to which roles
2. Write the COMPLETE prompt template that a language model will answer

## Dataset: {profile.dataset_id}
{f"## Description: {description}" if description else ""}
## Columns with sample values:
{chr(10).join(schema_lines)}

## Output Format
Return a JSON object with these fields:

"question_field": Column name for the main question/input text (REQUIRED)
"answer_field": Column name for ground truth answer (null if none)
"choices_fields": For MCQ — list of column names containing answer choices. Examples:
  - Single column with list: ["choices"]
  - Spread across columns: ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
  - null if not MCQ
"context_field": Column providing supporting context the model needs to read alongside the question (passage for reading comprehension, second sentence for paraphrase). null if question is self-contained. NEVER use solution/explanation/rationale columns.
"task_type": One of "mcq", "open_numeric", "open_text", "classification", "yes_no", "safety_refusal"
"label_names": For classification, list of possible labels (e.g., ["positive", "negative"]). Empty list otherwise.
"n_choices": Number of MCQ choices (0 if not MCQ)
"is_text_only": false if questions reference images/audio/video not available as text. true otherwise.
"prompt_template": The COMPLETE prompt string with placeholders. This is the most important field. Rules:
  - Must be a natural, grammatically correct instruction that makes sense to someone who has NEVER seen this dataset
  - ONLY use these exact placeholders: {{question}}, {{context}}, {{choices}}, {{instruction}}. Do NOT invent custom placeholders like {{pronoun}} or {{quote}} — those won't be filled in.
  - {{question}} = the text from question_field
  - {{context}} = the text from context_field (omit if context_field is null)
  - {{choices}} = auto-formatted lettered options from choices_fields (omit if not MCQ)
  - {{instruction}} = where perturbation instructions get inserted (usually near end, before response format)
  - Must end with a clear response format (e.g., 'Respond with JSON: {{"answer": "<letter>"}}')
  - For classification: must mention the possible labels
  - Do NOT include internal pipeline metadata, label descriptions, or answer extraction hints
"question_extraction": If the question_field contains nested/structured data (e.g., a JSON object or list of dicts with sub-fields like 'question', 'options'), specify how to extract the actual question text. Format: a dot-path like "0.question" meaning "take the first element, then the 'question' key". null if the field is already a plain string.
"choices_extraction": If choices are inside a nested structure (e.g., inside the question field as an 'options' sub-field), specify the dot-path. e.g., "0.options". null if choices_fields already points to the right columns.
"sample_prompt": The fully rendered prompt for row 0, with all placeholders filled in using actual data. For MCQ, show the actual choices as A), B), C), D).

## Critical Rules
- Solution/explanation/reasoning/rationale columns are NEVER context — they are metadata to exclude
- If questions reference "the image", "the figure", "the audio", "the video" etc., set is_text_only to false
- The prompt_template must make complete sense on its own. A person reading it should understand exactly what task to perform without any external context.
- For datasets where the question field contains raw data without framing (e.g., just a sentence for sentiment analysis, or two sentences for paraphrase detection), YOU must add the framing instruction.
- NEVER use custom placeholders. Only {{question}}, {{context}}, {{choices}}, {{instruction}} are valid.

Output ONLY the JSON object:"""

        from ..utils.retry import retry_async

        async def _do_llm_detect():
            responses = await api(
                model_id=model_id,
                prompt=Prompt(messages=[
                    ChatMessage(role=MessageRole.user, content=prompt_text),
                ]),
                n=1,
                temperature=0.0,
                max_tokens=2048,
            )
            text = responses[0].completion if responses else ""

            # Record LLM call in tracer
            if _ds_trace:
                from ..discovery_tracer import DiscoveryLLMCall
                _ds_trace.adapter_llm = DiscoveryLLMCall(
                    stage="detect_with_llm", prompt=prompt_text,
                    response=text, model=model_id, temperature=0.0,
                    label=profile.dataset_id,
                )

            json_match = re.search(r'\{[\s\S]*\}', text)
            if not json_match:
                raise ValueError(f"No JSON in LLM response for {profile.dataset_id}")

            return json.loads(json_match.group())

        try:
            result = await retry_async(
                _do_llm_detect,
                stage_name="detect_with_llm",
                item_id=profile.dataset_id,
                api=api,
            )
        except Exception as e:
            logger.error(
                f"LLM detection FAILED for {profile.dataset_id} after retries: {e}, "
                f"keeping heuristic profile"
            )
            return profile

        # Check is_text_only
        if not result.get("is_text_only", True):
            logger.warning(
                f"LLM: {profile.dataset_id} requires non-text modality"
            )
            profile.task_type = "non_text"
            return profile

        # Apply field mapping
        if result.get("question_field") and result["question_field"] in column_names:
            profile.question_field = result["question_field"]

        if result.get("answer_field"):
            if result["answer_field"] in column_names:
                profile.answer_field = result["answer_field"]
        else:
            profile.answer_field = None

        if result.get("context_field"):
            if result["context_field"] in column_names:
                profile.context_field = result["context_field"]
        else:
            profile.context_field = None

        # Handle choices_fields (now always a list or null)
        choices_fields = result.get("choices_fields")
        if choices_fields and isinstance(choices_fields, list):
            valid_cols = [c for c in choices_fields if c in column_names]
            if len(valid_cols) == 1:
                profile.choices_field = valid_cols[0]
            elif len(valid_cols) > 1:
                profile.choices_field = "__multi__" + ",".join(valid_cols)
            profile.n_choices = result.get("n_choices", len(valid_cols))
        else:
            profile.choices_field = None

        if result.get("task_type"):
            profile.task_type = result["task_type"]
        if result.get("label_names"):
            profile.label_names = result["label_names"]
        if result.get("n_choices"):
            profile.n_choices = result["n_choices"]

        # Store nested extraction paths
        if result.get("question_extraction"):
            profile.question_extraction = result["question_extraction"]
        if result.get("choices_extraction"):
            profile.choices_extraction = result["choices_extraction"]

        # MCQ without choices → downgrade
        if profile.task_type == "mcq" and not profile.choices_field:
            logger.warning(
                f"MCQ but no choices for {profile.dataset_id}, "
                f"downgrading to open_text"
            )
            profile.task_type = "open_text"
            profile.n_choices = 0

        # Use the LLM-generated prompt template
        llm_template = result.get("prompt_template", "")
        if llm_template and "{question}" in llm_template and "{instruction}" in llm_template:
            profile.prompt_template = llm_template
            # Detect instruction placement from template position
            q_pos = llm_template.find("{question}")
            i_pos = llm_template.find("{instruction}")
            if i_pos < q_pos:
                profile.instruction_placement = "prepend"
            else:
                profile.instruction_placement = "append"
        else:
            # Fallback: generate template from heuristics
            logger.warning(
                f"LLM template invalid for {profile.dataset_id}, "
                f"using heuristic template"
            )
            profile.prompt_template = self._generate_prompt_template(
                task_type=profile.task_type,
                context_field=profile.context_field,
                choices_field=profile.choices_field,
                n_choices=profile.n_choices,
                label_names=profile.label_names,
                instruction_placement=profile.instruction_placement,
            )

        # Detect answer extraction
        profile.answer_extraction = self._detect_answer_extraction(
            profile.task_type, samples, profile.answer_field,
        )

        # Validate: check sample_prompt contains actual question text
        sample_prompt = result.get("sample_prompt", "")
        if sample_prompt:
            q_text = str(samples[0].get(profile.question_field, ""))[:50]
            if q_text and q_text not in sample_prompt:
                logger.warning(
                    f"LLM sample_prompt doesn't contain question text for "
                    f"{profile.dataset_id}, template may be wrong"
                )

        # Cache
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = hashlib.md5(
                f"{profile.dataset_id}:{profile.config_name or 'default'}".encode()
            ).hexdigest()
            with open(self.cache_dir / f"{cache_key}.json", "w") as f:
                json.dump(profile.to_dict(), f, indent=2)

        profile.llm_detected = True
        logger.info(
            f"LLM-detected: task_type={profile.task_type}, "
            f"question={profile.question_field}, answer={profile.answer_field}, "
            f"choices={profile.choices_field}, context={profile.context_field}"
        )
        return profile

    async def ideate_answer_labels(
        self,
        profile: DatasetProfile,
        api: "InferenceAPI",
        model_id: str = "claude-haiku-4-5-20251001",
        _ds_trace: Optional["DatasetProfilingTrace"] = None,
    ) -> DatasetProfile:
        """Run LLM-based label ideation for open_text and safety_refusal datasets.

        Analyzes sample Q+A pairs to identify extractable features for
        flip detection. Call this after detect() for open_text/safety datasets.

        For other task types (mcq, classification, etc.), returns unchanged.

        Args:
            profile: The detected DatasetProfile
            api: InferenceAPI for LLM calls
            model_id: Model to use for ideation

        Returns:
            Updated DatasetProfile with answer_labels populated
        """
        # Label ideation runs for all task types — discovers richer features
        # beyond just answer correctness (reasoning quality, confidence, etc.)

        if profile.answer_labels:
            # Already ideated (e.g., loaded from cache)
            return profile

        from .label_ideation import LabelIdeator

        # Load samples — prefer cached from detect(), fallback to re-stream
        samples = getattr(profile, '_cached_samples', None)
        if not samples:
            try:
                samples, _ = self.retry_hf_call(
                    lambda: self._load_samples(
                        profile.dataset_id, profile.config_name,
                        profile.split, n=5,
                    ),
                    timeout=30.0,
                )
            except Exception as e:
                logger.warning(f"Cannot load samples for label ideation: {e}")
                return profile

        ideator = LabelIdeator(api, model_id)
        labels, placement = await ideator.ideate_labels(
            dataset_id=profile.dataset_id,
            task_type=profile.task_type,
            samples=samples,
            question_field=profile.question_field,
            answer_field=profile.answer_field,
            label_names=profile.label_names,
            _ds_trace=_ds_trace,
        )

        profile.answer_labels = [l.to_dict() for l in labels]

        # Update instruction placement if LLM suggested a different one
        if placement != profile.instruction_placement:
            logger.info(
                f"LLM suggested placement '{placement}' "
                f"(was '{profile.instruction_placement}')"
            )
            profile.instruction_placement = placement
            # Regenerate template with new placement
            profile.prompt_template = self._generate_prompt_template(
                profile.task_type,
                profile.context_field,
                profile.choices_field,
                profile.n_choices,
                profile.label_names,
                instruction_placement=placement,
            )

        # Update cache
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = hashlib.md5(
                f"{profile.dataset_id}:{profile.config_name or 'default'}".encode()
            ).hexdigest()
            with open(self.cache_dir / f"{cache_key}.json", "w") as f:
                json.dump(profile.to_dict(), f, indent=2)

        return profile

    def _load_builder(self, dataset_id: str, config_name: Optional[str] = None):
        """Load dataset builder with token if available."""
        return load_dataset_builder(dataset_id, config_name, token=self._hf_token)

    def _load_dataset(self, dataset_id: str, config_name: Optional[str] = None, split: str = "train", streaming: bool = False):
        """Load dataset with token and trust_remote_code.

        Args:
            streaming: If True, returns an iterable dataset that doesn't
                download the full data upfront. Use for profiling/sampling.
        """
        import logging as _logging
        _ds_logger = _logging.getLogger("datasets.load")
        _orig_level = _ds_logger.level
        _ds_logger.setLevel(_logging.CRITICAL)  # suppress [ERROR] spam before we catch it
        try:
            return load_dataset(
                dataset_id, config_name, split=split,
                trust_remote_code=True, token=self._hf_token,
                streaming=streaming,
            )
        except RuntimeError as e:
            if "Dataset scripts are no longer supported" not in str(e):
                raise
            # Fallback: find raw data files (json/jsonl/csv/parquet) in the
            # repo and load them directly, bypassing the unsupported .py script.
            logger.info(
                f"[FALLBACK] {dataset_id}: legacy script detected, trying raw files / parquet branch"
            )
            return self._load_dataset_from_raw_files(
                dataset_id, config_name, split, streaming,
            )
        finally:
            _ds_logger.setLevel(_orig_level)

    def _load_dataset_from_raw_files(
        self,
        dataset_id: str,
        config_name: Optional[str],
        split: str,
        streaming: bool,
    ):
        """Fallback loader: find raw data files in the HF repo and load directly.

        Used when the dataset has a legacy .py script that datasets>=4.x rejects.
        Tries two approaches:
        1. Raw data files on main branch (json/jsonl/csv/parquet)
        2. Auto-converted parquet on refs/convert/parquet branch (HF generates
           these for dataset previews)
        """
        from huggingface_hub import HfApi
        api = HfApi()

        # Approach 1: raw data files on main branch (including zip-extracted CSVs)
        info = api.dataset_info(dataset_id, token=self._hf_token)
        zip_files = [
            s.rfilename for s in info.siblings
            if s.rfilename.endswith(".zip") and not s.rfilename.startswith(".")
        ]
        data_files = [
            s.rfilename for s in info.siblings
            if s.rfilename.endswith((".json", ".jsonl", ".csv", ".parquet"))
            and not s.rfilename.startswith(".")
        ]
        # Approach 1a: zip file containing CSVs (e.g. lmlmcat/cmmlu)
        if not data_files and zip_files:
            import zipfile
            from huggingface_hub import hf_hub_download
            zip_path = hf_hub_download(
                dataset_id, zip_files[0], repo_type="dataset", token=self._hf_token
            )
            with zipfile.ZipFile(zip_path) as z:
                csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                # Prefer split-matching files; fall back to all
                split_csvs = [n for n in csv_names if split in n.lower()]
                chosen_csvs = split_csvs or csv_names
                import pandas as pd
                dfs = []
                for name in chosen_csvs:
                    with z.open(name) as f:
                        dfs.append(pd.read_csv(f))
                df = pd.concat(dfs, ignore_index=True)
            from datasets import Dataset
            logger.info(
                f"Loading {dataset_id} from zip {zip_files[0]}: {len(df)} rows"
            )
            ds = Dataset.from_pandas(df)
            return ds if not streaming else ds.to_iterable_dataset()
        if data_files:
            split_matches = [f for f in data_files if split in f.lower()]
            chosen = split_matches[0] if split_matches else data_files[0]
            fmt = "parquet" if chosen.endswith(".parquet") else "json" if chosen.endswith((".json", ".jsonl")) else "csv"
            hf_path = f"hf://datasets/{dataset_id}/{chosen}"
            logger.info(f"Loading {dataset_id} from raw file: {chosen} (format={fmt})")
            return load_dataset(
                fmt, data_files=hf_path, split="train",
                token=self._hf_token, streaming=streaming,
            )

        # Approach 2: HF auto-converted parquet on refs/convert/parquet branch
        try:
            parquet_files = []
            for item in api.list_repo_tree(
                dataset_id, repo_type="dataset",
                revision="refs/convert/parquet", recursive=True,
            ):
                if hasattr(item, "path") and item.path.endswith(".parquet"):
                    parquet_files.append(item.path)

            if parquet_files:
                # Match config_name and split in the path
                # Pattern: {config}/{split}/0000.parquet
                candidates = parquet_files
                if config_name:
                    candidates = [f for f in candidates if f.startswith(f"{config_name}/")]
                split_matches = [f for f in candidates if f"/{split}/" in f]
                if split_matches:
                    chosen = split_matches[0]
                elif candidates:
                    chosen = candidates[0]
                else:
                    chosen = parquet_files[0]

                hf_path = f"hf://datasets/{dataset_id}@refs/convert/parquet/{chosen}"
                logger.info(
                    f"Loading {dataset_id} from auto-converted parquet: {chosen}"
                )
                return load_dataset(
                    "parquet", data_files=hf_path, split="train",
                    token=self._hf_token, streaming=streaming,
                )
        except Exception as e:
            logger.debug(f"No parquet convert branch for {dataset_id}: {e}")

        raise RuntimeError(
            f"Dataset {dataset_id} uses an unsupported .py script and "
            f"has no raw data files or auto-converted parquet to fall back on"
        )

    def _load_samples(self, dataset_id: str, config_name: Optional[str], split: str, n: int = 20) -> tuple[list[dict], int]:
        """Load a small sample from a dataset using streaming only.

        Results are cached per (dataset_id, config_name, split) so repeated
        calls within the same detection run don't re-stream from HF.
        Returns up to `n` samples from cache (may have more if a previous
        call requested more).

        Returns:
            (samples, n_total) — n_total is estimated from dataset info,
            falls back to len(samples) if unavailable.
        """
        # Normalize cache key: if dataset_id contains ":config", extract it
        # so the key is always "org/name:config:split" regardless of how
        # dataset_id and config_name are passed.
        if ":" in dataset_id and config_name is None:
            base_id, embedded_config = dataset_id.rsplit(":", 1)
            cache_key = f"{base_id}:{embedded_config}:{split}"
        else:
            cache_key = f"{dataset_id}:{config_name or 'default'}:{split}"

        # Return cached samples if we have enough (in-memory first)
        if cache_key in self._sample_cache:
            cached_samples, cached_n_total = self._sample_cache[cache_key]
            if len(cached_samples) >= n:
                return cached_samples[:n], cached_n_total

        # Check disk cache (survives across runs)
        if self.cache_dir:
            import hashlib
            disk_key = hashlib.md5(cache_key.encode()).hexdigest()
            disk_path = self.cache_dir / f"samples_{disk_key}.json"
            if disk_path.exists():
                try:
                    with open(disk_path) as f:
                        cached = json.load(f)
                    samples = cached["samples"]
                    n_total = cached["n_total"]
                    self._sample_cache[cache_key] = (samples, n_total)
                    logger.debug(f"Loaded {len(samples)} cached samples for {dataset_id}")
                    return samples[:n], n_total
                except Exception:
                    pass  # Corrupted cache, re-fetch

        import itertools
        import concurrent.futures

        # Wrap load_dataset in a thread with timeout — some datasets hang
        # on "Resolving data files" when they have thousands of shards.
        def _load_and_slice():
            ds = self._load_dataset(dataset_id, config_name, split, streaming=True)
            fetch_n = max(n, 20)
            return list(itertools.islice(ds, fetch_n)), ds

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_load_and_slice)
        try:
            samples, ds = future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"Dataset {dataset_id} timed out resolving data files "
                f"(>30s). Likely too many shards — skipping."
            )
        finally:
            executor.shutdown(wait=False)

        # Get total size from dataset info if available
        n_total = 0
        if hasattr(ds, 'info') and ds.info and ds.info.splits:
            split_info = ds.info.splits.get(split)
            if split_info:
                n_total = split_info.num_examples

        # If we couldn't get n_total from info, estimate from builder
        if n_total == 0:
            try:
                builder = self._load_builder(dataset_id, config_name)
                if builder.info.splits and split in builder.info.splits:
                    n_total = builder.info.splits[split].num_examples
            except Exception:
                n_total = len(samples)  # Lower bound

        self._sample_cache[cache_key] = (samples, n_total)

        # Persist to disk (text-serializable samples only)
        if self.cache_dir:
            try:
                import hashlib
                disk_key = hashlib.md5(cache_key.encode()).hexdigest()
                disk_path = self.cache_dir / f"samples_{disk_key}.json"
                # Only cache if all values are JSON-serializable (skip image/audio)
                json.dumps(samples[0] if samples else {})
                with open(disk_path, "w") as f:
                    json.dump({"samples": samples, "n_total": n_total}, f)
            except (TypeError, ValueError):
                pass  # Non-serializable (images, audio) — skip disk cache

        return samples[:n], n_total

    @staticmethod
    def retry_hf_call(fn, max_retries: int = 3, base_delay: float = 300.0, timeout: float = 30.0):
        """Retry a HuggingFace Hub call with backoff for 429 errors.

        HF rate limits reset every ~5 minutes, so retries wait 300s (5 min)
        on first 429, then 600s on second attempt.

        Public so that other modules (e.g., discovery) can reuse the
        retry/timeout logic without reimplementing it.

        Args:
            fn: Callable to execute
            max_retries: Max retry attempts for 429 errors
            base_delay: Base delay between retries (300s = 5 min for HF rate limits)
            timeout: Max seconds per attempt (prevents hanging on slow HF calls)
        """
        import time as _time
        import concurrent.futures

        for attempt in range(max_retries):
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(fn)
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                executor.shutdown(wait=False)
                raise TimeoutError(
                    f"HuggingFace call timed out after {timeout}s"
                )
            except Exception as e:
                executor.shutdown(wait=False)
                if "429" in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"HF rate limited (429), waiting {delay:.0f}s "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    _time.sleep(delay)
                else:
                    raise

    @staticmethod
    def _extract_first_config(error_msg: str) -> Optional[str]:
        """Extract the first config name from a 'config missing' error message."""
        # Look for patterns like: ['config1', 'config2']
        match = re.search(r"\['([^']+)'", error_msg)
        if match:
            return match.group(1)
        return None

    def _detect_split_from_builder(
        self, builder, dataset_id: str, config_name: Optional[str],
    ) -> str:
        """Detect the best split using an already-loaded builder (no extra API call).

        Prefers test > validation > train. Falls back to first available split.
        Only hits HF API as last resort if builder has no split info.
        """
        available_splits = []
        if builder and builder.info and builder.info.splits:
            available_splits = list(builder.info.splits.keys())

        for preferred in ["test", "validation", "val", "dev", "train"]:
            if preferred in available_splits:
                return preferred

        # Use first available split if none of the preferred names matched
        if available_splits:
            chosen = available_splits[0]
            logger.info(
                f"No standard split found for {dataset_id}, "
                f"using first available: '{chosen}' (from {available_splits})"
            )
            return chosen

        # Last resort: try streaming to check if a split exists
        # (only happens when builder has no split info at all)
        for preferred in ["test", "validation", "train"]:
            try:
                import itertools as _itertools
                ds = self.retry_hf_call(
                    lambda p=preferred: self._load_dataset(
                        dataset_id, config_name, split=p, streaming=True,
                    )
                )
                list(_itertools.islice(ds, 1))
                return preferred
            except Exception:
                continue

        return "train"

    def _detect_field(
        self,
        column_names: list[str],
        candidates: list[str],
        samples: list[dict],
        role: str,
        required: bool = True,
    ) -> Optional[str]:
        """Detect a field by matching column names against candidates.

        Priority:
        1. Exact match (case-insensitive)
        2. Contains match (for question role: prefer longer text when
           multiple columns match the same candidate, e.g. question_title
           vs question_content)
        3. Longest string field (fallback for question)
        """
        lower_cols = {c.lower(): c for c in column_names}

        # Exact match
        for i, candidate in enumerate(candidates):
            if candidate.lower() in lower_cols:
                matched = lower_cols[candidate.lower()]
                self._detection_log.append(
                    f"{role}: exact match '{matched}' (priority {i} in {candidates[:6]}{'...' if len(candidates) > 6 else ''})"
                )
                return matched

        # Contains match — collect all matches for each candidate
        for candidate in candidates:
            matches = [
                col_orig for col_lower, col_orig in lower_cols.items()
                if candidate.lower() in col_lower
            ]
            if matches:
                # For question role with multiple matches, prefer the one
                # with longer avg text (e.g., question_content > question_title)
                if role == "question" and len(matches) > 1 and samples:
                    best = max(
                        matches,
                        key=lambda col: sum(
                            len(str(s.get(col, ""))) for s in samples[:5]
                        ),
                    )
                    self._detection_log.append(
                        f"{role}: substring match '{best}' (contains '{candidate}', "
                        f"picked over {matches} by avg text length)"
                    )
                    return best
                self._detection_log.append(
                    f"{role}: substring match '{matches[0]}' (contains '{candidate}')"
                )
                return matches[0]

        # For question field: fallback to longest string column by avg length
        if role == "question" and samples:
            string_cols = []
            for col in column_names:
                vals = [s.get(col) for s in samples if isinstance(s.get(col), str)]
                if vals:
                    avg_len = sum(len(v) for v in vals) / len(vals)
                    string_cols.append((col, avg_len))
            if string_cols:
                string_cols.sort(key=lambda x: x[1], reverse=True)
                self._detection_log.append(
                    f"{role}: no name match, using longest string column '{string_cols[0][0]}' "
                    f"(avg {string_cols[0][1]:.0f} chars, checked: {[c[0] for c in string_cols[:5]]})"
                )
                logger.warning(
                    f"Could not find {role} field by name, using longest string "
                    f"column: {string_cols[0][0]}"
                )
                return string_cols[0][0]

        if not required:
            self._detection_log.append(f"{role}: no match (optional)")
        if required:
            raise ValueError(
                f"Cannot detect {role} field. Columns: {column_names}. "
                f"Tried: {candidates}"
            )
        return None

    def _detect_choices_field(
        self,
        column_names: list[str],
        features: Any,
        samples: list[dict],
    ) -> Optional[str]:
        """Detect MCQ choices field.

        Checks for:
        1. Named choices field (choices, options)
        2. List-of-strings column
        3. Multiple fields matching ans0/ans1/ans2 or choice_a/choice_b
        """
        lower_cols = {c.lower(): c for c in column_names}

        # Named choices field
        for candidate in _CHOICES_FIELDS:
            if candidate.lower() in lower_cols:
                matched = lower_cols[candidate.lower()]
                self._detection_log.append(f"choices: named match '{matched}' (in {_CHOICES_FIELDS})")
                return matched

        # List-of-strings column (HF Sequence feature)
        if features:
            from datasets import Sequence, Value
            for col_name in column_names:
                feat = features.get(col_name)
                if feat is not None:
                    if isinstance(feat, Sequence) and isinstance(feat.feature, Value):
                        if feat.feature.dtype == "string":
                            self._detection_log.append(
                                f"choices: HF Sequence(string) feature '{col_name}'"
                            )
                            return col_name

        # Check sample values for list-of-strings
        for col in column_names:
            vals = [s.get(col) for s in samples[:5]]
            if all(isinstance(v, list) and len(v) > 1 and all(isinstance(x, str) for x in v) for v in vals if v is not None):
                self._detection_log.append(
                    f"choices: sample values in '{col}' are lists of strings (len={len(vals[0])})"
                )
                return col

        # Multi-field choices (ans0, ans1, ans2 or choice_a, choice_b)
        ans_pattern = re.compile(r'^(ans|answer|choice|option)[_\s]?(\d+|[a-d])$', re.I)
        multi_fields = [c for c in column_names if ans_pattern.match(c)]
        if len(multi_fields) >= 2:
            self._detection_log.append(
                f"choices: spread columns {multi_fields} (matched pattern ans/choice/option + digit/letter)"
            )
            return f"__multi__{','.join(sorted(multi_fields))}"

        self._detection_log.append("choices: no match")
        return None

    def _detect_task_type(
        self,
        features: Any,
        samples: list[dict],
        answer_field: Optional[str],
        choices_field: Optional[str],
        description: str,
    ) -> tuple[str, list[str], int]:
        """Detect task type from field types and sample values.

        Returns:
            (task_type, label_names, n_choices)
        """
        # MCQ: choices field exists
        if choices_field is not None:
            n_choices = 4  # default
            if not choices_field.startswith("__multi__"):
                sample_choices = [s.get(choices_field) for s in samples[:5] if s.get(choices_field)]
                if sample_choices and isinstance(sample_choices[0], list):
                    n_choices = len(sample_choices[0])
            else:
                multi_fields = choices_field.replace("__multi__", "").split(",")
                n_choices = len(multi_fields)
            self._detection_log.append(f"task_type: mcq (choices_field='{choices_field}', n_choices={n_choices})")
            return TaskType.MCQ.value, [], n_choices

        if answer_field is None:
            desc_lower = description.lower()
            if any(kw in desc_lower for kw in _SAFETY_KEYWORDS):
                self._detection_log.append(f"task_type: open (no answer field, safety keywords in description → safety dataset)")
                return TaskType.OPEN.value, [], 0
            self._detection_log.append(f"task_type: open (no answer field)")
            return TaskType.OPEN.value, [], 0

        # Get answer values from samples
        answer_vals = [s.get(answer_field) for s in samples if s.get(answer_field) is not None]

        if not answer_vals:
            self._detection_log.append(f"task_type: open (answer field '{answer_field}' has no values in samples)")
            return TaskType.OPEN.value, [], 0

        sample_answers = [str(v)[:50] for v in answer_vals[:3]]

        # Classification: small set of integer labels → extract label names for ideation
        label_names = []
        if all(isinstance(v, int) for v in answer_vals):
            unique_vals = set(answer_vals)
            if len(unique_vals) <= 10:
                if features and answer_field in features:
                    feat = features[answer_field]
                    if hasattr(feat, 'names'):
                        label_names = feat.names
                    elif hasattr(feat, 'num_classes'):
                        label_names = [str(i) for i in range(feat.num_classes)]
                if not label_names:
                    label_names = [str(v) for v in sorted(unique_vals)]
                src = "ClassLabel.names" if (features and hasattr(features.get(answer_field, None), 'names')) else "unique values"
                self._detection_log.append(
                    f"task_type: open (int labels with {len(unique_vals)} unique values, "
                    f"labels={label_names[:5]} from {src} → label ideation will handle extraction)"
                )
                return TaskType.OPEN.value, label_names, 0

        # Boolean / Yes-No answers
        if all(isinstance(v, bool) for v in answer_vals):
            self._detection_log.append(f"task_type: open (boolean answers → label ideation will handle extraction)")
            return TaskType.OPEN.value, ["true", "false"], 0

        str_vals = [str(v).strip().lower() for v in answer_vals]
        yes_no_set = {"yes", "no", "true", "false"}
        if all(v in yes_no_set for v in str_vals):
            self._detection_log.append(f"task_type: open (yes/no answers, samples: {sample_answers} → label ideation will handle)")
            return TaskType.OPEN.value, list(yes_no_set), 0

        # MCQ answers (single letters A-E) without choices field
        if all(re.match(r'^[a-eA-E]$', v.strip()) for v in str_vals):
            n_choices = max(ord(v.strip().upper()) - ord('A') + 1 for v in str_vals)
            self._detection_log.append(
                f"task_type: mcq (answer values are single letters A-E, samples: {sample_answers}, n_choices={max(n_choices, 2)})"
            )
            return TaskType.MCQ.value, [], max(n_choices, 2)

        # Numeric answers
        numeric_count = 0
        for v in str_vals:
            v_clean = v.replace(",", "").replace("$", "").replace("%", "")
            try:
                float(v_clean)
                numeric_count += 1
            except ValueError:
                if "####" in v or "\\boxed" in v:
                    numeric_count += 1
        if numeric_count >= len(str_vals) * 0.8:
            self._detection_log.append(
                f"task_type: open ({numeric_count}/{len(str_vals)} answers are numeric, samples: {sample_answers} → label ideation will handle)"
            )
            return TaskType.OPEN.value, [], 0

        self._detection_log.append(f"task_type: open (default, samples: {sample_answers} → label ideation will handle extraction)")
        return TaskType.OPEN.value, [], 0

    def _detect_answer_extraction(
        self,
        task_type: str,
        samples: list[dict],
        answer_field: Optional[str],
    ) -> str:
        """Determine how to extract the answer from model output.

        For MCQ: extract letter. For open: feature_based (label ideation
        handles it). Legacy sub-types still recognized for cached profiles.
        """
        if task_type == TaskType.MCQ.value:
            return "json_letter"
        elif task_type == TaskType.OPEN.value:
            return "feature_based"
        # Legacy sub-types (cached profiles may still have these)
        elif task_type == TaskType.OPEN_NUMERIC.value:
            if answer_field:
                vals = [str(s.get(answer_field, "")) for s in samples[:5]]
                if any("\\boxed" in v for v in vals):
                    return "boxed"
                if any("####" in v for v in vals):
                    return "last_number"
            return "last_number"
        elif task_type in (TaskType.CLASSIFICATION.value, TaskType.YES_NO.value):
            return "exact_match"
        elif task_type == TaskType.SAFETY_REFUSAL.value:
            return "refusal_classifier"
        else:
            return "feature_based"

    def _generate_prompt_template(
        self,
        task_type: str,
        context_field: Optional[str],
        choices_field: Optional[str],
        n_choices: int,
        label_names: list[str],
        instruction_placement: str = "append",
        task_instruction: Optional[str] = None,
        response_format_override: Optional[str] = None,
    ) -> str:
        """Generate a prompt template based on task type and instruction placement.

        Builds the template structure first, then inserts {instruction}
        at the position specified by instruction_placement.

        Args:
            task_instruction: LLM-generated task framing (e.g., "What does the
                ambiguous pronoun refer to?"). Overrides the default preamble
                for the given task_type when provided.
            response_format_override: Per-axis response format instruction.
                When set, replaces the default response format (e.g.,
                'Respond with JSON: {"label": "..."}').

        Templates use {question}, {context}, {choices}, {instruction} placeholders.
        """
        # Build template parts
        parts: list[tuple[str, str]] = []  # (tag, content)

        if task_type == TaskType.MCQ.value:
            letters = ", ".join(chr(ord("A") + i) for i in range(n_choices))
            preamble = task_instruction or f"Answer the following question by selecting one of {letters}."
            # Ensure preamble mentions the letter options
            if task_instruction and letters not in preamble:
                preamble = f"{preamble} Select one of {letters}."
            parts.append(("preamble", f"{preamble}\n\n"))
            if context_field:
                parts.append(("context", "Context:\n{context}\n\n"))
            parts.append(("question", "Question: {question}\n\n"))
            parts.append(("choices", "{choices}\n\n"))
            parts.append(("response_format", f'Respond with JSON: {{"answer": "<letter>"}}'))

        elif task_type == TaskType.OPEN_NUMERIC.value:
            preamble = task_instruction or "Solve the following problem. Put your final numerical answer in \\boxed{}."
            parts.append(("preamble", f"{preamble}\n\n"))
            if context_field:
                parts.append(("context", "Context:\n{context}\n\n"))
            parts.append(("question", "Problem:\n{question}\n\n"))

        elif task_type == TaskType.CLASSIFICATION.value:
            labels_str = ", ".join(label_names) if label_names else "{label_names}"
            preamble = task_instruction or f"Classify the following into one of: {labels_str}."
            parts.append(("preamble", f"{preamble}\n\n"))
            if context_field:
                parts.append(("context", "Text 1: {context}\n\n"))
                parts.append(("question", "Text 2: {question}\n\n"))
            else:
                parts.append(("question", "Text: {question}\n\n"))
            parts.append(("response_format", f'Respond with JSON: {{"label": "<label>"}}'))

        elif task_type == TaskType.YES_NO.value:
            preamble = task_instruction or "Answer the following question with Yes or No."
            parts.append(("preamble", f"{preamble}\n\n"))
            if context_field:
                parts.append(("context", "Context:\n{context}\n\n"))
            parts.append(("question", "Question: {question}\n\n"))
            parts.append(("response_format", "Answer:"))

        elif task_type == TaskType.OPEN.value:
            # Generic open template — label ideation determines extraction
            preamble = task_instruction or "Answer the following."
            parts.append(("preamble", f"{preamble}\n\n"))
            if context_field:
                parts.append(("context", "Context:\n{context}\n\n"))
            parts.append(("question", "{question}\n\n"))

        # Legacy sub-types (for cached profiles)
        elif task_type == TaskType.SAFETY_REFUSAL.value:
            parts.append(("question", "{question}"))

        else:  # OPEN_TEXT, OPEN_NUMERIC, etc. (legacy)
            if context_field:
                parts.append(("context", "{context}\n\n"))
            parts.append(("question", "{question}\n\n"))

        # Override response_format if per-axis format is provided
        if response_format_override:
            parts = [(tag, content) for tag, content in parts if tag != "response_format"]
            parts.append(("response_format", response_format_override))

        # Insert {instruction} at the specified position
        insertion_point = self._get_insertion_index(parts, instruction_placement)
        parts.insert(insertion_point, ("instruction", "{instruction}\n\n"))

        return "".join(content for _, content in parts)

    @staticmethod
    def _get_insertion_index(
        parts: list[tuple[str, str]],
        placement: str,
    ) -> int:
        """Determine where to insert {instruction} in the parts list.

        Returns the index to insert at.
        """
        tag_indices = {tag: i for i, (tag, _) in enumerate(parts)}

        if placement == "prepend":
            return 0
        elif placement == "after_context" and "context" in tag_indices:
            return tag_indices["context"] + 1
        elif placement == "after_question" and "question" in tag_indices:
            return tag_indices["question"] + 1
        elif placement == "after_choices" and "choices" in tag_indices:
            return tag_indices["choices"] + 1
        else:  # "append" or fallback
            # Insert before response_format if present, else at end
            if "response_format" in tag_indices:
                return tag_indices["response_format"]
            return len(parts)

    def _detect_instruction_placement(self, task_type: str, description: str) -> str:
        """Heuristic instruction placement (used before LLM ideation runs).

        The LLM ideation may override this with a better placement.
        """
        if task_type == TaskType.SAFETY_REFUSAL.value:
            return "prepend"
        return "append"


# =============================================================================
# Dataset Adapter
# =============================================================================


class DatasetAdapter:
    """Adapts any HuggingFace dataset for the perturbation pipeline.

    Provides the same interface as BaseBenchmark but is dynamically
    constructed from an auto-detected DatasetProfile.
    """

    def __init__(self, profile: DatasetProfile):
        self.profile = profile
        self._dataset = None

    def _ensure_loaded(self, n_samples: int = 50) -> Any:
        """Lazy-load the dataset via streaming.

        Always uses streaming to avoid downloading the full dataset.
        Only fetches enough rows to sample from (2x n_samples, capped at 1000).

        Args:
            n_samples: Number of samples needed. Streams ~2x this amount
                for randomness.
        """
        if self._dataset is not None:
            return self._dataset

        import os
        import itertools
        from datasets import Dataset

        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )

        # Stream a small buffer — never download the full dataset.
        fetch_n = min(max(n_samples * 2, 50), 1000)
        import logging as _logging
        from huggingface_hub import HfApi as _HfApi
        _ds_logger = _logging.getLogger("datasets.load")
        _orig_level = _ds_logger.level
        _ds_logger.setLevel(_logging.CRITICAL)  # suppress [ERROR] spam we're about to handle
        try:
            ds_stream = load_dataset(
                self.profile.dataset_id,
                self.profile.config_name,
                split=self.profile.split,
                trust_remote_code=True,
                token=token,
                streaming=True,
            )
            _ds_logger.setLevel(_orig_level)
        except RuntimeError as _e:
            _ds_logger.setLevel(_orig_level)
            if "Dataset scripts are no longer supported" not in str(_e):
                raise
            logger.info(
                f"[FALLBACK] {self.profile.dataset_id}: legacy script, trying raw files / parquet branch"
            )
            # Fallback: try raw data files then auto-converted parquet
            _api = _HfApi()
            _info = _api.dataset_info(self.profile.dataset_id, token=token)
            _raw = [
                s.rfilename for s in _info.siblings
                if s.rfilename.endswith((".json", ".jsonl", ".csv", ".parquet"))
                and not s.rfilename.startswith(".")
            ]
            if _raw:
                _split_match = [f for f in _raw if self.profile.split in f.lower()]
                _chosen = _split_match[0] if _split_match else _raw[0]
                _fmt = "parquet" if _chosen.endswith(".parquet") else "json" if _chosen.endswith((".json", ".jsonl")) else "csv"
                ds_stream = load_dataset(_fmt, data_files=f"hf://datasets/{self.profile.dataset_id}/{_chosen}", split="train", token=token, streaming=True)
            else:
                # Try refs/convert/parquet branch
                _parquet = []
                try:
                    for _item in _api.list_repo_tree(self.profile.dataset_id, repo_type="dataset", revision="refs/convert/parquet", recursive=True):
                        if hasattr(_item, "path") and _item.path.endswith(".parquet"):
                            _parquet.append(_item.path)
                except Exception:
                    pass
                if not _parquet:
                    # Last resort: zip file containing CSVs
                    _zip_files = [
                        s.rfilename for s in _info.siblings
                        if s.rfilename.endswith(".zip") and not s.rfilename.startswith(".")
                    ]
                    if not _zip_files:
                        raise
                    import zipfile as _zipfile
                    import pandas as _pd
                    from huggingface_hub import hf_hub_download as _hf_dl
                    _zip_path = _hf_dl(self.profile.dataset_id, _zip_files[0], repo_type="dataset", token=token)
                    with _zipfile.ZipFile(_zip_path) as _z:
                        _csvs = [n for n in _z.namelist() if n.endswith(".csv")]
                        _split_csvs = [n for n in _csvs if self.profile.split in n.lower()]
                        _chosen_csvs = _split_csvs or _csvs
                        _dfs = []
                        for _name in _chosen_csvs:
                            with _z.open(_name) as _f:
                                _dfs.append(_pd.read_csv(_f))
                    _df = _pd.concat(_dfs, ignore_index=True)
                    logger.info(f"[FALLBACK] {self.profile.dataset_id}: loaded {len(_df)} rows from zip {_zip_files[0]}")
                    from datasets import Dataset as _Dataset
                    _ds_obj = _Dataset.from_pandas(_df)
                    self._dataset = _ds_obj
                    return self._dataset
                _cfg = self.profile.config_name
                _cands = [f for f in _parquet if f.startswith(f"{_cfg}/")] if _cfg else _parquet
                _sm = [f for f in (_cands or _parquet) if f"/{self.profile.split}/" in f]
                _chosen = _sm[0] if _sm else (_cands[0] if _cands else _parquet[0])
                ds_stream = load_dataset("parquet", data_files=f"hf://datasets/{self.profile.dataset_id}@refs/convert/parquet/{_chosen}", split="train", token=token, streaming=True)
        rows = list(itertools.islice(ds_stream, fetch_n))
        self._dataset = Dataset.from_list(rows)
        logger.info(
            f"Streamed {len(rows)} rows from {self.profile.dataset_id} "
            f"(requested {n_samples} samples)"
        )
        return self._dataset

    def load_examples(
        self,
        n_samples: int,
        random_seed: int = 42,
    ) -> list[AdaptedExample]:
        """Load examples from the dataset.

        Args:
            n_samples: Number of examples to load
            random_seed: Random seed for reproducibility

        Returns:
            List of AdaptedExample objects
        """
        ds = self._ensure_loaded(n_samples=n_samples)

        rng = random.Random(random_seed)
        indices = rng.sample(range(len(ds)), min(n_samples, len(ds)))

        examples = []
        for i, idx in enumerate(indices):
            item = ds[idx]
            question = str(item.get(self.profile.question_field, ""))

            # Handle nested question extraction (e.g., RACE: "0.question")
            if self.profile.question_extraction:
                question = self._extract_nested(
                    item.get(self.profile.question_field),
                    self.profile.question_extraction,
                ) or question

            # Extract answer
            ground_truth = ""
            if self.profile.answer_field:
                raw_answer = item.get(self.profile.answer_field, "")
                ground_truth = self._extract_ground_truth(raw_answer)

            # Extract choices (may also come from nested extraction)
            choices = self._extract_choices(item)
            if not choices and self.profile.choices_extraction:
                raw = item.get(self.profile.question_field)
                nested_choices = self._extract_nested(
                    raw, self.profile.choices_extraction,
                )
                if isinstance(nested_choices, list):
                    choices = [str(c) for c in nested_choices]

            # Extract context
            context = None
            if self.profile.context_field:
                context = str(item.get(self.profile.context_field, ""))

            # Collect extra metadata
            metadata = {}
            known_fields = {
                self.profile.question_field,
                self.profile.answer_field,
                self.profile.choices_field,
                self.profile.context_field,
            }
            for k, v in item.items():
                if k not in known_fields and isinstance(v, (str, int, float, bool)):
                    metadata[k] = v

            examples.append(AdaptedExample(
                idx=i,
                question=question,
                ground_truth_answer=ground_truth,
                choices=choices,
                context=context,
                metadata=metadata,
            ))

        return examples

    @staticmethod
    def _extract_nested(raw: Any, dot_path: str) -> Any:
        """Extract a value from nested data using a dot-path.

        E.g., dot_path="0.question" on [{"question": "What?", "options": [...]}]
        returns "What?".

        Handles stringified JSON (common in HF datasets where lists are stored as strings).
        """
        if raw is None:
            return None

        # Parse stringified JSON
        if isinstance(raw, str) and raw.strip().startswith(("[", "{")):
            try:
                raw = json.loads(raw.replace("'", '"'))
            except (json.JSONDecodeError, ValueError):
                return None

        # Walk the dot-path
        current = raw
        for key in dot_path.split("."):
            if current is None:
                return None
            if isinstance(current, list):
                try:
                    current = current[int(key)]
                except (IndexError, ValueError):
                    return None
            elif isinstance(current, dict):
                current = current.get(key)
            else:
                return None

        return str(current) if not isinstance(current, (list, dict)) else current

    # Fields that look like solutions/explanations — never use as context
    _SOLUTION_FIELD_NAMES = {
        "solution", "explanation", "rationale", "reasoning", "cot",
        "chain_of_thought", "steps", "work", "derivation", "proof",
        "url", "link", "source", "reference", "id", "idx", "index",
    }

    def _detect_pair_context(self, ds) -> None:
        """Detect and set context_field for pair-input tasks at load time.

        Only triggers for classification/yes_no tasks where pair inputs
        are common (NLI, paraphrase, entailment). Skips math/code/open-ended
        where extra text fields are usually solutions, not context.
        """
        # Only check pair-input for classification and yes_no tasks
        if self.profile.task_type not in ("classification", "yes_no"):
            return

        assigned = {
            self.profile.question_field,
            self.profile.answer_field,
            self.profile.choices_field,
        }
        assigned.discard(None)

        # Sample a few items to check text fields
        sample_items = [ds[i] for i in range(min(5, len(ds)))]
        columns = list(sample_items[0].keys()) if sample_items else []

        candidates = []
        for col in columns:
            if col in assigned:
                continue
            # Skip solution/metadata field names
            if col.lower() in self._SOLUTION_FIELD_NAMES:
                continue
            vals = [
                item.get(col) for item in sample_items
                if isinstance(item.get(col), str)
            ]
            if vals:
                avg_len = sum(len(v) for v in vals) / len(vals)
                if avg_len > 20:
                    candidates.append((col, avg_len))

        if not candidates:
            return

        # Pick the longest unassigned text field
        candidates.sort(key=lambda x: x[1], reverse=True)
        context_field = candidates[0][0]
        self.profile.context_field = context_field
        logger.info(
            f"Pair-input detected at load time: using '{context_field}' as context"
        )

        # Rebuild prompt template with context
        detector = DatasetDetector()
        self.profile.prompt_template = detector._generate_prompt_template(
            task_type=self.profile.task_type,
            context_field=context_field,
            choices_field=self.profile.choices_field,
            n_choices=self.profile.n_choices,
            label_names=self.profile.label_names,
            instruction_placement=self.profile.instruction_placement,
        )

    def _extract_ground_truth(self, raw_answer: Any) -> str:
        """Extract ground truth answer from raw value."""
        if isinstance(raw_answer, int):
            # Classification label index — convert to label name if available
            if self.profile.label_names and raw_answer < len(self.profile.label_names):
                return self.profile.label_names[raw_answer]
            return str(raw_answer)
        raw_str = str(raw_answer)
        # Handle "...#### <answer>" suffix-style answer format
        if "####" in raw_str:
            return raw_str.split("####")[-1].strip()
        return raw_str.strip()

    def _extract_choices(self, item: dict) -> Optional[list[str]]:
        """Extract MCQ choices from an item."""
        if self.profile.choices_field is None:
            return None

        if self.profile.choices_field.startswith("__multi__"):
            # Multi-field choices
            field_names = self.profile.choices_field.replace("__multi__", "").split(",")
            return [str(item.get(f, "")) for f in field_names]

        raw = item.get(self.profile.choices_field)
        if isinstance(raw, list):
            return [str(c) for c in raw]
        # Handle dict with 'text' key (e.g., ARC: {'text': [...], 'label': [...]})
        if isinstance(raw, dict):
            if "text" in raw and isinstance(raw["text"], list):
                return [str(c) for c in raw["text"]]
            # Try first list-valued key
            for v in raw.values():
                if isinstance(v, list) and v and isinstance(v[0], str):
                    return [str(c) for c in v]
        # Handle stringified lists: "['A', 'B', 'C']"
        if isinstance(raw, str) and raw.strip().startswith(("[", "{")):
            try:
                parsed = json.loads(raw.replace("'", '"'))
                if isinstance(parsed, list):
                    return [str(c) for c in parsed]
                if isinstance(parsed, dict) and "text" in parsed:
                    return [str(c) for c in parsed["text"]]
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    def _format_choices(self, choices: list[str]) -> str:
        """Format MCQ choices as lettered options."""
        lines = []
        for i, choice in enumerate(choices):
            letter = chr(ord("A") + i)
            lines.append(f"{letter}) {choice}")
        return "\n".join(lines)

    def make_baseline_prompt(
        self,
        example: AdaptedExample,
        instruction: str = "",
    ) -> str:
        """Create baseline prompt for an example.

        Args:
            example: The adapted example
            instruction: Optional instruction

        Returns:
            Formatted prompt string
        """
        return self._build_prompt(example, instruction)

    def make_lever_prompt(
        self,
        example: AdaptedExample,
        lever_instruction: str,
        baseline_instruction: str = "",
    ) -> str:
        """Create lever prompt for an example.

        Args:
            example: The adapted example
            lever_instruction: The lever instruction to add
            baseline_instruction: Ignored (lever replaces baseline)

        Returns:
            Formatted prompt string
        """
        return self._build_prompt(example, lever_instruction)

    def _build_prompt(self, example: AdaptedExample, instruction: str = "") -> str:
        """Build prompt from profile template, filling in placeholders."""
        return self._build_prompt_from_template(
            example, instruction, self.profile.prompt_template,
        )

    def _build_prompt_from_template(
        self,
        example: AdaptedExample,
        instruction: str,
        template: str,
    ) -> str:
        """Build prompt from a given template, filling in placeholders.

        Handles both heuristic-generated and LLM-generated templates.
        Standard placeholders: {question}, {context}, {choices}, {instruction}, {label_names}
        Extra placeholders (e.g., {repo}, {version}) are filled from example.metadata.
        """
        replacements = {
            "{question}": example.question,
            "{instruction}": instruction or "",
        }

        if example.context is not None:
            replacements["{context}"] = example.context
        else:
            # Remove any line containing {context} if no context available
            template = re.sub(r'[^\n]*\{context\}[^\n]*\n?', '', template)
            replacements["{context}"] = ""

        if example.choices is not None:
            replacements["{choices}"] = self._format_choices(example.choices)
        else:
            replacements["{choices}"] = ""

        if self.profile.label_names:
            replacements["{label_names}"] = ", ".join(self.profile.label_names)

        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        # Fill remaining {placeholders} from example.metadata
        # (handles LLM-generated templates with custom fields like {repo}, {version})
        if example.metadata:
            for key, value in example.metadata.items():
                placeholder = "{" + key + "}"
                if placeholder in result:
                    result = result.replace(placeholder, str(value))

        # Clean up empty instruction lines and triple+ newlines
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
        return result.strip()

    def make_edited_prompt(
        self,
        example: AdaptedExample,
        edits: list,
        instruction: str = "",
    ) -> str:
        """Build prompt with problem edits applied, then instruction placed.

        Args:
            example: The original adapted example
            edits: List of ProblemEdit operations
            instruction: Optional instruction to include

        Returns:
            Formatted prompt string with edits applied
        """
        from .edit_utils import apply_field_edits

        modified = apply_field_edits(example, edits)
        return self._build_prompt(modified, instruction)

    # -----------------------------------------------------------------
    # Per-axis prompt methods (use axis-specific preamble + response format)
    # -----------------------------------------------------------------

    def _find_axis_label(self, axis_name: str) -> Optional[dict]:
        """Find the label dict for a given axis name."""
        for label in self.profile.answer_labels:
            if isinstance(label, dict) and label.get("name") == axis_name:
                return label
        return None

    def _build_axis_template(self, axis_label: dict) -> str:
        """Build a prompt template with axis-specific preamble + response format."""
        return DatasetDetector._generate_prompt_template(
            DatasetDetector(),
            task_type=self.profile.task_type,
            context_field=self.profile.context_field,
            choices_field=self.profile.choices_field,
            n_choices=self.profile.n_choices,
            label_names=self.profile.label_names,
            instruction_placement=self.profile.instruction_placement,
            task_instruction=axis_label.get("axis_preamble"),
            response_format_override=axis_label.get("axis_response_format"),
        )

    def make_axis_baseline_prompt(
        self,
        example: AdaptedExample,
        axis_name: str,
        instruction: str = "",
    ) -> str:
        """Create baseline prompt with axis-specific preamble + response format.

        Falls back to default make_baseline_prompt if the axis has no
        axis_preamble defined.
        """
        label = self._find_axis_label(axis_name)
        if label is None or not label.get("axis_preamble"):
            return self._build_prompt(example, instruction)
        template = self._build_axis_template(label)
        return self._build_prompt_from_template(example, instruction, template)

    def make_axis_lever_prompt(
        self,
        example: AdaptedExample,
        axis_name: str,
        lever_instruction: str,
        baseline_instruction: str = "",
    ) -> str:
        """Create lever prompt with axis-specific preamble + response format.

        Falls back to default make_lever_prompt if the axis has no
        axis_preamble defined.
        """
        label = self._find_axis_label(axis_name)
        if label is None or not label.get("axis_preamble"):
            return self._build_prompt(example, lever_instruction)
        template = self._build_axis_template(label)
        return self._build_prompt_from_template(example, lever_instruction, template)

    def make_axis_edited_prompt(
        self,
        example: AdaptedExample,
        axis_name: str,
        edits: list,
        instruction: str = "",
    ) -> str:
        """Build axis-specific prompt with problem edits applied.

        Falls back to default make_edited_prompt if the axis has no
        axis_preamble defined.
        """
        from .edit_utils import apply_field_edits
        modified = apply_field_edits(example, edits)
        return self.make_axis_baseline_prompt(modified, axis_name, instruction)

    def get_problem_for_attribution(self, example: AdaptedExample) -> str:
        """Get the problem text for Phase 2 attribution display.

        Returns the problem WITH benchmark-specific framing but WITHOUT
        the lever/baseline instruction.
        """
        return self._build_prompt(example, instruction="")

    def get_edited_problem_for_attribution(
        self,
        example: AdaptedExample,
        edits: list,
    ) -> str:
        """Get edited problem text for Phase 2 attribution display.

        Returns the problem WITH edits applied and benchmark framing,
        but WITHOUT lever/baseline instruction.
        """
        return self.make_edited_prompt(example, edits, instruction="")

    def build_attribution_question(self) -> str:
        """Generate attribution question from task type and ideated labels.

        Follows the pattern from existing YAML configs:
        - math: "Would the numeric value of your final answer be different?"
        - safety: "Would your refusal behavior in your response change?"
        - fairness: "Would your answer choice be different?"
        - coding: "Would your code's behavior be different?"
        """
        task_type = self.profile.task_type
        labels = self.profile.answer_labels

        # Task-type defaults (match existing YAML configs)
        defaults = {
            TaskType.MCQ.value: "Would your answer choice be different?",
            TaskType.OPEN_NUMERIC.value: "Would the numeric value of your final answer be different?",
            TaskType.YES_NO.value: "Would your yes/no answer be different?",
            TaskType.CLASSIFICATION.value: "Would your classification be different?",
            TaskType.SAFETY_REFUSAL.value: "Would your refusal behavior change?",
            TaskType.OPEN_TEXT.value: "Would your response be meaningfully different?",
        }

        base_q = defaults.get(task_type, defaults[TaskType.OPEN_TEXT.value])

        # If labels are available, enrich with specific features
        if labels:
            descriptions = [
                l.get("description", l.get("name", ""))
                for l in labels[:3]
            ]
            features_str = ", ".join(descriptions)
            return f"{base_q} Consider these aspects: {features_str}."

        return base_q

    def build_yes_no_explanations(
        self,
        parsed_answer: str,
        features: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """Build YES/NO explanations grounded on actual features.

        Follows existing domain pattern:
        - Math: 'your answer "42" would be different' / 'would remain "42"'
        - Safety: 'you would comply' / 'you would still refuse'

        Uses extracted features when available to be specific.

        Args:
            parsed_answer: The parsed answer string
            features: Extracted feature dict from extract_features()

        Returns:
            Tuple of (yes_explanation, no_explanation)
        """
        if not features:
            answer_preview = str(parsed_answer)[:50]
            return (
                f'your answer "{answer_preview}" would be different',
                f'your answer would remain "{answer_preview}"',
            )

        # Build grounded explanations from actual feature values
        feature_parts = []
        for name, value in list(features.items())[:3]:
            feature_parts.append(f'{name}="{value}"')
        features_str = ", ".join(feature_parts)

        return (
            f"your response would differ (currently: {features_str})",
            f"your response would remain the same ({features_str})",
        )

    def create_verifier(self):
        """Create verifier, routing to domain-specific verifiers when possible.

        Routing strategy:
        - open_numeric → MathVerifier: Symbolic equivalence (3/6 == 0.5)
          that regex cannot replicate.
        - mcq, open_text, classification, yes_no → AnswerParser (with
          MCQMixin for MCQ): Label-driven parsing; MCQ uses the same
          battle-tested 8-level letter extraction as CodeVerifier/MathVerifier.

        Not yet routed (interface gaps):
        - safety_refusal: SafetyVerifier requires async classify_response()
          before answers_match(), and needs query kwarg. EmpiricalVerifier
          calls answers_match(a, b) without query context.
        - coding: No CODING TaskType in auto-detector yet. CodeVerifier
          needs entry_point/function_prompt which auto-adapted datasets
          lack. AST features would be valuable once we add coding detection.

        Returns:
            BaseVerifier instance for answer comparison
        """
        task_type = self.profile.task_type

        if task_type == TaskType.OPEN_NUMERIC.value:
            from prompt_attribution.eval.domains.math.verifier import MathVerifier
            return MathVerifier()

        from .answer_parser import AnswerParser
        return AnswerParser(
            answer_labels=self.profile.answer_labels,
            task_type=task_type,
        )
