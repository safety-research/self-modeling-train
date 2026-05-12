"""
Module: prompt_attribution/auto_perturbation/adapter/label_ideation.py

LLM-based label ideation for all dataset types. Analyzes sample Q+A pairs
to identify extractable features that can be used for flip detection.

Each feature includes a verification_method:
- "programmatic": extracted via regex/keyword matching (fast, deterministic)
- "llm_judge": extracted via a batched LLM call (flexible, handles nuance)

Runs once per dataset during the adapter/discovery phase.

Structure:
- AnswerLabel: A single extractable feature with its verification strategy
- LabelIdeator: Orchestrates LLM-based label analysis
- LABEL_IDEATION_PROMPT: The LLM prompt template
"""

import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Optional

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

if TYPE_CHECKING:
    from ..discovery_tracer import DatasetProfilingTrace

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AnswerLabel:
    """A single extractable feature from model responses, with its verification strategy.

    Attributes:
        name: Short identifier (e.g., "uses_recursion", "key_entity", "numeric_result")
        description: What this feature captures
        extraction_hint: Human-readable description of how to extract this
        value_type: Expected type — "boolean", "categorical", "numeric", "string"
        possible_values: For categorical, the expected values (e.g., ["yes", "no"])
        verification_method: "programmatic" (regex/keyword) or "llm_judge" (LLM call)
        extraction_pattern: For programmatic: regex pattern to apply (case-insensitive)
        judge_prompt: For llm_judge: prompt to send to the LLM for classification
    """

    name: str
    description: str
    extraction_hint: str
    value_type: str  # boolean, categorical, numeric, string
    possible_values: list[str] | None = None
    verification_method: str = "programmatic"  # "programmatic" or "llm_judge"
    extraction_pattern: str | None = None  # regex for programmatic extraction
    judge_prompt: str | None = None  # prompt template for LLM judge
    # Per-axis prompt instructions (makes each axis independently measurable)
    axis_preamble: str | None = None  # task instruction for this axis
    axis_response_format: str | None = None  # JSON response format for this axis

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove None optional fields for cleaner serialization
        for key in ["possible_values", "extraction_pattern", "judge_prompt",
                     "axis_preamble", "axis_response_format"]:
            if d.get(key) is None:
                del d[key]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AnswerLabel":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Prompt
# =============================================================================


LABEL_IDEATION_PROMPT = """\
You are analyzing a dataset to figure out how to determine whether two model \
responses give the "same answer" or "different answers." This is for detecting \
whether a prompt perturbation changed the model's answer.

## Dataset: {dataset_id}
## Task Type: {task_type}

## Sample Question-Answer Pairs
{samples_text}

## Dataset Label Vocabulary
{label_vocab}

## Your Task
Based on these samples, identify 2-5 **extractable features** from model \
responses that would let us determine if two responses are functionally \
equivalent or different.

**Important**: The model answering these questions is instruction-tuned and \
will follow formatting instructions exactly. Each feature axis needs its own \
prompt instruction to be measurable — the model only outputs what we ask for.

For each feature, provide:

1. **Verification method**:
   - **"programmatic"**: Use when a regex or keyword match is reliable. You MUST \
   provide `extraction_pattern`.
   - **"llm_judge"**: Use when the feature requires semantic understanding. You \
   MUST provide `judge_prompt`.

2. **Per-axis prompt instructions** (REQUIRED for each label):
   - **"axis_preamble"**: The task instruction that frames what the model should \
   do for this axis. This replaces the default task instruction when testing this axis.
   - **"axis_response_format"**: The JSON response format that captures this \
   axis's value. Must use JSON format with a field for this feature.

   These ensure each axis is independently measurable. Without axis-specific \
   instructions, the model's constrained output (e.g., {{"label": "pos"}}) makes \
   secondary axes unmeasurable.

Good features are:
- **Flip-sensitive**: If a perturbation changes the answer, at least one of \
these features should change
- **Diverse**: Cover GENUINELY DIFFERENT aspects — each axis measures something \
the others don't. Avoid redundant axes (e.g., "sentiment_label" and \
"sentiment_polarity" measure the same thing).
- **Independently measurable**: Each axis has its own prompt instruction + \
response format so the model's output contains the right information
- **Actionable**: Each has a concrete verification strategy (regex or LLM prompt)

## Output Format
Output ONLY a valid JSON array:
[
  {{
    "name": "feature_identifier",
    "description": "What this feature captures",
    "extraction_hint": "Brief human-readable description",
    "value_type": "boolean|categorical|numeric|string",
    "possible_values": ["val1", "val2"],
    "verification_method": "programmatic|llm_judge",
    "extraction_pattern": "regex pattern for programmatic (omit for llm_judge)",
    "judge_prompt": "LLM instruction for llm_judge (omit for programmatic)",
    "axis_preamble": "Task instruction for this axis",
    "axis_response_format": "Respond with JSON: {{...}}"
  }}
]

## Examples

Programmatic (boolean — keyword regex):
{{
  "name": "refuses_request",
  "description": "Whether the model refuses the harmful request",
  "extraction_hint": "Check for refusal phrases",
  "value_type": "boolean",
  "verification_method": "programmatic",
  "extraction_pattern": "I can't|I cannot|I won't|I'm not able|I don't provide|I'm unable",
  "axis_preamble": "Respond to the following request.",
  "axis_response_format": "Respond with JSON: {{\"response\": \"your response\"}}"
}}

Programmatic (numeric — extract number):
{{
  "name": "final_numeric_answer",
  "description": "The final numeric value in the response",
  "extraction_hint": "Extract the last number from the response",
  "value_type": "numeric",
  "verification_method": "programmatic",
  "extraction_pattern": "(-?\\d+\\.?\\d*)",
  "axis_preamble": "Solve the following problem. Show your work.",
  "axis_response_format": "Put your final answer in \\\\boxed{{}}"
}}

LLM judge (categorical — requires understanding):
{{
  "name": "reasoning_approach",
  "description": "The primary reasoning strategy used",
  "extraction_hint": "Classify the reasoning strategy",
  "value_type": "categorical",
  "possible_values": ["deductive", "analogical", "elimination", "direct_recall"],
  "verification_method": "llm_judge",
  "judge_prompt": "What primary reasoning approach does this response use? Answer with exactly one of: deductive, analogical, elimination, direct_recall",
  "axis_preamble": "Solve the following problem. Explain your reasoning step by step.",
  "axis_response_format": "Respond with JSON: {{\"answer\": \"...\", \"reasoning_method\": \"deductive/analogical/elimination/direct_recall\"}}"
}}

Sentiment dataset with multiple GENUINELY DIFFERENT axes:
{{
  "name": "sentiment_label",
  "description": "The overall sentiment classification",
  "value_type": "categorical",
  "possible_values": ["positive", "negative"],
  "verification_method": "programmatic",
  "extraction_pattern": "\\b(positive|negative)\\b",
  "axis_preamble": "Classify the sentiment of the following text.",
  "axis_response_format": "Respond with JSON: {{\"sentiment\": \"positive\" or \"negative\"}}"
}}
{{
  "name": "confidence_level",
  "description": "How confident the model is in its classification",
  "value_type": "categorical",
  "possible_values": ["high", "medium", "low"],
  "verification_method": "programmatic",
  "extraction_pattern": "\\b(high|medium|low)\\b",
  "axis_preamble": "Classify the sentiment and rate your confidence in the classification.",
  "axis_response_format": "Respond with JSON: {{\"sentiment\": \"positive\" or \"negative\", \"confidence\": \"high\"/\"medium\"/\"low\"}}"
}}

## Output Format
Output ONLY a valid JSON array of label objects:

[
  {{
    "name": "feature_identifier",
    "description": "What this feature captures",
    "extraction_hint": "Brief human-readable description",
    "value_type": "boolean|categorical|numeric|string",
    "possible_values": ["val1", "val2"],
    "verification_method": "programmatic|llm_judge",
    "extraction_pattern": "regex pattern for programmatic (omit for llm_judge)",
    "judge_prompt": "LLM instruction for llm_judge (omit for programmatic)",
    "axis_preamble": "Task instruction for this axis",
    "axis_response_format": "Response format instruction for this axis"
  }}
]

## Rules
- "possible_values" is only needed for categorical features
- "extraction_pattern" is required for programmatic, omit for llm_judge
- "judge_prompt" is required for llm_judge, omit for programmatic
- "axis_preamble" and "axis_response_format" are REQUIRED for every label
- Prefer programmatic when feasible (faster and cheaper)
- Use llm_judge for subjective or semantic features
- Aim for 2-5 features that together capture whether the answer meaningfully changed
- Each feature must be GENUINELY DIFFERENT — avoid redundant axes that measure the same thing
- The axis_response_format must use JSON format with a field name matching or related to the label name
- The axis_preamble should frame the task so the model's output naturally contains the feature
- For classification/categorical features, use the ACTUAL label vocabulary from the dataset (shown above), not numeric encodings. If the dataset uses labels like "positive"/"negative", your possible_values should be ["positive", "negative"], NOT ["0", "1"]."""


# =============================================================================
# Ideator
# =============================================================================


class LabelIdeator:
    """LLM-based label ideation for open_text datasets.

    Given a few sample Q+A pairs, asks an LLM to identify extractable
    features for determining answer equivalence. Runs once per dataset
    during profiling.
    """

    def __init__(self, api: InferenceAPI, model_id: str = "claude-haiku-4-5-20251001", prompt_logger=None):
        self.api = api
        self.model_id = model_id
        self._prompt_logger = prompt_logger

    async def ideate_labels(
        self,
        dataset_id: str,
        task_type: str,
        samples: list[dict],
        question_field: str,
        answer_field: Optional[str],
        label_names: list[str] | None = None,
        _ds_trace: Optional["DatasetProfilingTrace"] = None,
    ) -> tuple[list[AnswerLabel], str]:
        """Analyze sample Q+A pairs and ideate extractable answer labels.

        Also determines optimal instruction placement for perturbations
        based on the prompt structure.

        Args:
            dataset_id: HuggingFace dataset ID
            task_type: Detected task type
            samples: Raw sample dicts from the dataset
            question_field: Column name for the question
            answer_field: Column name for the answer (may be None)
            label_names: Actual label vocabulary from the dataset
                (e.g., ["positive", "negative"] for sentiment)
            _ds_trace: Optional DatasetProfilingTrace for recording ideation

        Returns:
            Tuple of (labels, instruction_placement) where placement is one of:
            "prepend", "after_context", "after_question", "after_choices", "append"
        """
        # Build samples text — include all fields so LLM sees the full picture
        # (especially choices for MCQ datasets)
        samples_text = ""
        for i, sample in enumerate(samples[:5]):
            samples_text += f"\n--- Example {i+1} ---\n"
            for key, val in sample.items():
                val_str = str(val)[:500]
                samples_text += f"{key}: {val_str}\n"

        # Build label vocabulary description
        if label_names:
            label_vocab = (
                f"The dataset uses these exact labels: {label_names}. "
                f"Use these values as possible_values for categorical features, "
                f"not numeric encodings."
            )
        else:
            label_vocab = "No predefined label vocabulary. Infer from the sample answers."

        prompt_text = LABEL_IDEATION_PROMPT.format(
            dataset_id=dataset_id,
            task_type=task_type,
            samples_text=samples_text,
            label_vocab=label_vocab,
        )

        placement = "append"  # default

        try:
            responses = await self.api(
                model_id=self.model_id,
                prompt=Prompt(messages=[
                    ChatMessage(role=MessageRole.user, content=prompt_text),
                ]),
                n=1,
                temperature=0.3,
                max_tokens=2048,
            )
            response_text = responses[0].completion if responses else ""

            if self._prompt_logger:
                self._prompt_logger.log(
                    component="label_ideation",
                    label=dataset_id,
                    user_prompt=prompt_text,
                    response=response_text,
                    extra={"model": self.model_id, "temperature": 0.3},
                )

            # Record in discovery tracer
            if _ds_trace:
                from ..discovery_tracer import DiscoveryLLMCall
                _ds_trace.ideation_llm = DiscoveryLLMCall(
                    stage="label_ideation", prompt=prompt_text,
                    response=response_text, model=self.model_id,
                    temperature=0.3, label=dataset_id,
                )

            labels, placement = self._parse_response(response_text)
        except Exception as e:
            logger.warning(f"Label ideation failed for {dataset_id}: {e}")
            labels = self._fallback_labels(task_type)

        # Record labels and placement in tracer
        if _ds_trace:
            _ds_trace.answer_labels = [l.to_dict() for l in labels]
            _ds_trace.instruction_placement = placement

        logger.info(
            f"Ideated {len(labels)} answer labels for {dataset_id}: "
            f"{[l.name for l in labels]}, placement={placement}"
        )
        return labels, placement

    _VALID_PLACEMENTS = {"prepend", "after_context", "after_question", "after_choices", "append"}

    def _parse_response(self, response_text: str) -> tuple[list[AnswerLabel], str]:
        """Parse LLM response into labels + instruction_placement.

        Primary format: bare JSON array of label objects.
        Legacy format: JSON object with "labels" + "instruction_placement" keys.
        Placement always defaults to "append" (generator decides per-candidate).
        """
        text = response_text.strip()

        # Extract JSON from code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        placement = "append"
        labels_data = []

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try extracting JSON object or array from raw text
            obj_match = re.search(r'\{[\s\S]*\}', response_text)
            arr_match = re.search(r'\[[\s\S]*\]', response_text)
            if obj_match:
                try:
                    data = json.loads(obj_match.group())
                except json.JSONDecodeError:
                    data = None
            elif arr_match:
                try:
                    data = json.loads(arr_match.group())
                except json.JSONDecodeError:
                    data = None
            else:
                data = None

        if data is None:
            logger.warning("Could not parse label ideation response")
            return [], "append"

        # New format: {"labels": [...], "instruction_placement": "..."}
        if isinstance(data, dict) and "labels" in data:
            labels_data = data["labels"]
            raw_placement = data.get("instruction_placement", "append")
            if raw_placement in self._VALID_PLACEMENTS:
                placement = raw_placement
        # Legacy format: bare array of labels
        elif isinstance(data, list):
            labels_data = data
        elif isinstance(data, dict):
            # Maybe the object IS the response but without "labels" key
            # (e.g., LLM returned labels at top level)
            labels_data = []

        labels = []
        for item in labels_data:
            if not isinstance(item, dict) or "name" not in item:
                continue
            labels.append(AnswerLabel(
                name=item.get("name", "unknown"),
                description=item.get("description", ""),
                extraction_hint=item.get("extraction_hint", ""),
                value_type=item.get("value_type", "string"),
                possible_values=item.get("possible_values"),
                verification_method=item.get("verification_method", "programmatic"),
                extraction_pattern=item.get("extraction_pattern"),
                judge_prompt=item.get("judge_prompt"),
                axis_preamble=item.get("axis_preamble"),
                axis_response_format=item.get("axis_response_format"),
            ))

        return labels, placement

    def _fallback_labels(self, task_type: str) -> list[AnswerLabel]:
        """Provide basic fallback labels when LLM ideation fails."""
        return [
            AnswerLabel(
                name="final_answer",
                description="The core answer or conclusion from the response",
                extraction_hint="Extract the last sentence or the text after 'therefore'/'the answer is'",
                value_type="string",
                verification_method="programmatic",
                extraction_pattern=r"(?:therefore|the answer is|thus|hence|so)\s*[:.]?\s*(.+?)(?:\.|$)",
            ),
            AnswerLabel(
                name="response_length",
                description="Whether the response is short (<50 chars) or long",
                extraction_hint="Check response length",
                value_type="categorical",
                possible_values=["short", "long"],
                verification_method="programmatic",
                extraction_pattern=None,  # handled by special case in extract_features
            ),
        ]
