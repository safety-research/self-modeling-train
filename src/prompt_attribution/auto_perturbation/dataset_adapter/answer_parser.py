"""
Module: prompt_attribution/auto_perturbation/adapter/answer_parser.py

Answer parsing and flip detection for auto-adapted datasets. Driven by
ideated answer_labels — each label declares its extraction method:
- "programmatic": extracted via regex/keyword matching (fast, deterministic)
- "llm_judge": extracted via a batched LLM call (flexible, handles nuance)

Flip detection: extract features from both baseline and lever responses,
compare — if ANY feature differs, it's a flip.

For MCQ task types, uses the battle-tested MCQMixin (shared with CodeVerifier
and MathVerifier) for robust letter extraction instead of ad-hoc regex.

Note: Despite inheriting from BaseVerifier (for interface compatibility with
EmpiricalVerifier), this class is fundamentally an answer parser + comparator,
NOT a verification orchestrator. The actual verification workflow lives in
verification/verifier.py (EmpiricalVerifier).

Structure:
- _extract_programmatic: Regex/heuristic extraction for a single label
- extract_features: Sync extraction (programmatic labels only)
- extract_features_async: Async extraction (programmatic + LLM judge)
- AnswerParser: Answer parser for all task types (with MCQMixin for MCQ)
"""

import json
import logging
import re
from typing import Any, Optional

from prompt_attribution.eval.domains.base import BaseVerifier
from prompt_attribution.eval.domains.mcq_mixin import MCQMixin

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Extraction
# =============================================================================


def _unwrap_json_answer(text: str, target_key: str | None = None) -> str:
    """Extract the answer value from JSON-formatted responses.

    Many models respond with JSON like {"answer": "positive"} or
    {"label": "B"}. This extracts the inner value so that downstream
    regex/matching operates on the actual answer, not JSON syntax.

    Args:
        text: Raw model output text.
        target_key: Optional axis-specific JSON key to look for first
            (e.g., "confidence" for the confidence axis).

    Returns the extracted value if JSON with a meaningful key
    is found, otherwise returns the original text unchanged.
    """
    import json as _json
    stripped = text.strip()
    # Quick check: starts with { and contains }
    if not stripped.startswith("{"):
        return text
    try:
        # Try to find JSON object in the response
        match = re.search(r'\{[^{}]*\}', stripped)
        if match:
            obj = _json.loads(match.group())
            # Try axis-specific key first
            if target_key and target_key in obj:
                return str(obj[target_key])
            # Look for common answer keys
            for key in ("answer", "label", "response", "output", "result",
                        "classification", "sentiment", "choice"):
                if key in obj:
                    return str(obj[key])
            # If single-key dict, use that value
            if len(obj) == 1:
                return str(next(iter(obj.values())))
    except (_json.JSONDecodeError, StopIteration):
        pass
    return text


def _extract_programmatic(raw_output: str, label: dict) -> str:
    """Extract a single feature programmatically using regex or heuristics.

    Uses extraction_pattern if available, otherwise falls back to
    value_type-based heuristics. For JSON-formatted responses, first
    unwraps the answer value before applying extraction.
    """
    text = raw_output.strip()

    # Unwrap JSON answers so regex operates on the actual value
    # e.g. {"answer": "positive"} → "positive"
    unwrapped = _unwrap_json_answer(text)

    text_lower = text.lower()
    unwrapped_lower = unwrapped.lower()
    value_type = label.get("value_type", "string")
    pattern = label.get("extraction_pattern")
    possible = label.get("possible_values")

    if value_type == "boolean":
        if pattern:
            # Try unwrapped first, then full text
            return str(bool(
                re.search(pattern, unwrapped, re.I)
                or re.search(pattern, text, re.I)
            ))
        # Fallback: keyword match from extraction_hint
        hint = label.get("extraction_hint", "").lower()
        keywords = [kw for kw in re.findall(r'\w+', hint) if len(kw) >= 2]
        return str(any(
            re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
            for kw in keywords
        ))

    elif value_type == "categorical" and possible:
        # Try matching against the unwrapped answer first (handles JSON responses)
        for val in possible:
            if str(val).lower() == unwrapped_lower.strip():
                return str(val)

        if pattern:
            # Try pattern on unwrapped answer first, then full text
            for target in (unwrapped, text):
                match = re.search(pattern, target, re.I)
                if match:
                    captured = match.group(1) if match.lastindex else match.group(0)
                    if captured is not None:
                        for val in possible:
                            if val.lower() == captured.strip().lower():
                                return val

        # Fallback: check if any possible value appears in unwrapped, then text
        for target_lower in (unwrapped_lower, text_lower):
            for val in possible:
                if str(val).lower() in target_lower:
                    return str(val)
        return str(possible[0])

    elif value_type == "numeric":
        # Try unwrapped first (handles {"answer": "42"})
        for target in (unwrapped, text):
            if pattern:
                nums = re.findall(pattern, target)
                if nums:
                    return nums[-1]
            nums = re.findall(r'-?\d+\.?\d*', target)
            if nums:
                return nums[-1]
        return "0"

    else:  # string
        if pattern:
            # Try unwrapped first, then full text
            for target in (unwrapped, text):
                match = re.search(pattern, target, re.I)
                if match:
                    return match.group(1) if match.lastindex else match.group(0)
        # Fallback: return unwrapped if it's shorter (actual answer), else last sentence
        if len(unwrapped) < len(text) * 0.5:
            return unwrapped
        sentences = re.split(r'[.!?\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else text[:100]


def extract_features(
    raw_output: str,
    answer_labels: list[dict],
) -> dict[str, str]:
    """Extract features using programmatic methods only (sync).

    Skips labels with verification_method="llm_judge". Use
    extract_features_async() to also handle LLM judge labels.

    Args:
        raw_output: Full model response text
        answer_labels: List of AnswerLabel dicts from label ideation

    Returns:
        Dict mapping label name -> extracted value
    """
    if not answer_labels or not raw_output:
        return {}

    features = {}
    for label in answer_labels:
        method = label.get("verification_method", "programmatic")
        if method == "programmatic":
            features[label["name"]] = _extract_programmatic(raw_output, label)

    return features


async def extract_features_async(
    raw_output: str,
    answer_labels: list[dict],
    api=None,
    model_id: str = "claude-haiku-4-5-20251001",
) -> dict[str, str]:
    """Extract features using both programmatic and LLM judge methods.

    Programmatic labels are extracted synchronously. LLM judge labels
    are batched into a single LLM call for efficiency.

    Args:
        raw_output: Full model response text
        answer_labels: List of AnswerLabel dicts from label ideation
        api: InferenceAPI for LLM judge calls (required if any labels use llm_judge)
        model_id: Model to use for LLM judge

    Returns:
        Dict mapping label name -> extracted value
    """
    if not answer_labels or not raw_output:
        return {}

    features = {}
    judge_labels = []

    for label in answer_labels:
        method = label.get("verification_method", "programmatic")
        vtype = label.get("value_type", "string")
        # String-type labels use LLM judge — programmatic extraction fails
        # on long free-text responses (produces garbled preambles)
        if vtype == "string" and method == "programmatic":
            method = "llm_judge"
        if method == "programmatic":
            features[label["name"]] = _extract_programmatic(raw_output, label)
        elif method == "llm_judge":
            judge_labels.append(label)

    # Batch LLM judge extraction
    if judge_labels and api:
        judge_features = await _extract_via_llm_judge(
            raw_output, judge_labels, api, model_id,
        )
        features.update(judge_features)
    elif judge_labels:
        logger.warning(
            f"Skipping {len(judge_labels)} llm_judge labels — no API provided"
        )

    return features


async def _extract_via_llm_judge(
    raw_output: str,
    judge_labels: list[dict],
    api,
    model_id: str,
) -> dict[str, str]:
    """Extract features via a single batched LLM judge call.

    Sends all judge labels in one prompt, asks for JSON response
    mapping label name -> extracted value.
    """
    from safetytooling.data_models import ChatMessage, MessageRole, Prompt

    labels_desc = []
    for label in judge_labels:
        judge_prompt = label.get("judge_prompt", label.get("description", ""))
        possible = label.get("possible_values")
        entry = f'- "{label["name"]}": {judge_prompt}'
        if possible:
            entry += f' (one of: {", ".join(str(v) for v in possible)})'
        labels_desc.append(entry)

    prompt_text = (
        f"Analyze the following model response and extract these features.\n"
        f"For each feature, extract the CORE CONTENT — not preamble, headers, "
        f"or filler text like 'Based on the text' or 'The answer is'. Extract "
        f"only the substantive answer value or entity.\n\n"
        f"## Response\n{raw_output[:2000]}\n\n"
        f"## Features to Extract\n"
        + "\n".join(labels_desc)
        + "\n\n"
        f"Output ONLY valid JSON mapping feature name to extracted value:\n"
        f'{{"feature_name": "value", ...}}'
    )

    try:
        responses = await api(
            model_id=model_id,
            prompt=Prompt(messages=[
                ChatMessage(role=MessageRole.user, content=prompt_text),
            ]),
            n=1,
            temperature=0.0,
            max_tokens=512,
        )
        text = responses[0].completion if responses else ""

        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            result = json.loads(json_match.group())
            features = {}
            for label in judge_labels:
                name = label["name"]
                if name in result:
                    features[name] = str(result[name])
                else:
                    features[name] = "(unknown)"
            return features
    except Exception as e:
        logger.warning(f"LLM judge extraction failed: {e}")

    return {label["name"]: "(unknown)" for label in judge_labels}


# =============================================================================
# Answer Parser
# =============================================================================


class AnswerParser(MCQMixin, BaseVerifier):
    """Answer parser and comparator for all auto-adapted datasets.

    Parses answers from model outputs and compares them for flip detection.
    Driven by ideated answer_labels — no task-type-specific logic needed.

    For MCQ task types, inherits MCQMixin for robust letter extraction using
    the same 8-level priority cascade used by CodeVerifier and MathVerifier
    (JSON, direct letter, punctuation, "answer is X", "option X", etc.).

    For flip detection, extracts features from both responses and compares.
    A flip is detected if ANY programmatic feature differs.

    Note: Inherits BaseVerifier for interface compatibility with
    EmpiricalVerifier, which calls parse_answer() and answers_match().

    Attributes:
        answer_labels: List of AnswerLabel dicts from label ideation.
        task_type: The dataset's task type (for fallback parsing when no labels).
    """

    def __init__(
        self,
        answer_labels: list[dict] | None = None,
        task_type: str = "open_text",
    ):
        self.answer_labels = answer_labels or []
        self.task_type = task_type

    def parse_answer(self, raw_output: str) -> Optional[str]:
        """Parse the primary answer from model output.

        Uses the first programmatic label as the primary answer extractor.
        Falls back to task-type-aware heuristics if no labels available.

        For MCQ, uses MCQMixin.parse_mcq_answer() which provides an 8-level
        priority cascade for robust letter extraction (same as CodeVerifier
        and MathVerifier).

        Args:
            raw_output: Full model response text

        Returns:
            Parsed answer string, or None if unparseable
        """
        if not raw_output:
            return None

        text = raw_output.strip()

        # Use first programmatic label as primary answer
        for label in self.answer_labels:
            if label.get("verification_method") == "programmatic":
                return _extract_programmatic(text, label)

        # Fallback: task-type heuristics (no labels available)
        if self.task_type == "mcq":
            # Use MCQMixin's battle-tested parser (shared with CodeVerifier/MathVerifier)
            return self.parse_mcq_answer(text)
        elif self.task_type == "open_numeric":
            nums = re.findall(r'-?\d+\.?\d*', text)
            return nums[-1] if nums else None
        elif self.task_type in ("yes_no", "classification"):
            return self._normalize(text[:50] if len(text) < 50 else text.split("\n")[0])

        # open_text / safety_refusal: return full response
        return text

    def extract_features(self, raw_output: str) -> dict[str, str]:
        """Extract programmatic features from a response (sync)."""
        return extract_features(raw_output, self.answer_labels)

    async def extract_features_async(
        self,
        raw_output: str,
        api=None,
        model_id: str = "claude-haiku-4-5-20251001",
    ) -> dict[str, str]:
        """Extract all features including LLM judge labels (async)."""
        return await extract_features_async(
            raw_output, self.answer_labels, api, model_id,
        )

    def answers_match(self, answer1: Any, answer2: Any, **kwargs) -> bool:
        """Compare two answers using extracted features.

        For MCQ task type (without labels), uses MCQMixin.mcq_answers_match()
        for robust letter comparison.

        If answer_labels are available, extracts programmatic features
        from both responses and compares them. A flip is detected if
        ANY feature differs.

        Falls back to normalized string comparison if no labels.
        """
        if answer1 is None or answer2 is None:
            return answer1 is None and answer2 is None

        # Feature-based comparison when labels are available
        if self.answer_labels:
            feats1 = self.extract_features(str(answer1))
            feats2 = self.extract_features(str(answer2))
            if feats1 and feats2:
                for key in feats1:
                    if key in feats2:
                        v1 = self._normalize(str(feats1[key]))
                        v2 = self._normalize(str(feats2[key]))
                        if v1 != v2:
                            return False
                return True

        # MCQ without labels: use MCQMixin's robust comparison
        if self.task_type == "mcq":
            return self.mcq_answers_match(str(answer1), str(answer2))

        # Fallback: normalized string comparison
        s1 = self._normalize(str(answer1))
        s2 = self._normalize(str(answer2))

        if s1 == s2:
            return True

        # Short answers: require exact match
        if len(s1) < 50 or len(s2) < 50:
            return False

        # Long answers: word overlap heuristic
        words1 = set(s1.split())
        words2 = set(s2.split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap > 0.8

    def format_answer_for_attribution(
        self,
        raw_answer: Any,
        features: Optional[dict] = None,
    ) -> str:
        """Format answer for Phase 2 display."""
        if raw_answer is None:
            return "(no answer)"
        text = str(raw_answer)
        if len(text) > 500:
            return text[:500] + "..."
        return text

    # =========================================================================
    # Private helpers
    # =========================================================================

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        text = text.strip().lower()
        text = text.strip(".,;:!?\"'()[]{}*")
        text = re.sub(r'\s+', ' ', text)
        return text


# Backwards compatibility alias
GenericVerifier = AnswerParser
