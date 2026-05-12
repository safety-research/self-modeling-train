"""
Module: prompt_attribution/auto_perturbation/adapter/edit_utils.py

Shared utilities for applying problem edits and computing edit metrics.
Used by both DatasetAdapter and BenchmarkAdapter.

Structure:
- apply_field_edits: Apply ProblemEdit find-and-replace operations to an example
- compute_edit_metrics: Compute edit distance/fraction between two prompts
"""

import difflib
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import ProblemEdit
    from .dataset_adapter import AdaptedExample

logger = logging.getLogger(__name__)


def _strip_latex(text: str) -> str:
    """Strip LaTeX $ delimiters for fuzzy matching."""
    return re.sub(r'\$', '', text)


def _robust_replace(text: str, original: str, replacement: str) -> str:
    """Replace original with replacement, with fallback for formatting mismatches.

    LLM-generated edits sometimes strip LaTeX delimiters ($x$ → x),
    normalize quotes, or change whitespace. This function tries exact match
    first, then falls back to matching on LaTeX-stripped text.
    """
    # Try exact match first
    if original in text:
        return text.replace(original, replacement, 1)

    # Fallback: strip $ from text and try again
    stripped_text = _strip_latex(text)
    stripped_original = _strip_latex(original)

    if stripped_original in stripped_text:
        # Found match in stripped version. Now find the corresponding
        # span in the original text by scanning character positions.
        idx = stripped_text.find(stripped_original)

        # Build position map: stripped_pos → original_pos
        orig_positions = []  # orig_positions[stripped_idx] = original_idx
        for i, ch in enumerate(text):
            if ch != '$':
                orig_positions.append(i)

        if idx + len(stripped_original) <= len(orig_positions):
            start = orig_positions[idx]
            end = orig_positions[idx + len(stripped_original) - 1] + 1
            # Extend end to include trailing $ if present
            while end < len(text) and text[end] == '$':
                end += 1
            return text[:start] + replacement + text[end:]

    # Fallback: LLM may have included template framing (e.g., "Text: ..."
    # or "Question: ...") in the original. Try matching a suffix of the
    # original against the text — if the tail matches, the LLM just
    # prepended template labels.
    for prefix_len in range(min(30, len(original) - 5)):
        suffix = original[prefix_len:]
        if len(suffix) >= 10 and suffix in text:
            # Also strip the same prefix from the replacement if present
            if replacement.startswith(original[:prefix_len]):
                adj_replacement = replacement[prefix_len:]
            else:
                adj_replacement = replacement
            return text.replace(suffix, adj_replacement, 1)

    # Fallback: LLM may have truncated the original (copied only a prefix
    # of the actual substring). Find where the original starts in the text
    # and replace up to len(original) characters.
    if len(original) >= 20:
        # Try matching the original as a prefix of a longer substring
        idx = text.find(original[:20])
        if idx >= 0:
            # Verify more of the original matches at this position
            match_len = 0
            for j in range(min(len(original), len(text) - idx)):
                if text[idx + j] == original[j]:
                    match_len = j + 1
                else:
                    break
            if match_len >= len(original) * 0.8:
                # Good prefix match — replace the matched portion
                logger.debug(
                    f"Edit original truncated, prefix-matched {match_len}/{len(original)} chars"
                )
                return text[:idx] + replacement + text[idx + match_len:]

    logger.warning(
        f"Edit original not found (even with fuzzy match): '{original[:60]}'"
    )
    return text


def apply_field_edits(
    example: "AdaptedExample",
    edits: list["ProblemEdit"],
) -> "AdaptedExample":
    """Create a shallow copy of example with field-level edits applied.

    For "question"/"context": simple str.replace on the field.
    For "choices": str.replace on each individual choice string.

    Args:
        example: The original adapted example
        edits: List of ProblemEdit operations to apply

    Returns:
        New AdaptedExample with edits applied
    """
    from .dataset_adapter import AdaptedExample as AE

    question = example.question
    context = example.context
    choices = list(example.choices) if example.choices else None

    for edit in edits:
        if edit.field == "question":
            question = _robust_replace(question, edit.original, edit.replacement)
        elif edit.field == "context" and context is not None:
            context = _robust_replace(context, edit.original, edit.replacement)
        elif edit.field == "context" and context is None:
            # LLM targeted "context" but this dataset has no context field —
            # the text lives in "question". Fall back to editing question.
            if edit.original in question:
                logger.debug(
                    f"Edit targets 'context' but context is None, "
                    f"falling back to 'question'"
                )
                question = _robust_replace(question, edit.original, edit.replacement)
        elif edit.field == "choices" and choices is not None:
            choices = [_robust_replace(c, edit.original, edit.replacement) for c in choices]

    return AE(
        idx=example.idx,
        question=question,
        ground_truth_answer=example.ground_truth_answer,
        choices=choices,
        context=context,
        metadata=example.metadata,
    )


def compute_edit_metrics(
    baseline_prompt: str,
    lever_prompt: str,
) -> tuple[int, float]:
    """Compute edit distance and edit fraction between two prompts.

    Uses SequenceMatcher ratio to compute a character-level similarity,
    then derives edit distance as (1 - ratio) * max_len.

    Args:
        baseline_prompt: The baseline (original) prompt text
        lever_prompt: The lever (modified) prompt text

    Returns:
        (edit_distance, edit_fraction) where:
        - edit_distance: approximate char-level edit distance
        - edit_fraction: edit_distance / len(baseline_prompt)
    """
    if not baseline_prompt:
        return len(lever_prompt), 1.0

    # SequenceMatcher.ratio() gives 2.0 * M / T where M = matching chars,
    # T = total chars in both strings. We derive distance from this.
    ratio = difflib.SequenceMatcher(None, baseline_prompt, lever_prompt).ratio()
    max_len = max(len(baseline_prompt), len(lever_prompt))
    edit_distance = int((1.0 - ratio) * max_len)
    edit_fraction = edit_distance / len(baseline_prompt) if baseline_prompt else 0.0

    return edit_distance, round(edit_fraction, 4)
