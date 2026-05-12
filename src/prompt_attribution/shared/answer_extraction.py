"""
Module: prompt_attribution/shared/answer_extraction.py

Shared answer extraction and flip computation used by both:
- Auto-perturbation pipeline (empirical_verification/verifier.py)
- GT cache generation (training/ground_truth/phase1_inference.py)

Single source of truth for:
- LLM judge config (model, temperature, prompt template)
- Programmatic extraction (delegates to answer_parser._extract_programmatic)
- JSON unwrapping (delegates to answer_parser._unwrap_json_answer)
- Flip computation from extracted axis values

Structure:
- JUDGE_* constants: Shared config for LLM judge calls
- extract_axis_value(): Extract a single axis value from a model response
- compute_flip(): Compare baseline vs lever responses, return FlipResult
- FlipResult: Dataclass for flip computation results
"""

import asyncio
import json
import logging
import re
import traceback
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Shared judge config ──────────────────────────────────────────────────────
# All answer extraction (pipeline + GT cache) uses these settings.
# Change here → changes everywhere.

JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_TEMPERATURE = 0.0  # Deterministic extraction
JUDGE_MAX_TOKENS = 4096  # Aligned with pipeline verifier (verifier.py) for consistent extraction
JUDGE_TIMEOUT_S = 120  # Seconds before giving up on a judge call
FAILURE_VALUE = "(unknown)"  # Returned when extraction fails (matches pipeline convention)


# ── Extraction ───────────────────────────────────────────────────────────────

async def extract_axis_value(
    response: str,
    label: dict,
    api=None,
    judge_model: str = JUDGE_MODEL,
) -> str:
    """Extract a single axis value from a model response.

    Mirrors the pipeline's extraction logic (verifier.py lines 131-198):
    - String-type labels with method="programmatic" are promoted to llm_judge
    - Programmatic extraction returns raw value (no normalization)
    - LLM judge extraction returns raw value (no normalization)
    - Normalization happens in compute_flip() comparison step, NOT here

    Returns:
        Raw extracted value. "(unknown)" on failure.
    """
    if not response:
        return FAILURE_VALUE

    method = label.get("verification_method", "llm_judge")
    vtype = label.get("value_type", "string")

    # Promote string-type programmatic to llm_judge
    # (matches pipeline verifier.py line 146: "if vtype == 'string' or method == 'llm_judge'")
    if vtype == "string" and method == "programmatic":
        method = "llm_judge"

    if method == "programmatic":
        return _extract_programmatic_raw(response, label)

    # LLM judge path
    if api is None:
        logger.warning("No API provided for llm_judge extraction, falling back to programmatic")
        return _extract_programmatic_raw(response, label)

    return await _extract_via_judge(response, label, api, judge_model)


def _extract_programmatic_raw(response: str, label: dict) -> str:
    """Programmatic extraction — returns raw value, no normalization.

    Delegates to the pipeline's _extract_programmatic (answer_parser.py).
    Normalization happens in compute_flip() comparison step.
    """
    from prompt_attribution.auto_perturbation.dataset_adapter.answer_parser import (
        _extract_programmatic,
    )
    value = _extract_programmatic(response, label)
    return str(value) if value else FAILURE_VALUE


async def _extract_via_judge(
    response: str,
    label: dict,
    api,
    judge_model: str,
) -> str:
    """Extract a value using LLM judge with shared config.

    Mirrors the exact logic from the pipeline's _extract_via_llm_judge
    (answer_parser.py lines 249-310) for consistency:
    - Same prompt template (CORE CONTENT instruction)
    - Same T=0.0, max_tokens=512
    - Same JSON parsing (regex → json.loads)
    - Same failure value: "(unknown)" on missing key or exception
    - NO normalization here — callers normalize after extraction
      (matching pipeline verifier.py lines 168-169)

    Wrapped in a timeout to prevent blocking on slow API calls (not in pipeline,
    but needed for GT cache which runs outside the pipeline's retry framework).
    """
    # Match pipeline's judge_prompt resolution: try judge_prompt, then description
    judge_prompt = label.get("judge_prompt", label.get("description", ""))
    name = label.get("name", "")
    possible = label.get("possible_values")

    # Build prompt — identical to pipeline's _extract_via_llm_judge
    feature_desc = f'- "{name}": {judge_prompt}'
    if possible:
        feature_desc += f' (one of: {", ".join(str(v) for v in possible)})'

    prompt_text = (
        f"Analyze the following model response and extract these features.\n"
        f"For each feature, extract the CORE CONTENT — not preamble, headers, "
        f"or filler text like 'Based on the text' or 'The answer is'. Extract "
        f"only the substantive answer value or entity.\n\n"
        f"## Response\n{response[:2000]}\n\n"
        f"## Features to Extract\n"
        f"{feature_desc}\n\n"
        f"Output ONLY valid JSON mapping feature name to extracted value:\n"
        f'{{"feature_name": "value", ...}}'
    )

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            # On retry, use random temperature to bust safetytooling cache
            # (same prompt + same temp returns the same cached bad response)
            import random
            retry_temp = round(random.uniform(0.1, 0.5), 4) if attempt > 0 else None
            result_text = await asyncio.wait_for(
                _call_judge_api(api, judge_model, prompt_text, temperature_override=retry_temp),
                timeout=JUDGE_TIMEOUT_S,
            )

            # Parse JSON response — same regex as pipeline
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                obj = json.loads(json_match.group())
                if name in obj:
                    # Return raw str(value) — NO normalization
                    # (pipeline returns str(result[name]) without .strip().lower())
                    return str(obj[name])
                else:
                    if attempt < max_retries:
                        continue  # Retry — key missing
                    return FAILURE_VALUE

            # No JSON found — retry or return failure
            if attempt < max_retries:
                continue
            return FAILURE_VALUE

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                logger.debug(f"Judge JSON parse failed (attempt {attempt + 1}), retrying: {e}")
                continue
            logger.warning(f"\033[31m[JUDGE FAIL]\033[0m JSON parse failed after {max_retries + 1} attempts: {e}")
            return FAILURE_VALUE
        except asyncio.TimeoutError:
            if attempt < max_retries:
                logger.debug(f"Judge timeout (attempt {attempt + 1}), retrying")
                continue
            logger.warning(f"\033[31m[JUDGE TIMEOUT]\033[0m after {max_retries + 1} attempts ({JUDGE_TIMEOUT_S}s each)")
            return FAILURE_VALUE
        except (OSError, IOError) as e:
            # NFS stale file handle / transient I/O — retryable
            if attempt < max_retries:
                logger.debug(f"Judge I/O error (attempt {attempt + 1}), retrying: {e}")
                continue
            logger.warning(f"\033[31m[JUDGE FAIL]\033[0m {type(e).__name__} after {max_retries + 1} attempts: {e}")
            return FAILURE_VALUE
        except Exception as e:
            tb = traceback.format_exc()
            logger.warning(f"\033[31m[JUDGE FAIL]\033[0m {type(e).__name__}: {e}\n{tb}")
            return FAILURE_VALUE

    return FAILURE_VALUE


async def _call_judge_api(
    api, model: str, prompt_text: str, temperature_override: float | None = None,
) -> str:
    """Call the judge model via safetytooling API with shared config."""
    from safetytooling.data_models import ChatMessage, MessageRole, Prompt

    temp = temperature_override if temperature_override is not None else JUDGE_TEMPERATURE
    responses = await api(
        model_id=model,
        prompt=Prompt(
            messages=[ChatMessage(role=MessageRole.user, content=prompt_text)]
        ),
        n=1,
        temperature=temp,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    return responses[0].completion if responses else ""


# ── Flip computation ─────────────────────────────────────────────────────────

@dataclass
class FlipResult:
    """Result of flip computation between baseline and lever responses."""
    flip_count: int
    n_runs: int
    flip_fraction: float
    flipped: bool
    bl_values: list[str]
    lv_values: list[str]


async def compute_flip(
    bl_responses: list[str],
    lv_responses: list[str],
    target_label: dict | None,
    api=None,
    judge_model: str = JUDGE_MODEL,
) -> FlipResult:
    """Compute flip by extracting axis values and comparing.

    For each (baseline, lever) response pair, extracts the target axis value
    and compares. A flip is counted when values differ.

    Args:
        bl_responses: Baseline model responses (one per run).
        lv_responses: Lever model responses (one per run).
        target_label: Label definition dict (name, verification_method, etc.).
            If None, falls back to normalized string comparison.
        api: InferenceAPI for LLM judge calls.
        judge_model: Model ID for judge extraction.

    Returns:
        FlipResult with flip count, fraction, and extracted values.
    """
    n_runs = min(len(bl_responses), len(lv_responses))
    if n_runs == 0:
        return FlipResult(0, 0, 0.0, False, [], [])

    bl_values: list[str] = []
    lv_values: list[str] = []

    if target_label:
        # Extract axis values for all responses in parallel
        tasks = []
        for resp in bl_responses[:n_runs] + lv_responses[:n_runs]:
            tasks.append(extract_axis_value(resp, target_label, api, judge_model))
        all_values = await asyncio.gather(*tasks, return_exceptions=True)
        all_values = [v if isinstance(v, str) else FAILURE_VALUE for v in all_values]

        bl_values = all_values[:n_runs]
        lv_values = all_values[n_runs:]
    else:
        # Fallback: normalized string comparison
        for bl_resp, lv_resp in zip(bl_responses[:n_runs], lv_responses[:n_runs]):
            bl_values.append(bl_resp.strip().lower()[:500])
            lv_values.append(lv_resp.strip().lower()[:500])

    # Normalize before comparing — matches pipeline verifier.py lines 168-169:
    #   b_norm = str(b_val).strip().lower() if b_val else ""
    #   l_norm = str(l_val).strip().lower() if l_val else ""
    def _norm(val: str) -> str:
        return str(val).strip().lower() if val else ""

    # If ANY extraction returned "(unknown)" after retries, flip label is unreliable.
    # Set flip_fraction=0.5 (uncertain) instead of silently biasing toward no-flip.
    any_unknown = any(
        _norm(v) == _norm(FAILURE_VALUE)
        for v in bl_values + lv_values
    )
    if any_unknown:
        logger.warning(
            f"\033[33m[UNCERTAIN]\033[0m Extraction returned (unknown) — "
            f"setting flip_fraction=0.5. "
            f"bl={bl_values}, lv={lv_values}"
        )
        return FlipResult(
            flip_count=0,
            n_runs=n_runs,
            flip_fraction=0.5,
            flipped=False,
            bl_values=bl_values,
            lv_values=lv_values,
        )

    flip_count = sum(
        1 for bl_val, lv_val in zip(bl_values, lv_values)
        if _norm(bl_val) != _norm(lv_val)
    )
    flip_fraction = flip_count / n_runs
    flipped = flip_fraction > 0.5

    return FlipResult(
        flip_count=flip_count,
        n_runs=n_runs,
        flip_fraction=round(flip_fraction, 3),
        flipped=flipped,
        bl_values=bl_values,
        lv_values=lv_values,
    )
