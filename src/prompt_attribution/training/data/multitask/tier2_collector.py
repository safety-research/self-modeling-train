"""
Module: prompt_attribution/training/data/multitask/tier2_collector.py

Collects additional Phase 1 inference data for Tier 2 training tasks.
Requires a running vLLM server (target model) and optionally Claude API
(for decomposition in E7).

Structure:
- ModelClient: Lightweight async vLLM client with concurrency control
- Tier2Collector: Main collector with per-task async methods
  - collect_e4: Correctness probability (resample + GT check)
  - collect_e5: Confidence calibration (confidence-eliciting Phase 1)
  - collect_e7: Component ablation (decompose + ablation resampling)
  - collect_e8: Propose flip instruction (model proposes + we test)
  - collect_e10: Margin & second choice (logprobs for MCQ)
"""

import asyncio
import json
import logging
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import openai  # type: ignore[import-untyped]
from openai import AsyncOpenAI

from prompt_attribution.training.data.multitask.task_prompts import _strip_respond_line

from prompt_attribution.training.data.multitask.schema import (
    MultitaskRecord,
    TaskType,
)
from prompt_attribution.training.data.multitask.task_prompts import (
    build_e4_prompt,
    build_e5_phase1_prompt,
    build_e5_phase2_prompt,
    build_e7_prompt,
    build_e8_prompt,
    build_e10_prompt,
)

logger = logging.getLogger(__name__)

# ANSI colors
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Lightweight async model client
# ---------------------------------------------------------------------------

GPT_OSS_SYSTEM_PROMPT = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: 2026-04-13\n\n"
    "Reasoning: high\n\n"
    "# Valid channels: analysis, commentary, final. "
    "Channel must be included for every message."
)


@dataclass
class ModelClientConfig:
    """Configuration for the async model client."""

    vllm_url: str  # e.g., "http://HOST:PORT/v1"
    model_id: str  # e.g., "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 2048
    max_concurrent: int = 32
    system_prompt: str | None = None  # e.g., GPT_OSS_SYSTEM_PROMPT for thinking


class ModelClient:
    """Lightweight async client for vLLM inference with concurrency control."""

    def __init__(self, config: ModelClientConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.vllm_url,
            api_key="not-needed",
        )
        self._sem = asyncio.Semaphore(config.max_concurrent)
        self._call_count = 0
        # Auto-detect thinking mode from model name
        from prompt_attribution.training.config import ModelFormat
        self._model_format = ModelFormat.from_model_name(
            config.model_id, max_tokens=config.max_tokens,
        )
        self._extra_body = self._model_format.get_thinking_extra_body() or None

    async def call(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call the model and return response text."""
        async with self._sem:
            self._call_count += 1
            try:
                messages = []
                if self.config.system_prompt:
                    messages.append({"role": "system", "content": self.config.system_prompt})
                messages.append({"role": "user", "content": prompt})
                kwargs: dict = dict(
                    model=self.config.model_id,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                )
                if self._extra_body:
                    kwargs["extra_body"] = self._extra_body
                resp = await self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except openai.BadRequestError as e:
                if "maximum context length" in str(e):
                    logger.warning(f"Skipping prompt exceeding context length: {e}")
                    return ""
                raise

    async def call_with_logprobs(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 5,
        top_logprobs: int = 10,
    ) -> tuple[str, list[dict]]:
        """Call the model and return (response_text, token_logprobs).

        token_logprobs is a list of dicts: [{"token": "A", "logprob": -0.5}, ...]
        """
        async with self._sem:
            self._call_count += 1
            try:
                resp = await self.client.chat.completions.create(
                    model=self.config.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature if temperature is not None else self.config.temperature,
                    max_tokens=max_tokens,
                    logprobs=True,
                    top_logprobs=top_logprobs,
                )
            except openai.BadRequestError as e:
                if "maximum context length" in str(e):
                    logger.warning(f"Skipping prompt exceeding context length: {e}")
                    return "", []
                raise
            text = resp.choices[0].message.content or ""
            logprobs_data = []
            if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
                for token_info in resp.choices[0].logprobs.content:
                    entry = {"token": token_info.token, "logprob": token_info.logprob}
                    if token_info.top_logprobs:
                        entry["top_logprobs"] = {
                            tp.token: tp.logprob for tp in token_info.top_logprobs
                        }
                    logprobs_data.append(entry)
            return text, logprobs_data

    async def resample(
        self,
        prompt: str,
        n: int,
        temperature: Optional[float] = None,
    ) -> list[str]:
        """Call model n times with the same prompt, return all responses."""
        tasks = [self.call(prompt, temperature=temperature) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def get_choice_logprobs(
        self,
        prompt: str,
        choices: list[str],
    ) -> dict[str, float]:
        """Get exact logprobs for each MCQ choice letter.

        Uses the completions endpoint with echo=True to get the exact
        logprob of each choice letter token. One call per choice.
        No top-N limitation, no thinking overhead.

        Returns dict of {letter: logprob}.
        """
        logprob_prompt = prompt + "\n\nAnswer with ONLY a single letter. Do not output anything else.\n\n"

        async def _get_one(letter: str) -> tuple[str, float]:
            async with self._sem:
                self._call_count += 1
                try:
                    r = await self.client.completions.create(
                        model=self.config.model_id,
                        prompt=logprob_prompt + letter,
                        max_tokens=0,
                        echo=True,
                        logprobs=1,
                    )
                    tl = r.choices[0].logprobs.token_logprobs
                    if tl and tl[-1] is not None:
                        return letter, tl[-1]
                    return letter, -100.0
                except Exception as e:
                    logger.warning(f"  Choice logprob failed for '{letter}': {e}")
                    return letter, -100.0

        results = await asyncio.gather(*[_get_one(l) for l in choices])
        return dict(results)


# ---------------------------------------------------------------------------
# Haiku API client (drop-in replacement for ModelClient)
# ---------------------------------------------------------------------------


class HaikuModelClient:
    """Async Anthropic API client with the same interface as ModelClient.

    Use instead of ModelClient when the target model is a Claude API model
    (no vLLM required). call_with_logprobs() is emulated via resampling
    since the Anthropic API does not expose token logprobs.
    """

    def __init__(self, model_id: str, api_key: str = "", max_concurrent: int = 20,
                 temperature: float = 0.7, max_tokens: int = 2048):
        import anthropic
        import os
        resolved_key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or ""
        )
        self._aclient = anthropic.AsyncAnthropic(api_key=resolved_key)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._sem = asyncio.Semaphore(max_concurrent)
        self._call_count = 0

    async def call(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call the Anthropic API and return response text."""
        async with self._sem:
            self._call_count += 1
            resp = await self._aclient.messages.create(
                model=self.model_id,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text if resp.content else ""

    async def call_with_logprobs(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 5,
        top_logprobs: int = 10,
        n_resample: int = 20,
    ) -> tuple[str, list[dict]]:
        """Emulate logprobs via resampling (N=20).

        WARNING: Closed-source API models (Haiku, GPT-4, etc.) do not expose
        real token logprobs. This method approximates choice probabilities by
        sampling N responses and counting letter frequencies. The resulting
        ground truth is coarse (resolution = 1/N) and unreliable for low-N.

        E10 (margin + second choice) should NOT be included in training data
        for closed-source models — the GT is too noisy to provide a useful
        learning signal. Use E10 only with open-weight models via vLLM where
        real token logprobs are available.
        """
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "call_with_logprobs: using resampling emulation (N=%d) — "
            "logprobs are approximated, not real. E10 GT will be unreliable. "
            "Use vLLM with an open-weight model for accurate E10 data.",
            n_resample,
        )
        responses = await self.resample(prompt, n=n_resample, temperature=0.7)
        # Count MCQ letter occurrences
        counts: dict[str, int] = defaultdict(int)
        for r in responses:
            letters = re.findall(r'\b([A-D])\b', r[:50])  # Check first 50 chars
            if letters:
                counts[letters[0]] += 1
        total = sum(counts.values()) or 1
        # Convert to logprob format
        logprobs_data = [
            {"token": letter, "logprob": math.log(count / total + 1e-9),
             "top_logprobs": {letter: math.log(count / total + 1e-9)
                              for letter, count in sorted(counts.items())}}
            for letter, count in sorted(counts.items(), key=lambda x: -x[1])
        ]
        # Return most common response + synthetic logprobs
        most_common = max(counts, key=lambda k: counts[k], default="")
        return most_common, logprobs_data

    async def resample(
        self,
        prompt: str,
        n: int,
        temperature: Optional[float] = None,
    ) -> list[str]:
        """Call model n times with the same prompt, return all responses."""
        tasks = [self.call(prompt, temperature=temperature) for _ in range(n)]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Answer matching utilities
# ---------------------------------------------------------------------------

def _normalize_ground_truth(ground_truth: str, prompt_baseline: str) -> str:
    """Normalize ground truth answer for E4 comparison.

    Maps numeric indices (0,1,2,3) to MCQ letter labels (A,B,C,D) when the
    prompt_baseline contains lettered answer choices. Returns the original
    GT for non-MCQ formats or when parsing fails.
    """
    gt_stripped = ground_truth.strip()

    # Only attempt mapping for single-digit numeric strings
    if not gt_stripped.isdigit():
        return gt_stripped

    idx = int(gt_stripped)
    if idx > 25:
        return gt_stripped

    # Check if prompt_baseline has lettered MCQ choices (A), B), etc.)
    import re as _re
    # Match patterns like "A)", "A.", "A:", "(A)" at the start of a line or after whitespace
    choice_pattern = _re.findall(
        r'(?:^|\n)\s*(?:\(?([A-Z])\)?[\.\):\s])', prompt_baseline
    )
    if choice_pattern and idx < len(choice_pattern):
        return choice_pattern[idx]

    # Fallback: map 0→A, 1→B, etc. if the prompt has any letter-based choices
    if _re.search(r'\b[A-D]\)', prompt_baseline) or _re.search(r'\b[A-D]\.\s', prompt_baseline):
        return chr(65 + idx)

    return gt_stripped


def _parse_answer_simple(response: str, ground_truth: str) -> bool:
    """Simple answer matching: check if GT appears in response."""
    response_lower = response.lower().strip()
    gt_lower = ground_truth.lower().strip()

    # Try JSON extraction
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            for key in ("answer", "final_answer", "result"):
                if key in data:
                    return str(data[key]).lower().strip() == gt_lower
    except (json.JSONDecodeError, TypeError):
        pass

    # Exact match
    if response_lower == gt_lower:
        return True

    # Numeric match
    try:
        r_num = float(re.sub(r"[^\d.\-]", "", response_lower))
        g_num = float(re.sub(r"[^\d.\-]", "", gt_lower))
        return abs(r_num - g_num) < 1e-6
    except (ValueError, TypeError):
        pass

    # MCQ letter match
    letters_in_response = re.findall(r'\b([A-E])\b', response)
    if letters_in_response and gt_lower in [l.lower() for l in letters_in_response]:
        return True

    # Substring containment
    return gt_lower in response_lower


def _parse_confidence(response: str) -> Optional[float]:
    """Parse confidence value from model response."""
    # Try JSON
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "confidence" in data:
            return float(data["confidence"])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Try regex
    match = re.search(r'"confidence"\s*:\s*([0-9.]+)', response)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Try bare float
    match = re.search(r'\b(0\.\d+|1\.0|0|1)\b', response)
    if match:
        try:
            val = float(match.group(1))
            if 0.0 <= val <= 1.0:
                return val
        except ValueError:
            pass

    return None


def _parse_json_answer(response: str, field: str) -> Optional[str]:
    """Parse a field from JSON response."""
    try:
        data = json.loads(response)
        if isinstance(data, dict) and field in data:
            return str(data[field])
    except (json.JSONDecodeError, TypeError):
        pass

    # Try regex fallback
    match = re.search(rf'"{field}"\s*:\s*"([^"]*)"', response)
    if match:
        return match.group(1)

    return None


def _extract_answer_key(response: str, capability_tags: list[str] | None = None) -> str:
    """Extract the answer key from a model response for flip comparison.

    Routes by capability_tags to extract the meaningful answer (letter, number,
    label) rather than comparing full response strings (which differ in reasoning
    even when the final answer is the same).

    Returns a normalized lowercase answer string.
    """
    tags = capability_tags or []
    tags_joined = " ".join(t.lower() for t in tags)

    # 1. Try JSON answer field first
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            for key in ("answer", "answer_letter", "final_answer", "result", "label"):
                if key in data:
                    return str(data[key]).strip().lower()
    except (json.JSONDecodeError, TypeError):
        pass

    text = response.strip()

    # 2. Math/numeric: extract last number
    if any(k in tags_joined for k in ("math", "numeric", "arithmetic")):
        nums = re.findall(r'-?\d+(?:\.\d+)?', text)
        if nums:
            return nums[-1].lower()

    # 3. MCQ: extract first standalone letter A-E
    if any(k in tags_joined for k in ("mcq", "multiple_choice", "knowledge", "commonsense",
                                       "reading_comprehension", "science", "logical")):
        letters = re.findall(r'\b([A-E])\b', text)
        if letters:
            return letters[0].lower()

    # 4. Classification/yes_no: first line normalized
    if any(k in tags_joined for k in ("classification", "yes_no")):
        first_line = text.split("\n")[0].strip()
        return first_line[:100].lower()

    # 5. Default: try JSON field, then first short non-empty line
    match = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    if match:
        return match.group(1).strip().lower()

    for line in text.split("\n"):
        line = line.strip()
        if line:
            return line[:200].lower()
    return text[:200].lower()


# ---------------------------------------------------------------------------
# Tags that require LLM judge (open-ended — programmatic extraction unreliable)
# ---------------------------------------------------------------------------

_OPEN_ENDED_TAGS = frozenset({
    "instruction_following", "text_generation", "dialog", "code",
    "code_generation", "creative", "summarization",
})


def _needs_llm_judge(capability_tags: list[str]) -> bool:
    """Check if this example needs an LLM judge for flip detection."""
    tags_joined = " ".join(t.lower() for t in capability_tags)
    return any(tag in tags_joined for tag in _OPEN_ENDED_TAGS)


# ---------------------------------------------------------------------------
# LLM Judge templates
# ---------------------------------------------------------------------------

FLIP_JUDGE_TEMPLATE = """\
Compare these two model responses and determine if they give the SAME answer \
or DIFFERENT answers. Focus on the core answer/conclusion, not reasoning or style.

Response A:
---
{response_a}
---

Response B:
---
{response_b}
---

Do these responses give the same core answer? \
Respond in JSON: {{"same_answer": true}} or {{"same_answer": false}}"""


CORRECTNESS_JUDGE_TEMPLATE = """\
Given this prompt and the model's response, determine if the response is correct.

Prompt:
---
{prompt}
---

Model's response:
---
{response}
---

Is the model's response correct? Consider the task requirements and whether the \
core answer (not formatting or style) is right.

Respond in JSON: {{"correct": true}} or {{"correct": false}}"""


DECOMPOSER_TEMPLATE = """\
Given the following prompt, identify exactly 3 distinct factors that influence \
what answer an AI model would give. These factors can overlap in the text — \
they are semantic aspects, not exclusive text partitions.

Requirements:
- CRITICAL: Each factor must be SMALL and PRECISE — a single word, phrase, \
number, or short clause. Examples of good factors: "the word 'never'", \
"the 2-hour delay", "the constraint to use recursion", "the mention of \
being a student". Examples of BAD factors: "the problem description", \
"the context paragraph", "the task instructions". If you can't point to \
a specific short phrase, the factor is too broad.
- Identify factors at DIFFERENT levels of influence: one that strongly \
determines the answer, one that moderately shapes it, and one with minor effect.
- Each factor should be a concrete aspect of THIS prompt (e.g., \
"the constraint that solutions must use recursion" not "problem constraints").
- For each factor, provide a MODIFIED version of the full prompt where that \
small factor is neutralized (e.g., removed, replaced with a neutral alternative, \
or generalized). The rest of the prompt should be unchanged.
- The modified prompt MUST still be coherent and answerable.
- Use NEUTRAL language — do not evaluate whether any factor is correct or misleading.

Prompt:
---
{full_prompt}
---

Respond in JSON:
{{"components": [
    {{"name": "<concrete factor name>", "text": "<the relevant text or aspect>", \
"ablated_prompt": "<full prompt with this factor neutralized>"}},
    {{"name": "<concrete factor name>", "text": "<the relevant text or aspect>", \
"ablated_prompt": "<full prompt with this factor neutralized>"}},
    {{"name": "<concrete factor name>", "text": "<the relevant text or aspect>", \
"ablated_prompt": "<full prompt with this factor neutralized>"}}
]}}"""


# ---------------------------------------------------------------------------
# Lightweight Anthropic judge client
# ---------------------------------------------------------------------------

def _parse_judge_json(raw: str) -> dict | None:
    """Robustly parse JSON from a judge response that may contain surrounding text."""
    # 1. Try direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    # 2. Extract first {...} block
    match = re.search(r'\{[^}]+\}', raw)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    # 3. Look for key patterns directly (e.g., "same_answer": true)
    if "same_answer" in raw:
        if re.search(r'same_answer["\s:]+true', raw, re.I):
            return {"same_answer": True}
        if re.search(r'same_answer["\s:]+false', raw, re.I):
            return {"same_answer": False}
    if "correct" in raw:
        if re.search(r'"correct"["\s:]+true', raw, re.I):
            return {"correct": True}
        if re.search(r'"correct"["\s:]+false', raw, re.I):
            return {"correct": False}
    return None


def _parse_decomposer_json(raw: str) -> dict | None:
    """Robustly parse decomposer JSON that may have invalid escapes or truncation.

    Haiku's decomposer output often has:
    - Invalid \\escapes in verbatim text (LaTeX, code)
    - Truncated strings when ablated_prompt is long (hits max_tokens)
    - Missing commas between fields
    """
    # 1. Try strict parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Try with strict=False (handles invalid escapes like \\boxed)
    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        pass

    # 3. Extract outermost {...} and try again
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        fragment = match.group()
        try:
            return json.loads(fragment, strict=False)
        except json.JSONDecodeError:
            pass

        # 4. Try to repair truncated JSON: close open strings and brackets
        repaired = fragment
        # Close any unterminated string
        if repaired.count('"') % 2 == 1:
            repaired += '"'
        # Close brackets/braces
        open_braces = repaired.count('{') - repaired.count('}')
        open_brackets = repaired.count('[') - repaired.count(']')
        repaired += ']' * max(open_brackets, 0)
        repaired += '}' * max(open_braces, 0)
        try:
            return json.loads(repaired, strict=False)
        except json.JSONDecodeError:
            pass

    return None


class JudgeClient:
    """Async Anthropic client for LLM judge calls (Haiku for flip, Opus for correctness)."""

    def __init__(
        self,
        api_key: str = "",
        haiku_model: str = "claude-haiku-4-5-20251001",
        opus_model: str = "claude-opus-4-5-20251101",
        max_concurrent_haiku: int = 300,
        max_concurrent_opus: int = 200,
    ):
        import anthropic
        import os
        resolved_key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or ""
        )
        self._client = anthropic.AsyncAnthropic(api_key=resolved_key)
        self.haiku_model = haiku_model
        self.opus_model = opus_model
        self._sem_haiku = asyncio.Semaphore(max_concurrent_haiku)
        self._sem_opus = asyncio.Semaphore(max_concurrent_opus)
        self._call_count = 0

    async def call(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        max_retries: int = 5,
    ) -> str:
        """Call Anthropic API with exponential backoff retry on transient errors."""
        import anthropic
        _RETRYABLE = (
            anthropic.APIStatusError,
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
        )
        use_model = model or self.haiku_model
        sem = self._sem_opus if use_model == self.opus_model else self._sem_haiku
        async with sem:
            self._call_count += 1
            for attempt in range(max_retries):
                try:
                    resp = await self._client.messages.create(
                        model=use_model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return resp.content[0].text if resp.content else ""
                except _RETRYABLE as e:
                    status = getattr(e, "status_code", None)
                    # Only retry on transient errors (429, 529, 500, 502, 503, 504)
                    if status and status < 429 and status not in (408,):
                        raise
                    wait = min(2 ** attempt + random.random(), 30)
                    logger.warning(
                        f"Judge API error (attempt {attempt + 1}/{max_retries}, "
                        f"status={status}): {e}. Retrying in {wait:.1f}s..."
                    )
                    await asyncio.sleep(wait)
            # All retries exhausted — return empty (callers handle gracefully)
            logger.warning(
                f"{YELLOW}[JUDGE]{RESET} All {max_retries} retries exhausted for "
                f"model={use_model}. Returning empty response."
            )
            return ""

    async def judge_flip(self, response_a: str, response_b: str) -> bool:
        """Ask Haiku whether two responses give the same answer.

        Returns True if they are DIFFERENT (i.e., a flip occurred).
        Falls back to False (no flip) on any failure.
        """
        prompt = FLIP_JUDGE_TEMPLATE.format(
            response_a=response_a[:2000],
            response_b=response_b[:2000],
        )
        try:
            raw = await self.call(prompt, model=self.haiku_model)
            if not raw:
                logger.warning(f"{YELLOW}[JUDGE]{RESET} Flip judge returned empty — defaulting to no-flip")
                return False
            parsed = _parse_judge_json(raw)
            if parsed is None:
                logger.warning(f"{YELLOW}[JUDGE]{RESET} Flip judge unparseable: {raw[:120]} — defaulting to no-flip")
                return False
            return not parsed.get("same_answer", True)
        except Exception as e:
            logger.warning(f"{YELLOW}[JUDGE]{RESET} Flip judge failed ({e}) — defaulting to no-flip")
            return False

    async def judge_correctness(self, prompt_text: str, response: str) -> bool:
        """Ask Opus whether a model's response is correct.

        Uses Opus for higher-quality correctness judgments (E4 training data
        will be used for training where accuracy matters).
        Falls back to False (incorrect) on any failure.
        """
        judge_prompt = CORRECTNESS_JUDGE_TEMPLATE.format(
            prompt=prompt_text[:3000],
            response=response[:2000],
        )
        try:
            raw = await self.call(judge_prompt, model=self.opus_model)
            if not raw:
                logger.warning(f"{YELLOW}[JUDGE]{RESET} Correctness judge returned empty — defaulting to incorrect")
                return False
            parsed = _parse_judge_json(raw)
            if parsed is None:
                logger.warning(f"{YELLOW}[JUDGE]{RESET} Correctness judge unparseable: {raw[:120]} — defaulting to incorrect")
                return False
            return parsed.get("correct", False)
        except Exception as e:
            logger.warning(f"{YELLOW}[JUDGE]{RESET} Correctness judge failed ({e}) — defaulting to incorrect")
            return False

    async def decompose_prompt(self, full_prompt: str) -> list[dict] | None:
        """Ask Haiku to semantically decompose a prompt into 3 components.

        Returns list of 3 dicts with name, text, ablated_prompt.
        Falls back to None on failure (caller uses rule-based fallback).
        """
        prompt = DECOMPOSER_TEMPLATE.format(full_prompt=full_prompt[:4000])
        try:
            raw = await self.call(prompt, model=self.haiku_model, max_tokens=4096)
            if not raw:
                logger.warning(f"{YELLOW}[JUDGE]{RESET} Decomposer returned empty — falling back to rule-based")
                return None
            parsed = _parse_decomposer_json(raw)
            if (
                parsed
                and isinstance(parsed, dict)
                and "components" in parsed
                and len(parsed["components"]) == 3
            ):
                components = []
                for comp in parsed["components"]:
                    components.append({
                        "description": comp.get("name", comp.get("text", "")[:200]),
                        "text": comp.get("text", ""),
                        "ablated_prompt": comp.get("ablated_prompt", ""),
                    })
                return components
            logger.warning(f"{YELLOW}[JUDGE]{RESET} Decomposer parse failed — falling back to rule-based")
        except Exception as e:
            logger.warning(f"{YELLOW}[JUDGE]{RESET} Decomposer failed ({e}) — falling back to rule-based")
        return None


# ---------------------------------------------------------------------------
# Hybrid flip detection (programmatic + LLM judge)
# ---------------------------------------------------------------------------

async def _detect_flip_hybrid(
    response_a: str,
    response_b: str,
    capability_tags: list[str],
    judge_client: Optional["JudgeClient"] = None,
) -> bool:
    """Detect if two responses give different answers.

    Uses programmatic extraction for structured tasks, LLM judge for open-ended.
    """
    # 1. Programmatic extraction (fast, free)
    key_a = _extract_answer_key(response_a, capability_tags)
    key_b = _extract_answer_key(response_b, capability_tags)

    if not _needs_llm_judge(capability_tags):
        # Programmatic is reliable for structured tasks
        return key_a != key_b

    # 2. For open-ended tasks: if programmatic keys clearly differ, trust them
    if key_a != key_b and len(key_a) < 50 and len(key_b) < 50:
        return True

    # 3. LLM judge for open-ended tasks
    if judge_client:
        return await judge_client.judge_flip(response_a, response_b)

    # 4. Fallback: raw string comparison (conservative)
    return key_a != key_b


def _decompose_prompt_simple(prompt: str, n_components: int = 3) -> list[dict]:
    """Simple rule-based prompt decomposition into components.

    Splits the prompt into roughly equal segments (sentences/paragraphs).
    Used as fallback when Claude API decomposition fails.
    """
    # Split by paragraphs first
    paragraphs = [p.strip() for p in prompt.split("\n\n") if p.strip()]

    if len(paragraphs) >= n_components:
        # Group paragraphs into n_components
        chunk_size = max(1, len(paragraphs) // n_components)
        components = []
        for i in range(n_components):
            start = i * chunk_size
            end = start + chunk_size if i < n_components - 1 else len(paragraphs)
            text = "\n\n".join(paragraphs[start:end])
            components.append({
                "description": text,
                "text": text,
            })
        return components

    # Fall back to sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', prompt)
    if len(sentences) >= n_components:
        chunk_size = max(1, len(sentences) // n_components)
        components = []
        for i in range(n_components):
            start = i * chunk_size
            end = start + chunk_size if i < n_components - 1 else len(sentences)
            text = " ".join(sentences[start:end])
            components.append({
                "description": text,
                "text": text,
            })
        return components

    # Too short — split roughly
    n = len(prompt)
    chunk = n // n_components
    components = []
    for i in range(n_components):
        start = i * chunk
        end = start + chunk if i < n_components - 1 else n
        text = prompt[start:end]
        components.append({
            "description": text,
            "text": text,
        })
    return components


# ---------------------------------------------------------------------------
# Tier2Collector
# ---------------------------------------------------------------------------

class Tier2Collector:
    """Collects Phase 1 data for Tier 2 tasks via async model inference.

    Requires a running vLLM server for target model calls.
    """

    def __init__(
        self,
        client: ModelClient,
        corpus_dir: str,
        n_resample: int = 5,
        seed: int = 42,
        max_rows: int = 2000,
        cache_dir: Path | None = None,
        judge_client: Optional[JudgeClient] = None,
    ):
        self.client = client
        self.corpus_dir = corpus_dir
        self.n_resample = n_resample
        self.rng = random.Random(seed)
        self.max_rows = max_rows
        self._cache_dir = cache_dir
        self.judge = judge_client

    def _save_incremental(self, task_type: str, records: list) -> None:
        """Save records to cache after each batch (overwrites previous save)."""
        if not self._cache_dir or not records:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / f"{task_type}.jsonl"
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    def _load_incremental(self, task_type: str) -> list | None:
        """Load partial records from cache. Returns None if no cache."""
        if not self._cache_dir:
            return None
        path = self._cache_dir / f"{task_type}.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            return None
        from prompt_attribution.training.data.multitask.schema import MultitaskRecord
        records = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(MultitaskRecord.from_dict(json.loads(line)))
        logger.info(
            f"{CYAN}[RESUME]{RESET} Loaded {len(records)} partial records for {task_type}"
        )
        return records

    def _subsample(self, rows: list[dict]) -> list[dict]:
        """Subsample rows if exceeding max_rows."""
        if len(rows) <= self.max_rows:
            return rows
        sampled = self.rng.sample(rows, self.max_rows)
        logger.info(
            f"  Subsampled {len(rows)} -> {self.max_rows} rows"
        )
        return sampled

    async def collect_e4(self, rows: list[dict]) -> list[MultitaskRecord]:
        """E4: Correctness Probability.

        Prioritizes unique problems (baseline prompt), then fills with
        perturbation variants (lever prompt) if more records are needed.
        Each record: resample N times, check against GT, compute accuracy.
        """
        # Separate unique (baseline) vs fill (lever) rows
        seen = set()
        unique_rows = []
        fill_rows = []
        for row in rows:
            if not row.get("ground_truth_answer", "").strip():
                continue
            key = (row.get("dataset_id", ""), row.get("example_idx", 0))
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)
            else:
                fill_rows.append(row)

        # Select: unique first, then fill with lever variants
        to_process: list[tuple[dict, str]] = []  # (row, prompt_key)
        for row in unique_rows[:self.max_rows]:
            to_process.append((row, "prompt_baseline"))

        if len(to_process) < self.max_rows:
            self.rng.shuffle(fill_rows)
            for row in fill_rows[:self.max_rows - len(to_process)]:
                to_process.append((row, "prompt_lever"))

        logger.info(
            f"  {len(to_process)} rows to process "
            f"({min(len(unique_rows), self.max_rows)} baseline + "
            f"{len(to_process) - min(len(unique_rows), self.max_rows)} lever fill)"
        )

        records = []
        total = len(to_process)

        for batch_start in range(0, total, 500):
            batch = to_process[batch_start:batch_start + 500]

            # Phase 1: Launch all resamples in parallel
            resample_coros = []
            resample_meta = []  # (row, variant, prompt_key)
            for row, prompt_key in batch:
                prompt = row.get(prompt_key, "")
                if not prompt:
                    continue
                variant = "e4_baseline" if prompt_key == "prompt_baseline" else "e4_lever"
                resample_coros.append(self.client.resample(prompt, self.n_resample))
                resample_meta.append((row, variant, prompt_key))

            resample_results = await asyncio.gather(*resample_coros, return_exceptions=True)

            # Phase 2: Launch all judge calls in parallel
            judge_coros = []
            judge_map: list[tuple[int, int]] = []  # (meta_idx, resp_idx)
            valid_items: list[tuple[int, dict, str, str, list[str]]] = []

            for meta_idx, ((row, variant, p_key), result) in enumerate(zip(resample_meta, resample_results)):
                if isinstance(result, BaseException):
                    logger.warning(f"  E4 resample failed for {row.get('unique_id', '?')}: {result}")
                    continue
                responses = result
                if not responses:
                    continue
                valid_items.append((meta_idx, row, variant, p_key, responses))

                if self.judge:
                    eval_prompt = row.get(p_key, row.get("prompt_baseline", ""))
                    for resp_idx, resp in enumerate(responses):
                        judge_coros.append(self.judge.judge_correctness(eval_prompt, resp))
                        judge_map.append((meta_idx, resp_idx))

            judge_results_all = []
            if judge_coros:
                judge_results_all = await asyncio.gather(*judge_coros, return_exceptions=True)

            # Phase 3: Assemble records
            for meta_idx, row, variant, p_key, responses in valid_items:
                prompt_baseline = row.get("prompt_baseline", "")
                gt_answer_raw = row.get("ground_truth_answer", "")
                gt_answer = _normalize_ground_truth(gt_answer_raw, prompt_baseline)

                if self.judge and judge_results_all:
                    n_correct = sum(
                        1 for (mi, ri), jr in zip(judge_map, judge_results_all)
                        if mi == meta_idx and isinstance(jr, bool) and jr
                    )
                else:
                    n_correct = sum(
                        1 for resp in responses
                        if _parse_answer_simple(resp, gt_answer)
                    )
                empirical_accuracy = n_correct / len(responses)

                prompt_e4 = build_e4_prompt(
                    problem_text=prompt_baseline or row.get("question", ""),
                    ground_truth_answer=gt_answer,
                    capability_tags=row.get("capability_tags", []),
                    hide_gt=self.judge is not None,
                )

                records.append(MultitaskRecord(
                    task_type=TaskType.E4_CORRECTNESS_PROBABILITY.value,
                    template_variant=variant,
                    task_prompt=prompt_e4,
                    gt_value=empirical_accuracy,
                    gt_type="continuous",
                    unique_id=row.get("unique_id", ""),
                    corpus_dir=self.corpus_dir,
                    dataset_id=row.get("dataset_id", ""),
                    example_idx=row.get("example_idx", 0),
                    question=row.get("question", ""),
                    ground_truth_answer=gt_answer,
                    capability_tags=row.get("capability_tags", []),
                    prompt_baseline=row.get("prompt_baseline", ""),
                    prompt_lever=row.get("prompt_lever", ""),
                ))

            done = min(batch_start + 500, total)
            logger.info(
                f"  E4 progress: {done}/{total} problems, "
                f"{len(records)} records so far"
            )
            self._save_incremental("e4_correctness_probability", records)

        logger.info(
            f"{MAGENTA}[E4]{RESET} Collected {len(records)} records. "
            f"Mean accuracy: {sum(r.gt_value or 0 for r in records) / max(len(records), 1):.3f}"
        )
        return records

    async def collect_e5(self, rows: list[dict]) -> list[MultitaskRecord]:
        """E5: Confidence Calibration.

        For each row, run Phase 1 with confidence-eliciting prompt,
        parse confidence values, compute mean as GT.
        """
        rows = self._subsample(rows)
        records = []
        total = len(rows)
        skipped_parse = 0

        for batch_start in range(0, total, 500):
            batch = rows[batch_start:batch_start + 500]

            # Phase 1: Launch all resamples in parallel
            resample_coros = []
            resample_meta = []  # (row, variant, full_prompt)
            for i, row in enumerate(batch):
                global_idx = batch_start + i
                if global_idx % 2 == 0:
                    full_prompt = row.get("prompt_baseline", "")
                    variant = "e5_baseline"
                else:
                    full_prompt = row.get("prompt_lever", "")
                    variant = "e5_lever"

                if not full_prompt:
                    continue

                phase1_prompt = build_e5_phase1_prompt(full_prompt)
                resample_coros.append(self.client.resample(phase1_prompt, self.n_resample))
                resample_meta.append((row, variant, full_prompt))

            resample_results = await asyncio.gather(*resample_coros, return_exceptions=True)

            # Phase 2: Assemble records
            for (row, variant, full_prompt), result in zip(resample_meta, resample_results):
                if isinstance(result, BaseException):
                    logger.warning(f"  E5 resample failed: {result}")
                    continue
                responses = result

                confidences = []
                for resp in responses:
                    c = _parse_confidence(resp)
                    if c is not None:
                        confidences.append(c)

                if not confidences:
                    skipped_parse += 1
                    continue

                mean_confidence = sum(confidences) / len(confidences)

                prompt_e5 = build_e5_phase2_prompt(
                    full_prompt=full_prompt,
                    capability_tags=row.get("capability_tags", []),
                )

                records.append(MultitaskRecord(
                    task_type=TaskType.E5_CONFIDENCE_CALIBRATION.value,
                    template_variant=variant,
                    task_prompt=prompt_e5,
                    gt_value=mean_confidence,
                    gt_type="continuous",
                    unique_id=row.get("unique_id", ""),
                    corpus_dir=self.corpus_dir,
                    dataset_id=row.get("dataset_id", ""),
                    example_idx=row.get("example_idx", 0),
                    question=row.get("question", ""),
                    ground_truth_answer=row.get("ground_truth_answer", ""),
                    lever_text=row.get("lever_text", ""),
                    category=row.get("category", ""),
                    empirical_flip_fraction=row.get("empirical_flip_fraction"),
                    capability_tags=row.get("capability_tags", []),
                    prompt_baseline=row.get("prompt_baseline", ""),
                    prompt_lever=row.get("prompt_lever", ""),
                ))

            done = min(batch_start + 500, total)
            logger.info(
                f"  E5 progress: {done}/{total} rows, "
                f"{len(records)} records, {skipped_parse} parse failures"
            )
            self._save_incremental("e5_confidence_calibration", records)

        logger.info(
            f"{MAGENTA}[E5]{RESET} Collected {len(records)} records "
            f"({skipped_parse} skipped due to parse failure). "
            f"Mean confidence: {sum(r.gt_value or 0 for r in records) / max(len(records), 1):.3f}"
        )
        return records

    async def collect_e7(
        self, rows: list[dict], target_records: int = 1500,
    ) -> list[MultitaskRecord]:
        """E7: Component Ablation.

        For each row: decompose lever prompt into 3 components,
        ablation-test each by removing it and resampling.
        GT = letter of most influential component.

        Processes rows until target_records is reached (early stopping),
        rather than subsampling a fixed number of rows upfront.
        """
        # Shuffle rows for diversity, but process all until target reached
        self.rng.shuffle(rows)
        records = []
        total = len(rows)
        skipped = 0

        E7_BATCH = 100  # Moderate batches — too large causes gather hangs
        for batch_start in range(0, total, E7_BATCH):
            batch = rows[batch_start:batch_start + E7_BATCH]

            # Phase 1a: Decompose all rows (Claude API if available, else rule-based)
            decomposed: list[tuple[dict, str, list[dict]]] = []  # (row, lever_prompt, components)
            if self.judge:
                decompose_tasks = []
                decompose_rows = []
                for row in batch:
                    lever_prompt = row.get("prompt_lever", "")
                    if not lever_prompt:
                        skipped += 1
                        continue
                    lever_prompt_clean = _strip_respond_line(lever_prompt)
                    decompose_tasks.append(self.judge.decompose_prompt(lever_prompt_clean))
                    decompose_rows.append((row, lever_prompt, lever_prompt_clean))

                decompose_results = await asyncio.gather(*decompose_tasks, return_exceptions=True)
                for (row, lever_prompt, lever_prompt_clean), result in zip(decompose_rows, decompose_results):
                    if isinstance(result, BaseException) or result is None:
                        # Fallback to rule-based
                        components = _decompose_prompt_simple(lever_prompt_clean, 3)
                    else:
                        components = result
                    if len(components) < 3:
                        skipped += 1
                        continue
                    decomposed.append((row, lever_prompt, components))
            else:
                for row in batch:
                    lever_prompt = row.get("prompt_lever", "")
                    if not lever_prompt:
                        skipped += 1
                        continue
                    lever_prompt_clean = _strip_respond_line(lever_prompt)
                    components = _decompose_prompt_simple(lever_prompt_clean, 3)
                    if len(components) < 3:
                        skipped += 1
                        continue
                    decomposed.append((row, lever_prompt, components))

            # Phase 1b: Launch resample tasks in parallel
            # Use temperature=0 for deterministic outputs — avoids false flips
            # from stochastic sampling noise (especially math/MCQ)
            batch_items: list[tuple] = []
            for row, lever_prompt, components in decomposed:
                lever_task = self.client.resample(lever_prompt, self.n_resample, temperature=0.0)
                ablation_tasks = []
                for comp in components:
                    # Use ablated_prompt from decomposer if available, else string replace
                    ablated = comp.get("ablated_prompt") or lever_prompt.replace(comp["text"], "").strip()
                    comp["_ablated_prompt_used"] = ablated  # Store for debugging
                    ablation_tasks.append(
                        self.client.resample(ablated, self.n_resample, temperature=0.0)
                    )
                batch_items.append((row, lever_prompt, components, lever_task, ablation_tasks))

            # Phase 2: Await all tasks concurrently
            # Flatten all coroutines for this batch and gather
            all_coros = []
            coro_map: list[tuple[int, str]] = []  # (item_idx, "lever" | "abl_0" etc.)
            for item_idx, (row, lever_prompt, components, lever_task, ablation_tasks) in enumerate(batch_items):
                all_coros.append(lever_task)
                coro_map.append((item_idx, "lever"))
                for abl_idx, abl_task in enumerate(ablation_tasks):
                    all_coros.append(abl_task)
                    coro_map.append((item_idx, f"abl_{abl_idx}"))

            if not all_coros:
                done = min(batch_start + 20, total)
                logger.info(f"  E7 progress: {done}/{total} rows, {len(records)} records")
                continue

            all_results = await asyncio.gather(*all_coros, return_exceptions=True)

            # Phase 3: Reassemble results per row
            results_by_item: dict[int, dict] = defaultdict(dict)
            for (item_idx, key), result in zip(coro_map, all_results):
                if isinstance(result, BaseException):
                    logger.warning(f"  E7 resample failed: {result}")
                    results_by_item[item_idx][key] = []
                else:
                    results_by_item[item_idx][key] = result

            # Phase 3b: Collect ALL flip tasks across all rows/components, gather once
            all_flip_tasks = []
            flip_task_map: list[tuple[int, int, int]] = []  # (item_idx, comp_idx, pair_idx)
            valid_items: list[tuple[int, dict, str, list[dict], list, dict]] = []

            for item_idx, (row, lever_prompt, components, _, _) in enumerate(batch_items):
                item_results = results_by_item[item_idx]
                lever_responses = item_results.get("lever", [])
                if not lever_responses or all(r == "" for r in lever_responses):
                    skipped += 1
                    continue

                cap_tags = row.get("capability_tags", [])
                valid_items.append((item_idx, row, lever_prompt, components, lever_responses, item_results))

                for comp_idx in range(len(components)):
                    ablated_responses = item_results.get(f"abl_{comp_idx}", [])
                    for lr in lever_responses:
                        for ar in ablated_responses:
                            pair_idx = len(all_flip_tasks)
                            all_flip_tasks.append(
                                _detect_flip_hybrid(lr, ar, cap_tags, self.judge)
                            )
                            flip_task_map.append((item_idx, comp_idx, pair_idx))

            # Single gather for all flip tasks in this batch
            all_flip_results = []
            if all_flip_tasks:
                all_flip_results = await asyncio.gather(*all_flip_tasks)

            # Phase 3c: Reassemble flip rates and build records
            for item_idx, row, lever_prompt, components, lever_responses, item_results in valid_items:
                cap_tags = row.get("capability_tags", [])
                n_comps = len(components)

                ablation_flip_rates = []
                for comp_idx in range(n_comps):
                    ablated_responses = item_results.get(f"abl_{comp_idx}", [])
                    # Store responses for debugging
                    components[comp_idx]["_lever_response"] = lever_responses[0] if lever_responses else ""
                    components[comp_idx]["_ablated_response"] = ablated_responses[0] if ablated_responses else ""

                    # Find flip results for this item+component
                    comp_flips = [
                        all_flip_results[pair_idx]
                        for (ii, ci, pair_idx) in flip_task_map
                        if ii == item_idx and ci == comp_idx
                    ]
                    if comp_flips:
                        flip_rate = sum(1 for f in comp_flips if f) / len(comp_flips)
                    else:
                        flip_rate = 0.0
                    ablation_flip_rates.append(flip_rate)

                # Shuffle components into A/B/C
                indices = list(range(3))
                self.rng.shuffle(indices)
                shuffled_components = [components[i] for i in indices]
                shuffled_flip_rates = [ablation_flip_rates[i] for i in indices]

                letters = ["A", "B", "C"]
                for letter, comp, fr in zip(letters, shuffled_components, shuffled_flip_rates):
                    comp["letter"] = letter
                    comp["ablation_flip_rate"] = fr

                max_rate = max(shuffled_flip_rates)
                min_rate = min(shuffled_flip_rates)

                # Skip 3-way ties (no training signal)
                if max_rate == min_rate:
                    skipped += 1
                    continue

                tied = [i for i, r in enumerate(shuffled_flip_rates) if r == max_rate]
                gt_all = [letters[i] for i in tied]
                gt_letter = self.rng.choice(gt_all)

                prompt_e7 = build_e7_prompt(
                    lever_prompt=lever_prompt,
                    components=shuffled_components,
                    capability_tags=cap_tags,
                )

                records.append(MultitaskRecord(
                    task_type=TaskType.E7_COMPONENT_ABLATION.value,
                    template_variant="e7_ablation",
                    task_prompt=prompt_e7,
                    gt_label=gt_letter,
                    gt_labels=gt_all,
                    gt_type="mcq",
                    unique_id=row.get("unique_id", ""),
                    corpus_dir=self.corpus_dir,
                    dataset_id=row.get("dataset_id", ""),
                    example_idx=row.get("example_idx", 0),
                    question=row.get("question", ""),
                    lever_text=row.get("lever_text", ""),
                    category=row.get("category", ""),
                    empirical_flip_fraction=row.get("empirical_flip_fraction"),
                    capability_tags=cap_tags,
                    prompt_baseline=row.get("prompt_baseline", ""),
                    prompt_lever=lever_prompt,
                    e7_components=[
                        {
                            "letter": c["letter"],
                            "description": c["description"],
                            "text": c.get("text", ""),
                            "ablation_flip_rate": c["ablation_flip_rate"],
                            "ablated_prompt": c.get("_ablated_prompt_used", ""),
                            "lever_response": c.get("_lever_response", ""),
                            "ablated_response": c.get("_ablated_response", ""),
                        }
                        for c in shuffled_components
                    ],
                ))

            logger.info(
                f"  E7 progress: {min(batch_start + E7_BATCH, total)}/{total} rows, "
                f"{len(records)}/{target_records} records"
            )
            self._save_incremental("e7_component_ablation", records)

            # Early stopping: stop processing once we have enough records
            if len(records) >= target_records:
                logger.info(
                    f"  E7 reached target ({len(records)} >= {target_records}), stopping early"
                )
                break

        logger.info(
            f"{MAGENTA}[E7]{RESET} Collected {len(records)} records "
            f"(skipped {skipped}). "
            f"GT dist: { {l: sum(1 for r in records if r.gt_label == l) for l in 'ABC'} }"
        )
        return records

    async def collect_e8(self, rows: list[dict]) -> list[MultitaskRecord]:
        """E8: Propose Flip Instruction.

        Ask model to propose minimal edits, then test if they cause a flip.
        """
        rows = self._subsample(rows)
        records = []
        total = len(rows)
        skipped = 0

        for batch_start in range(0, total, 500):
            batch = rows[batch_start:batch_start + 500]

            proposal_tasks = []
            for i, row in enumerate(batch):
                global_idx = batch_start + i
                # Alternate base/pert condition
                if global_idx % 2 == 0:
                    edit_prompt = row.get("prompt_baseline", "")
                    variant = "e8_base"
                else:
                    edit_prompt = row.get("prompt_lever", "")
                    variant = "e8_pert"

                if not edit_prompt:
                    skipped += 1
                    continue

                propose_prompt = build_e8_prompt(
                    prompt_text=edit_prompt,
                    capability_tags=row.get("capability_tags", []),
                )
                proposal_tasks.append(
                    (row, variant, edit_prompt, global_idx,
                     self.client.call(propose_prompt, temperature=0.7))
                )

            for row, variant, edit_prompt, global_idx, task in proposal_tasks:
                try:
                    proposal_response = await task
                except Exception as e:
                    logger.warning(f"  E8 proposal failed: {e}")
                    skipped += 1
                    continue

                # Parse proposed edit
                edited_text = _parse_json_answer(proposal_response, "edited_text")
                if not edited_text:
                    skipped += 1
                    continue

                # Compute edit distance
                matcher = SequenceMatcher(None, edit_prompt, edited_text)
                edit_similarity = matcher.ratio()
                edit_distance = 1.0 - edit_similarity

                # Test proposed edit: resample both in parallel
                try:
                    original_responses, edited_responses = await asyncio.gather(
                        self.client.resample(edit_prompt, self.n_resample),
                        self.client.resample(edited_text, self.n_resample),
                    )
                except Exception as e:
                    logger.warning(f"  E8 test failed: {e}")
                    skipped += 1
                    continue

                # Check if answers differ (flip) — hybrid: programmatic + LLM judge
                cap_tags = row.get("capability_tags", [])
                flip_tasks = []
                for orig_r in original_responses:
                    for edit_r in edited_responses:
                        flip_tasks.append(
                            _detect_flip_hybrid(orig_r, edit_r, cap_tags, self.judge)
                        )
                if flip_tasks:
                    flip_results = await asyncio.gather(*flip_tasks)
                    n_differ = sum(1 for f in flip_results if f)
                    flip_success = (n_differ / len(flip_tasks)) >= 0.5
                else:
                    flip_success = False

                # Combined reward: flip_acc * (1 - edit_dist) — multiplicative + linear
                # Rewards minimal edits that actually flip the answer.
                flip_acc = float(flip_success)
                edit_quality = 1.0 - edit_distance
                combined_reward = flip_acc * edit_quality

                records.append(MultitaskRecord(
                    task_type=TaskType.E8_PROPOSE_FLIP.value,
                    template_variant=variant,
                    task_prompt=build_e8_prompt(
                        prompt_text=edit_prompt,
                        capability_tags=row.get("capability_tags", []),
                    ),
                    gt_label="flip" if flip_success else "no_flip",
                    gt_value=combined_reward,
                    gt_type="continuous",
                    gt_text=edited_text,
                    unique_id=row.get("unique_id", ""),
                    corpus_dir=self.corpus_dir,
                    dataset_id=row.get("dataset_id", ""),
                    example_idx=row.get("example_idx", 0),
                    question=row.get("question", ""),
                    lever_text=row.get("lever_text", ""),
                    category=row.get("category", ""),
                    capability_tags=row.get("capability_tags", []),
                    prompt_baseline=row.get("prompt_baseline", ""),
                    prompt_lever=row.get("prompt_lever", ""),
                    e8_proposed_edit=edited_text,
                    e8_flip_success=flip_success,
                    e8_edit_distance=edit_distance,
                ))

            done = min(batch_start + 500, total)
            logger.info(
                f"  E8 progress: {done}/{total} rows, "
                f"{len(records)} records, {skipped} skipped"
            )

        n_flip = sum(1 for r in records if r.e8_flip_success)
        logger.info(
            f"{MAGENTA}[E8]{RESET} Collected {len(records)} records "
            f"(skipped {skipped}). "
            f"Flip success: {n_flip}/{len(records)} ({n_flip / max(len(records), 1) * 100:.1f}%)"
        )
        return records

    @staticmethod
    def _extract_choice_logprobs(
        logprobs_data: list[dict],
        valid_letters: list[str],
    ) -> dict[str, float] | None:
        """Extract choice letter logprobs from a token sequence.

        Skips past thinking/reasoning blocks before looking for the choice
        letter. Handles multiple model formats:
        - Qwen3: <think>...</think> (single special tokens)
        - GPT-OSS: <|channel|>analysis<|message|>...<|end|> (channel tokens)
        - Other: <think>...</think> as multi-token text

        Returns dict of {letter: logprob} or None if no choice letter found.
        """
        # Scan tokens to find where thinking ends.
        # Accumulate raw text for multi-token tag detection, and also check
        # individual tokens for special token patterns.
        accumulated = ""
        start_idx = 0
        in_thinking = False

        for i, token_info in enumerate(logprobs_data):
            token = token_info.get("token", "")
            accumulated += token

            # Qwen3: <think> as single special token
            if token.strip() == "<think>":
                in_thinking = True
            if in_thinking and token.strip() == "</think>":
                start_idx = i + 1
                break

            # GPT-OSS: <|channel|> analysis <|message|> ... <|end|>
            # The analysis channel contains thinking. Skip until <|end|>
            # after an analysis channel marker.
            if token.strip() == "analysis" and in_thinking:
                pass  # still in analysis
            if token.strip() in ("<|channel|>",):
                # Check if next token is "analysis" by peeking at accumulated
                if "analysis" in accumulated[-30:]:
                    in_thinking = True
            if in_thinking and token.strip() == "<|end|>":
                start_idx = i + 1
                in_thinking = False
                # Don't break — there might be more channel segments before final
                continue

            # Multi-token <think>...</think> detection via accumulated text
            if "<think>" in accumulated and not in_thinking:
                in_thinking = True
            if in_thinking and "</think>" in accumulated:
                start_idx = i + 1
                break

        for token_info in logprobs_data[start_idx:]:
            token_text = token_info.get("token", "").strip()
            # Check if this token is a choice letter
            if token_text in valid_letters:
                # Found it — extract top_logprobs at this position
                choice_logprobs: dict[str, float] = {l: -100.0 for l in valid_letters}
                if "top_logprobs" in token_info:
                    for tok, lp in token_info["top_logprobs"].items():
                        tok_clean = tok.strip()
                        if tok_clean in choice_logprobs:
                            choice_logprobs[tok_clean] = lp
                # Also include this token's own logprob
                if token_text in choice_logprobs and token_info.get("logprob") is not None:
                    choice_logprobs[token_text] = max(
                        choice_logprobs[token_text], token_info["logprob"]
                    )
                return choice_logprobs
        return None

    async def collect_e10(
        self, rows: list[dict],
    ) -> tuple[list[MultitaskRecord], list[MultitaskRecord]]:
        """E10: Margin & Second Choice → split into E10a (margin) + E10b (second).

        Uses get_choice_logprobs for exact per-choice probabilities:
        1. Model generates (with optional thinking)
        2. Full response rendered via tokenizer
        3. Completions endpoint with echo → exact logprob per choice letter

        Returns (margin_records, second_records).
        """

        # Filter to MCQ rows
        mcq_rows = [
            r for r in rows
            if r.get("choices") and len(r.get("choices", [])) >= 2
        ]
        if not mcq_rows:
            logger.info(f"{MAGENTA}[E10]{RESET} No MCQ rows found, skipping")
            return [], []

        # Separate unique (baseline) vs fill (lever) rows
        seen: set[tuple[str, int]] = set()
        unique_rows = []
        fill_rows = []
        for row in mcq_rows:
            key = (row.get("dataset_id", ""), row.get("example_idx", 0))
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)
            else:
                fill_rows.append(row)

        # Select: unique first, then fill
        to_process: list[tuple[dict, str]] = []  # (row, prompt_key)
        for row in unique_rows[:self.max_rows]:
            to_process.append((row, "prompt_baseline"))

        if len(to_process) < self.max_rows:
            self.rng.shuffle(fill_rows)
            for row in fill_rows[:self.max_rows - len(to_process)]:
                to_process.append((row, "prompt_lever"))

        logger.info(
            f"  {len(to_process)} MCQ rows to process "
            f"({min(len(unique_rows), self.max_rows)} baseline + "
            f"{len(to_process) - min(len(unique_rows), self.max_rows)} lever fill)"
        )

        margin_records: list[MultitaskRecord] = []
        second_records: list[MultitaskRecord] = []
        total = len(to_process)
        skipped = 0

        for batch_start in range(0, total, 500):
            batch = to_process[batch_start:batch_start + 500]

            # Launch all logprob calls in parallel
            logprob_coros = []
            logprob_meta = []  # (row, prompt_key, letters)
            for row, prompt_key in batch:
                choices = row.get("choices", [])
                prompt = row.get(prompt_key, "")
                if not prompt or len(choices) < 2:
                    skipped += 1
                    continue
                letters = [chr(65 + j) for j in range(len(choices))]
                logprob_coros.append(
                    self.client.get_choice_logprobs(prompt, letters)
                )
                logprob_meta.append((row, prompt_key, letters))

            logprob_results = await asyncio.gather(*logprob_coros, return_exceptions=True)

            for (row, prompt_key, letters), result in zip(logprob_meta, logprob_results):
                if isinstance(result, BaseException):
                    logger.warning(f"  E10 logprobs failed: {result}")
                    skipped += 1
                    continue

                choice_logprobs = result
                choices = row.get("choices", [])

                # Softmax to get probabilities
                max_lp = max(choice_logprobs.values())
                if max_lp <= -99.0:
                    skipped += 1
                    continue

                exp_lps = {l: math.exp(lp - max_lp) for l, lp in choice_logprobs.items()}
                total_exp = sum(exp_lps.values())
                choice_probs = {l: e / total_exp for l, e in exp_lps.items()}

                sorted_probs = sorted(
                    choice_probs.items(), key=lambda x: x[1], reverse=True
                )
                _, top1_prob = sorted_probs[0]
                top2_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
                margin = top1_prob - top2_prob
                top2_letters = [l for l, p in sorted_probs[1:] if p == top2_prob] if len(sorted_probs) > 1 else []

                # Shared fields
                uid = row.get("unique_id", "")
                ds_id = row.get("dataset_id", "")
                ex_idx = row.get("example_idx", 0)
                question = row.get("question", "")
                cap_tags = row.get("capability_tags", [])
                p_base = row.get("prompt_baseline", "")
                p_lever = row.get("prompt_lever", "")

                # E10a: Margin
                prompt_margin, _ = build_e10_prompt(
                    question=question, choices=choices, variant_idx=0,
                )
                margin_records.append(MultitaskRecord(
                    task_type=TaskType.E10A_MARGIN.value,
                    template_variant="e10_margin",
                    task_prompt=prompt_margin,
                    gt_value=margin,
                    gt_type="continuous",
                    unique_id=uid, corpus_dir=self.corpus_dir,
                    dataset_id=ds_id, example_idx=ex_idx,
                    question=question, capability_tags=cap_tags,
                    prompt_baseline=p_base, prompt_lever=p_lever,
                    e10_choice_probs=choice_probs,
                ))

                # E10b: Second choice
                prompt_second, _ = build_e10_prompt(
                    question=question, choices=choices, variant_idx=1,
                )
                top2_label = self.rng.choice(top2_letters) if top2_letters else "?"
                second_records.append(MultitaskRecord(
                    task_type=TaskType.E10B_SECOND.value,
                    template_variant="e10_second",
                    task_prompt=prompt_second,
                    gt_label=top2_label,
                    gt_labels=top2_letters,
                    gt_type="mcq",
                    unique_id=uid, corpus_dir=self.corpus_dir,
                    dataset_id=ds_id, example_idx=ex_idx,
                    question=question, capability_tags=cap_tags,
                    prompt_baseline=p_base, prompt_lever=p_lever,
                    e10_choice_probs=choice_probs,
                ))

            done = min(batch_start + 500, total)
            logger.info(
                f"  E10 progress: {done}/{total} rows, "
                f"{len(margin_records)} margin + {len(second_records)} second"
            )
            self._save_incremental("e10a_margin", margin_records)
            self._save_incremental("e10b_second", second_records)

        logger.info(
            f"{MAGENTA}[E10]{RESET} Collected {len(margin_records)} margin + "
            f"{len(second_records)} second records (skipped {skipped}). "
            f"Mean margin: {sum(r.gt_value or 0 for r in margin_records) / max(len(margin_records), 1):.3f}"
        )
        return margin_records, second_records
