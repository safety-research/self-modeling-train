"""
Module: prompt_attribution/training/envs/multitask_env.py

ProblemEnv subclass for multi-task introspection training.
Dispatches reward computation based on gt_type (binary/continuous/mcq/text).
E8 uses Tinker sampling_client for online flip verification.

Structure:
- SamplingClientHolder: Mutable wrapper so env can access current sampling_client
- MultitaskEnv: Single-turn env with gt_type-dispatched reward
- MultitaskEnvGroupBuilder: Creates K identical envs for RL group sampling
- make_multitask_env_group_builder: Factory function
"""

import asyncio
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Optional

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import Action, Metrics, StepResult

from prompt_attribution.eval.self_modeling.parsers import extract_json
from prompt_attribution.training.data.multitask.schema import MultitaskRecord

logger = logging.getLogger(__name__)

FORMAT_BONUS = 0.5

# ANSI colors for log readability
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_RED = "\033[31m"
_DIM = "\033[2m"
_RESET = "\033[0m"


class SamplingClientHolder:
    """Mutable wrapper for Tinker sampling_client.

    The sampling_client updates at each checkpoint. This holder lets
    MultitaskEnv always access the current one without reconstruction.
    """

    def __init__(self) -> None:
        self.client: Any = None


class FlipJudgeHolder:
    """Holds an async Anthropic client for LLM-based flip detection in E8.

    For open-ended tasks where programmatic answer extraction is unreliable,
    calls Haiku to compare responses.
    """

    _FLIP_PROMPT = (
        "Compare these two model responses and determine if they give the SAME answer "
        "or DIFFERENT answers. Focus on the core answer/conclusion, not reasoning or style.\n\n"
        "Response A:\n---\n{response_a}\n---\n\n"
        "Response B:\n---\n{response_b}\n---\n\n"
        'Do these responses give the same core answer? '
        'Respond in JSON: {{"same_answer": true}} or {{"same_answer": false}}'
    )

    _OPEN_ENDED_TAGS = frozenset({
        "instruction_following", "text_generation", "dialog", "code",
        "code_generation", "creative", "summarization",
    })

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_concurrent: int = 50):
        import anthropic
        import os
        key = os.environ.get("ANTHROPIC_API_KEY") or ""
        self._client = anthropic.AsyncAnthropic(api_key=key)
        self._model = model
        self._sem = asyncio.Semaphore(max_concurrent)

    def needs_judge(self, capability_tags: list[str]) -> bool:
        tags_joined = " ".join(t.lower() for t in capability_tags)
        return any(tag in tags_joined for tag in self._OPEN_ENDED_TAGS)

    async def judge_flip(self, response_a: str, response_b: str) -> bool:
        """Returns True if responses give DIFFERENT answers."""
        import json as _json, re as _re
        prompt = self._FLIP_PROMPT.format(
            response_a=response_a[:2000], response_b=response_b[:2000],
        )
        async with self._sem:
            try:
                resp = await self._client.messages.create(
                    model=self._model, max_tokens=256, temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text if resp.content else ""
                # Parse JSON
                try:
                    parsed = _json.loads(raw)
                except _json.JSONDecodeError:
                    match = _re.search(r'\{[^}]+\}', raw)
                    parsed = _json.loads(match.group()) if match else {}
                if not isinstance(parsed, dict):
                    parsed = {}
                return not parsed.get("same_answer", True)
            except Exception:
                return False  # Conservative: no flip on failure


class MultitaskEnv(ProblemEnv):
    """Single-turn multi-task introspection environment.

    Presents a pre-built task_prompt and rewards based on gt_type:
    - binary: accuracy (answer == gt_label)
    - continuous: 1 - (pred - gt)²
    - mcq: accuracy (answer in gt_labels)
    - text: SequenceMatcher similarity

    E8 (propose_flip) uses Tinker sampling for online flip verification.
    """

    def __init__(
        self,
        record: MultitaskRecord,
        renderer: renderers.Renderer,
        sampling_client_holder: Optional[SamplingClientHolder] = None,
        flip_judge: Optional[FlipJudgeHolder] = None,
    ) -> None:
        super().__init__(renderer=renderer, format_coef=0.0)
        self._record = record
        self._sampling_holder = sampling_client_holder
        self._flip_judge = flip_judge

    def get_question(self) -> str:
        """Return the pre-built task prompt."""
        return self._record.task_prompt

    def check_format(self, sample_str: str) -> bool:
        """Check if response has parseable JSON with answer/edited_text."""
        parsed = extract_json(sample_str)
        if not isinstance(parsed, dict):
            return False
        if self._record.task_type == "e8_propose_flip":
            return "edited_text" in parsed
        return "answer" in parsed

    def check_answer(self, sample_str: str) -> bool:
        """Check if answer is correct (for logging, not reward)."""
        answer = self._extract_answer(sample_str)
        if answer is None:
            return False
        gt_type = self._record.gt_type
        if gt_type == "binary":
            return str(answer).strip() == self._record.gt_label
        elif gt_type == "mcq":
            labels = self._record.gt_labels or [self._record.gt_label]
            return str(answer).strip().upper() in [l.upper() for l in labels]
        elif gt_type == "continuous":
            try:
                pred = float(answer)
                gt = self._record.gt_value or 0.0
                return abs(pred - gt) < 0.2
            except (ValueError, TypeError):
                return False
        return False

    def get_reference_answer(self) -> str:
        """Return human-readable GT summary."""
        gt_type = self._record.gt_type
        if gt_type == "binary":
            return self._record.gt_label
        elif gt_type == "continuous":
            return f"{self._record.gt_value:.3f}" if self._record.gt_value is not None else "?"
        elif gt_type == "mcq":
            labels = self._record.gt_labels or [self._record.gt_label]
            return "/".join(labels)
        elif gt_type == "text":
            return self._record.gt_text
        return "?"

    async def step(self, action: Action) -> StepResult:
        """Parse model output and compute task-specific reward."""
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        format_ok = parse_success and self.check_format(content)

        # Compute reward
        if self._record.task_type == "e8_propose_flip":
            reward, metrics = await self._compute_e8_reward(content, format_ok)
        else:
            reward = self._compute_reward(content, format_ok)
            metrics = self._build_metrics(content, format_ok, reward)

        # Per-completion logging removed — too noisy even at debug level.
        # GT and reward are visible in Tinker's per-step metrics section.

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )

    def _extract_answer(self, content: str) -> Optional[str]:
        """Extract the answer field from JSON response."""
        parsed = extract_json(content)
        if not isinstance(parsed, dict):
            return None
        if self._record.task_type == "e8_propose_flip":
            val = parsed.get("edited_text")
            return str(val) if val is not None else None
        answer = parsed.get("answer")
        return str(answer) if answer is not None else None

    def _compute_reward(self, content: str, format_ok: bool) -> float:
        """Compute reward dispatching on gt_type.

        Returns format_bonus + base_reward (total in [0, 1.5]).
        Parse failures get 0.0.
        """
        if not format_ok:
            return 0.0

        answer = self._extract_answer(content)
        if answer is None:
            return 0.0

        gt_type = self._record.gt_type
        base = 0.0

        if gt_type == "binary":
            base = 1.0 if answer.strip() == self._record.gt_label else 0.0

        elif gt_type == "continuous":
            try:
                pred = float(answer)
                # Score variants (1-10 scale) need normalization to 0-1
                is_score_variant = "score" in (self._record.template_variant or "")
                if is_score_variant:
                    pred = pred / 10.0
                pred = max(0.0, min(1.0, pred))
                gt = max(0.0, min(1.0, self._record.gt_value or 0.0))

                # MSE reward for all continuous tasks (including score variants)
                base = 1.0 - (pred - gt) ** 2
            except (ValueError, TypeError):
                base = 0.0

        elif gt_type == "mcq":
            labels = self._record.gt_labels or [self._record.gt_label]
            labels_upper = [l.upper() for l in labels if l]
            base = 1.0 if answer.strip().upper() in labels_upper else 0.0

        elif gt_type == "text":
            base = SequenceMatcher(None, str(answer), str(self._record.gt_text)).ratio()

        return FORMAT_BONUS + max(0.0, min(1.0, base))

    async def _compute_e8_reward(
        self, content: str, format_ok: bool
    ) -> tuple[float, Metrics]:
        """Compute E8 reward: flip_acc * (1 - edit_dist).

        Uses Tinker sampling_client to test if the proposed edit flips
        the current model's answer. Multiplicative formula rewards minimal
        edits that actually flip.
        """
        metrics: Metrics = {
            "format": float(format_ok),
        }

        if not format_ok:
            metrics["reward"] = 0.0
            metrics["flip_acc"] = 0.0
            metrics["edit_dist"] = 1.0
            return 0.0, metrics

        parsed = extract_json(content)
        edited_text = parsed.get("edited_text", "") if parsed else ""
        if not edited_text:
            metrics["reward"] = 0.0
            metrics["flip_acc"] = 0.0
            metrics["edit_dist"] = 1.0
            return 0.0, metrics

        # Reconstruct full prompts for sampling.
        # The editable region had the respond line stripped (_strip_respond_line),
        # so edited_text is content-only. We need to re-append the format
        # instruction for the flip test, otherwise the model won't respond in
        # the expected format.
        edited_text = str(edited_text)
        original_prompt = str(self._record.prompt_baseline or self._record.prompt_lever or "")

        # Extract the respond line from the original prompt
        _respond_line = ""
        if original_prompt:
            _orig_lines = original_prompt.rstrip("\n").splitlines()
            if _orig_lines and _orig_lines[-1].strip().lower().startswith("respond"):
                _respond_line = "\n" + _orig_lines[-1]

        # Compare edit distance on content only (without respond line)
        _orig_content = original_prompt.rstrip("\n")
        if _respond_line:
            _orig_content = _orig_content[:-(len(_respond_line))].rstrip("\n")
        if _orig_content:
            sim = SequenceMatcher(None, edited_text, _orig_content).ratio()
            edit_dist = 1.0 - sim
        else:
            edit_dist = 0.5

        # Full prompts for sampling (content + respond line)
        edited_full_prompt = edited_text + _respond_line
        original_full_prompt = original_prompt

        # Flip accuracy component (online via Tinker sampling)
        # Compare PARSED answers (not full responses) to avoid false positives
        # from different reasoning text with the same final answer.
        flip_acc = 0.0
        original_answer = ""
        edited_answer = ""
        original_parsed = False
        edited_parsed = False
        if self._sampling_holder and self._sampling_holder.client is not None:
            try:
                sc = self._sampling_holder.client

                async def _sample_response(prompt_text: str) -> str:
                    """Sample model response to a prompt via Tinker."""
                    messages: list[dict] = [{"role": "user", "content": prompt_text}]
                    model_input = self.renderer.build_generation_prompt(messages)  # type: ignore[arg-type]
                    result = await sc.sample_async(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=tinker.SamplingParams(
                            max_tokens=512,
                            temperature=0.0,
                            stop=self.renderer.get_stop_sequences(),
                        ),
                    )
                    resp_message, _ = self.renderer.parse_response(result.sequences[0].tokens)
                    return renderers.get_text_content(resp_message)

                def _parse_answer_for_flip(response: str) -> tuple[str, bool]:
                    """Parse answer from model response for flip comparison.

                    Returns (answer, parsed_ok) where parsed_ok indicates whether
                    the answer was extracted via a structured method (JSON unwrap +
                    task-specific parser). If False, fell back to raw text.
                    """
                    import re as _re
                    tags = self._record.capability_tags or []
                    tags_joined = " ".join(t.lower() for t in tags)

                    # Unwrap JSON first — extract the answer value, then parse it
                    text = response.strip()
                    try:
                        json_parsed = extract_json(response)
                        if isinstance(json_parsed, dict) and "answer" in json_parsed:
                            text = str(json_parsed["answer"]).strip()
                    except (TypeError, ValueError):
                        pass

                    # Math: extract last number
                    if any(k in tags_joined for k in ("math", "numeric", "arithmetic")):
                        nums = _re.findall(r'-?\d+(?:\.\d+)?', text)
                        if nums:
                            return nums[-1].lower(), True

                    # MCQ: extract letter
                    if any(k in tags_joined for k in ("mcq", "multiple_choice", "knowledge",
                                                       "commonsense", "reading_comprehension",
                                                       "science", "logical")):
                        from prompt_attribution.eval.self_modeling.parsers import parse_letter
                        letter = parse_letter(text)
                        if letter:
                            return letter.lower(), True

                    # Classification/yes_no: first line
                    if any(k in tags_joined for k in ("classification", "yes_no")):
                        first_line = text.split("\n")[0].strip()
                        if first_line:
                            return first_line[:100].lower(), True

                    # Default: first non-empty line (unparsed)
                    for line in text.split("\n"):
                        line = line.strip()
                        if line:
                            return line[:200].lower(), False
                    return text[:200].lower(), False

                # Sample current model on EDITED prompt (with respond line restored)
                edited_response = await _sample_response(edited_full_prompt)
                edited_answer, edited_parsed = _parse_answer_for_flip(edited_response)

                # Use stored full response (gt_text) — parse answer from it
                stored_response = self._record.gt_text
                if stored_response:
                    original_answer, original_parsed = _parse_answer_for_flip(stored_response)
                else:
                    original_response = await _sample_response(original_full_prompt)
                    original_answer, original_parsed = _parse_answer_for_flip(original_response)

                cap_tags = self._record.capability_tags or []

                if self._flip_judge and self._flip_judge.needs_judge(cap_tags):
                    # Open-ended tasks: always use LLM judge
                    flipped = await self._flip_judge.judge_flip(
                        stored_response or original_answer, edited_response
                    )
                    flip_acc = 1.0 if flipped else 0.0
                elif original_parsed and edited_parsed:
                    # Both parsed via structured method — compare answers
                    flip_acc = 1.0 if original_answer != edited_answer else 0.0
                elif not original_parsed and not edited_parsed:
                    # Neither parseable + has judge — use judge
                    if self._flip_judge:
                        flipped = await self._flip_judge.judge_flip(
                            stored_response or original_answer, edited_response
                        )
                        flip_acc = 1.0 if flipped else 0.0
                    else:
                        flip_acc = 0.0
                else:
                    # One parsed, other didn't — format broke, not a real flip
                    flip_acc = 0.0
                logger.debug(
                    f"\033[35m[E8]\033[0m orig='{original_answer}' | edit='{edited_answer}' | flip={flip_acc:.0f} | edit_dist={edit_dist:.2f}"
                )
            except Exception as e:
                import traceback
                logger.warning(f"E8 online flip check failed: {e}\n{traceback.format_exc()}")
                flip_acc = 0.0

        # Combined: flip_acc * (1.0 - edit_dist) — multiplicative + linear
        # Rewards minimal edits that actually flip the answer.
        # Old formula (0.5*flip + 0.5*(1-dist²)) had near-zero within-group variance.
        edit_quality = 1.0 - edit_dist
        base = flip_acc * edit_quality
        reward = FORMAT_BONUS + max(0.0, min(1.0, base))

        metrics["reward"] = reward
        metrics["flip_acc"] = flip_acc
        metrics["edit_dist"] = edit_dist
        # String metrics for trajectory logging (dict_mean patched to skip these)
        metrics["edited_text"] = edited_text  # type: ignore[assignment]
        metrics["orig_answer"] = original_answer  # type: ignore[assignment]
        metrics["edit_answer"] = edited_answer  # type: ignore[assignment]
        metrics["edit_quality"] = edit_quality
        return reward, metrics

    def _build_metrics(self, content: str, format_ok: bool, reward: float) -> Metrics:
        """Build metrics dict for logging.

        All values must be numeric (float/int) — Tinker's dict_mean
        computes np.mean over all metric values across the batch.
        """
        answer = self._extract_answer(content) if format_ok else None
        metrics: Metrics = {
            "format": float(format_ok),
            "reward": reward,
        }

        # GT display — dict_mean is patched to skip non-numeric values
        gt_display = self.get_reference_answer()
        metrics["gt"] = gt_display  # type: ignore[assignment]

        if self._record.gt_type == "text":
            if answer is not None:
                sim = SequenceMatcher(None, str(answer), str(self._record.gt_text)).ratio()
                metrics["gt_similarity"] = sim
        elif self._record.gt_type == "continuous":
            gt = max(0.0, min(1.0, self._record.gt_value or 0.0))
            if answer is not None:
                try:
                    pred = float(answer)
                    is_score_variant = "score" in (self._record.template_variant or "")
                    if is_score_variant:
                        pred = pred / 10.0
                    pred = max(0.0, min(1.0, pred))
                    metrics["predicted_value"] = pred
                    metrics["true_value"] = gt
                    metrics["mse"] = (pred - gt) ** 2
                except (ValueError, TypeError):
                    pass
        elif self._record.gt_type in ("binary", "mcq"):
            labels = self._record.gt_labels or [self._record.gt_label]
            labels_upper = [l.upper() for l in labels if l]
            if answer is not None:
                metrics["correct"] = float(answer.strip().upper() in labels_upper)

        return metrics


@dataclass(frozen=True)
class MultitaskEnvGroupBuilder(ProblemGroupBuilder):
    """Creates K identical MultitaskEnv instances for one record."""

    env_thunk: Callable[[], ProblemEnv]
    num_envs: int
    dataset_name: str = "multitask"
    task_type: str = ""
    gt_type: str = ""

    def logging_tags(self) -> list[str]:
        """Return tags for per-task metric aggregation."""
        return [self.dataset_name, self.task_type]


def make_multitask_env_group_builder(
    record: MultitaskRecord,
    renderer: renderers.Renderer,
    k_completions: int,
    sampling_client_holder: Optional[SamplingClientHolder] = None,
    flip_judge: Optional[FlipJudgeHolder] = None,
) -> MultitaskEnvGroupBuilder:
    """Factory to create a MultitaskEnvGroupBuilder for a single record."""

    def env_thunk() -> MultitaskEnv:
        return MultitaskEnv(
            record=record,
            renderer=renderer,
            sampling_client_holder=sampling_client_holder,
            flip_judge=flip_judge,
        )

    return MultitaskEnvGroupBuilder(
        env_thunk=env_thunk,
        num_envs=k_completions,
        dataset_name="multitask",
        task_type=record.task_type,
        gt_type=record.gt_type,
    )
