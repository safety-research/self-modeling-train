"""
Module: prompt_attribution/training/validation/attribution_evaluator.py

SamplingClientEvaluator adapter for attribution training.
Plugs into the Tinker cookbook's evaluator_builders config.

Structure:
- AttributionEvaluator: Evaluates via SamplingClient, computes MSE + C-index
- parse_flip_probability(): Shared parsing logic (extracted from TrainingEvaluator)
"""

import json
import logging
import re

import tinker
from transformers import PreTrainedTokenizer

from prompt_attribution.training.data.dataset import TrainingRecord
from prompt_attribution.training.data.prompt_builder import TrainingPromptBuilder

logger = logging.getLogger(__name__)


def parse_flip_probability(text: str) -> float:
    """Parse flip_probability from model output.

    Handles thinking models (Qwen3, DeepSeek) where reasoning may be truncated.

    Parsing priority:
    1. Strip <think>...</think> tags if complete
    2. If thinking is truncated (has <think> but no </think>), search full text
    3. Try JSON: {"flip_probability": 0.7}
    4. Try keyword match: flip_probability: 0.7
    5. Try bare float in [0, 1] range
    """
    original_text = text

    has_think_open = "<think>" in text
    has_think_close = "</think>" in text
    truncated_thinking = has_think_open and not has_think_close

    if has_think_close:
        think_end = text.rfind("</think>")
        text = text[think_end + len("</think>"):]

    text = text.strip()

    result = _try_parse_flip_value(text)
    if result is not None:
        return result

    if truncated_thinking:
        result = _try_parse_flip_value(original_text)
        if result is not None:
            return result

    return 0.5


def _try_parse_flip_value(text: str) -> float | None:
    """Try to extract a flip_probability value from text."""
    # JSON extraction
    try:
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            data = json.loads(json_match.group())
            if "flip_probability" in data:
                prob = float(data["flip_probability"])
                return max(0.0, min(1.0, prob))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Keyword match
    float_match = re.search(r'(?:flip_probability|probability)["\s:]*([0-9]*\.?[0-9]+)', text)
    if float_match:
        try:
            prob = float(float_match.group(1))
            return max(0.0, min(1.0, prob))
        except ValueError:
            pass

    # Loose keyword match
    loose_match = re.search(
        r'(?:flip_probability|probability)\b.{0,30}?(0\.\d+|1\.0|0|1)\b', text
    )
    if loose_match:
        try:
            prob = float(loose_match.group(1))
            return max(0.0, min(1.0, prob))
        except ValueError:
            pass

    # Bare float
    bare_match = re.search(r'^["\s]*([0-1](?:\.\d+)?)\s*$', text.strip())
    if bare_match:
        try:
            prob = float(bare_match.group(1))
            return max(0.0, min(1.0, prob))
        except ValueError:
            pass

    return None


def _concordance_index(y_true: list[float], y_pred: list[float]) -> float:
    """Compute concordance index (C-index) for ranking quality."""
    n = len(y_true)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
               (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return concordant / total if total > 0 else 0.5


class AttributionEvaluator:
    """Evaluates attribution flip probability prediction via Tinker SamplingClient.

    Conforms to cookbook's SamplingClientEvaluator interface so it can be
    plugged into evaluator_builders for run_evaluations_parallel().
    """

    name = "attribution"

    def __init__(
        self,
        test_records: list[TrainingRecord],
        prompt_builder: TrainingPromptBuilder,
        tokenizer: PreTrainedTokenizer,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        n_samples: int = 50,
        transcript_dir: str | None = None,
    ) -> None:
        self._records = test_records[:n_samples] if n_samples > 0 else test_records
        self._prompt_builder = prompt_builder
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._transcript_dir = transcript_dir
        self._eval_count = 0

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """Run evaluation on test records and return metrics."""
        pairs = [self._prompt_builder.build(r) for r in self._records]

        # Pre-tokenize all prompts
        tokenized_inputs = []
        for pair in pairs:
            messages = [{"role": "user", "content": pair.prompt}]
            tokens = self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            model_input = tinker.ModelInput(
                chunks=[tinker.EncodedTextChunk(tokens=list(tokens))]
            )
            tokenized_inputs.append(model_input)

        sampling_params = tinker.SamplingParams(
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        # Sample all concurrently via asyncio.gather
        import asyncio

        async def _predict_one(mi: tinker.ModelInput) -> tinker.SampleResponse:
            return await sampling_client.sample_async(
                prompt=mi, num_samples=1, sampling_params=sampling_params,
            )

        responses = await asyncio.gather(
            *[_predict_one(mi) for mi in tokenized_inputs],
            return_exceptions=True,
        )

        # Collect predictions with full context
        predictions: list[dict] = []

        for record, pair, response in zip(self._records, pairs, responses):
            output_text = ""
            predicted_prob = 0.5
            error_msg = ""

            try:
                if isinstance(response, Exception):
                    raise response
                if response.sequences:
                    output_tokens = response.sequences[0].tokens
                    output_text = self._tokenizer.decode(
                        output_tokens, skip_special_tokens=False
                    )
                    predicted_prob = parse_flip_probability(output_text)
            except Exception as e:
                logger.warning(f"Eval prediction failed for {record.unique_id}: {e}")
                error_msg = str(e)

            predictions.append({
                "unique_id": record.unique_id,
                "category": record.category or "",
                "perturbation_type": record.perturbation_type or "",
                "true_flip_rate": record.empirical_flip_fraction or 0.0,
                "predicted_probability": predicted_prob,
                "model_output": output_text,
                "prompt": pair.prompt,
                "target_completion": pair.completion,
                "error": error_msg,
            })

        y_true = [p["true_flip_rate"] for p in predictions]
        y_pred = [p["predicted_probability"] for p in predictions]

        # Compute metrics
        mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / max(1, len(y_true))
        c_index = _concordance_index(y_true, y_pred)

        # Log sample predictions to console
        self._eval_count += 1
        n_show = min(3, len(predictions))
        logger.info(
            f"\033[35m[EVAL]\033[0m attribution: MSE={mse:.4f}, "
            f"C-index={c_index:.4f} (n={len(y_true)})"
        )
        for p in predictions[:n_show]:
            logger.info(
                f"  {p['unique_id'][:12]} | true={p['true_flip_rate']:.2f} "
                f"pred={p['predicted_probability']:.2f} | "
                f"output={p['model_output'][:120]}"
            )

        # Save transcripts to JSONL
        if self._transcript_dir:
            self._save_transcripts(predictions)

        # Log wandb table with full predictions
        self._log_wandb_table(predictions)

        return {
            "attribution/mse": mse,
            "attribution/c_index": c_index,
            "attribution/n_examples": float(len(y_true)),
        }

    def _save_transcripts(self, predictions: list[dict]) -> None:
        """Save eval predictions to JSONL for debugging."""
        from pathlib import Path

        transcript_dir = Path(self._transcript_dir)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        path = transcript_dir / f"eval_{self._eval_count:04d}.jsonl"
        with open(path, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")
        logger.info(f"Saved {len(predictions)} eval transcripts to {path}")

    def _log_wandb_table(self, predictions: list[dict]) -> None:
        """Log full eval predictions as a wandb Table."""
        try:
            import wandb
            if wandb.run is None:
                return

            columns = [
                "unique_id", "category", "perturbation_type",
                "true_flip_rate", "predicted_probability", "error_abs",
                "model_output", "prompt", "target_completion",
            ]
            table = wandb.Table(columns=columns)
            for p in predictions:
                table.add_data(
                    p["unique_id"],
                    p["category"],
                    p["perturbation_type"],
                    round(p["true_flip_rate"], 3),
                    round(p["predicted_probability"], 3),
                    round(abs(p["true_flip_rate"] - p["predicted_probability"]), 3),
                    p["model_output"],
                    p["prompt"],
                    p["target_completion"],
                )
            wandb.log({f"eval_predictions/step_{self._eval_count}": table})
        except Exception as e:
            logger.warning(f"Failed to log wandb eval table: {e}")
