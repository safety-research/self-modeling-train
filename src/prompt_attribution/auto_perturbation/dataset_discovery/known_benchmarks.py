"""
Module: prompt_attribution/auto_perturbation/discovery/known_benchmarks.py

Hand-verified benchmark name → HuggingFace ID mapping, plus the
BenchmarkMention dataclass shared across discovery modules.

This is the "known" mapping that the research agent checks first before
falling back to HF Hub search or LLM-guided trial-and-error.

Complexity axes on each entry:
- context_source: single_source | multi_source | multimodal_context
- context_length: short | long | multi_document
- interaction_mode: static | tool_use | multi_turn
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional


# =============================================================================
# Complexity axis enums
# =============================================================================


class ContextSource(str, Enum):
    """How many information sources the task requires."""

    SINGLE_SOURCE = "single_source"  # One passage/question/problem
    MULTI_SOURCE = "multi_source"  # Multiple passages to integrate
    MULTIMODAL_CONTEXT = "multimodal_context"  # Code repos, tools, environments


class ContextLength(str, Enum):
    """Typical input length of the task."""

    SHORT = "short"  # < 500 tokens
    LONG = "long"  # 500-5000 tokens
    MULTI_DOCUMENT = "multi_document"  # > 5000 tokens or multiple docs


class InteractionMode(str, Enum):
    """What interaction pattern the task requires."""

    STATIC = "static"  # Single-turn Q&A
    TOOL_USE = "tool_use"  # Requires tool/API calls
    MULTI_TURN = "multi_turn"  # Conversational / multi-turn


# Defaults for untagged entries
DEFAULT_CONTEXT_SOURCE = ContextSource.SINGLE_SOURCE.value
DEFAULT_CONTEXT_LENGTH = ContextLength.SHORT.value
DEFAULT_INTERACTION_MODE = InteractionMode.STATIC.value


@dataclass
class BenchmarkMention:
    """A benchmark extracted from a tech report or known map.

    Attributes:
        name: Benchmark name as mentioned in the report (e.g., "GPQA Diamond")
        hf_id: Mapped HuggingFace dataset ID (e.g., "Idavidrein/gpqa:gpqa_diamond")
        category: Capability category (math, code, knowledge, reasoning, etc.)
        difficulty_tier: "frontier" (still challenging), "moderate", "saturated" (>95%)
        source_report: Which tech report this was found in
        context: Snippet from the report mentioning this benchmark
    """

    name: str
    hf_id: Optional[str] = None
    category: str = ""
    difficulty_tier: str = "frontier"  # frontier, moderate, saturated
    source_model: str = ""  # Which model's report this was found in (e.g., "GPT-5")
    source_report: str = ""  # URL of the tech report
    context: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Known benchmark name → HuggingFace ID mapping
#
# Hand-verified. The research agent checks this first before HF search
# or LLM-guided trial-and-error. Update this when new benchmarks emerge.
# =============================================================================

KNOWN_BENCHMARK_MAP: dict[str, dict] = {
    # =========================================================================
    # Math — single_source/short/static (self-contained problems)
    # =========================================================================
    "MATH-500": {"hf_id": "HuggingFaceH4/MATH-500", "category": "math", "tier": "frontier",
                 "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "MATH": {"hf_id": "EleutherAI/hendrycks_math:algebra", "category": "math", "tier": "moderate",
             "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "AIME 2024": {"hf_id": "AI-MO/aimo-validation-aime", "category": "math", "tier": "frontier",
                  "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "AIME": {"hf_id": "AI-MO/aimo-validation-aime", "category": "math", "tier": "frontier",
             "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "AMC": {"hf_id": "AI-MO/aimo-validation-amc", "category": "math", "tier": "frontier",
            "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "GSM8K": {"hf_id": "openai/gsm8k", "category": "math", "tier": "saturated",
              "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "MathVista": {"hf_id": "AI4Math/MathVista:testmini", "category": "math", "tier": "frontier",
                  "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "Minerva Math": {"hf_id": "EleutherAI/hendrycks_math:algebra", "category": "math", "tier": "moderate",
                     "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "MGSM": {"hf_id": "juletxara/mgsm", "category": "math", "tier": "moderate",
             "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},

    # =========================================================================
    # Code — mixed context sources
    # =========================================================================
    # SWE-bench: multimodal_context (issue + repo + test suite)
    "SWE-bench Verified": {"hf_id": "princeton-nlp/SWE-bench_Verified", "category": "code", "tier": "frontier",
                           "context_source": "multimodal_context", "context_length": "long", "interaction_mode": "tool_use"},
    "SWE-bench": {"hf_id": "princeton-nlp/SWE-bench", "category": "code", "tier": "frontier",
                  "context_source": "multimodal_context", "context_length": "long", "interaction_mode": "tool_use"},
    # LiveCodeBench: single function problems
    "LiveCodeBench": {"hf_id": "livecodebench/code_generation_lite", "category": "code", "tier": "frontier",
                      "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "HumanEval": {"hf_id": "openai/openai_humaneval", "category": "code", "tier": "saturated",
                  "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "HumanEval+": {"hf_id": "evalplus/humanevalplus", "category": "code", "tier": "moderate",
                   "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "MBPP": {"hf_id": "google-research-datasets/mbpp", "category": "code", "tier": "saturated",
             "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "MBPP+": {"hf_id": "evalplus/mbppplus", "category": "code", "tier": "moderate",
              "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},

    # =========================================================================
    # Knowledge / QA — single_source/short/static
    # =========================================================================
    "MMLU-Pro": {"hf_id": "TIGER-Lab/MMLU-Pro", "category": "knowledge", "tier": "frontier",
                 "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "MMLU": {"hf_id": "cais/mmlu:all", "category": "knowledge", "tier": "saturated",
             "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "GPQA Diamond": {"hf_id": "Idavidrein/gpqa:gpqa_diamond", "category": "knowledge", "tier": "frontier",
                     "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "GPQA": {"hf_id": "Idavidrein/gpqa:gpqa_diamond", "category": "knowledge", "tier": "frontier",
             "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "SimpleQA": {"hf_id": "OpenEvals/SimpleQA", "category": "knowledge", "tier": "frontier",
                 "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "TruthfulQA": {"hf_id": "truthfulqa/truthful_qa:multiple_choice", "category": "knowledge", "tier": "moderate",
                   "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "ARC-Challenge": {"hf_id": "allenai/ai2_arc:ARC-Challenge", "category": "knowledge", "tier": "moderate",
                      "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},

    # =========================================================================
    # Reasoning — mostly single_source/short/static
    # =========================================================================
    "BIG-Bench Hard": {"hf_id": "maveriq/bigbenchhard:boolean_expressions", "category": "reasoning", "tier": "frontier",
                       "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "BBH": {"hf_id": "lukaemon/bbh:boolean_expressions", "category": "reasoning", "tier": "frontier",
            "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "ZebraLogic": {"hf_id": "allenai/ZebraLogicBench-private:main", "category": "reasoning", "tier": "frontier",
                   "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "IFEval": {"hf_id": "google/IFEval", "category": "reasoning", "tier": "frontier",
               "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "HellaSwag": {"hf_id": "Rowan/hellaswag", "category": "reasoning", "tier": "saturated",
                  "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "WinoGrande": {"hf_id": "allenai/winogrande:winogrande_xl", "category": "reasoning", "tier": "saturated",
                   "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "PIQA": {"hf_id": "baber/piqa", "category": "reasoning", "tier": "saturated",
             "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "CommonsenseQA": {"hf_id": "tau/commonsense_qa", "category": "reasoning", "tier": "moderate",
                      "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    # DROP: single passage but requires discrete reasoning operations
    "DROP": {"hf_id": "ucinlp/drop", "category": "reasoning", "tier": "frontier",
             "context_source": "single_source", "context_length": "long", "interaction_mode": "static"},
    # MuSR: multi-step soft reasoning (still single passage per problem)
    "MuSR": {"hf_id": "TAUR-Lab/MuSR", "category": "reasoning", "tier": "frontier",
             "context_source": "single_source", "context_length": "long", "interaction_mode": "static"},

    # =========================================================================
    # Multi-hop reasoning — multi_source (multiple passages to integrate)
    # =========================================================================
    "HotpotQA": {"hf_id": "hotpotqa/hotpot_qa:distractor", "category": "reasoning", "tier": "frontier",
                 "context_source": "multi_source", "context_length": "long", "interaction_mode": "static",
                 "paper_url": "https://arxiv.org/abs/1809.09600"},
    "MuSiQue": {"hf_id": "bdsaglam/musique", "category": "reasoning", "tier": "frontier",
                "context_source": "multi_source", "context_length": "long", "interaction_mode": "static",
                "paper_url": "https://arxiv.org/abs/2108.00573"},
    "2WikiMultihopQA": {"hf_id": "xanhho/2WikiMultihopQA", "category": "reasoning", "tier": "frontier",
                        "context_source": "multi_source", "context_length": "long", "interaction_mode": "static",
                        "paper_url": "https://arxiv.org/abs/2011.01060"},

    # =========================================================================
    # Long-document QA — single_source/long/static
    # =========================================================================
    "QuALITY": {"hf_id": "emozilla/quality", "category": "reading_comprehension", "tier": "frontier",
                "context_source": "single_source", "context_length": "long", "interaction_mode": "static",
                "paper_url": "https://arxiv.org/abs/2112.08608"},
    "NarrativeQA": {"hf_id": "deepmind/narrativeqa", "category": "reading_comprehension", "tier": "frontier",
                    "context_source": "single_source", "context_length": "long", "interaction_mode": "static",
                    "paper_url": "https://arxiv.org/abs/1712.07040"},
    "Scrolls": {"hf_id": "tau/scrolls:qasper", "category": "reading_comprehension", "tier": "frontier",
                "context_source": "single_source", "context_length": "long", "interaction_mode": "static",
                "paper_url": "https://arxiv.org/abs/2201.03533"},

    # =========================================================================
    # Instruction following — mixed interaction modes
    # =========================================================================
    "AlpacaEval": {"hf_id": "tatsu-lab/alpaca_eval", "category": "instruction", "tier": "frontier",
                   "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    # MT-Bench: multi-turn (2-turn user prompts)
    "MT-Bench": {"hf_id": "HuggingFaceH4/mt_bench_prompts", "category": "instruction", "tier": "frontier",
                 "context_source": "single_source", "context_length": "short", "interaction_mode": "multi_turn"},
    "Arena-Hard": {"hf_id": "lmarena-ai/arena-hard-auto-v0.1", "category": "instruction", "tier": "frontier",
                   "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    # WildBench: real user queries, multi-turn
    "WildBench": {"hf_id": "allenai/WildBench:v2", "category": "instruction", "tier": "frontier",
                  "context_source": "single_source", "context_length": "long", "interaction_mode": "multi_turn",
                  "paper_url": "https://arxiv.org/abs/2406.04770"},
    # LMSYS-Chat-1M: conversation logs
    "LMSYS-Chat-1M": {"hf_id": "lmsys/lmsys-chat-1m", "category": "instruction", "tier": "moderate",
                      "context_source": "single_source", "context_length": "long", "interaction_mode": "multi_turn"},
    # ShareGPT: conversation logs
    "ShareGPT": {"hf_id": "Aeala/ShareGPT_Vicuna_unfiltered", "category": "instruction", "tier": "moderate",
                 "context_source": "single_source", "context_length": "long", "interaction_mode": "multi_turn"},

    # =========================================================================
    # Tool-use benchmarks — tool_use interaction mode
    # =========================================================================
    "BFCL": {"hf_id": "gorilla-llm/Berkeley-Function-Calling-Leaderboard", "category": "code", "tier": "frontier",
             "context_source": "single_source", "context_length": "short", "interaction_mode": "tool_use",
             "paper_url": "https://arxiv.org/abs/2402.15671"},

    # =========================================================================
    # Agent benchmarks — multimodal_context (tools + web + environments)
    # =========================================================================
    "GAIA": {"hf_id": "gaia-benchmark/GAIA", "category": "reasoning", "tier": "frontier",
             "context_source": "multimodal_context", "context_length": "long", "interaction_mode": "tool_use",
             "paper_url": "https://arxiv.org/abs/2311.12983"},

    # =========================================================================
    # Agent safety — multimodal_context
    # =========================================================================
    # PropensityBench: ICLR 2026, ScaleAI. HF ID TBD — may need custom loader.
    "PropensityBench": {"hf_id": None, "category": "safety", "tier": "frontier",
                        "context_source": "multimodal_context", "context_length": "long", "interaction_mode": "tool_use",
                        "paper_url": "https://arxiv.org/abs/2502.10834"},

    # =========================================================================
    # Safety / Ethics — single_source/short/static
    # =========================================================================
    "WildGuardMix": {"hf_id": "allenai/wildguardmix", "category": "safety", "tier": "frontier",
                     "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "ToxiGen": {"hf_id": "skg/toxigen-data", "category": "safety", "tier": "moderate",
                "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "BBQ": {"hf_id": "heegyu/bbq", "category": "safety", "tier": "moderate",
            "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "BeaverTails": {"hf_id": "PKU-Alignment/BeaverTails", "category": "safety", "tier": "moderate",
                    "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "RealToxicityPrompts": {"hf_id": "allenai/real-toxicity-prompts", "category": "safety", "tier": "moderate",
                            "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "CivilComments": {"hf_id": "google/civil_comments", "category": "safety", "tier": "moderate",
                      "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "Do-Not-Answer": {"hf_id": "LibrAI/do-not-answer", "category": "safety", "tier": "frontier",
                      "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "XSTest": {"hf_id": "natolambert/xstest-v2-copy", "category": "safety", "tier": "frontier",
               "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "HarmBench": {"hf_id": "harmbench/harmbench_behaviors_text_all", "category": "safety", "tier": "frontier",
                  "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "SafeRLHF": {"hf_id": "PKU-Alignment/PKU-SafeRLHF", "category": "safety", "tier": "moderate",
                 "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},

    # =========================================================================
    # Language Understanding — single_source/short/static
    # =========================================================================
    "MNLI": {"hf_id": "nyu-mll/multi_nli", "category": "language", "tier": "saturated",
             "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "RTE": {"hf_id": "yangwang825/rte", "category": "language", "tier": "saturated",
            "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "BoolQ": {"hf_id": "google/boolq", "category": "language", "tier": "moderate",
              "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "CoLA": {"hf_id": "nyu-mll/glue:cola", "category": "language", "tier": "saturated",
             "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "WiC": {"hf_id": "super_glue:wic", "category": "language", "tier": "moderate",
            "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "PAWS": {"hf_id": "google-research-datasets/paws:labeled_final", "category": "language", "tier": "moderate",
             "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "MultiRC": {"hf_id": "super_glue:multirc", "category": "language", "tier": "moderate",
                "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},

    # =========================================================================
    # Multilingual
    # =========================================================================
    "MMMLU": {"hf_id": "openai/MMMLU", "category": "multilingual", "tier": "frontier",
              "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},

    # =========================================================================
    # Classification (mostly saturated)
    # =========================================================================
    "SST-2": {"hf_id": "stanfordnlp/sst2", "category": "classification", "tier": "saturated",
              "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
    "IMDB": {"hf_id": "stanfordnlp/imdb", "category": "classification", "tier": "saturated",
             "context_source": "single_source", "context_length": "short", "interaction_mode": "static"},
}
