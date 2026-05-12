"""
Module: prompt_attribution/shared/inference/api_factory.py

Factory for safetytooling.InferenceAPI instances used by the corpus generator.
"""

from pathlib import Path
from typing import Optional

from safetytooling.apis.inference.api import InferenceAPI


def get_regular_api(
    concurrency: int = 50,
    api_key: Optional[str] = None,
    cache_dir: Path = Path(".cache"),
    no_cache: bool = False,
) -> InferenceAPI:
    """Build an InferenceAPI for small tasks (high concurrency, large cache budget)."""
    return InferenceAPI(
        cache_dir=cache_dir,
        anthropic_num_threads=concurrency,
        openai_num_threads=concurrency,
        anthropic_api_key=api_key,
        no_cache=no_cache,
        max_mem_usage_mb=20_000,
    )
