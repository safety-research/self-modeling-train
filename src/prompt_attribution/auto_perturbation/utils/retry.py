"""
Module: prompt_attribution/auto_perturbation/utils/retry.py

Shared retry utility for LLM API calls across all pipeline stages.
Handles transient errors (cache race conditions, 529 overloaded, timeouts).

Root cause of cache race: safetytooling's FileBasedCacheManager evicts
entries from in_memory_cache while concurrent coroutines hold stale
references, causing KeyError with PosixPath as the exception value.
We detect this and clear the stale cache state before retrying.
"""

import asyncio
import logging
from typing import Callable, Optional, TypeVar

from safetytooling.apis import InferenceAPI

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _is_cache_race_error(e: Exception) -> bool:
    """Detect safetytooling cache eviction race condition.

    The KeyError has a PosixPath as its value, which shows up as
    'PosixPath(...)' in the exception string.
    """
    return isinstance(e, KeyError) or "PosixPath" in str(e)


async def retry_async(
    fn: Callable,
    *,
    max_retries: int = 3,
    stage_name: str = "API call",
    item_id: str = "",
    api: Optional[InferenceAPI] = None,
) -> T:
    """Retry an async callable with exponential backoff.

    On cache race conditions (KeyError/PosixPath), clears the safetytooling
    in-memory cache before retrying to avoid hitting the same stale entry.

    Args:
        fn: Async callable to retry.
        max_retries: Max attempts before raising.
        stage_name: Name for log messages (e.g., "decomposer", "generator").
        item_id: ID of the item being processed (for logs).
        api: Optional InferenceAPI instance — if provided and a cache race
            is detected, clears its in-memory cache before retrying.

    Returns:
        Result of fn().

    Raises:
        The last exception if all retries are exhausted.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return await fn()
        except Exception as e:
            last_error = e
            is_cache_race = _is_cache_race_error(e)

            if attempt < max_retries - 1:
                delay = 2 ** attempt
                if is_cache_race:
                    delay = max(delay, 3)
                    # Clear in-memory cache to break the eviction race cycle
                    if api and hasattr(api, 'cache_manager') and api.cache_manager:
                        cm = api.cache_manager
                        if hasattr(cm, 'in_memory_cache'):
                            cm.in_memory_cache.clear()
                            if hasattr(cm, 'sizes'):
                                cm.sizes.clear()
                            if hasattr(cm, 'total_usage_mb'):
                                cm.total_usage_mb = 0
                            logger.warning(
                                f"[{stage_name}] Cache race detected for {item_id}, "
                                f"cleared in-memory cache. Retrying in {delay}s..."
                            )
                        else:
                            logger.warning(
                                f"[{stage_name}] Cache race for {item_id}: {e}. "
                                f"Retrying in {delay}s..."
                            )
                    else:
                        logger.warning(
                            f"[{stage_name}] Attempt {attempt+1}/{max_retries} failed "
                            f"for {item_id}: {e}. Retrying in {delay}s..."
                        )
                else:
                    logger.warning(
                        f"[{stage_name}] Attempt {attempt+1}/{max_retries} failed "
                        f"for {item_id}: {e}. Retrying in {delay}s..."
                    )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"[{stage_name}] FAILED after {max_retries} attempts "
                    f"for {item_id}: {e}"
                )
    raise last_error
