"""
Module: prompt_attribution/training/hooks/checkpoint_vllm.py

Checkpoint download and vLLM LoRA compatibility post-processing.

Structure:
- prepare_checkpoint_for_vllm(): Strip incompatible LoRA modules from Tinker checkpoints
- download_checkpoint(): Download Tinker checkpoint archive to local disk
- load_lora_into_vllm(): Dynamically load a LoRA adapter into a running vLLM server
- unload_lora_from_vllm(): Unload a LoRA adapter
- warmup_vllm_lora(): Verify a LoRA adapter is serving
"""

import json
import logging
import tarfile
import time
from io import BytesIO
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

VLLM_COMPATIBLE_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def prepare_checkpoint_for_vllm(checkpoint_dir: Path) -> None:
    """Strip vLLM-incompatible layers from a Tinker LoRA checkpoint.

    vLLM only supports a subset of LoRA target modules. Tinker's "all-linear"
    includes modules like lm_head, embed_tokens which cause vLLM load failures.
    This function filters adapter_config.json and safetensors to keep only
    compatible modules.
    """
    config_path = checkpoint_dir / "adapter_config.json"
    if not config_path.exists():
        return

    with open(config_path) as f:
        config = json.load(f)

    target_modules = config.get("target_modules", [])
    needs_fix = False

    if target_modules == "all-linear" or target_modules == ["all-linear"]:
        needs_fix = True
    elif isinstance(target_modules, list):
        incompatible = [m for m in target_modules if m not in VLLM_COMPATIBLE_MODULES]
        if incompatible:
            logger.info(f"Stripping incompatible LoRA modules for vLLM: {incompatible}")
            needs_fix = True
        else:
            return
    else:
        return

    config["target_modules"] = VLLM_COMPATIBLE_MODULES
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    safetensors_path = checkpoint_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file, save_file

            tensors = load_file(str(safetensors_path))
            filtered = {
                k: v for k, v in tensors.items()
                if any(mod in k for mod in VLLM_COMPATIBLE_MODULES)
            }
            removed = len(tensors) - len(filtered)
            if removed > 0:
                logger.info(f"Removed {removed} incompatible tensors from safetensors")
                save_file(filtered, str(safetensors_path))
        except Exception as e:
            logger.warning(f"Failed to filter safetensors: {e}")


def download_checkpoint(tinker_path: str, local_dir: Path) -> Path:
    """Download a Tinker checkpoint archive to local disk (sync version).

    Use download_checkpoint_async() when calling from an async context.

    Args:
        tinker_path: Tinker sampler path returned by save_weights_for_sampler().
        local_dir: Local directory to extract into.

    Returns:
        Path to the local checkpoint directory.
    """
    import tinker

    rest_client = tinker.ServiceClient().create_rest_client()
    archive_resp = rest_client.get_checkpoint_archive_url_from_tinker_path(
        tinker_path
    ).result()

    logger.info(f"Downloading checkpoint to {local_dir}...")
    # Retry — Tinker may still be creating the archive when the URL is returned
    import time as _time
    for attempt in range(5):
        resp = httpx.get(archive_resp.url, follow_redirects=True, timeout=300)
        if resp.status_code == 200:
            break
        logger.warning(
            f"Download attempt {attempt + 1}/5 failed (HTTP {resp.status_code}), "
            f"retrying in 30s..."
        )
        _time.sleep(30)
        # Re-fetch URL in case it expired
        archive_resp = rest_client.get_checkpoint_archive_url_from_tinker_path(
            tinker_path
        ).result()
    resp.raise_for_status()

    local_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(resp.content)) as tar:
        tar.extractall(path=local_dir)

    logger.info(f"Checkpoint downloaded to {local_dir}")
    return local_dir


async def download_checkpoint_async(
    tinker_path: str, local_dir: Path, service_client=None
) -> Path:
    """Download a Tinker checkpoint archive to local disk (async version).

    Args:
        tinker_path: Tinker sampler path returned by save_weights_for_sampler().
        local_dir: Local directory to extract into.
        service_client: Existing tinker.ServiceClient to reuse (avoids creating
            a new session, which can fail with connection refused).

    Returns:
        Path to the local checkpoint directory.
    """
    import tinker

    if service_client is None:
        service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    archive_resp = await rest_client.get_checkpoint_archive_url_from_tinker_path_async(
        tinker_path
    )

    logger.info(f"Downloading checkpoint to {local_dir}...")
    async with httpx.AsyncClient() as client:
        resp = await client.get(archive_resp.url, follow_redirects=True, timeout=300)
    resp.raise_for_status()

    local_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(resp.content)) as tar:
        tar.extractall(path=local_dir)

    logger.info(f"Checkpoint downloaded to {local_dir}")
    return local_dir


def load_lora_into_vllm(lora_name: str, lora_path: str, vllm_url: str) -> None:
    """Dynamically load a LoRA adapter into a running vLLM server."""
    base_url = vllm_url.removesuffix("/v1").rstrip("/")
    resp = httpx.post(
        f"{base_url}/v1/load_lora_adapter",
        json={"lora_name": lora_name, "lora_path": lora_path},
        timeout=60,
    )
    resp.raise_for_status()
    logger.info(f"Loaded LoRA adapter '{lora_name}' from {lora_path}")


def unload_lora_from_vllm(lora_name: str, vllm_url: str) -> None:
    """Unload a LoRA adapter from vLLM."""
    base_url = vllm_url.removesuffix("/v1").rstrip("/")
    resp = httpx.post(
        f"{base_url}/v1/unload_lora_adapter",
        json={"lora_name": lora_name},
        timeout=30,
    )
    resp.raise_for_status()


def warmup_vllm_lora(
    lora_name: str, vllm_url: str, max_retries: int = 5, delay: float = 2.0
) -> None:
    """Send a lightweight request to verify the LoRA adapter is serving.

    Prevents the 404 race condition where concurrent refresh requests
    hit vLLM before the adapter swap completes.
    """
    import openai

    client = openai.OpenAI(base_url=vllm_url, api_key="unused")

    for attempt in range(max_retries):
        try:
            client.chat.completions.create(
                model=lora_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            logger.info(f"LoRA '{lora_name}' warm-up OK (attempt {attempt + 1})")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                logger.debug(
                    f"LoRA warm-up attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.warning(
                    f"LoRA '{lora_name}' warm-up failed after {max_retries} attempts: {e}. "
                    "Proceeding anyway — individual requests have retry logic."
                )
