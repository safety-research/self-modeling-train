"""
Module: prompt_attribution/shared/inference/target_model_client.py

Unified async client for target model inference. Supports both:
- vLLM: Local OpenAI-compatible server (auto-launched or user-managed)
- API: safetytooling InferenceAPI (Anthropic, OpenAI, Together, etc.)

Manages vLLM subprocess lifecycle when auto_launch_vllm=True:
- Finds a free port automatically
- Launches `vllm serve` as a background subprocess
- Polls /health until ready
- Kills subprocess on shutdown (via finally block, atexit, signals)

Structure:
- TargetModelConfig: Configuration for the target model
- TargetModelClient: Async client with .call(prompt) and lifecycle management
"""

import asyncio
import atexit
import hashlib
import logging
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

logger = logging.getLogger(__name__)


@dataclass
class TargetModelConfig:
    """Configuration for the target model used in verification.

    Attributes:
        model_id: Model name for vLLM or API model identifier.
        vllm_url: Optional vLLM server URL (e.g., "http://localhost:8234/v1").
            If set, connects to this server. If empty and auto_launch_vllm=True,
            launches a new vLLM subprocess on a free port.
        auto_launch_vllm: Whether to auto-launch a vLLM subprocess when
            vllm_url is empty. Set False to use safetytooling API instead.
        temperature: Sampling temperature for generation.
        max_tokens: Max tokens for generation.
        max_model_len: Max model context length for vLLM (passed to --max-model-len).
    """

    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    vllm_url: str = ""
    auto_launch_vllm: bool = True
    temperature: float = 0.0  # Deterministic for stable, reproducible flip labels
    max_tokens: int = 2048
    max_model_len: int = 8192  # prompt + completion must fit

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TargetModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _find_free_port() -> int:
    """Find a free port on localhost using OS assignment."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TargetModelClient:
    """Unified async client for target model inference.

    Manages its own vLLM subprocess lifecycle when auto_launch_vllm=True.
    Use as an async context manager for automatic cleanup:

        async with TargetModelClient(config) as client:
            response = await client.call("What is 2+2?")
    """

    def __init__(
        self,
        config: TargetModelConfig,
        api: Optional[InferenceAPI] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.config = config
        self._api = api
        # Auto-detect thinking mode
        from prompt_attribution.training.config import ModelFormat
        self._model_format = ModelFormat.from_model_name(
            config.model_id, max_tokens=config.max_tokens,
        )
        self._thinking_extra = self._model_format.get_thinking_extra_body() or None
        self._is_gptoss = "gpt-oss" in config.model_id.lower()
        # Dedicated no-cache API for stability runs (bypasses safetytooling file cache)
        self._nocache_api: Optional[InferenceAPI] = None
        if api is not None:
            self._nocache_api = InferenceAPI(
                anthropic_num_threads=getattr(api, '_anthropic_num_threads', 10),
                no_cache=True,
            )
        self._vllm_client = None
        self._vllm_process: Optional[subprocess.Popen] = None
        self._vllm_url = config.vllm_url
        self._cache_dir = cache_dir
        self._cache: dict[str, str] = {}
        self._atexit_registered = False

    async def start(self) -> None:
        """Start the target model client.

        If vllm_url is empty and auto_launch_vllm is True, launches a
        vLLM subprocess and waits for it to be ready.
        """
        if self._vllm_url:
            logger.info(f"Using existing vLLM server at {self._vllm_url}")
            return

        if self.config.auto_launch_vllm:
            await self._launch_vllm()
        else:
            if self._api is None:
                raise ValueError(
                    "No vllm_url, auto_launch_vllm=False, and no API provided. "
                    "Either set vllm_url, enable auto_launch_vllm, or provide "
                    "an InferenceAPI instance."
                )
            logger.info(
                f"Using safetytooling API for target model {self.config.model_id}"
            )

    async def call(self, prompt_text: str, no_cache: bool = False, cache_suffix: str = "") -> str:
        """Send a prompt to the target model and return the response text.

        Args:
            prompt_text: The prompt to send.
            no_cache: If True, bypass cache lookup and don't store the result.
            cache_suffix: Appended to cache key for per-run caching (e.g., "_run0").
                Allows caching stability runs independently while still being
                reusable across restarts.
        """
        if not no_cache:
            cache_key = self._cache_key(prompt_text) + cache_suffix
            if cache_key in self._cache:
                return self._cache[cache_key]

        if self._vllm_url:
            result = await self._call_vllm(prompt_text)
        else:
            result = await self._call_api(prompt_text, no_cache=no_cache)

        if not no_cache:
            cache_key = self._cache_key(prompt_text) + cache_suffix
            self._cache[cache_key] = result
        return result

    async def shutdown(self) -> None:
        """Shut down the vLLM subprocess if we launched it."""
        if self._vllm_process is not None:
            logger.info(
                f"Shutting down vLLM subprocess (PID {self._vllm_process.pid})"
            )
            try:
                self._vllm_process.terminate()
                try:
                    self._vllm_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("vLLM didn't terminate gracefully, killing")
                    self._vllm_process.kill()
                    self._vllm_process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error shutting down vLLM: {e}")
            finally:
                self._vllm_process = None
                logger.info("vLLM subprocess shut down")

    async def __aenter__(self) -> "TargetModelClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()

    # =========================================================================
    # vLLM subprocess management
    # =========================================================================

    async def _launch_vllm(self) -> None:
        """Launch a vLLM server subprocess on a free port."""
        port = _find_free_port()
        self._vllm_url = f"http://localhost:{port}/v1"

        cmd = [
            "vllm", "serve", self.config.model_id,
            "--port", str(port),
            "--dtype", "bfloat16",
            "--max-model-len", str(self.config.max_model_len),
            "--disable-log-requests",
        ]

        logger.info(
            f"Launching vLLM server: {' '.join(cmd)}"
        )

        # Launch as subprocess, redirect output to devnull
        self._vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,  # new process group for clean kill
        )

        # Register cleanup handlers
        self._register_cleanup()

        # Wait for server to be ready
        await self._wait_for_ready(port, timeout=300)

        logger.info(
            f"vLLM server ready at {self._vllm_url} "
            f"(PID {self._vllm_process.pid})"
        )

    async def _wait_for_ready(
        self, port: int, timeout: int = 300
    ) -> None:
        """Poll the vLLM health endpoint until ready."""
        import httpx

        health_url = f"http://localhost:{port}/health"
        start = time.time()

        while time.time() - start < timeout:
            # Check if process died
            if self._vllm_process and self._vllm_process.poll() is not None:
                stderr = self._vllm_process.stderr.read().decode()[-500:]
                raise RuntimeError(
                    f"vLLM process exited with code "
                    f"{self._vllm_process.returncode}. "
                    f"Last stderr: {stderr}"
                )

            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(health_url, timeout=5)
                    if resp.status_code == 200:
                        return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass

            await asyncio.sleep(2)

        raise TimeoutError(
            f"vLLM server not ready after {timeout}s on port {port}"
        )

    def _register_cleanup(self) -> None:
        """Register atexit and signal handlers for cleanup."""
        if self._atexit_registered:
            return

        def _cleanup():
            if self._vllm_process is not None:
                try:
                    os.killpg(os.getpgid(self._vllm_process.pid), signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass

        atexit.register(_cleanup)
        self._atexit_registered = True

    # =========================================================================
    # Inference methods
    # =========================================================================

    async def _call_vllm(self, prompt_text: str) -> str:
        """Call via vLLM OpenAI-compatible API.

        Follows the established pattern from
        training/ground_truth/phase1_inference.py.
        """
        if self._vllm_client is None:
            import openai

            self._vllm_client = openai.AsyncOpenAI(
                base_url=self._vllm_url,
                api_key="unused",
                timeout=120.0,
            )

        for attempt in range(3):
            try:
                messages = []
                # GPT-OSS: system prompt enables thinking via analysis channel
                if self._is_gptoss:
                    from prompt_attribution.training.data.multitask.tier2_collector import GPT_OSS_SYSTEM_PROMPT
                    messages.append({"role": "system", "content": GPT_OSS_SYSTEM_PROMPT})
                messages.append({"role": "user", "content": prompt_text})
                kwargs: dict = dict(
                    model=self.config.model_id,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                # Qwen3/DeepSeek: extra_body for thinking budget
                if self._thinking_extra:
                    kwargs["extra_body"] = self._thinking_extra
                response = await self._vllm_client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content:
                    return content
                logger.warning(
                    f"vLLM returned empty content (attempt {attempt + 1}/3)"
                )
            except Exception as e:
                logger.warning(
                    f"vLLM call failed (attempt {attempt + 1}/3): {e}"
                )
                if attempt < 2:
                    await asyncio.sleep(1)
        return ""

    async def _call_api(self, prompt_text: str, no_cache: bool = False) -> str:
        """Call via safetytooling InferenceAPI.

        Follows the established pattern from
        empirical_verification/verifier.py _call_model.

        Args:
            prompt_text: The prompt to send.
            no_cache: If True, use the no-cache API instance to bypass
                safetytooling file cache.
        """
        api = self._nocache_api if (no_cache and self._nocache_api) else self._api
        if api is None:
            raise ValueError("No API instance available for API-based inference")

        try:
            responses = await api(
                model_id=self.config.model_id,
                prompt=Prompt(
                    messages=[
                        ChatMessage(role=MessageRole.user, content=prompt_text),
                    ]
                ),
                n=1,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return responses[0].completion if responses else ""
        except Exception as e:
            logger.warning(f"API call failed: {e}")
            return ""

    # =========================================================================
    # Caching
    # =========================================================================

    def _cache_key(self, prompt_text: str) -> str:
        """Generate a cache key from model config + prompt."""
        key_data = (
            f"{self.config.model_id}::"
            f"{self.config.temperature}::"
            f"{prompt_text}"
        )
        return hashlib.sha256(key_data.encode()).hexdigest()
