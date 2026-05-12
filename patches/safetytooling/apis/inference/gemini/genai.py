import asyncio
import logging
import os
import time
from pathlib import Path
from traceback import format_exc
from typing import Callable, List, Optional, Tuple

import google.generativeai as genai
from google.api_core.exceptions import InvalidArgument
from google.generativeai.types import GenerationConfig, HarmBlockThreshold, HarmCategory

from safetytooling.data_models import GeminiStopReason, LLMResponse, Prompt

from ....data_models.utils import (
    GEMINI_MODELS,
    GEMINI_RPM,
    GEMINI_TPM,
    GeminiRateTracker,
    RecitationRateFailureError,
    get_stop_reason,
    parse_safety_ratings,
)
from ..model import InferenceAPIModel

LOGGER = logging.getLogger(__name__)


class GeminiModel(InferenceAPIModel):
    def __init__(
        self,
        prompt_history_dir: Path = None,
        recitation_rate_check_volume: int = 100,  # Number of prompts we need to have processed before checking recitation rate
        recitation_rate_threshold: float = 0.5,  # Recitation rate threshold above which we end the current run and wait due to high recitation rates
        api_key_tag: str = "GOOGLE_API_KEY",
        empty_completion_threshold: float = 0,
    ):
        self.prompt_history_dir = prompt_history_dir
        self.model_ids = set()
        self.rate_trackers = {}

        self.recitation_rate_check_volume = recitation_rate_check_volume
        self.total_generate_calls = 0
        self.total_processed_prompts = 0
        self.total_recitation_retries = 0
        self.total_recitation_failures = 0
        self.recitation_retry_rate = 0
        self.recitation_failure_rate = 0
        self.recitation_rate_threshold = recitation_rate_threshold
        self.empty_completion_threshold = empty_completion_threshold
        self.lock = asyncio.Lock()

        self.kwarg_change_name = {
            "max_tokens": "max_output_tokens",
        }

        # Maps simple input of the safety threshold values (None, few, some, most) to the HarmBlockThreshold class values
        self.map_safety_block_name = {
            None: HarmBlockThreshold.BLOCK_NONE,
            "few": HarmBlockThreshold.BLOCK_ONLY_HIGH,
            "some": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "most": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
        self.is_initialized = False
        if api_key_tag in os.environ:
            genai.configure(api_key=os.environ[api_key_tag])
            self.is_initialized = True

    @staticmethod
    def _print_prompt_and_response(prompt, responses):
        raise NotImplementedError

    def add_model_id(self, model_id: str):
        if model_id not in GEMINI_MODELS and not model_id.startswith("gemini-"):
            raise ValueError(f"Unsupported model: {model_id}")
        if model_id not in self.model_ids:
            self.model_ids.add(model_id)
            # Use default rate limits for models not in GEMINI_RPM
            rpm = GEMINI_RPM.get(model_id, 360)
            self.rate_trackers[model_id] = GeminiRateTracker(rpm_limit=rpm, tpm_limit=GEMINI_TPM)

    def get_generation_config(self, kwargs):
        for k, v in self.kwarg_change_name.items():
            if k in kwargs:
                kwargs[v] = kwargs[k]
                del kwargs[k]
        return GenerationConfig(**kwargs)

    def get_safety_settings(self, safety_threshold):
        safety_settings = {
            category: self.map_safety_block_name[safety_threshold]
            for category in [
                HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                HarmCategory.HARM_CATEGORY_HARASSMENT,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]
        }
        return safety_settings

    async def run_query(
        self,
        model_id: str,
        prompt: Prompt,
        safety_settings: dict[str, HarmBlockThreshold],
        generation_config: GenerationConfig = None,
    ) -> Tuple[str, List[genai.types.file_types.File]]:
        if not self.is_initialized:
            raise RuntimeError(
                "Gemini is not initialized. Please set GOOGLE_API_KEY in .env before running your script"
            )
        content, system_message = prompt.gemini_format()
        if generation_config:
            model = genai.GenerativeModel(
                model_name=model_id,
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=system_message,
            )
        else:
            model = genai.GenerativeModel(
                model_name=model_id, safety_settings=safety_settings, system_instruction=system_message
            )

        upload_files = [i for i in content if isinstance(i, genai.types.file_types.File)]
        response = await model.generate_content_async(contents=content)
        return response, upload_files

    async def check_recitation_rates(self):
        if self.total_processed_prompts >= self.recitation_rate_check_volume:
            self.recitation_retry_rate = self.total_recitation_retries / self.total_generate_calls
            self.recitation_failure_rate = self.total_recitation_failures / self.total_processed_prompts
            # We've hit the threshold of prompts processed required to start checking if we're
            # Hitting high recitation rates

            if self.recitation_failure_rate > self.recitation_rate_threshold:
                LOGGER.error(
                    f"ERROR: CURRENT RECITATION FAILURE RATE {self.recitation_failure_rate * 100:.2f}% IS ABOVE RECITATION FAILURE THRESHOLD {self.recitation_rate_threshold}"
                )
                return True
            else:
                return False

    async def _make_genai_api_call(
        self,
        model_id: str,
        prompt: Prompt,
        thinking_budget: int = 0,
        max_output_tokens: int = 1000,
        max_attempts: int = 5,
        **kwargs,
    ) -> list[LLMResponse]:
        """Make a Gemini API call using the new google-genai SDK.

        Used for ALL Gemini calls (with or without thinking).
        Much faster than the old google-generativeai SDK.
        """
        from google import genai as genai_new
        from google.genai import types as genai_types

        api_key = os.environ.get("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
        client = genai_new.Client(api_key=api_key)

        content, system_message = prompt.gemini_format()
        prompt_parts = []
        for part in content:
            if isinstance(part, str):
                prompt_parts.append(part)
            elif hasattr(part, "text"):
                prompt_parts.append(part.text)
        prompt_text = "\n".join(prompt_parts) if prompt_parts else str(content)

        config_kwargs = {"max_output_tokens": max_output_tokens}
        if thinking_budget and thinking_budget > 0:
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=thinking_budget,
                include_thoughts=True,
            )
        if system_message:
            config_kwargs["system_instruction"] = system_message
        config = genai_types.GenerateContentConfig(**config_kwargs)

        start = time.time()
        for attempt in range(max_attempts):
            api_start = time.time()
            try:
                response = await client.aio.models.generate_content(
                    model=model_id,
                    contents=prompt_text,
                    config=config,
                )

                api_duration = time.time() - api_start
                duration = time.time() - start

                thinking_parts = []
                answer_parts = []
                if response.candidates and response.candidates[0].content:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "thought") and part.thought:
                            thinking_parts.append(part.text)
                        else:
                            answer_parts.append(part.text)

                completion = "\n".join(answer_parts)
                thinking_text = "\n\n".join(thinking_parts)

                return [LLMResponse(
                    model_id=model_id,
                    completion=completion,
                    thinking=thinking_text,
                    stop_reason="stop",
                    duration=duration,
                    api_duration=api_duration,
                    cost=0,
                )]
            except Exception as e:
                LOGGER.warning(f"Gemini genai API attempt {attempt}: {type(e).__name__}: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1.5 ** attempt)

        duration = time.time() - start
        LOGGER.error(f"Gemini genai API failed after {max_attempts} attempts for {model_id}")
        return [LLMResponse(
            model_id=model_id,
            completion="",
            stop_reason="api_error",
            duration=duration,
            api_duration=0,
            cost=0,
        )]

    async def _make_api_call(
        self,
        model_id: str,
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid: Callable[[str], bool] = lambda x: x.completion != "",
        safety_threshold: str = None,
        **kwargs,
    ) -> list[LLMResponse]:
        # Use new google-genai SDK for all gemini-2.5+ and gemini-3+ models
        # (faster, supports thinking, no rate tracker bottleneck)
        thinking_budget = kwargs.pop("thinking_budget", None) or 0
        if model_id.startswith("gemini-2.5") or model_id.startswith("gemini-3"):
            max_output_tokens = kwargs.pop("max_output_tokens", kwargs.pop("max_tokens", 1000))
            return await self._make_genai_api_call(
                model_id=model_id,
                prompt=prompt,
                thinking_budget=thinking_budget,
                max_output_tokens=max_output_tokens,
                **kwargs,
            )
        # For thinking on older models, also use new SDK
        if thinking_budget > 0:
            max_output_tokens = kwargs.pop("max_output_tokens", kwargs.pop("max_tokens", 1000))
            return await self._make_genai_api_call(
                model_id=model_id,
                prompt=prompt,
                thinking_budget=thinking_budget,
                max_output_tokens=max_output_tokens,
                **kwargs,
            )

        start = time.time()

        api_start = time.time()
        generation_config = self.get_generation_config(kwargs)
        safety_settings = self.get_safety_settings(safety_threshold)

        async_response, upload_files = await self.run_query(
            model_id=model_id, prompt=prompt, safety_settings=safety_settings, generation_config=generation_config
        )

        api_duration = time.time() - api_start

        response_data = async_response._result

        if response_data is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        # Deal with RECITATION issue
        try:
            # Sometimes when there is a failure due to RECITATION, the response output will not be in the expected format
            # I.e. candidates will not exist
            if not response_data.candidates:
                LOGGER.warning(
                    """
                               ***************************************
                               BLOCKED CONTENT: No candidate responses
                               ***************************************
                               """
                )
                block_reason = response_data.prompt_feedback.block_reason
                LOGGER.info(f"PROMPT CAUSING BLOCKING: \n {prompt}")
                LOGGER.info(
                    f"Specific block reason = {block_reason}, which will be recorded as GeminiStopReason.BLOCKED"
                )
                LOGGER.info(f"No candidates return on prompt {prompt}")
                LOGGER.info(f"BLOCKED RESPONSE DATA:\n  {response_data}")

                response = LLMResponse(
                    model_id=model_id,
                    completion="",
                    stop_reason=GeminiStopReason.BLOCKED,
                    safety_ratings={},
                    duration=duration,
                    api_duration=api_duration,
                    cost=0,
                )
            else:
                try:
                    # Normal response where we can access text object from candidates output
                    stop_reason = response_data.candidates[0].finish_reason

                    response = LLMResponse(
                        model_id=model_id,
                        completion=response_data.candidates[0].content.parts[0].text,
                        stop_reason=get_stop_reason(stop_reason),
                        safety_ratings=parse_safety_ratings(safety_ratings=response_data.candidates[0].safety_ratings),
                        duration=duration,
                        api_duration=api_duration,
                        cost=0,
                    )
                except Exception:
                    ## Handle RECITATION blocking generation
                    ## When response is blocked by recitation, candidates will not be in list format
                    LOGGER.warning(
                        """
                               **********************************
                               RECITATION: no candidate responses
                               **********************************
                               """
                    )
                    LOGGER.info("CANDIDATE RESPONSE IS NOT LIST")
                    LOGGER.info("PROMPT CAUSING NO RESPONSE: \n {prompt}")
                    LOGGER.info(f"RESPONSE: {response_data}")

                    # Convert candidates into a list so that we can access its contents
                    stop_reason = list(response_data.candidates)[0].finish_reason

                    response = LLMResponse(
                        model_id=model_id,
                        completion="",
                        stop_reason=get_stop_reason(stop_reason),
                        safety_ratings={},
                        duration=duration,
                        api_duration=api_duration,
                        cost=0,
                    )

        except Exception as e:
            ## Handle all other exceptions
            LOGGER.error(f"ERROR PARSING RESPONSE: {str(e)}")
            LOGGER.error(f"RESPONSE: {response_data}")
            response = LLMResponse(
                model_id=model_id,
                completion="",
                stop_reason=GeminiStopReason.RECITATION,
                safety_ratings={},
                duration=duration,
                api_duration=api_duration,
                cost=0,
            )

        return [response]

    async def __call__(
        self,
        model_id: str,
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid: Callable[[str], bool] = lambda x: x.completion != "",
        safety_threshold: str = None,
        **kwargs,
    ) -> List[LLMResponse]:
        start = time.time()

        # Fast path: gemini-2.5+ and gemini-3+ use new SDK directly,
        # bypassing the old rate tracker and lock (which serializes requests)
        if model_id.startswith("gemini-2.5") or model_id.startswith("gemini-3"):
            return await self._make_api_call(
                model_id, prompt, print_prompt_and_response, max_attempts, is_valid, **kwargs
            )

        self.add_model_id(model_id)

        async def attempt_api_call(model_id):
            async with self.lock:
                rate_tracker = self.rate_trackers[model_id]
                if rate_tracker.can_make_request(GEMINI_RPM.get(model_id, 360)):
                    try:
                        token_count = genai.GenerativeModel(model_name=model_id).count_tokens(prompt.gemini_format()).total_tokens
                    except Exception:
                        token_count = 1000  # Estimate for models not in old SDK
                    rate_tracker.add_request(token_count)
                else:
                    return None

            # try:
            return await self._make_api_call(
                model_id, prompt, print_prompt_and_response, max_attempts, is_valid, **kwargs
            )
            # except Exception as e:
            #     LOGGER.warning(f"Error calling {model_id}: {str(e)}")
            #     return None

        responses: Optional[List[LLMResponse]] = None

        for i in range(max_attempts):
            # Update tracking of recitation retries by generate attempt (not at the prompt level)
            self.total_generate_calls += 1
            try:
                responses = await attempt_api_call(model_id)
                if (
                    responses is not None
                    and len([response for response in responses if is_valid(response)]) / len(responses)
                    < self.empty_completion_threshold
                ):
                    raise RuntimeError(f"All invalid responses according to is_valid {responses}")
            except (TypeError, InvalidArgument) as e:
                raise e
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying in {1.5**i} seconds. (Attempt {i})")
                await asyncio.sleep(1.5**i)
            else:
                break

        # Update tracking of full recitation failure
        recitation_error = await self.check_recitation_rates()
        if recitation_error:
            raise RecitationRateFailureError("Recitation rate failure is too high!")
        if responses is None:
            self.total_recitation_failures += 1
            LOGGER.error(f"Failed to get a response for {prompt} from any API after {max_attempts} attempts.")
            LOGGER.info(f"Current recitation retry rate: {self.recitation_retry_rate* 100:.2f}")
            LOGGER.info(f"Current recitation failure rate: {self.recitation_failure_rate* 100:.2f}")
            return [
                LLMResponse(
                    model_id=model_id,
                    completion="",
                    stop_reason=GeminiStopReason.RECITATION,
                    safety_ratings={},
                    cost=0,
                    duration=0,
                    api_duration=0,
                    api_failures=1,
                )
            ]
        else:
            # Add number of attempts to response object for recitation retries
            for response in responses:
                response.recitation_retries = i
                self.total_recitation_retries += i

            if print_prompt_and_response:
                prompt.pretty_print(responses)
            LOGGER.debug(f"Completed call to Gemini in {time.time() - start}s.")

            # Update tracking for total processed prompts
            self.total_processed_prompts += 1

            # Check the current recitation retries and failure rates
            recitation_error = await self.check_recitation_rates()
            LOGGER.info(f"Current total generate calls: {self.total_generate_calls}")
            LOGGER.info(f"Current total processed prompts: {self.total_processed_prompts}")
            LOGGER.info(f"Current total recitation retries: {self.total_recitation_retries}")
            LOGGER.info(f"Current total recitation failures: {self.total_recitation_failures}")
            LOGGER.info(f"Current recitation retry rate: {self.recitation_retry_rate* 100:.2f}")
            LOGGER.info(f"Current recitation failure rate: {self.recitation_failure_rate* 100:.2f}")

            return responses
