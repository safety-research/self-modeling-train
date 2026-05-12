"""
Module: prompt_attribution/shared/config/sweep_defaults.py

Provider-detection prefixes used by `get_provider`.
"""

# Model ID prefixes that map to the Together AI provider.
TOGETHER_PREFIXES = (
    "meta-llama/",
    "Qwen/",
    "deepseek-ai/",
    "mistralai/",
    "google/",
    "databricks/",
    "NousResearch/",
    "togethercomputer/",
)
