"""Eval-side primitives reused by the data-gen + training pipelines.

Structure:
- attribution/: Phase 2 prompt config + template (used by training prompt_builder)
- domains/:     BaseVerifier + MathVerifier (used by the dataset_adapter)
- self_modeling/: JSON parsers (used by multitask_env)
"""
