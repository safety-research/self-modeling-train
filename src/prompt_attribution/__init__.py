"""
Self-modeling training pipeline — data generation + GRPO fine-tuning.

Structure:
- auto_perturbation/   corpus generation (Hugging Face track)
- training/            Tinker GRPO trainers (single-task + multitask)
- eval/                shared benchmark loaders + domain verifiers
- shared/config/       perturbation YAMLs, ModelFormat, get_provider
"""

__version__ = "0.1.0"
