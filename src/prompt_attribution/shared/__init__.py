"""
Shared infrastructure for the training pipeline.

Structure:
- config/:    PerturbationConfig, ModelConfig, get_provider, ModelFormat,
              perturbation YAMLs + registry.yaml
- inference/: safetytooling InferenceAPI factory, key rotation, TargetModelClient
"""
