"""
Module: prompt_attribution/training/data/multitask/

Multi-task introspection training data generation.
Converts auto_perturbation corpus output into per-task training JSONL
for E1 (flip prediction), E2 (output prediction), E6 (perturbation ranking),
E9 (feature presence), and E3 (flip probability, passthrough).
"""
