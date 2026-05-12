"""
Thin CLI wrapper around the auto-perturbation corpus generator.

Generates a perturbation corpus from a target model + a set of seed benchmarks.
Output: one JSONL per (benchmark, perturbation) pair plus a combined corpus dir.

Usage:
    uv run python scripts/data_gen/hf/generate_corpus.py \\
        --target-model meta-llama/Llama-3.1-8B-Instruct \\
        --vllm-url http://NODE:PORT/v1 \\
        --output-dir outputs/auto_perturbation/corpus_<MODEL>_<DATE>

Full flag set: see prompt_attribution/auto_perturbation/run_full_corpus.py.
"""
import sys
from prompt_attribution.auto_perturbation.run_full_corpus import main

if __name__ == "__main__":
    sys.exit(main())
