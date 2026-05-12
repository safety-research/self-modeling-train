# Self-Modeling Training

Data generation + RL fine-tuning pipeline that produces the trained
checkpoints evaluated by the
[Self-Modeling Eval](https://github.com/safety-research/self-modeling-eval) suite (E1–E10). Two data
sources, one trainer.

## Layout

```
src/prompt_attribution/
  auto_perturbation/        corpus generator (Hugging Face benchmark track)
  training/                 Tinker RL trainer + dataclasses
                            (single-task and multitask)
  eval/{benchmarks,domains} shared loaders + verifiers
  shared/                   ModelFormat, PerturbationConfig, inference helpers
patches/safetytooling/      overlay files for installed safetytooling
scripts/
  setup/apply_safetytooling_patches.py
  data_gen/
    hf/
      generate_corpus.py            # perturbation corpus from HF benchmarks
      generate_multitask_data.py    # 11-task training data from a corpus
    bloom/
      generate_bloom_fork.py        # BLOOM behavioral fork pipeline
      convert_bloom_to_multitask.py # → 11-task training format
  training/
    run_rl_single_task.py         # single-eval RL (canonical)
    run_rl_multitask.py           # multitask RL across E1–E10
```

## Setup

```bash
# 1. Install deps
uv sync

# 2. Apply safetytooling patches (provider thinking/reasoning extraction)
uv run python scripts/setup/apply_safetytooling_patches.py

# 3. Tinker cookbook (training utilities — checkpoint helpers, renderers).
#    Clone alongside this repo and put it on PYTHONPATH for training:
git clone https://github.com/thinking-machines-lab/tinker-cookbook ../tinker-cookbook
echo 'export PYTHONPATH="$PYTHONPATH:$(realpath ../tinker-cookbook)"' >> ~/.bashrc

# 4. API keys
cp .env.example .env && $EDITOR .env   # ANTHROPIC_API_KEY, OPENAI_API_KEY, TINKER_API_KEY, …
```

## Data generation

### Hugging Face benchmark track (E1–E10 training data)

Three steps: discover suitable HF datasets, generate a perturbation
corpus from a target model on those datasets, then fold the corpus into
per-task training records.

```bash
# 0. (One-time) Build the dataset cache the corpus generator reads from.
#    Searches HuggingFace Hub + curated seeds, classifies capability
#    domains with Claude, writes outputs/auto_perturbation/.discovery_cache/datasets.jsonl.
uv run python -m prompt_attribution.auto_perturbation.dataset_discovery.profile_datasets \
    --search_hub \
    --max_datasets 250 \
    --concurrency 20 \
    --output outputs/auto_perturbation/.discovery_cache/datasets.jsonl

# 1. Launch a vLLM server for the target model).
#
#    Non-thinking model (Llama):
#       uv run python -m vllm.entrypoints.openai.api_server \
#           --model meta-llama/Llama-3.1-8B-Instruct \
#           --port 8240 --dtype bfloat16 --no-enable-log-requests
#
#    Thinking model (Qwen3): add `--reasoning-parser qwen3`.
#    GPT-OSS tool-call model:  add `--tool-call-parser openai`.

# 2. Generate the perturbation corpus (CPU-only client, calls the vLLM server).
uv run python scripts/data_gen/hf/generate_corpus.py \
    --target_model_id meta-llama/Llama-3.1-8B-Instruct \
    --target_model_url http://HOST:PORT/v1 \
    --datasets_cache outputs/auto_perturbation/.discovery_cache/datasets.jsonl \
    --output_dir outputs/auto_perturbation/corpus_<MODEL>_<DATE>

# 3. Build the 11-task multitask training data.
uv run python scripts/data_gen/hf/generate_multitask_data.py \
    --corpus-dir outputs/auto_perturbation/corpus_<MODEL>_<DATE> \
    --tasks e1 e2 e3 e4 e5 e6 e7 e8 e9 e10a e10b \
    --vllm-url http://HOST:PORT/v1 \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --enable-judge --target-per-task 1500 \
    --output-dir outputs/training/multitask_<MODEL>_<DATE>
```

Tier 1 tasks (`e1, e2, e3, e6, e9`) need only the corpus.
Tier 2 (`e4, e5, e7, e8, e10a, e10b`) calls the vLLM target model
for additional resamples / judge labels, so the server must be running.

### BLOOM track (behavioral training data)

The **ungrounded fork** pipeline. For each behavior the runner
ideates scenarios, runs a base rollout, generates adversarial forks,
executes them, judges behavior presence, and does two multi-turn
feedback rounds with the judge's reasoning. The result is then
converted into the same 11-task multitask format that RL consumes —
the trainer sees the ungrounded prompt (scenario + change only) and
learns to predict the 1–10 judge score.

```bash
# vLLM target (Llama / Qwen / GPT-OSS): pass --target-vllm-url.
uv run python scripts/data_gen/bloom/generate_bloom_fork.py \
    --target-model qwen3-8b --target-vllm-url http://HOST:PORT/v1 \
    --scenarios 15 --forks-per-scenario 5 --feedback-rounds 2 \
    --max-per-behavior 75 \
    --output-dir outputs/training/bloom_fork_qwen3_8b_<DATE>

# Convert to the 11-task RL training schema:
uv run python scripts/data_gen/bloom/convert_bloom_to_multitask.py \
    --input-dir outputs/training/bloom_fork_<MODEL>_<DATE> \
    --output-dir outputs/training/multitask_bloom_<MODEL>_<DATE>
```

The BLOOM pipeline expects the `bloom` package (behavior catalog +
prompts) — point `pyproject.toml`'s `bloom` source at your local clone
if you want to run this track.

## Training

RL via Tinker LoRA. Single-task and multitask scripts share the same
canonical config (batch=64, k=16, lr=2e-5 cosine, lora=32, n_steps=200).
See each script's `--help` for the full flag list.

### Single-task

```bash
uv run python scripts/training/run_rl_single_task.py \
    --task e3 \
    --data-dir outputs/training/multitask_<MODEL>_<DATE> \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --max-generation-tokens 2048 \
    --wandb-project self-modeling-rl \
    --run-name e3_<MODEL>_<DATE>
```

### Multitask (all 11)

```bash
uv run python scripts/training/run_rl_multitask.py \
    --tasks all \
    --data-dir outputs/training/multitask_<MODEL>_<DATE> \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --max-generation-tokens 2048 \
    --wandb-project self-modeling-rl \
    --run-name mtl_<MODEL>_<DATE>
```

Subset training: `--tasks e1,e3,e6`.

### BLOOM track

The BLOOM converter writes records in the same MultitaskRecord schema
under `task_type=e3_flip_probability`, so the multitask trainer is the
same — just point `--data-dir` at the converter output and restrict to
E3:

```bash
uv run python scripts/training/run_rl_multitask.py \
    --tasks e3 \
    --data-dir outputs/training/multitask_bloom_<MODEL>_<DATE> \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --max-generation-tokens 2048 \
    --wandb-project self-modeling-rl \
    --run-name bloom_e3_<MODEL>_<DATE>
```

Training runs on Tinker (no local GPU needed). The LoRA checkpoint
can be downloaded from Tinker and merged for vLLM serving as part of the
post-training eval.


## CLI reference (training)

| flag | purpose |
|---|---|
| `--task ID` / `--tasks ID,...` | which of E1–E10 to train on |
| `--data-dir DIR` | output of `generate_multitask_data.py` (or BLOOM converter) |
| `--base-model HF_ID` | base model to fine-tune (Tinker resolves the weights) |
| `--max-generation-tokens N` | max tokens per rollout (default 2048) |
| `--batch-size N` | rollout batch size (default 64) |
| `--k-completions N` | rollouts per prompt for RL advantage (default 16) |
| `--lr F` | learning rate (default 2e-5, cosine decay) |
| `--lora-rank N` | LoRA rank (default 32) |
| `--n-steps N` | training steps (default 200) |
| `--load-checkpoint PATH` | warm-init from a previous Tinker checkpoint |
| `--wandb-project NAME` | wandb project name |
| `--run-name NAME` | unique run identifier (auto-timestamped if omitted) |
