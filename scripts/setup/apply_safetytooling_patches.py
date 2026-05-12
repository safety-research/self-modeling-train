"""
Overlay our patched safetytooling files onto the freshly-installed package.

safetytooling is pinned in pyproject.toml at a known commit, but several of
its files need provider-specific edits before the self-modeling eval works
end-to-end:

- data_models/inference.py: adds `thinking: str = ""` to LLMResponse
- apis/inference/anthropic.py: extracts type=="thinking" blocks into thinking
- apis/inference/together.py: extracts message.reasoning / reasoning_content
- apis/inference/openai/chat.py: same for OpenAI + adds temperature=1 +
  max_completion_tokens=10 to the dummy header for reasoning models so the
  initial capability probe doesn't 400
- apis/inference/gemini/genai.py: switches gemini-2.5+ / 3+ to the async
  google-genai SDK and bypasses the legacy lock+rate tracker

Run this once after `uv sync`. Re-run after any safetytooling upgrade.

Usage:
    uv run python scripts/setup/apply_safetytooling_patches.py
"""

import shutil
import sys
from importlib.util import find_spec
from pathlib import Path

PATCH_DIR = Path(__file__).resolve().parents[2] / "patches" / "safetytooling"


def site_packages_safetytooling() -> Path:
    spec = find_spec("safetytooling")
    if spec is None or not spec.submodule_search_locations:
        sys.exit(
            "safetytooling is not installed in the active environment. "
            "Run `uv sync` first."
        )
    return Path(next(iter(spec.submodule_search_locations)))


def main() -> None:
    target_root = site_packages_safetytooling()
    print(f"Patching safetytooling at: {target_root}")

    n_copied = 0
    for src in PATCH_DIR.rglob("*.py"):
        rel = src.relative_to(PATCH_DIR)
        dst = target_root / rel
        if not dst.parent.exists():
            sys.exit(f"Expected directory missing: {dst.parent}")
        shutil.copy2(src, dst)
        print(f"  copied {rel}")
        n_copied += 1

    if n_copied == 0:
        sys.exit("No patch files found under patches/safetytooling/")
    print(f"\nDone. {n_copied} file(s) patched.")


if __name__ == "__main__":
    main()
