"""
Module: prompt_attribution/training/data/multitask/tier1_converter.py

Converts auto_perturbation corpus output into per-task training data
for Tier 1 tasks (no new inference needed).

Structure:
- Tier1Converter: Main converter class
  - convert_e1: Flip prediction (binary Yes/No)
  - convert_e2: Output prediction (free text)
  - convert_e6: Perturbation ranking (2/3-way MCQ)
  - convert_e8: Propose flip (prompt only, reward computed online)
  - convert_e9: Feature presence (binary 0/1)
"""

import logging
import random
from collections import defaultdict

from prompt_attribution.training.data.multitask.schema import (
    MultitaskRecord,
    TaskType,
)
from prompt_attribution.training.data.multitask.task_prompts import (
    build_e1_prompt,
    build_e2_prompt,
    build_e3_prompt,
    build_e6_prompt,
    build_e8_prompt,
    build_e9_prompt,
)

logger = logging.getLogger(__name__)


def unique_base_filter(
    rows: list[dict],
    seed: int = 42,
    test_split: float = 0.15,
) -> list[dict]:
    """Replicate Dataset B sampling: one perturbation per unique problem.

    1. Apply canonical train/test split (seed=42, 85/15)
    2. Group train rows by (dataset_id, example_idx)
    3. Pick exactly one perturbation per problem (random, seed=42)

    Returns the filtered train rows (~1.5K from a 19K corpus).
    """
    rng = random.Random(seed)

    # Filter to rows with valid flip_fraction and answer_labels
    valid = [
        r for r in rows
        if r.get("empirical_flip_fraction") is not None
        and r.get("answer_labels")
    ]

    # Shuffle deterministically and split
    indices = list(range(len(valid)))
    rng_split = random.Random(seed)
    rng_split.shuffle(indices)
    n_test = int(len(valid) * test_split)
    train_indices = set(indices[n_test:])
    train_rows = [valid[i] for i in range(len(valid)) if i in train_indices]

    # Group by (dataset_id, example_idx)
    by_problem: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in train_rows:
        key = (r.get("dataset_id", ""), r.get("example_idx", 0))
        by_problem[key].append(r)

    # Pick one perturbation per problem
    unique_base = [rng.choice(v) for v in by_problem.values()]

    logger.info(
        f"[UNIQUE-BASE] {len(rows)} corpus rows -> {len(valid)} valid -> "
        f"{len(train_rows)} train -> {len(unique_base)} unique problems"
    )
    return unique_base


def slim_select(
    rows: list[dict],
    seed: int = 42,
    test_split: float = 0.15,
) -> dict[str, list[dict]]:
    """Select 3 perturbations per problem: 1 flip, 1 non_flip, 1 boundary.

    Returns dict with keys "flip", "non_flip", "boundary", each a list of rows.
    The boundary candidates are the ones that need re-verification with
    n_resample > 1 to get continuous flip probability GT.

    For problems missing a category, falls back to the closest available.
    """
    rng = random.Random(seed)

    # Filter and split (same as unique_base_filter)
    valid = [
        r for r in rows
        if r.get("empirical_flip_fraction") is not None
        and r.get("answer_labels")
    ]
    indices = list(range(len(valid)))
    rng_split = random.Random(seed)
    rng_split.shuffle(indices)
    n_test = int(len(valid) * test_split)
    train_indices = set(indices[n_test:])
    train_rows = [valid[i] for i in range(len(valid)) if i in train_indices]

    # Group by problem
    by_problem: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in train_rows:
        key = (r.get("dataset_id", ""), r.get("example_idx", 0))
        by_problem[key].append(r)

    flip_rows = []
    non_flip_rows = []
    boundary_rows = []
    skipped = 0

    for _key, perts in by_problem.items():
        # Partition by category
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for p in perts:
            cat = p.get("category", "")
            by_cat[cat].append(p)

        # Select best from each category
        # flip_inducing: pick the one with highest flip_fraction
        flip_candidates = by_cat.get("flip_inducing", [])
        # non_flip: pick the one with lowest flip_fraction
        non_flip_candidates = by_cat.get("non_flip", [])
        # boundary: pick the one closest to 0.5
        boundary_candidates = by_cat.get("boundary", [])

        # Need at least 2 categories for useful training data
        available_cats = sum(1 for c in [flip_candidates, non_flip_candidates, boundary_candidates] if c)
        if available_cats < 2:
            skipped += 1
            continue

        if flip_candidates:
            best_flip = max(flip_candidates, key=lambda r: r.get("empirical_flip_fraction", 0.0))
            flip_rows.append(best_flip)

        if non_flip_candidates:
            best_nf = min(non_flip_candidates, key=lambda r: r.get("empirical_flip_fraction", 1.0))
            non_flip_rows.append(best_nf)

        if boundary_candidates:
            best_bound = min(
                boundary_candidates,
                key=lambda r: abs(r.get("empirical_flip_fraction", 0.5) - 0.5),
            )
            boundary_rows.append(best_bound)
        elif flip_candidates:
            # Fallback: pick the flip candidate closest to 0.5
            fallback = min(
                flip_candidates,
                key=lambda r: abs(r.get("empirical_flip_fraction", 0.5) - 0.5),
            )
            boundary_rows.append(fallback)

    logger.info(
        f"[SLIM-SELECT] {len(by_problem)} problems -> "
        f"flip: {len(flip_rows)}, non_flip: {len(non_flip_rows)}, "
        f"boundary: {len(boundary_rows)} (skipped {skipped} with <2 categories)"
    )

    return {
        "flip": flip_rows,
        "non_flip": non_flip_rows,
        "boundary": boundary_rows,
    }


class Tier1Converter:
    """Converts corpus rows to multi-task training records.

    All conversions are deterministic given the same input + seed.
    """

    def __init__(self, corpus_dir: str, seed: int = 42, force_noshow: bool = False):
        self.corpus_dir = corpus_dir
        self.seed = seed
        # Set True when stored baseline answers are from a different model
        # (e.g. Llama corpus used for Haiku training). Skips SHOW variants
        # in E1/E2 that embed the model's prior answer.
        self.force_noshow = force_noshow

    def convert_e1(self, rows: list[dict]) -> list[MultitaskRecord]:
        """E1: Flip Prediction — binary Yes/No from empirical_flip_fraction.

        Every corpus row with a valid flip_fraction becomes one E1 example.
        Template variant (SHOW/NOSHOW) assigned deterministically by index.
        """
        records = []
        skipped = 0

        for i, row in enumerate(rows):
            flip_frac = row.get("empirical_flip_fraction")
            if flip_frac is None:
                skipped += 1
                continue

            prompt, variant = build_e1_prompt(
                prompt_baseline=row.get("prompt_baseline", ""),
                prompt_lever=row.get("prompt_lever", ""),
                empirical_baseline_answer=row.get("empirical_baseline_answer", ""),
                capability_tags=row.get("capability_tags", []),
                variant_idx=i,
                force_noshow=self.force_noshow,
            )

            gt_label = "Yes" if flip_frac >= 0.5 else "No"

            records.append(MultitaskRecord(
                task_type=TaskType.E1_FLIP_PREDICTION.value,
                template_variant=variant,
                task_prompt=prompt,
                gt_label=gt_label,
                gt_type="binary",
                unique_id=row.get("unique_id", ""),
                corpus_dir=self.corpus_dir,
                dataset_id=row.get("dataset_id", ""),
                example_idx=row.get("example_idx", 0),
                question=row.get("question", ""),
                ground_truth_answer=row.get("ground_truth_answer", ""),
                lever_text=row.get("lever_text", ""),
                perturbation_type=row.get("perturbation_type", ""),
                category=row.get("category", ""),
                empirical_flip_fraction=flip_frac,
                capability_tags=row.get("capability_tags", []),
                target_label_axis=row.get("target_label_axis", ""),
                prompt_baseline=row.get("prompt_baseline", ""),
                prompt_lever=row.get("prompt_lever", ""),
            ))

        logger.info(
            f"[E1] Converted {len(records)} rows "
            f"(skipped {skipped} without flip_fraction). "
            f"Yes: {sum(1 for r in records if r.gt_label == 'Yes')}, "
            f"No: {sum(1 for r in records if r.gt_label == 'No')}"
        )
        return records

    def convert_e3(self, rows: list[dict]) -> list[MultitaskRecord]:
        """E3: Flip Probability — continuous 0-1 from empirical_flip_fraction.

        Deduplicates to unique (dataset_id, example_idx, perturbation_id) keys.
        Only one row per unique prompt pair.
        """
        records = []
        skipped = 0
        seen = set()

        for row in rows:
            flip_frac = row.get("empirical_flip_fraction")
            if flip_frac is None:
                skipped += 1
                continue

            # Deduplicate by unique prompt pair
            key = (
                row.get("dataset_id", ""),
                row.get("example_idx", 0),
                row.get("perturbation_id", row.get("unique_id", "")),
            )
            if key in seen:
                skipped += 1
                continue
            seen.add(key)

            prompt = build_e3_prompt(
                question=row.get("question", ""),
                prompt_baseline=row.get("prompt_baseline", ""),
                prompt_lever=row.get("prompt_lever", ""),
                empirical_baseline_answer=row.get("empirical_baseline_answer", ""),
                capability_tags=row.get("capability_tags", []),
            )

            records.append(MultitaskRecord(
                task_type=TaskType.E3_FLIP_PROBABILITY.value,
                template_variant="e3_flip_probability",
                task_prompt=prompt,
                gt_value=flip_frac,
                gt_type="continuous",
                unique_id=row.get("unique_id", ""),
                corpus_dir=self.corpus_dir,
                dataset_id=row.get("dataset_id", ""),
                example_idx=row.get("example_idx", 0),
                question=row.get("question", ""),
                ground_truth_answer=row.get("ground_truth_answer", ""),
                lever_text=row.get("lever_text", ""),
                perturbation_type=row.get("perturbation_type", ""),
                category=row.get("category", ""),
                empirical_flip_fraction=flip_frac,
                capability_tags=row.get("capability_tags", []),
                target_label_axis=row.get("target_label_axis", ""),
                prompt_baseline=row.get("prompt_baseline", ""),
                prompt_lever=row.get("prompt_lever", ""),
            ))

        flip_values = [r.gt_value for r in records if r.gt_value is not None]
        logger.info(
            f"[E3] Converted {len(records)} unique prompt pairs "
            f"(skipped {skipped}). "
            f"Mean flip_fraction: {sum(flip_values) / max(len(flip_values), 1):.3f}"
        )
        return records

    def convert_e2(self, rows: list[dict]) -> list[MultitaskRecord]:
        """E2: Output Prediction — predict the response text.

        Uses stored empirical_baseline_responses and empirical_lever_responses.
        Rows without stored responses are skipped.
        """
        records = []
        skipped = 0

        for i, row in enumerate(rows):
            baseline_responses = row.get("empirical_baseline_responses", [])
            lever_responses = row.get("empirical_lever_responses", [])

            if not baseline_responses and not lever_responses:
                skipped += 1
                continue

            prompt, variant = build_e2_prompt(
                prompt_baseline=row.get("prompt_baseline", ""),
                prompt_lever=row.get("prompt_lever", ""),
                question=row.get("question", ""),
                empirical_baseline_answer=row.get("empirical_baseline_answer", ""),
                capability_tags=row.get("capability_tags", []),
                variant_idx=i,
                force_noshow=self.force_noshow,
            )

            # GT text depends on template variant:
            # A_show, A_noshow → predict lever output
            # B → predict baseline output
            # C → predict lever output
            v = i % 4
            if v == 2:  # Template B: baseline
                gt_text = baseline_responses[0] if baseline_responses else ""
            else:  # Templates A_show, A_noshow, C: lever
                gt_text = lever_responses[0] if lever_responses else ""

            if not gt_text:
                skipped += 1
                continue

            records.append(MultitaskRecord(
                task_type=TaskType.E2_OUTPUT_PREDICTION.value,
                template_variant=variant,
                task_prompt=prompt,
                gt_text=gt_text,
                gt_type="text",
                unique_id=row.get("unique_id", ""),
                corpus_dir=self.corpus_dir,
                dataset_id=row.get("dataset_id", ""),
                example_idx=row.get("example_idx", 0),
                question=row.get("question", ""),
                ground_truth_answer=row.get("ground_truth_answer", ""),
                lever_text=row.get("lever_text", ""),
                perturbation_type=row.get("perturbation_type", ""),
                category=row.get("category", ""),
                empirical_flip_fraction=row.get("empirical_flip_fraction"),
                capability_tags=row.get("capability_tags", []),
                target_label_axis=row.get("target_label_axis", ""),
                prompt_baseline=row.get("prompt_baseline", ""),
                prompt_lever=row.get("prompt_lever", ""),
            ))

        logger.info(
            f"[E2] Converted {len(records)} rows "
            f"(skipped {skipped} without stored responses). "
            f"Variants: { {v: sum(1 for r in records if r.template_variant == v) for v in ['e2_a_show', 'e2_a_noshow', 'e2_b', 'e2_c']} }"
        )
        return records

    def convert_e6(self, rows: list[dict]) -> list[MultitaskRecord]:
        """E6: Perturbation Ranking — 2-way or 3-way MCQ from grouped perturbations.

        Groups corpus rows by (dataset_id, example_idx), selects problems
        with 2+ perturbations, picks diverse options, shuffles order.

        For problems with 5+ perturbations, generates a second ranking from
        a different subset to increase data volume.
        """
        rng = random.Random(self.seed)

        # Group by problem — include hash(question) to avoid collisions
        # between super_glue subtasks that share the same dataset_id
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            if row.get("empirical_flip_fraction") is None:
                continue
            key = (
                row.get("dataset_id", ""),
                row.get("example_idx", 0),
                hash(row.get("question", "")),
            )
            groups[key] = groups.get(key, [])
            groups[key].append(row)

        records = []
        skipped_too_few = 0
        skipped_all_tied = 0
        n_2way = 0
        n_3way = 0

        for key, group in sorted(groups.items()):
            dataset_id = key[0]
            example_idx = key[1]
            # Filter to perturbations with non-empty lever_text
            group = [r for r in group if r.get("lever_text", "").strip()]

            if len(group) < 2:
                skipped_too_few += 1
                continue

            # Sort by flip fraction to pick diverse options
            sorted_group = sorted(
                group, key=lambda r: r.get("empirical_flip_fraction", 0.0)
            )

            # Generate option sets: primary (low/mid/high) + secondary if 5+
            option_sets = []

            # Primary: lowest + highest (+ middle if 3+)
            lowest = sorted_group[0]
            highest = sorted_group[-1]
            if len(sorted_group) >= 3:
                middle = sorted_group[len(sorted_group) // 2]
                option_sets.append(([lowest, middle, highest], "e6_3way"))
            else:
                option_sets.append(([lowest, highest], "e6_2way"))

            # Secondary ranking from different subset (for problems with 5+)
            if len(sorted_group) >= 5:
                q1 = sorted_group[len(sorted_group) // 4]
                q3 = sorted_group[3 * len(sorted_group) // 4]
                # Pick a different middle than primary
                alt_mid_idx = len(sorted_group) // 4 + len(sorted_group) // 2
                alt_mid_idx = min(alt_mid_idx, len(sorted_group) - 1)
                alt_mid = sorted_group[alt_mid_idx]
                option_sets.append(([q1, alt_mid, q3], "e6_3way"))

            ref_row = group[0]

            for set_idx, (selected, variant) in enumerate(option_sets):
                letters = ["A", "B", "C"] if len(selected) == 3 else ["A", "B"]

                options = []
                for r in selected:
                    options.append({
                        "lever_text": r.get("lever_text", ""),
                        "prompt_lever": r.get("prompt_lever", ""),
                        "flip_fraction": r.get("empirical_flip_fraction", 0.0),
                        "unique_id": r.get("unique_id", ""),
                        "mechanism_name": r.get("mechanism_name", ""),
                        "category": r.get("category", ""),
                    })

                # Shuffle positions
                shuffled_indices = list(range(len(options)))
                rng.shuffle(shuffled_indices)
                shuffled_options = [options[i] for i in shuffled_indices]

                for letter, opt in zip(letters, shuffled_options):
                    opt["letter"] = letter

                # GT = letter(s) of highest flip fraction
                max_frac = max(o["flip_fraction"] for o in shuffled_options)
                min_frac = min(o["flip_fraction"] for o in shuffled_options)

                # Skip records where all options have the same flip fraction
                if max_frac == min_frac:
                    if set_idx == 0:
                        skipped_all_tied += 1
                    continue

                tied_indices = [i for i in range(len(shuffled_options)) if shuffled_options[i]["flip_fraction"] == max_frac]
                gt_all = [letters[i] for i in tied_indices]
                gt_letter = rng.choice(gt_all)

                if variant == "e6_3way":
                    n_3way += 1
                else:
                    n_2way += 1

                uid_suffix = f"_v{set_idx}" if set_idx > 0 else ""

                prompt = build_e6_prompt(
                    problem_text=ref_row.get("prompt_baseline", "") or ref_row.get("question", ""),
                    options=shuffled_options,
                    capability_tags=ref_row.get("capability_tags", []),
                )

                records.append(MultitaskRecord(
                    task_type=TaskType.E6_PERTURBATION_RANKING.value,
                    template_variant=variant,
                    task_prompt=prompt,
                    gt_label=gt_letter,
                    gt_labels=gt_all,
                    gt_type="mcq",
                    unique_id=f"e6_{dataset_id}::{example_idx}{uid_suffix}",
                    corpus_dir=self.corpus_dir,
                    dataset_id=dataset_id,
                    example_idx=example_idx,
                    question=ref_row.get("question", ""),
                    ground_truth_answer=ref_row.get("ground_truth_answer", ""),
                    capability_tags=ref_row.get("capability_tags", []),
                    target_label_axis=ref_row.get("target_label_axis", ""),
                    prompt_baseline=ref_row.get("prompt_baseline", ""),
                    e6_options=[
                        {
                            "letter": o["letter"],
                            "lever_text": o["lever_text"],
                            "flip_fraction": o["flip_fraction"],
                            "mechanism_name": o["mechanism_name"],
                            "category": o["category"],
                        }
                        for o in shuffled_options
                    ],
                ))

        logger.info(
            f"[E6] Converted {len(records)} problems "
            f"({n_3way} 3-way + {n_2way} 2-way, "
            f"skipped {skipped_too_few} with <2 perturbations, "
            f"{skipped_all_tied} with all-tied flip fractions). "
            f"GT distribution: { {l: sum(1 for r in records if r.gt_label == l) for l in 'ABC'} }"
        )
        return records

    def convert_e8(self, rows: list[dict]) -> list[MultitaskRecord]:
        """E8: Propose Flip — prompt only, reward computed online at Tinker rollout.

        For each corpus row, builds the E8 prompt (asking model to propose
        a minimal edit that changes the answer). GT is computed online:
        run the proposed edit through vLLM, check flip, compute edit distance.

        Stores prompt_baseline/prompt_lever for the online reward function.
        """
        records = []
        skipped = 0

        for i, row in enumerate(rows):
            # Alternate between baseline and lever condition
            if i % 2 == 0:
                edit_prompt = row.get("prompt_baseline", "")
                variant = "e8_base"
                # Store full response (not parsed answer) for reliable flip comparison
                baseline_resps = row.get("empirical_baseline_responses", [])
                stored_full_response = baseline_resps[0] if baseline_resps else ""
            else:
                edit_prompt = row.get("prompt_lever", "")
                variant = "e8_pert"
                lever_resps = row.get("empirical_lever_responses", [])
                stored_full_response = lever_resps[0] if lever_resps else ""

            if not edit_prompt:
                skipped += 1
                continue

            prompt = build_e8_prompt(
                prompt_text=edit_prompt,
                capability_tags=row.get("capability_tags", []),
            )

            records.append(MultitaskRecord(
                task_type=TaskType.E8_PROPOSE_FLIP.value,
                template_variant=variant,
                task_prompt=prompt,
                gt_type="continuous",  # online reward: flip_acc * (1 - edit_dist)
                gt_text=stored_full_response,  # full model response for flip comparison
                unique_id=row.get("unique_id", ""),
                corpus_dir=self.corpus_dir,
                dataset_id=row.get("dataset_id", ""),
                example_idx=row.get("example_idx", 0),
                question=row.get("question", ""),
                lever_text=row.get("lever_text", ""),
                category=row.get("category", ""),
                empirical_flip_fraction=row.get("empirical_flip_fraction"),
                capability_tags=row.get("capability_tags", []),
                prompt_baseline=row.get("prompt_baseline", ""),
                prompt_lever=row.get("prompt_lever", ""),
            ))

        logger.info(
            f"[E8] Converted {len(records)} rows "
            f"(skipped {skipped} without prompt). "
            f"Variants: base={sum(1 for r in records if r.template_variant == 'e8_base')}, "
            f"pert={sum(1 for r in records if r.template_variant == 'e8_pert')}"
        )
        return records

    def convert_e9(self, rows: list[dict]) -> list[MultitaskRecord]:
        """E9: Feature Presence — from features_baseline + lever negatives.

        Generates two types of records:
        - Baseline records: feature present in baseline response (mostly 1.0)
        - Lever negatives: feature present in baseline but absent/changed in
          lever response → ask about lever prompt with gt=0.0

        This produces balanced training data instead of 90% ones.
        """
        records = []
        skipped = 0
        n_lever_negatives = 0

        for row in rows:
            target_axis = row.get("target_label_axis", "")
            answer_labels = row.get("answer_labels", [])

            if not target_axis or not answer_labels:
                skipped += 1
                continue

            features_baseline = row.get("features_baseline", {})

            def _is_present(value: str) -> bool:
                return bool(
                    value
                    and str(value).strip()
                    and str(value).strip().lower() not in ("unknown", "none", "n/a", "")
                )

            # --- Baseline record (ask about baseline prompt) ---
            baseline_value = features_baseline.get(target_axis, "")
            gt_value = 1.0 if _is_present(baseline_value) else 0.0

            feat_desc = next(
                (l.get("description", target_axis) for l in answer_labels if l.get("name") == target_axis),
                target_axis,
            )

            prompt = build_e9_prompt(
                problem_text=row.get("prompt_baseline", "") or row.get("question", ""),
                capability_tags=row.get("capability_tags", []),
                target_label_axis=target_axis,
                answer_labels=answer_labels,
            )

            records.append(MultitaskRecord(
                task_type=TaskType.E9_FEATURE_PRESENCE.value,
                template_variant="e9_feature",
                task_prompt=prompt,
                gt_value=gt_value,
                gt_type="continuous",
                unique_id=row.get("unique_id", ""),
                corpus_dir=self.corpus_dir,
                dataset_id=row.get("dataset_id", ""),
                example_idx=row.get("example_idx", 0),
                question=row.get("question", ""),
                ground_truth_answer=row.get("ground_truth_answer", ""),
                lever_text=row.get("lever_text", ""),
                perturbation_type=row.get("perturbation_type", ""),
                category=row.get("category", ""),
                empirical_flip_fraction=row.get("empirical_flip_fraction"),
                capability_tags=row.get("capability_tags", []),
                target_label_axis=target_axis,
                prompt_baseline=row.get("prompt_baseline", ""),
                prompt_lever=row.get("prompt_lever", ""),
                e9_feature_name=target_axis,
                e9_feature_description=feat_desc,
            ))

            # --- Lever negative: feature flipped from baseline → lever ---
            features_lever = row.get("features_lever", {})
            if features_lever and _is_present(baseline_value):
                lever_value = features_lever.get(target_axis, "")
                # If feature changed or disappeared in lever, it's a natural negative
                if not _is_present(lever_value) or str(lever_value).strip().lower() != str(baseline_value).strip().lower():
                    lever_prompt = build_e9_prompt(
                        problem_text=row.get("prompt_lever", "") or row.get("question", ""),
                        capability_tags=row.get("capability_tags", []),
                        target_label_axis=target_axis,
                        answer_labels=answer_labels,
                    )
                    records.append(MultitaskRecord(
                        task_type=TaskType.E9_FEATURE_PRESENCE.value,
                        template_variant="e9_lever_neg",
                        task_prompt=lever_prompt,
                        gt_value=0.0,
                        gt_type="continuous",
                        unique_id=row.get("unique_id", "") + "_lever",
                        corpus_dir=self.corpus_dir,
                        dataset_id=row.get("dataset_id", ""),
                        example_idx=row.get("example_idx", 0),
                        question=row.get("question", ""),
                        ground_truth_answer=row.get("ground_truth_answer", ""),
                        lever_text=row.get("lever_text", ""),
                        perturbation_type=row.get("perturbation_type", ""),
                        category=row.get("category", ""),
                        empirical_flip_fraction=row.get("empirical_flip_fraction"),
                        capability_tags=row.get("capability_tags", []),
                        target_label_axis=target_axis,
                        prompt_baseline=row.get("prompt_baseline", ""),
                        prompt_lever=row.get("prompt_lever", ""),
                        e9_feature_name=target_axis,
                        e9_feature_description=feat_desc,
                    ))
                    n_lever_negatives += 1

        n_present = sum(1 for r in records if r.gt_value == 1.0)
        n_absent = sum(1 for r in records if r.gt_value == 0.0)
        logger.info(
            f"[E9] Converted {len(records)} rows "
            f"(skipped {skipped} without target_label_axis or answer_labels). "
            f"Present: {n_present} ({n_present / max(len(records), 1) * 100:.1f}%), "
            f"Absent: {n_absent} ({n_absent / max(len(records), 1) * 100:.1f}%, "
            f"incl {n_lever_negatives} lever negatives)"
        )
        return records
