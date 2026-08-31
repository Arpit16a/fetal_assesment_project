"""
build_labeling_batch.py

Turns manual_label_template.csv (every candidate — currently 50,437
rows) into a much smaller, prioritized batch actually worth
hand-labeling with limited time.

Why not just label randomly
-----------------------------
A random sample over-represents whatever dataset happens to be
biggest (Four-IMU: 48,903 of the 50,437 candidates — 97%) and
under-represents the rest, and gives no guarantee of covering the
full range of candidate strength. This script instead:

    1. Samples per DATASET separately, so Cough and Oxford aren't
       drowned out by Four-IMU's sheer volume.
    2. Within each dataset, stratifies by peak_activity_score
       quartile — every candidate already cleared the detection
       threshold, so "low" here means "least extreme candidate",
       not "no movement." Sampling across the full range on purpose
       includes the borderline cases near the threshold, which are
       exactly the ones a model needs to learn to separate — a
       random sample would mostly hand you the easy, obvious cases.
    3. Spreads across as many distinct SUBJECTS as possible (capped
       per subject), because the baseline model needs subject
       diversity in the labeled set to do subject-independent
       validation at all — labeling 300 candidates from 5 subjects
       is far less useful than 300 from 40.

Output
-------
data/processed/artifact_features/priority_labeling_batch.csv

Label THIS file (same manual_label / confidence / reviewer_notes
columns as before), not the full manual_label_template.csv — this
subset is stamped with the same candidate_id, so label_validation.py
picks up whatever you've filled in here automatically (it already
merges on candidate_id against the full feature set).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "dataset.py").exists():
            return candidate
    raise RuntimeError(
        f"Could not locate the project root (looked for "
        f"src/dataset.py) starting from {start}."
    )


PROJECT_ROOT = _find_project_root(Path(__file__).parent)

FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "artifact_features"
TEMPLATE_PATH = FEATURES_DIR / "manual_label_template.csv"
OUTPUT_PATH = FEATURES_DIR / "priority_labeling_batch.csv"


# ============================================================
# CONFIG — adjust these based on how much labeling time you have
# ============================================================

# Target candidates per dataset. Oxford gets more per-candidate
# weight relative to its total size (only 1133 candidates from 16
# subjects) since every one of its subjects matters for eventual
# subject-independent validation there; Four-IMU's target is a small
# fraction of its 48,903 candidates on purpose.
TARGET_PER_DATASET = {
    "COUGH": 60,
    "FOUR_IMU": 220,
    "OXFORD": 140,
}

N_ACTIVITY_BINS = 4          # quartile stratification
MAX_CANDIDATES_PER_SUBJECT = 8  # forces subject spread, not depth


def build_batch(template: pd.DataFrame) -> pd.DataFrame:

    batches = []

    for dataset_name, target_n in TARGET_PER_DATASET.items():

        subset = template[template["dataset"] == dataset_name].copy()

        if subset.empty:
            print(f"{dataset_name}: no candidates found, skipping")
            continue

        # Quartile bins on peak_activity_score, computed WITHIN this
        # dataset (activity score scale differs across datasets --
        # BCG vs. accelerometer -- so bins must be dataset-relative).
        try:
            subset["activity_bin"] = pd.qcut(
                subset["peak_activity_score"],
                q=min(N_ACTIVITY_BINS, subset["peak_activity_score"].nunique()),
                duplicates="drop",
            )
        except ValueError:
            subset["activity_bin"] = 0  # too few distinct values to bin

        n_bins = subset["activity_bin"].nunique()
        per_bin_target = max(target_n // n_bins, 1)

        sampled_parts = []

        for _, bin_group in subset.groupby("activity_bin", observed=True):

            # Cap per subject so the sample spreads across many
            # subjects instead of repeatedly drawing from whichever
            # subject happens to have the most candidates in this
            # bin. Iterating explicitly rather than
            # groupby(...).apply(...) — apply() silently drops the
            # grouping column (subject_id) on some pandas versions,
            # caught by testing this against data at the real scale
            # before handing it over.
            capped_groups = []
            for _, subject_group in bin_group.groupby("subject_id"):
                n_take = min(len(subject_group), MAX_CANDIDATES_PER_SUBJECT)
                capped_groups.append(subject_group.sample(n_take, random_state=42))

            capped = pd.concat(capped_groups, ignore_index=True) if capped_groups else bin_group.iloc[:0]

            n_take = min(per_bin_target, len(capped))
            sampled_parts.append(capped.sample(n_take, random_state=42))

        dataset_sample = pd.concat(sampled_parts, ignore_index=True)

        print(
            f"{dataset_name}: sampled {len(dataset_sample)} candidates "
            f"from {dataset_sample['subject_id'].nunique()} subjects "
            f"(of {subset['subject_id'].nunique()} available)"
        )

        batches.append(dataset_sample.drop(columns=["activity_bin"]))

    return pd.concat(batches, ignore_index=True)


def main() -> None:

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"{TEMPLATE_PATH} not found. Run artifact_features.py first."
        )

    template = pd.read_csv(TEMPLATE_PATH)
    print(f"Full candidate pool: {len(template)} candidates")

    batch = build_batch(template)

    batch.to_csv(OUTPUT_PATH, index=False)

    print(f"\nPriority labeling batch: {len(batch)} candidates")
    print(f"Saved: {OUTPUT_PATH}")
    print(
        "\nOpen this file, fill in manual_label for each row "
        "(artifact / likely_fetal / uncertain), save it, then run "
        "label_validation.py as before -- it merges on candidate_id "
        "against the full feature set automatically, so labeling "
        "this smaller file is enough."
    )


if __name__ == "__main__":
    main()
