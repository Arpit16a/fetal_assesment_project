"""
03_label_validation.py

ARTIFACT LABEL VALIDATION
==========================

Purpose
-------
This is Step 4 of the artifact-handling layer.

It does NOT train any model.

It answers one question:

    "Before we trust any label, is it actually meaningful?"

Inputs
------
data/processed/artifact_features/
    cough_candidate_features.csv
    four_imu_candidate_features.csv
    oxford_candidate_features.csv
    manual_label_template.csv   (filled in by a human reviewer:
                                  manual_label column populated with
                                  artifact / likely_fetal / uncertain)

Outputs
-------
data/processed/artifact_labels/
    validated_dataset.csv       (features + resolved binary label,
                                  ready for Step 5 / 6)
    label_validation_report.txt

Label resolution priority
--------------------------
For every CANDIDATE EVENT (the labeling unit, since Phase 6/7 — a
candidate is a real detected activity event within a segment, not
the whole segment):

    1. Human manual_label, if present and not "uncertain"
         artifact       -> is_artifact = 1
         likely_fetal   -> is_artifact = 0

    2. COUGH weak label, if this is a COUGH candidate and true_label
       is available (true_label is the trial's talking/non-talking
       condition — see COUGH_WEAK_LABELS_ENABLED below for why this
       is disabled, not a source of real supervision).

    3. Otherwise -> unlabeled (excluded from the training split,
       kept in the file for transparency / future manual review).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

def _find_project_root(start: Path) -> Path:
    """
    Walk up from `start` until finding the directory that contains
    src/dataset.py, instead of hardcoding a fixed number of .parent
    hops -- a fixed hop-count broke once already when this script
    moved from src/signal_processing/ to src/Artifact_layer/new_ver/.
    dataset.py is this project's most stable anchor point.
    """
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "dataset.py").exists():
            return candidate
    raise RuntimeError(
        f"Could not locate the project root (looked for "
        f"src/dataset.py) starting from {start}."
    )


PROJECT_ROOT = _find_project_root(Path(__file__).parent)

# Phase 7 output — one feature row per CANDIDATE EVENT, not per
# whole continuous segment. This replaces the old
# artifact_inspection/*_segment_features.csv input: those files
# came from 02_artifact_inspection.py's whole-segment z-score
# heuristic, which this project explicitly moved away from in favor
# of the deterministic candidate-generation + feature-extraction
# pipeline (06_candidate_generation.py / 07_artifact_features.py).
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "artifact_features"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "artifact_labels"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_FEATURE_FILES = {
    "COUGH": FEATURES_DIR / "cough_candidate_features.csv",
    "FOUR_IMU": FEATURES_DIR / "four_imu_candidate_features.csv",
    "OXFORD": FEATURES_DIR / "oxford_candidate_features.csv",
}

MANUAL_TEMPLATE_PATH = FEATURES_DIR / "manual_label_template.csv"

# ============================================================
# WEAK-LABEL BOOTSTRAP (OPT-IN — READ THIS BEFORE ENABLING)
# ============================================================
#
# If you genuinely don't have time to hand-review every candidate,
# this lets labeling_helper.py's rule-based suggested_label stand
# in for manual_label where manual_label was left blank.
#
# BE HONEST ABOUT WHAT THIS ACTUALLY GIVES YOU: suggested_label
# comes from a simple, transparent heuristic (duration + multi-
# sensor sync + directional consistency), not a human who looked
# at the signal and judged it. A baseline model trained on these
# labels is measuring "can ML reproduce this heuristic," not "can
# ML separate real artifacts from real fetal movement." That's a
# legitimate thing to report as a placeholder / pipeline-validation
# milestone — it is NOT a legitimate thing to present as a
# validated artifact detector. Say so explicitly if you show this
# to your professor.
#
# Rows are tagged label_source="heuristic_suggestion" specifically
# so they are never silently confused with label_source="manual"
# downstream — anyone reading validated_dataset.csv can immediately
# tell which labels came from a human and which didn't.

USE_SUGGESTED_LABEL_AS_FALLBACK = True

SUGGESTIONS_PATH = FEATURES_DIR / "priority_labeling_batch_with_suggestions.csv"


# ============================================================
# CONFIG — COUGH TRIAL SEMANTICS (CONFIRMED, NOT A GUESS)
# ============================================================
#
# CONFIRMED against the actual dataset folder names:
#   ['Trial_1_No_Talking', 'Trial_2_Talking', 'Trial_3_Nonverbal']
#
# These are SPEECH/TALKING CONDITIONS during the recording, not
# activity-type labels (cough / laugh / walk / etc). There is no
# "quiet baseline vs. active motion" distinction encoded in the
# trial folder name at all — the earlier QUIET_TRIAL_KEYWORDS
# approach (matching "rest"/"quiet"/"still" substrings) was built
# on a wrong assumption about what these folder names meant and is
# now disabled below rather than left silently wrong.
#
# The real per-event activity labels (cough/laugh/speech/sneeze/
# etc, if they exist) live in a separate annotation JSON per trial
# — see the "Cough Event Analysis" section of
# 03_artifact_dataset_exploration.ipynb, which reads an
# `annotation_file`. Parsing that JSON into per-event ground truth
# is real, separate work, not something to fake from the folder
# name. Given the 5-day timeline, this is intentionally deferred:
# COUGH is used purely as a non-target-motion reference dataset for
# now, exactly as the project architecture already designates it —
# not as a source of weak supervision.

COUGH_WEAK_LABELS_ENABLED = False


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_candidate_features() -> pd.DataFrame:
    frames = []

    for dataset_name, path in CANDIDATE_FEATURE_FILES.items():

        if not path.exists():
            print(f"WARNING: missing {path}, skipping {dataset_name}")
            continue

        df = pd.read_csv(path)
        if "dataset" not in df.columns:
            df["dataset"] = dataset_name
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No candidate feature files were found. Run "
            "06_candidate_generation.py then 07_artifact_features.py first."
        )

    return pd.concat(frames, ignore_index=True)


def load_manual_labels() -> pd.DataFrame:
    if not MANUAL_TEMPLATE_PATH.exists():
        print(
            "WARNING: manual_label_template.csv not found. "
            "Continuing with zero manual labels."
        )
        return pd.DataFrame(columns=["candidate_id", "manual_label"])

    df = pd.read_csv(MANUAL_TEMPLATE_PATH)
    df["manual_label"] = df["manual_label"].fillna("").astype(str).str.strip().str.lower()

    return df[["candidate_id", "manual_label"]]


def load_suggested_labels() -> pd.DataFrame:
    """
    Only loaded if USE_SUGGESTED_LABEL_AS_FALLBACK is True. Returns
    candidate_id -> suggested_label for candidates whose manual_label
    was left blank in priority_labeling_batch_with_suggestions.csv
    (i.e. this NEVER overrides a real manual label someone actually
    filled in — see apply_suggested_label_fallback below).
    """

    if not SUGGESTIONS_PATH.exists():
        print(
            f"WARNING: USE_SUGGESTED_LABEL_AS_FALLBACK is True but "
            f"{SUGGESTIONS_PATH} was not found. Continuing without it."
        )
        return pd.DataFrame(columns=["candidate_id", "suggested_label", "manual_label"])

    df = pd.read_csv(SUGGESTIONS_PATH)
    df["manual_label"] = df["manual_label"].fillna("").astype(str).str.strip().str.lower()
    df["suggested_label"] = df["suggested_label"].fillna("").astype(str).str.strip().str.lower()

    return df[["candidate_id", "manual_label", "suggested_label"]]


def apply_suggested_label_fallback(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Fill manual_label from suggested_label ONLY where a row has no
    real manual_label already (blank, not "uncertain" — "uncertain"
    is a real human decision and is never overwritten either way).
    A candidate reviewed by hand always wins over the heuristic.
    """

    if not USE_SUGGESTED_LABEL_AS_FALLBACK:
        merged["label_source_detail"] = ""
        return merged

    suggestions = load_suggested_labels()
    if suggestions.empty:
        merged["label_source_detail"] = ""
        return merged

    suggestions = suggestions.rename(columns={
        "manual_label": "_suggestions_manual_label",
        "suggested_label": "_suggested_label",
    })

    merged = merged.merge(
        suggestions[["candidate_id", "_suggestions_manual_label", "_suggested_label"]],
        on="candidate_id", how="left",
    )

    was_blank = merged["manual_label"] == ""
    has_suggestion = merged["_suggested_label"].isin(["artifact", "likely_fetal"])
    use_fallback = was_blank & has_suggestion

    merged.loc[use_fallback, "manual_label"] = merged.loc[use_fallback, "_suggested_label"]
    merged["label_source_detail"] = np.where(use_fallback, "heuristic_suggestion", "")

    merged = merged.drop(columns=["_suggestions_manual_label", "_suggested_label"])

    print(
        f"\nWEAK-LABEL BOOTSTRAP ACTIVE: filled {use_fallback.sum()} blank "
        f"manual_label rows from suggested_label (heuristic, not human "
        f"review). These are tagged label_source='heuristic_suggestion' "
        f"in the output — treat this as a placeholder baseline, not a "
        f"validated result."
    )

    return merged


def apply_manual_labels(features: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:

    # candidate_id is unique across the whole project (it's built
    # from dataset + record + sensor + segment + candidate index in
    # 06_candidate_generation.py), so it — not the old
    # (dataset, record_identifier, segment_id) triple — is the join
    # key here. A segment can contain several candidates; a segment-
    # level key would have applied one manual label to all of them.
    merged = features.merge(
        manual,
        on="candidate_id",
        how="left",
    )

    merged["manual_label"] = merged["manual_label"].fillna("")

    return merged


def apply_cough_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    DISABLED (see COUGH_WEAK_LABELS_ENABLED above). trial_dir.name
    encodes a talking/non-talking condition, not an activity type
    — there is no valid quiet-vs-active split to derive from it.
    Returns the frame unchanged, with weak_label_cough left as NaN
    for every row so resolve_final_label() falls through to manual
    labels or "unlabeled" instead of a fabricated weak label.
    """

    df = df.copy()
    df["weak_label_cough"] = np.nan

    if not COUGH_WEAK_LABELS_ENABLED:
        return df

    # Unreachable while disabled — kept only so re-enabling this
    # requires flipping one flag once real per-event COUGH labels
    # (from the annotation JSON) are parsed, not rewriting this
    # function from scratch.
    is_cough = df["dataset"] == "COUGH"
    return df


def resolve_final_label(row: pd.Series) -> float:

    manual = row.get("manual_label", "")

    if manual == "artifact":
        return 1.0
    if manual == "likely_fetal":
        return 0.0
    # "uncertain" or blank falls through

    weak = row.get("weak_label_cough", np.nan)
    if pd.notna(weak):
        return weak

    return np.nan


def resolve_label_source(row: pd.Series) -> str:

    manual = row.get("manual_label", "")
    detail = row.get("label_source_detail", "")

    if manual in ("artifact", "likely_fetal"):
        return "heuristic_suggestion" if detail == "heuristic_suggestion" else "manual"
    if manual == "uncertain":
        return "uncertain_manual"

    weak = row.get("weak_label_cough", np.nan)
    if pd.notna(weak):
        return "cough_weak_label"

    return "unlabeled"


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_header("STEP 4 — ARTIFACT LABEL VALIDATION")

    # UNMISSABLE CONFIG BANNER — printed every run, before anything
    # else. Shows exactly which files and which mode are actually
    # active, so "did my setting take effect" is answered by looking
    # at the console, not by re-reading source code.
    print(f"\nWEAK-LABEL BOOTSTRAP: {'ENABLED' if USE_SUGGESTED_LABEL_AS_FALLBACK else 'DISABLED'}")
    print(f"  MANUAL_TEMPLATE_PATH exists: {MANUAL_TEMPLATE_PATH.exists()}  -> {MANUAL_TEMPLATE_PATH}")
    print(f"  SUGGESTIONS_PATH exists:     {SUGGESTIONS_PATH.exists()}  -> {SUGGESTIONS_PATH}")
    if not USE_SUGGESTED_LABEL_AS_FALLBACK:
        print(
            "  NOTE: bootstrap is OFF. Only real manual_label values "
            "count -- if manual_label_template.csv is still blank, "
            "labeled rows will be 0. Set USE_SUGGESTED_LABEL_AS_FALLBACK "
            "= True near the top of this file to use suggested_label "
            "as a fallback instead."
        )

    features = load_candidate_features()
    print(f"Loaded segment features: {features.shape}")

    manual = load_manual_labels()
    print(f"Loaded manual label rows: {len(manual)}")

    merged = apply_manual_labels(features, manual)
    merged = apply_suggested_label_fallback(merged)
    merged = apply_cough_weak_labels(merged)

    merged["is_artifact"] = merged.apply(resolve_final_label, axis=1)
    merged["label_source"] = merged.apply(resolve_label_source, axis=1)

    # --------------------------------------------------------------
    # REPORT
    # --------------------------------------------------------------

    report_lines = []

    report_lines.append("LABEL SOURCE COUNTS (all datasets)")
    report_lines.append(str(merged["label_source"].value_counts()))
    report_lines.append("")

    report_lines.append("CLASS BALANCE (resolved is_artifact, labeled rows only)")
    labeled = merged[merged["is_artifact"].notna()]
    report_lines.append(str(labeled["is_artifact"].value_counts()))
    report_lines.append("")

    report_lines.append("LABEL SOURCE BY DATASET")
    report_lines.append(str(pd.crosstab(merged["dataset"], merged["label_source"])))
    report_lines.append("")

    if "candidate_reason" in features.columns:
        pass  # candidate_events has this, not segment_features; skipped here.

    if "true_label" in merged.columns:
        cough_rows = merged[merged["dataset"] == "COUGH"]
        if not cough_rows.empty:
            report_lines.append(
                "COUGH: unique true_label (trial folder) values found "
                "-- these are talking/non-talking CONDITIONS, not "
                "activity types. COUGH weak-labeling is disabled; "
                "see COUGH_WEAK_LABELS_ENABLED."
            )
            report_lines.append(str(sorted(cough_rows["true_label"].dropna().unique())))
            report_lines.append("")

    manual_labeled = merged[merged["label_source"] == "manual"]
    if not manual_labeled.empty and "weak_label_cough" in merged.columns:
        overlap = manual_labeled[manual_labeled["weak_label_cough"].notna()]
        if not overlap.empty:
            agreement = (overlap["is_artifact"] == overlap["weak_label_cough"]).mean()
            report_lines.append(
                f"Agreement between human manual label and COUGH weak "
                f"label, where both exist: {agreement:.2%} (n={len(overlap)})"
            )
            report_lines.append("")

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = OUTPUT_DIR / "label_validation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    # --------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------

    output_path = OUTPUT_DIR / "validated_dataset.csv"
    merged.to_csv(output_path, index=False)

    print_header("STEP 4 COMPLETE")
    print(f"Validated dataset saved: {output_path}")
    print(f"Report saved:            {report_path}")
    print(
        f"\nLabeled rows ready for training: "
        f"{merged['is_artifact'].notna().sum()} / {len(merged)}"
    )
    print(
        "\nIf this count looks too low: COUGH weak-labeling is "
        "disabled (see COUGH_WEAK_LABELS_ENABLED), so COUGH rows "
        "only get labeled through manual review. FOUR_IMU and "
        "OXFORD rely entirely on manual labeling too -- widen the "
        "manual-review sample if this number is too small to train on."
    )


if __name__ == "__main__":
    main()