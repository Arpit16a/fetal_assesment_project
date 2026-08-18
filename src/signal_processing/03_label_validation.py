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
data/processed/artifact_inspection/
    cough_segment_features.csv
    four_imu_segment_features.csv
    oxford_segment_features.csv
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
For every segment (candidate or not):

    1. Human manual_label, if present and not "uncertain"
         artifact       -> is_artifact = 1
         likely_fetal   -> is_artifact = 0

    2. COUGH weak label, if this is a COUGH segment and true_label
       is available (requires the extract_segment_features() patch
       that stamps true_label onto each row).

    3. Otherwise -> unlabeled (excluded from the training split,
       kept in the file for transparency / future manual review).

The COUGH weak-label rule is intentionally conservative: it is a
starting hypothesis, not ground truth, and is reported separately
so its influence on the final baseline is auditable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INSPECTION_DIR = PROJECT_ROOT / "data" / "processed" / "artifact_inspection"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "artifact_labels"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_FILES = {
    "COUGH": INSPECTION_DIR / "cough_segment_features.csv",
    "FOUR-IMU": INSPECTION_DIR / "four_imu_segment_features.csv",
    "OXFORD": INSPECTION_DIR / "oxford_segment_features.csv",
}

MANUAL_TEMPLATE_PATH = INSPECTION_DIR / "manual_label_template.csv"


# ============================================================
# CONFIG — EDIT AFTER INSPECTING TRUE_LABEL VALUES
# ============================================================
#
# These are trial-name substrings from the Multimodal Cough
# Dataset that represent a QUIET / BASELINE condition (i.e.
# NOT expected to contain non-fetal motion artifacts).
# Everything else is treated as a candidate artifact-positive
# activity (cough, laugh, walk, sneeze, speech, etc.).
#
# IMPORTANT: run this script once, read the printed list of
# unique true_label values under COUGH, and correct this list
# before trusting the weak labels.

QUIET_TRIAL_KEYWORDS = ["rest", "quiet", "still", "baseline", "sit"]


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_segment_features() -> pd.DataFrame:
    frames = []

    for dataset_name, path in SEGMENT_FILES.items():

        if not path.exists():
            print(f"WARNING: missing {path}, skipping {dataset_name}")
            continue

        df = pd.read_csv(path)
        df["dataset"] = dataset_name
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No segment feature files were found. "
            "Run 02_artifact_inspection.py first."
        )

    return pd.concat(frames, ignore_index=True)


def load_manual_labels() -> pd.DataFrame:
    if not MANUAL_TEMPLATE_PATH.exists():
        print(
            "WARNING: manual_label_template.csv not found. "
            "Continuing with zero manual labels."
        )
        return pd.DataFrame(
            columns=["dataset", "record_identifier", "segment_id", "manual_label"]
        )

    df = pd.read_csv(MANUAL_TEMPLATE_PATH)
    df["manual_label"] = df["manual_label"].fillna("").astype(str).str.strip().str.lower()

    return df[["dataset", "record_identifier", "segment_id", "manual_label"]]


def apply_manual_labels(features: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:

    merged = features.merge(
        manual,
        on=["dataset", "record_identifier", "segment_id"],
        how="left",
    )

    merged["manual_label"] = merged["manual_label"].fillna("")

    return merged


def apply_cough_weak_labels(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    is_cough = df["dataset"] == "COUGH"

    if "true_label" not in df.columns:
        print(
            "NOTE: 'true_label' column not found — apply the "
            "extract_segment_features() patch and re-run "
            "02_artifact_inspection.py to enable COUGH weak labels."
        )
        df["weak_label_cough"] = np.nan
        return df

    def classify(label: str) -> float:
        if not isinstance(label, str) or not label:
            return np.nan
        label_lower = label.lower()
        is_quiet = any(keyword in label_lower for keyword in QUIET_TRIAL_KEYWORDS)
        return 0.0 if is_quiet else 1.0

    df["weak_label_cough"] = np.nan
    df.loc[is_cough, "weak_label_cough"] = df.loc[is_cough, "true_label"].apply(classify)

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
    if manual in ("artifact", "likely_fetal"):
        return "manual"
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

    features = load_segment_features()
    print(f"Loaded segment features: {features.shape}")

    manual = load_manual_labels()
    print(f"Loaded manual label rows: {len(manual)}")

    merged = apply_manual_labels(features, manual)
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
            report_lines.append("COUGH: unique true_label values found")
            report_lines.append(str(sorted(cough_rows["true_label"].dropna().unique())))
            report_lines.append("")
            report_lines.append(
                "COUGH: weak_label_cough vs true_label crosstab "
                "(sanity check the QUIET_TRIAL_KEYWORDS list above)"
            )
            report_lines.append(
                str(pd.crosstab(cough_rows["true_label"], cough_rows["weak_label_cough"]))
            )
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
        "\nIf this count looks too low, either label more candidates "
        "manually, or widen/verify QUIET_TRIAL_KEYWORDS above."
    )


if __name__ == "__main__":
    main()