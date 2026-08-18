"""
05_artifact_detector_integration.py

ARTIFACT DETECTOR INTEGRATION
===============================

Purpose
-------
Step 7 of the artifact-handling layer.

    timing
      |
      v
    artifact detector   <-- this script
      |
      v
    clean movement candidates

This script does NOT re-run the timing layer or re-extract features.
It consumes the SAME segment_features.csv files produced by
02_artifact_inspection.py (i.e. every valid continuous segment,
not just the flagged "candidates"), scores each one with the
Step 5/6 baseline model, and splits them into:

    - clean movement candidates  (predicted non-artifact)
    - artifact segments          (predicted artifact)

so that Step 4 in the OVERALL project pipeline (movement detection)
has a filtered, artifact-aware input instead of raw segments.

IMPORTANT
---------
This is a baseline detector. Its predictions are a filter, not a
ground-truth relabeling. Segments predicted "artifact" are set
aside, not deleted — they remain available for review.

Input
-----
data/processed/artifact_inspection/{dataset}_segment_features.csv
data/processed/artifact_baseline/best_model.joblib
data/processed/artifact_baseline/best_model_features.json

Outputs
-------
data/processed/artifact_detector_output/
    {dataset}_segments_scored.csv
    {dataset}_clean_movement_candidates.csv
    {dataset}_artifact_segments.csv
    integration_summary.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INSPECTION_DIR = PROJECT_ROOT / "data" / "processed" / "artifact_inspection"
BASELINE_DIR = PROJECT_ROOT / "data" / "processed" / "artifact_baseline"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "artifact_detector_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_FILES = {
    "COUGH": INSPECTION_DIR / "cough_segment_features.csv",
    "FOUR-IMU": INSPECTION_DIR / "four_imu_segment_features.csv",
    "OXFORD": INSPECTION_DIR / "oxford_segment_features.csv",
}

MODEL_PATH = BASELINE_DIR / "best_model.joblib"
FEATURES_PATH = BASELINE_DIR / "best_model_features.json"


# ============================================================
# CONFIG
# ============================================================

# Decision threshold on predicted artifact probability.
# 0.5 is the default; move it toward 0 to be more conservative
# about discarding segments (favors recall of clean fetal data),
# or toward 1 to be stricter about what counts as "clean".
ARTIFACT_PROBABILITY_THRESHOLD = 0.5

# Sanity-check bounds. If the fraction of segments flagged as
# artifact falls outside this range for a dataset, print a
# warning — it usually means the threshold or the feature
# pipeline needs a second look, not that the data is "wrong".
SANITY_MIN_FLAGGED_FRACTION = 0.02
SANITY_MAX_FLAGGED_FRACTION = 0.60


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_model_and_features() -> tuple[object, list[str]]:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run 04_artifact_baseline_models.py first."
        )

    model = joblib.load(MODEL_PATH)
    feature_columns = json.loads(FEATURES_PATH.read_text())

    return model, feature_columns


def score_dataset(
    dataset_name: str,
    path: Path,
    model,
    feature_columns: list[str],
) -> pd.DataFrame | None:

    if not path.exists():
        print(f"WARNING: {path} not found, skipping {dataset_name}")
        return None

    df = pd.read_csv(path)

    missing = [c for c in feature_columns if c not in df.columns]
    for col in missing:
        df[col] = np.nan

    X = df[feature_columns]

    proba = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["artifact_probability"] = proba
    df["predicted_artifact"] = (
        proba >= ARTIFACT_PROBABILITY_THRESHOLD
    ).astype(int)

    return df


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_header("STEP 7 — ARTIFACT DETECTOR INTEGRATION")

    model, feature_columns = load_model_and_features()
    print(f"Loaded model. Expecting {len(feature_columns)} features.")
    print(f"Decision threshold: {ARTIFACT_PROBABILITY_THRESHOLD}")

    summary_rows = []

    for dataset_name, path in SEGMENT_FILES.items():

        print_header(f"{dataset_name}")

        scored = score_dataset(dataset_name, path, model, feature_columns)

        if scored is None:
            summary_rows.append(
                {"dataset": dataset_name, "status": "SKIPPED (file missing)"}
            )
            continue

        total = len(scored)
        flagged = int(scored["predicted_artifact"].sum())
        flagged_fraction = flagged / total if total else 0.0

        print(f"Total segments scored : {total}")
        print(f"Flagged as artifact   : {flagged} ({flagged_fraction:.1%})")

        if flagged_fraction < SANITY_MIN_FLAGGED_FRACTION:
            print(
                "WARNING: unusually LOW artifact fraction — "
                "double-check the model isn't defaulting to 'clean' for everything."
            )
        elif flagged_fraction > SANITY_MAX_FLAGGED_FRACTION:
            print(
                "WARNING: unusually HIGH artifact fraction — "
                "double-check feature alignment / threshold before trusting this filter."
            )

        prefix = dataset_name.lower().replace("-", "_")

        scored_path = OUTPUT_DIR / f"{prefix}_segments_scored.csv"
        scored.to_csv(scored_path, index=False)

        clean = scored[scored["predicted_artifact"] == 0]
        clean_path = OUTPUT_DIR / f"{prefix}_clean_movement_candidates.csv"
        clean.to_csv(clean_path, index=False)

        artifact = scored[scored["predicted_artifact"] == 1]
        artifact_path = OUTPUT_DIR / f"{prefix}_artifact_segments.csv"
        artifact.to_csv(artifact_path, index=False)

        print(f"Saved: {scored_path.name}, {clean_path.name}, {artifact_path.name}")

        summary_rows.append(
            {
                "dataset": dataset_name,
                "total_segments": total,
                "flagged_artifact": flagged,
                "flagged_fraction": round(flagged_fraction, 4),
                "clean_movement_candidates": total - flagged,
                "status": "OK",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "integration_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print_header("STEP 7 COMPLETE")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved: {summary_path}")
    print(
        "\nNext layer: 04_movement_detection — consume "
        "*_clean_movement_candidates.csv as input, not raw segments."
    )


if __name__ == "__main__":
    main()