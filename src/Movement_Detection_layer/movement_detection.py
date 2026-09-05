"""
movement_detection.py

PHASE 9 — MOVEMENT DETECTION
==============================

The question changes from "is this a trustworthy signal / is this an
artifact?" (Phase 6-8) to "is this a genuine fetal movement event?"

KEY DESIGN DECISION — read before changing anything here
-----------------------------------------------------------
This does NOT re-run event detection from scratch. Phase 6
(candidate_generation.py) already finds precise start/end boundaries
for every activity burst. A "movement event" in this project IS a
Phase 6 candidate that survives artifact filtering — nothing more.
Re-detecting boundaries a second time here would duplicate Phase 6's
job and risk a second, inconsistent definition of "event."

TWO OPERATING MODES — this project currently has no trained,
deployable artifact model (Phase 8 correctly refuses to save one
until real human-reviewed labels exist). Rather than block this
entire phase on that, this script runs in whichever mode the
evidence on disk actually supports, and is explicit about which one
ran:

  FILTERED   — a real trained model (best_model.joblib) exists, or
               real human-reviewed labels (label_source == "manual")
               exist for some candidates. Those candidates are
               filtered by real evidence. is_artifact predictions
               come with a probability; artifact_filtered=True.

  UNFILTERED — no trustworthy filter exists for a candidate (today's
               actual state for effectively all candidates). It is
               STILL included as a provisional movement event, but
               explicitly tagged artifact_filtered=False. This is
               not the same claim as "confirmed movement" — treat
               downstream analysis of unfiltered movements as
               provisional until re-run with a real filter.

The moment a real model is trained (Phase 8, once real labels
exist), re-running this script automatically starts filtering —
no code change required here.

Output schema (per movement event), as specified:
    movement_id, subject_id, record_id, start_time, end_time,
    duration
plus traceability/audit columns: dataset, candidate_id, sensor_id,
artifact_filtered, filter_method, artifact_probability (NaN if
unfiltered).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
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
LABELS_DIR = PROJECT_ROOT / "data" / "processed" / "artifact_labels"
BASELINE_DIR = PROJECT_ROOT / "data" / "processed" / "artifact_baseline"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "movement_detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["COUGH", "FOUR_IMU", "OXFORD"]

# Probability threshold above which a candidate is filtered out as
# an artifact, when a real trained model is available to score it.
ARTIFACT_PROBABILITY_THRESHOLD = 0.5


# =============================================================================
# LOAD WHATEVER REAL FILTERING EVIDENCE EXISTS
# =============================================================================

def load_trained_model():
    """Returns (model, feature_columns) or (None, None) if no real,
    deployable model has been trained yet (Phase 8's correct gate)."""

    model_path = BASELINE_DIR / "best_model.joblib"
    features_path = BASELINE_DIR / "best_model_features.json"

    if not model_path.exists() or not features_path.exists():
        return None, None

    model = joblib.load(model_path)
    feature_columns = json.loads(features_path.read_text())
    return model, feature_columns


def load_manual_label_lookup() -> dict:
    """candidate_id -> is_artifact, for REAL human-reviewed labels
    only (label_source == "manual"). Heuristic-sourced labels are
    deliberately excluded here — see label_validation.py's own
    circularity discussion; they are not trustworthy evidence for
    filtering real movement events out or in."""

    validated_path = LABELS_DIR / "validated_dataset.csv"
    if not validated_path.exists():
        return {}

    df = pd.read_csv(validated_path)
    if "label_source" not in df.columns or "candidate_id" not in df.columns:
        return {}

    manual = df[df["label_source"] == "manual"]
    return dict(zip(manual["candidate_id"], manual["is_artifact"]))


# =============================================================================
# FILTER ONE DATASET'S CANDIDATES INTO MOVEMENT EVENTS
# =============================================================================

def filter_to_movement_events(
    candidates: pd.DataFrame,
    model,
    feature_columns: list[str] | None,
    manual_lookup: dict,
) -> pd.DataFrame:

    result = candidates.copy()
    result["artifact_filtered"] = False
    result["filter_method"] = "unfiltered"
    result["artifact_probability"] = np.nan
    result["is_movement"] = True  # default: included, provisionally

    # Priority 1 — real trained model, if one exists.
    if model is not None and feature_columns is not None:
        for col in feature_columns:
            if col not in result.columns:
                result[col] = np.nan

        proba = model.predict_proba(result[feature_columns])[:, 1]
        result["artifact_probability"] = proba
        result["is_movement"] = proba < ARTIFACT_PROBABILITY_THRESHOLD
        result["artifact_filtered"] = True
        result["filter_method"] = "trained_model"

    # Priority 2 — real manual labels, for whichever specific
    # candidates have them (independent of whether a model exists —
    # a direct human label always overrides a model's prediction for
    # that specific candidate).
    if manual_lookup:
        has_manual = result["candidate_id"].isin(manual_lookup)
        manual_is_artifact = result.loc[has_manual, "candidate_id"].map(manual_lookup)

        result.loc[has_manual, "is_movement"] = manual_is_artifact.to_numpy() == 0
        result.loc[has_manual, "artifact_filtered"] = True
        result.loc[has_manual, "filter_method"] = "manual_label"
        result.loc[has_manual, "artifact_probability"] = manual_is_artifact.to_numpy()

    return result[result["is_movement"]].drop(columns=["is_movement"])


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    model, feature_columns = load_trained_model()
    manual_lookup = load_manual_label_lookup()

    print("=" * 78)
    print("PHASE 9 — MOVEMENT DETECTION")
    print("=" * 78)
    print(f"Trained artifact model available: {model is not None}")
    print(f"Real manual labels available: {len(manual_lookup)} candidates")
    if model is None and not manual_lookup:
        print(
            "\nWARNING: no trustworthy artifact filter exists yet. "
            "Every candidate below will be included as a PROVISIONAL "
            "movement event (artifact_filtered=False). This is a "
            "pipeline-readiness run, not a clean result -- re-run "
            "after Phase 8 has real labels and a saved model."
        )

    all_movements = []
    summary_rows = []

    for dataset_name in DATASETS:

        features_path = FEATURES_DIR / f"{dataset_name.lower()}_candidate_features.csv"

        if not features_path.exists():
            print(f"\n{dataset_name}: SKIPPED -- {features_path} not found")
            summary_rows.append({"dataset": dataset_name, "status": "SKIPPED"})
            continue

        candidates = pd.read_csv(features_path, dtype={"true_label": str})
        if "dataset" not in candidates.columns:
            candidates["dataset"] = dataset_name

        movements = filter_to_movement_events(candidates, model, feature_columns, manual_lookup)

        movements = movements.rename(columns={
            "start_timestamp": "start_time",
            "end_timestamp": "end_time",
            "duration_seconds": "duration",
        })
        movements.insert(0, "movement_id", [
            f"{dataset_name}_mv{i:06d}" for i in range(1, len(movements) + 1)
        ])

        output_columns = [
            "movement_id", "subject_id", "record_id", "start_time", "end_time",
            "duration", "dataset", "candidate_id", "sensor_id",
            "artifact_filtered", "filter_method", "artifact_probability",
        ]
        output_columns = [c for c in output_columns if c in movements.columns]
        movements = movements[output_columns]

        output_path = OUTPUT_DIR / f"{dataset_name.lower()}_movement_events.csv"
        movements.to_csv(output_path, index=False)

        filtered_count = (movements["artifact_filtered"]).sum() if "artifact_filtered" in movements.columns else 0

        print(
            f"\n{dataset_name}: {len(candidates)} candidates -> "
            f"{len(movements)} movement events "
            f"({filtered_count} actually artifact-filtered, "
            f"{len(movements) - filtered_count} still unfiltered/provisional)"
        )
        print(f"Saved: {output_path}")

        all_movements.append(movements)
        summary_rows.append({
            "dataset": dataset_name,
            "status": "OK",
            "candidates_in": len(candidates),
            "movement_events_out": len(movements),
            "actually_filtered": int(filtered_count),
        })

    if all_movements:
        combined = pd.concat(all_movements, ignore_index=True)
        combined_path = OUTPUT_DIR / "all_movement_events.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined: {len(combined)} movement events -> {combined_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "movement_detection_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("PHASE 9 SUMMARY")
    print("=" * 78)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
