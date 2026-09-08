"""
movement_characterization.py

PHASE 10 — MOVEMENT CHARACTERIZATION
=======================================

Stop counting movements — describe what each one was actually like.

KEY DESIGN DECISION — this does NOT recompute signal features
------------------------------------------------------------------
Phase 7 (artifact_features.py) already computed RMS, peak, energy,
dominant frequency, spectral entropy, per-axis energy contribution,
and multi-sensor overlap for every candidate. These are generic
signal descriptors — they don't stop being valid measurements of
"what did this event look like" just because they were originally
computed to help classify artifacts. Recomputing them here would be
duplicate work and risks two features silently drifting out of sync
with different definitions.

What IS genuinely new here — because it requires looking ACROSS
events or sensors, not just within one candidate's own window:
    - inter_event_interval_seconds: needs the sequence of movements
      in a recording, not a single event in isolation.
    - resultant_direction: Phase 7 has "how concentrated" (directional
      consistency) but never named WHICH axis dominates. This adds
      that label.

Everything else below is Phase 7's existing features, re-presented
under the categories the project's own plan asked for (Temporal /
Amplitude / Frequency / Direction / Spatial), joined onto whichever
movement events survived Phase 9.
"""

from __future__ import annotations

from pathlib import Path

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

MOVEMENT_DIR = PROJECT_ROOT / "data" / "processed" / "movement_detection"
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "artifact_features"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "movement_characterization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["COUGH", "FOUR_IMU", "OXFORD"]

PRIMARY_AXES = {
    "COUGH": ["ax", "ay", "az"],
    "FOUR_IMU": ["ax", "ay", "az"],
    "OXFORD": ["bcg_x", "bcg_y", "bcg_z"],
}


# =============================================================================
# NEW FEATURE 1 — INTER-EVENT INTERVAL
# =============================================================================

def add_inter_event_interval(movements: pd.DataFrame) -> pd.DataFrame:
    """
    Time gap between this movement's start and the previous
    movement's end, WITHIN THE SAME RECORD (and sensor, for
    multi-sensor datasets) — computed on the sequence of movements
    that actually survived Phase 9, not the original candidate pool.
    NaN for the first movement in a record (no previous event to
    measure from).
    """

    movements = movements.copy()
    movements["inter_event_interval_seconds"] = np.nan

    group_columns = ["record_id"]
    if "sensor_id" in movements.columns:
        group_columns.append("sensor_id")

    for _, group in movements.groupby(group_columns):

        ordered = group.sort_values("start_time")
        previous_end = ordered["end_time"].shift(1)
        interval = ordered["start_time"] - previous_end

        movements.loc[ordered.index, "inter_event_interval_seconds"] = interval.to_numpy()

    return movements


# =============================================================================
# NEW FEATURE 2 — RESULTANT DIRECTION
# =============================================================================

def add_resultant_direction(features: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Names WHICH axis dominates an event's energy, using Phase 7's
    existing per-axis contribution columns
    (primary_{axis}_contribution). Phase 7 already tells you HOW
    concentrated an event is (primary_directional_consistency) —
    this adds the label for which direction that concentration is
    actually in.
    """

    features = features.copy()
    axes = PRIMARY_AXES.get(dataset_name, [])
    contribution_columns = [f"primary_{axis}_contribution" for axis in axes]
    available = [c for c in contribution_columns if c in features.columns]

    if not available:
        features["resultant_direction"] = None
        return features

    def resolve(row):
        values = row[available]
        if values.isna().all():
            return None
        best_column = values.idxmax()
        return best_column.replace("primary_", "").replace("_contribution", "")

    features["resultant_direction"] = features.apply(resolve, axis=1)
    return features


# =============================================================================
# ORCHESTRATION
# =============================================================================

FEATURE_COLUMNS_TO_KEEP = [
    # Temporal (start_time/end_time/duration already on the movement record)
    "inter_event_interval_seconds",
    # Amplitude / Energy
    "primary_rms", "primary_peak", "primary_peak_to_peak",
    "primary_std", "primary_variance", "primary_energy", "primary_mean_energy",
    # Frequency
    "primary_dominant_frequency_hz", "primary_spectral_centroid_hz",
    "primary_band_power", "primary_spectral_entropy",
    # Direction
    "primary_directional_consistency", "resultant_direction",
    # Secondary axis group (gyro, where available)
    "secondary_rms", "secondary_peak", "secondary_zero_crossing_rate",
    # Spatial (Four-IMU only)
    "sensors_involved", "max_sensor_overlap_fraction",
]


def process_dataset(dataset_name: str) -> pd.DataFrame | None:

    movements_path = MOVEMENT_DIR / f"{dataset_name.lower()}_movement_events.csv"
    features_path = FEATURES_DIR / f"{dataset_name.lower()}_candidate_features.csv"

    if not movements_path.exists():
        print(f"{dataset_name}: SKIPPED -- {movements_path} not found. Run movement_detection.py first.")
        return None

    if not features_path.exists():
        print(f"{dataset_name}: SKIPPED -- {features_path} not found.")
        return None

    movements = pd.read_csv(movements_path)
    if movements.empty:
        print(f"{dataset_name}: 0 movement events, nothing to characterize.")
        return movements

    features = pd.read_csv(features_path, dtype={"true_label": str})
    features = add_resultant_direction(features, dataset_name)

    join_columns = ["candidate_id"] + [c for c in FEATURE_COLUMNS_TO_KEEP if c in features.columns and c != "inter_event_interval_seconds"]
    characterized = movements.merge(features[join_columns], on="candidate_id", how="left")

    characterized = add_inter_event_interval(characterized)

    output_path = OUTPUT_DIR / f"{dataset_name.lower()}_movement_characteristics.csv"
    characterized.to_csv(output_path, index=False)

    print(f"{dataset_name}: {len(characterized)} movements characterized -> {output_path}")

    return characterized


def main() -> None:

    print("=" * 78)
    print("PHASE 10 — MOVEMENT CHARACTERIZATION")
    print("=" * 78)

    all_results = []

    for dataset_name in DATASETS:
        result = process_dataset(dataset_name)
        if result is not None and not result.empty:
            all_results.append(result)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = OUTPUT_DIR / "all_movement_characteristics.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined: {len(combined)} movements -> {combined_path}")

        print("\nResultant direction distribution:")
        print(combined["resultant_direction"].value_counts(dropna=False))

        print("\nInter-event interval (seconds), summary:")
        print(combined["inter_event_interval_seconds"].describe())

    print("\n" + "=" * 78)
    print("PHASE 10 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
