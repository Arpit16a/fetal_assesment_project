"""
behavioral_profiling.py

PHASE 11 — BEHAVIORAL PROFILING
==================================

Phase 10 produced one row per MOVEMENT. This phase collapses all of
one recording's movements into one row describing the PATTERN —
this is the layer that actually answers the project's founding
question: two recordings can have the identical movement count and
still behave completely differently (evenly spread all recording
long vs. long quiet stretches punctuated by bursts). Counting alone
cannot tell those apart; this can.

Four behavioral dimensions computed per (dataset, subject_id,
record_id):

    1. ACTIVITY LEVEL
       movement count, and movements-per-minute using the span
       between the first and last movement as a PROXY for observed
       duration (not the recording's true total length — that
       would need joining against the timing/segment layer's actual
       duration, which this script does not yet do; documented as
       a proxy, not silently treated as exact).

    2. TEMPORAL STRUCTURE (bursty vs. evenly spread)
       Coefficient of variation of inter-event intervals (std/mean)
       — a standard, simple burstiness indicator: low = regular
       spacing, high = clustered. Movements are also grouped into
       discrete "bursts" (consecutive movements closer together
       than BURST_GAP_SECONDS) and burst count/size is reported.

    3. MOVEMENT REPERTOIRE (how varied are the movements themselves)
       Spread (std) of duration, peak strength, and dominant
       frequency across the recording's movements, plus which
       direction (resultant_direction) dominates.

    4. TEMPORAL EVOLUTION (does activity change across the recording)
       Movement counts in the early/mid/late thirds of the observed
       span. Deliberately reports raw counts, NOT an "increasing/
       decreasing" trend label — with the small movement counts many
       recordings will have, declaring a confident trend from 3 bins
       is overclaiming. Let a human, or a later properly-validated
       method with enough data, decide what a trend means here.

HONESTY CARRY-FORWARD — READ BEFORE PRESENTING ANY OUTPUT OF THIS
SCRIPT
-------------------------------------------------------------------
Every movement feeding this is still PROVISIONAL (movement_detection.py's
artifact_filtered column) because no trustworthy artifact classifier
exists yet. A behavioral profile built from provisional movements
can still include real artifacts as if they were fetal activity.
This script does not hide that: every output file's summary states
the artifact_filtered fraction of what it was built from.
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

CHARACTERISTICS_DIR = PROJECT_ROOT / "data" / "processed" / "movement_characterization"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "behavioral_profiling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["COUGH", "FOUR_IMU", "OXFORD"]

# Movements closer together than this are grouped into the same
# "burst" rather than reported as separate isolated events. Not the
# same threshold as candidate_generation.py's merge_gap_seconds
# (0.5s, for merging raw detection fragments of the same event) —
# this is a much coarser, behavior-level grouping of genuinely
# distinct movements that happen to cluster in time.
BURST_GAP_SECONDS = 10.0


# =============================================================================
# PER-RECORDING PROFILE
# =============================================================================

def profile_one_recording(group: pd.DataFrame) -> dict:

    group = group.sort_values("start_time").reset_index(drop=True)

    movement_count = len(group)
    span_seconds = float(group["end_time"].max() - group["start_time"].min())
    movements_per_minute = (
        (movement_count / span_seconds) * 60.0 if span_seconds > 0 else np.nan
    )

    # --- Temporal structure ---
    intervals = group["inter_event_interval_seconds"].dropna()

    if len(intervals) >= 2:
        interval_cv = float(intervals.std() / intervals.mean()) if intervals.mean() > 0 else np.nan
    else:
        interval_cv = np.nan

    # Burst grouping: start a new burst whenever the gap since the
    # previous movement exceeds BURST_GAP_SECONDS (or is unknown —
    # the first movement always starts a burst).
    is_new_burst = (
        group["inter_event_interval_seconds"].isna()
        | (group["inter_event_interval_seconds"] > BURST_GAP_SECONDS)
    )
    burst_id = is_new_burst.cumsum()
    burst_sizes = burst_id.value_counts()

    burst_count = int(burst_id.nunique())
    mean_burst_size = float(burst_sizes.mean())
    max_burst_size = int(burst_sizes.max())

    # --- Movement repertoire ---
    duration_std = float(group["duration"].std()) if movement_count > 1 else 0.0
    peak_std = (
        float(group["primary_peak"].std())
        if "primary_peak" in group.columns and movement_count > 1
        else np.nan
    )
    dominant_freq_std = (
        float(group["primary_dominant_frequency_hz"].std())
        if "primary_dominant_frequency_hz" in group.columns and movement_count > 1
        else np.nan
    )

    direction_counts = (
        group["resultant_direction"].value_counts().to_dict()
        if "resultant_direction" in group.columns
        else {}
    )
    dominant_direction = max(direction_counts, key=direction_counts.get) if direction_counts else None

    # --- Temporal evolution: raw per-third counts, no trend claim ---
    if span_seconds > 0:
        relative_position = (group["start_time"] - group["start_time"].min()) / span_seconds
        third = pd.cut(
            relative_position, bins=[-0.001, 1/3, 2/3, 1.001],
            labels=["early", "mid", "late"],
        )
        third_counts = third.value_counts().reindex(["early", "mid", "late"]).fillna(0).astype(int)
    else:
        third_counts = pd.Series({"early": movement_count, "mid": 0, "late": 0})

    # --- Honesty carry-forward ---
    artifact_filtered_fraction = (
        float(group["artifact_filtered"].mean())
        if "artifact_filtered" in group.columns
        else np.nan
    )

    return {
        "movement_count": movement_count,
        "span_seconds": span_seconds,
        "movements_per_minute": movements_per_minute,

        "interval_cv": interval_cv,
        "burst_count": burst_count,
        "mean_burst_size": mean_burst_size,
        "max_burst_size": max_burst_size,

        "duration_std_seconds": duration_std,
        "peak_amplitude_std": peak_std,
        "dominant_frequency_std_hz": dominant_freq_std,
        "dominant_direction": dominant_direction,
        "direction_counts": direction_counts,

        "movements_early_third": int(third_counts["early"]),
        "movements_mid_third": int(third_counts["mid"]),
        "movements_late_third": int(third_counts["late"]),

        "artifact_filtered_fraction": artifact_filtered_fraction,
    }


# =============================================================================
# ORCHESTRATION
# =============================================================================

def process_dataset(dataset_name: str) -> pd.DataFrame | None:

    path = CHARACTERISTICS_DIR / f"{dataset_name.lower()}_movement_characteristics.csv"

    if not path.exists():
        print(f"{dataset_name}: SKIPPED -- {path} not found. Run movement_characterization.py first.")
        return None

    movements = pd.read_csv(path)

    if movements.empty:
        print(f"{dataset_name}: 0 movements, nothing to profile.")
        return None

    required = ["subject_id", "record_id", "start_time", "end_time", "duration"]
    missing = [c for c in required if c not in movements.columns]
    if missing:
        print(f"{dataset_name}: SKIPPED -- missing required columns: {missing}")
        return None

    profiles = []
    for (subject_id, record_id), group in movements.groupby(["subject_id", "record_id"]):
        profile = profile_one_recording(group)
        profile["dataset"] = dataset_name
        profile["subject_id"] = subject_id
        profile["record_id"] = record_id
        profiles.append(profile)

    profiles_df = pd.DataFrame(profiles)

    output_path = OUTPUT_DIR / f"{dataset_name.lower()}_behavioral_profiles.csv"
    profiles_df.to_csv(output_path, index=False)

    print(f"{dataset_name}: {len(profiles_df)} recordings profiled -> {output_path}")

    return profiles_df


def main() -> None:

    print("=" * 78)
    print("PHASE 11 — BEHAVIORAL PROFILING")
    print("=" * 78)

    all_profiles = []

    for dataset_name in DATASETS:
        profiles_df = process_dataset(dataset_name)
        if profiles_df is not None:
            all_profiles.append(profiles_df)

    if not all_profiles:
        print("\nNothing to combine -- no dataset produced profiles.")
        return

    combined = pd.concat(all_profiles, ignore_index=True)
    combined_path = OUTPUT_DIR / "all_behavioral_profiles.csv"
    combined.to_csv(combined_path, index=False)

    print(f"\nCombined: {len(combined)} recording profiles -> {combined_path}")

    print("\nMovements-per-minute distribution:")
    print(combined["movements_per_minute"].describe())

    print("\nBurstiness (interval_cv) distribution -- higher = more clustered:")
    print(combined["interval_cv"].describe())

    known_artifact_status = combined["artifact_filtered_fraction"].notna()
    mean_filtered = combined.loc[known_artifact_status, "artifact_filtered_fraction"].mean()

    print("\n" + "=" * 78)
    print("PHASE 11 COMPLETE")
    print("=" * 78)
    print(
        f"\nIMPORTANT: these profiles are built from PROVISIONAL movement "
        f"events -- {mean_filtered:.0%} average artifact-filtered fraction "
        f"across recordings with known status (NaN = status entirely "
        f"unknown). No trustworthy artifact classifier exists yet (see "
        f"Phase 8/9). Treat activity levels and burst patterns here as "
        f"pipeline output to validate the architecture, not as confirmed "
        f"fetal behavior, until Phase 8 has real human-reviewed labels."
    )


if __name__ == "__main__":
    main()
