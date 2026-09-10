"""
behavioral_profiling_audit.py

PHASE 11 VALIDATION / AUDIT
==============================

Not a new pipeline phase — a systematic check on what Phases 9-11
actually produced, before treating any of it as ready for
interpretation. Two things this catches that silent trust would not:

    1. Data integrity: NaN/Inf, negative or overlapping intervals,
       suspiciously huge gaps, near-empty recordings.
    2. Four-IMU cross-sensor overlap: are near-simultaneous events
       across different IMUs on the same recording being counted as
       separate movements, inflating movement_count/burst_count?

Also runs a burst-threshold sensitivity check: BURST_GAP_SECONDS in
behavioral_profiling.py (10.0s) is a provisional engineering choice,
not a scientifically validated definition of a "movement burst".
This reports how burst_count changes at 5s/10s/15s/20s so that
claim is never made by accident.

Output: a single printed report plus
data/processed/behavioral_profiling/phase11_audit_report.txt
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
PROFILES_DIR = PROJECT_ROOT / "data" / "processed" / "behavioral_profiling"

DATASETS = ["COUGH", "FOUR_IMU", "OXFORD"]

CROSS_SENSOR_PROXIMITY_SECONDS = 0.3

BURST_THRESHOLDS_TO_TEST = [5.0, 10.0, 15.0, 20.0]

MIN_MOVEMENTS_FOR_RELIABLE_PROFILE = 3


# =============================================================================
# STEP 1 — INTEGRITY
# =============================================================================

def audit_integrity(movements: pd.DataFrame, dataset_name: str) -> dict:

    report = {"dataset": dataset_name, "total_movements": len(movements)}

    numeric_cols = movements.select_dtypes(include=[np.number]).columns
    report["nan_counts"] = movements[numeric_cols].isna().sum().to_dict()
    report["inf_count"] = int(np.isinf(movements[numeric_cols]).sum().sum())

    if "inter_event_interval_seconds" in movements.columns:
        intervals = movements["inter_event_interval_seconds"].dropna()
        negative = intervals[intervals < 0]
        report["negative_interval_count"] = int(len(negative))
        report["negative_interval_fraction"] = (
            float(len(negative) / len(intervals)) if len(intervals) else np.nan
        )

        id_cols = ["record_id"] + (["sensor_id"] if "sensor_id" in movements.columns else [])
        example_cols = id_cols + ["start_time", "end_time", "inter_event_interval_seconds"]
        report["negative_interval_examples"] = (
            movements.loc[negative.index, example_cols].head(5).to_dict("records")
        )

        huge_gap_threshold = intervals.quantile(0.999) if len(intervals) > 100 else (intervals.max() if len(intervals) else np.nan)
        report["suspiciously_huge_gap_threshold_seconds"] = float(huge_gap_threshold) if pd.notna(huge_gap_threshold) else None
        report["suspiciously_huge_gap_count"] = int((intervals > huge_gap_threshold).sum()) if pd.notna(huge_gap_threshold) else 0

    return report


# =============================================================================
# STEP 2 — FOUR-IMU CROSS-SENSOR OVERLAP
# =============================================================================

def audit_cross_sensor_overlap(movements: pd.DataFrame) -> dict:
    """
    For Four-IMU only: how often do DIFFERENT sensors on the SAME
    record_id report events starting within
    CROSS_SENSOR_PROXIMITY_SECONDS of each other? This does not
    decide whether those are one physical movement seen by multiple
    sensors, or genuinely separate events — it quantifies how often
    the question even arises, which is the prerequisite for
    deciding whether a physical-event aggregation layer is needed.
    """

    if "sensor_id" not in movements.columns:
        return {"applicable": False}

    near_simultaneous_pairs = 0
    records_checked = 0
    records_with_overlap = 0

    for record_id, record_group in movements.groupby("record_id"):

        sensors_here = record_group["sensor_id"].unique()
        if len(sensors_here) < 2:
            continue

        records_checked += 1
        found_overlap_this_record = False

        starts = record_group[["sensor_id", "start_time"]].sort_values("start_time")
        times = starts["start_time"].to_numpy()
        sensors = starts["sensor_id"].to_numpy()

        for i in range(len(times) - 1):
            j = i + 1
            while j < len(times) and times[j] - times[i] <= CROSS_SENSOR_PROXIMITY_SECONDS:
                if sensors[j] != sensors[i]:
                    near_simultaneous_pairs += 1
                    found_overlap_this_record = True
                j += 1

        if found_overlap_this_record:
            records_with_overlap += 1

    return {
        "applicable": True,
        "proximity_threshold_seconds": CROSS_SENSOR_PROXIMITY_SECONDS,
        "records_with_multiple_sensors": records_checked,
        "records_with_near_simultaneous_cross_sensor_events": records_with_overlap,
        "near_simultaneous_cross_sensor_pairs": near_simultaneous_pairs,
        "records_affected_fraction": (
            records_with_overlap / records_checked if records_checked else np.nan
        ),
    }


# =============================================================================
# STEP 3 — BURST-THRESHOLD SENSITIVITY
# =============================================================================

def burst_sensitivity_check(movements: pd.DataFrame) -> pd.DataFrame:
    """
    Recomputes burst_count under several candidate thresholds to
    show how much the "how bursty is this recording" conclusion
    depends on an engineering choice that has not been validated
    against any independent ground truth.
    """

    results = []

    group_columns = ["record_id"]
    if "sensor_id" in movements.columns:
        group_columns.append("sensor_id")

    for threshold in BURST_THRESHOLDS_TO_TEST:

        burst_counts = []

        for _, group in movements.groupby(group_columns):
            group = group.sort_values("start_time")
            is_new_burst = (
                group["inter_event_interval_seconds"].isna()
                | (group["inter_event_interval_seconds"] > threshold)
            )
            burst_counts.append(int(is_new_burst.sum()))

        results.append({
            "threshold_seconds": threshold,
            "mean_burst_count": float(np.mean(burst_counts)) if burst_counts else np.nan,
            "median_burst_count": float(np.median(burst_counts)) if burst_counts else np.nan,
        })

    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    lines = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 78)
    emit("PHASE 11 VALIDATION")
    emit("=" * 78)

    profile_counts = {}
    all_integrity = []

    for dataset_name in DATASETS:

        char_path = CHARACTERISTICS_DIR / f"{dataset_name.lower()}_movement_characteristics.csv"
        profile_path = PROFILES_DIR / f"{dataset_name.lower()}_behavioral_profiles.csv"

        if not char_path.exists():
            emit(f"\n{dataset_name}: SKIPPED -- {char_path} not found.")
            continue

        header_cols = pd.read_csv(char_path, nrows=0).columns
        dtype_arg = {"true_label": str} if "true_label" in header_cols else None
        movements = pd.read_csv(char_path, dtype=dtype_arg)

        if profile_path.exists():
            profile_counts[dataset_name] = len(pd.read_csv(profile_path))

        emit(f"\n--- {dataset_name} ---")

        integrity = audit_integrity(movements, dataset_name)
        all_integrity.append(integrity)

        emit(f"Total movements: {integrity['total_movements']}")
        emit(f"Inf values: {integrity['inf_count']}")

        if "negative_interval_count" in integrity:
            emit(
                f"Negative inter-event intervals: {integrity['negative_interval_count']} "
                f"({integrity['negative_interval_fraction']:.2%})"
            )
            if integrity["negative_interval_count"] > 0:
                emit("  Examples:")
                for ex in integrity["negative_interval_examples"]:
                    emit(f"    {ex}")

        if dataset_name == "FOUR_IMU":
            emit("\nCross-sensor proximity check:")
            overlap = audit_cross_sensor_overlap(movements)
            if overlap.get("applicable"):
                emit(
                    f"  Records with >1 sensor: {overlap['records_with_multiple_sensors']}"
                )
                emit(
                    f"  Records with near-simultaneous (<={overlap['proximity_threshold_seconds']}s) "
                    f"cross-sensor events: {overlap['records_with_near_simultaneous_cross_sensor_events']} "
                    f"({overlap['records_affected_fraction']:.1%})"
                )
                emit(
                    f"  Total near-simultaneous cross-sensor event pairs: "
                    f"{overlap['near_simultaneous_cross_sensor_pairs']}"
                )

            emit("\nBurst-threshold sensitivity (mean burst_count per recording):")
            sensitivity = burst_sensitivity_check(movements)
            emit(sensitivity.to_string(index=False))

        group_cols = ["record_id"] + (["sensor_id"] if "sensor_id" in movements.columns else [])
        low_count_groups = movements.groupby(group_cols).size()
        n_low = int((low_count_groups < MIN_MOVEMENTS_FOR_RELIABLE_PROFILE).sum())
        emit(
            f"\nRecordings/sensor-streams with fewer than "
            f"{MIN_MOVEMENTS_FOR_RELIABLE_PROFILE} movements: {n_low} / {len(low_count_groups)}"
        )

    emit("\n" + "=" * 78)
    emit("SUMMARY")
    emit("=" * 78)
    emit("Profiles: " + ", ".join(f"{k} {v}" for k, v in profile_counts.items()))
    emit(f"TOTAL: {sum(profile_counts.values())}")

    total_negative = sum(i.get("negative_interval_count", 0) for i in all_integrity)
    emit(f"\nNegative-interval rows found across all datasets: {total_negative}")
    if total_negative > 0:
        emit(
            "ACTION REQUIRED: negative intervals indicate a data-integrity "
            "issue upstream, not a Phase 11 bug on its own -- traced to a "
            "Four-IMU record_id collision across sub-datasets, now fixed in "
            "dataset.py. Re-run the full pipeline from candidate_generation.py "
            "onward for Four-IMU after applying that fix, then re-run this "
            "audit and confirm this count drops to (near) zero."
        )

    emit("\nSTATUS: ARCHITECTURE VALIDATED / BEHAVIORAL INTERPRETATION NOT YET VALIDATED")
    emit(
        "Reasons: (1) no trustworthy artifact classifier exists yet -- all "
        "movements are provisional; (2) BURST_GAP_SECONDS is a provisional "
        "engineering threshold, not a validated definition -- see the "
        "sensitivity table above; (3) if any negative intervals were found, "
        "re-run after the record_id fix before trusting Four-IMU numbers."
    )

    report_path = PROFILES_DIR / "phase11_audit_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
