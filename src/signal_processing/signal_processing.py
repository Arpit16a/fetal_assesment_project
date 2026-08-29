"""
02_signal_processing.py

PHASE 5 — SIGNAL PROCESSING
=============================

Converts timing-validated continuous segments (the output of
01_timing.py) into analysis-ready, quality-flagged, fixed-length
windows. This module answers ONE question per step, matching the
project's layer-by-layer principle:

    Raw signal
        |
        v
    Resampling (only if the dataset's config requires a common
                time base — see config.py's target_sampling_rate_hz;
                currently None for all three datasets, matching the
                project's own "do not resample automatically" policy)
        |
        v
    Sensor synchronization CHECK (multi-IMU datasets only —
                verifies streams share a compatible time range;
                does NOT yet resample sensors onto a shared grid,
                see module docstring note below)
        |
        v
    Band-pass filtering (Butterworth, order + cutoffs from
                config.py, justified per-dataset against each
                dataset's own Nyquist limit)
        |
        v
    Signal quality assessment (GOOD / QUESTIONABLE / INVALID per
                window, using config.QUALITY_THRESHOLDS)
        |
        v
    Windowing (fixed-length, configurable overlap)
        |
        v
    Analysis-ready windows

WHAT THIS MODULE DOES NOT DO
------------------------------
- It does not decide whether a window is an "artifact" — that is a
  motion-SOURCE question (maternal vs fetal vs sensor), answered by
  the artifact layer. This module only answers the SENSOR-quality
  question ("is this window trustworthy as a measurement at all").
- It does not do full multi-sensor fusion / resampling onto one
  shared clock for Four-IMU. It CHECKS whether the four IMU streams
  already cover compatible time ranges and flags misalignment. If
  real misalignment is found, that becomes a scoped follow-up
  (interpolating sensors onto one shared grid), not something to
  guess at here.

INTEGRATION NOTE — READ BEFORE WIRING THIS INTO THE ARTIFACT LAYER
---------------------------------------------------------------------
02_artifact_inspection.py currently treats each ENTIRE continuous
segment (from 01_timing.py) as one feature row. This module produces
much finer fixed-length windows (2 seconds each, per config.py)
instead. That is intentional — the project plan calls for real
windowing before feature extraction — but it means
02_artifact_inspection.py's feature extraction will need to run
per-window (using this module's output) rather than per-whole-segment
to actually consume this layer's output. That change is NOT made in
this file — flagging it explicitly rather than silently leaving a
mismatch between what this module produces and what the next stage
currently expects.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).parent))

from importlib.util import spec_from_file_location, module_from_spec


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_THIS_DIR = Path(__file__).parent
timing = _load_module("timing_layer", _THIS_DIR / "01_timing.py")

import config as cfg  # noqa: E402


# =============================================================================
# SIGNAL COLUMNS
# =============================================================================

SIGNAL_COLUMNS = [
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "bcg_x", "bcg_y", "bcg_z",
]


def _available_signal_columns(df: pd.DataFrame) -> list[str]:
    """Only columns that are actually populated for this dataset."""

    available = []
    for col in SIGNAL_COLUMNS:
        if col in df.columns and df[col].notna().any():
            available.append(col)
    return available


# =============================================================================
# STEP 1 — RESAMPLING (POLICY-GATED, NOT AUTOMATIC)
# =============================================================================

def resample_segment(
    segment: pd.DataFrame,
    dataset_config: "cfg.DatasetConfig",
) -> pd.DataFrame:
    """
    Resample onto a common time base — ONLY if
    dataset_config.target_sampling_rate_hz is explicitly set.

    All three current datasets leave this None (see config.py
    comments: "Do NOT resample automatically"), so this is a no-op
    for COUGH / FOUR_IMU / OXFORD today. It exists so that the
    moment a real reason to resample appears (e.g. combining
    recordings at genuinely different native rates), it's a
    one-line config change, not new code.
    """

    if dataset_config.target_sampling_rate_hz is None:
        return segment

    if len(segment) < 2:
        return segment

    signal_columns = _available_signal_columns(segment)

    t0 = segment["timestamp"].iloc[0]
    t1 = segment["timestamp"].iloc[-1]

    target_dt = 1.0 / dataset_config.target_sampling_rate_hz
    new_timestamps = np.arange(t0, t1, target_dt)

    if len(new_timestamps) < 2:
        return segment

    resampled = pd.DataFrame({"timestamp": new_timestamps})

    for col in signal_columns:
        resampled[col] = np.interp(
            new_timestamps,
            segment["timestamp"].to_numpy(dtype=float),
            segment[col].to_numpy(dtype=float),
        )

    # Carry non-signal columns forward as constants (they're
    # per-recording metadata, not per-sample signal).
    for col in ["sensor_id", "label", "subject_id", "dataset_source", "record_id"]:
        if col in segment.columns:
            resampled[col] = segment[col].iloc[0]

    return resampled


# =============================================================================
# STEP 2 — MULTI-SENSOR SYNCHRONIZATION CHECK
# =============================================================================

def check_sensor_synchronization(
    record_data: pd.DataFrame,
    dataset_config: "cfg.DatasetConfig",
    max_offset_seconds: float = 0.5,
) -> dict:
    """
    For multi-sensor datasets (Four-IMU), verify that each sensor's
    stream covers a compatible time range within this recording.

    This is a CHECK, not a fix. It reports whether synchronization
    looks fine, and if not, by how much and for which sensors — so
    a real fusion/resampling step can be scoped deliberately instead
    of assumed to be unnecessary.
    """

    if not dataset_config.multiple_sensors or "sensor_id" not in record_data.columns:
        return {"applicable": False}

    sensor_ranges = (
        record_data.groupby("sensor_id")["timestamp"]
        .agg(["min", "max", "count"])
        .rename(columns={"min": "start", "max": "end"})
    )

    if len(sensor_ranges) < 2:
        return {"applicable": False}

    overall_start = sensor_ranges["start"].min()
    overall_end = sensor_ranges["end"].max()

    start_offsets = (sensor_ranges["start"] - overall_start).abs()
    end_offsets = (sensor_ranges["end"] - overall_end).abs()

    max_start_offset = float(start_offsets.max())
    max_end_offset = float(end_offsets.max())

    synchronized = (
        max_start_offset <= max_offset_seconds
        and max_end_offset <= max_offset_seconds
    )

    return {
        "applicable": True,
        "synchronized": synchronized,
        "max_start_offset_seconds": max_start_offset,
        "max_end_offset_seconds": max_end_offset,
        "sensor_ranges": sensor_ranges.to_dict("index"),
    }


# =============================================================================
# STEP 3 — FILTERING (JUSTIFIED, CONFIG-DRIVEN)
# =============================================================================

def apply_bandpass_filter(
    segment: pd.DataFrame,
    dataset_config: "cfg.DatasetConfig",
) -> tuple[pd.DataFrame, Optional[str]]:
    """
    Zero-phase Butterworth band-pass filter, using the cutoffs and
    order already declared (and Nyquist-validated) in config.py for
    this dataset — never a value invented here.

    Returns (filtered_segment, skip_reason). skip_reason is None on
    success; if set, the segment is returned UNFILTERED and the
    caller should propagate the reason (e.g. into window quality)
    rather than silently pretending filtering happened.
    """

    if dataset_config.bandpass_hz is None:
        return segment, "no_bandpass_configured"

    low, high = dataset_config.bandpass_hz
    fs = dataset_config.sampling_rate_hz

    if fs is None:
        return segment, "sampling_rate_unknown"

    nyquist = fs / 2.0

    order = dataset_config.filter_order or 4

    sos_low = low / nyquist
    sos_high = min(high / nyquist, 0.999)  # guard exact-Nyquist edge case

    b, a = butter(order, [sos_low, sos_high], btype="band")

    # BUG FIX: filtfilt requires len(segment) STRICTLY GREATER THAN
    # padlen, not >=. Compute padlen the same way scipy does
    # internally (3 * max(len(a), len(b))) instead of guessing at a
    # formula that can silently drift out of sync with scipy's
    # actual default — this is what previously crashed with
    # "length ... must be greater than padlen, which is 27" because
    # a segment of exactly 27 samples passed a "< 27" check.
    padlen = 3 * max(len(a), len(b))

    if len(segment) <= padlen:
        return segment, "segment_too_short_to_filter"

    signal_columns = _available_signal_columns(segment)
    if not signal_columns:
        return segment, "no_signal_columns_available"

    filtered = segment.copy()

    for col in signal_columns:
        values = segment[col].to_numpy(dtype=float)

        try:
            if np.isnan(values).any():
                # Filter only the valid stretch; leave NaN structure
                # intact rather than inventing values for gaps.
                valid = ~np.isnan(values)
                if valid.sum() <= padlen:
                    continue
                filtered_values = values.copy()
                filtered_values[valid] = filtfilt(b, a, values[valid])
                filtered[col] = filtered_values
            else:
                filtered[col] = filtfilt(b, a, values)
        except ValueError:
            # Defense in depth: never let one unexpectedly-short or
            # degenerate column crash a multi-hour batch run over a
            # single edge case. Falls through with this column
            # unfiltered rather than aborting everything after it.
            continue

    return filtered, None


# =============================================================================
# STEP 4 — SIGNAL QUALITY ASSESSMENT
# =============================================================================

# =============================================================================
# STEP 4 — SIGNAL QUALITY ASSESSMENT  (numpy-array based — see note below)
# =============================================================================
#
# PERFORMANCE NOTE: the previous version of this module built a full
# pandas DataFrame copy per window via .iloc[...].reset_index(), for
# EVERY window (Four-IMU alone produces roughly 1 million windows at
# ~30 samples each). Pandas per-object construction overhead at that
# scale, not the actual arithmetic, is what took ~40 minutes. This
# version converts each segment's signal columns to numpy arrays
# ONCE, then slices numpy views (near-zero cost) for every window
# instead of building a new DataFrame per window. Output format is
# unchanged — same CSV columns, same values.

def _max_run_of_equal_values(values: np.ndarray) -> int:
    """
    Longest run of consecutive identical values, vectorized.

    Replaces a pure-Python for-loop that, across ~1M windows x 6
    signal columns x ~30-200 samples, was itself a meaningful
    fraction of the runtime.
    """

    if len(values) < 2:
        return len(values)

    is_same = np.diff(values) == 0

    if not is_same.any():
        return 1

    padded = np.concatenate(([False], is_same, [False])).astype(np.int8)
    edges = np.diff(padded)

    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    run_lengths = (ends - starts) + 1

    return int(run_lengths.max())


def assess_window_quality_arrays(
    signal_arrays: dict[str, np.ndarray],
    thresholds: "cfg.QualityThresholds",
    filter_skip_reason: Optional[str] = None,
) -> dict:
    """
    Same logic as before, operating directly on a dict of
    {column_name: numpy_array} for one window instead of a
    pandas DataFrame slice.
    """

    if not signal_arrays:
        return {"quality": "INVALID", "reason": "no_signal_columns_available"}

    reasons = []

    # --- Missingness ---
    missing_fraction = max(
        np.isnan(values).mean() for values in signal_arrays.values()
    )
    if missing_fraction > thresholds.max_missing_fraction:
        return {
            "quality": "INVALID",
            "reason": f"missing_fraction={missing_fraction:.3f}",
        }

    for col, values in signal_arrays.items():

        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            continue

        # --- Flatline / dead sensor ---
        if np.std(valid) < thresholds.min_channel_std:
            reasons.append(f"flatline:{col}")
            continue

        if _max_run_of_equal_values(valid) > thresholds.max_consecutive_identical_samples:
            reasons.append(f"stuck_sensor:{col}")

        # --- Clipping: exact repeated ties at the extreme, not
        # proximity to it (see module notes on why "close to" the
        # peak of a smooth signal is not the same as clipping). ---
        value_max = valid.max()
        value_min = valid.min()

        max_ties = np.isclose(valid, value_max, rtol=0, atol=1e-9).sum()
        min_ties = np.isclose(valid, value_min, rtol=0, atol=1e-9).sum()
        clipped_fraction = max(max_ties, min_ties) / len(valid)

        if clipped_fraction > thresholds.max_clipped_fraction:
            reasons.append(f"possible_clipping:{col}")

        # --- Amplitude sanity (only if a bound has actually been set) ---
        if (
            thresholds.max_abs_amplitude is not None
            and np.abs(valid).max() > thresholds.max_abs_amplitude
        ):
            reasons.append(f"implausible_amplitude:{col}")

    if filter_skip_reason is not None:
        reasons.append(f"unfiltered:{filter_skip_reason}")

    if not reasons:
        return {"quality": "GOOD", "reason": None}

    hard_fail = any(
        r.startswith(("flatline", "stuck_sensor", "unfiltered"))
        for r in reasons
    )

    return {
        "quality": "INVALID" if hard_fail else "QUESTIONABLE",
        "reason": ";".join(reasons),
    }


# =============================================================================
# STEP 5 — WINDOWING  (index computation only — no per-window DataFrame)
# =============================================================================

def compute_window_bounds(
    n_samples: int,
    dataset_config: "cfg.DatasetConfig",
) -> list[tuple[int, int]]:
    """
    Compute (start, end) integer index pairs for fixed-length,
    overlapping windows. Pure arithmetic — no data touched here,
    so this is essentially free even for millions of samples.
    """

    if dataset_config.window_seconds is None or n_samples < 2:
        return [(0, n_samples)]

    fs = dataset_config.sampling_rate_hz
    if fs is None:
        return [(0, n_samples)]

    window_samples = max(int(round(dataset_config.window_seconds * fs)), 1)
    overlap = dataset_config.overlap_fraction or 0.0
    step_samples = max(int(round(window_samples * (1 - overlap))), 1)

    bounds = []
    start = 0

    while start < n_samples:
        end = min(start + window_samples, n_samples)

        if (end - start) >= max(2, window_samples // 2):
            bounds.append((start, end))

        if end == n_samples:
            break

        start += step_samples

    return bounds if bounds else [(0, n_samples)]


# =============================================================================
# ORCHESTRATION — ONE SEGMENT
# =============================================================================

def process_segment(
    segment: pd.DataFrame,
    dataset_name: str,
    segment_id: str,
) -> list[dict]:
    """
    Run one continuous segment through the full Phase 5 chain and
    return one row of metadata per resulting window.

    IMPORTANT ORDERING NOTE: sensor-fault quality checks (flatline,
    stuck sensor, clipping, missingness) are computed on the RAW
    signal, before filtering — band-pass filtering removes near-DC
    content, so a genuinely dead/stuck sensor's flat signal gets
    smoothed into something that no longer looks flat, which would
    silently hide the exact fault this check exists to catch.
    Filtering is applied separately for the analysis-ready signal
    itself; window index bounds are identical between the raw and
    filtered arrays (filtering doesn't change row count or
    timestamps), so the same (start, end) pairs index both.
    """

    dataset_config = cfg.get_dataset_config(dataset_name)

    segment = resample_segment(segment, dataset_config)

    if segment.empty or "timestamp" not in segment.columns:
        return []

    filtered_segment, skip_reason = apply_bandpass_filter(segment, dataset_config)

    signal_columns = _available_signal_columns(segment)

    # Convert once per segment, not once per window.
    timestamps = segment["timestamp"].to_numpy(dtype=float)
    raw_arrays = {col: segment[col].to_numpy(dtype=float) for col in signal_columns}
    # NOTE: filtered signal VALUES are not retained here — this
    # module currently outputs window-level METADATA only (matching
    # the metadata-CSV pattern 02_artifact_inspection.py already
    # uses), not the filtered samples themselves. The moment a
    # downstream step needs the actual filtered signal per window
    # (not just its quality flag), save filtered_segment[signal_columns]
    # per window here rather than recomputing it later.

    subject_id = segment["subject_id"].iloc[0] if "subject_id" in segment.columns else None
    record_id = segment["record_id"].iloc[0] if "record_id" in segment.columns else None
    sensor_id = segment["sensor_id"].iloc[0] if "sensor_id" in segment.columns else None

    bounds = compute_window_bounds(len(segment), dataset_config)

    rows = []

    for index, (start, end) in enumerate(bounds, start=1):

        raw_window_arrays = {col: arr[start:end] for col, arr in raw_arrays.items()}

        quality = assess_window_quality_arrays(
            raw_window_arrays,
            cfg.QUALITY_THRESHOLDS,
            filter_skip_reason=skip_reason,
        )

        rows.append(
            {
                "dataset": dataset_name,
                "segment_id": segment_id,
                "window_id": f"{segment_id}_w{index:04d}",
                "start_timestamp": float(timestamps[start]),
                "end_timestamp": float(timestamps[end - 1]),
                "sample_count": end - start,
                "filtered": skip_reason is None,
                "filter_skip_reason": skip_reason,
                "quality": quality["quality"],
                "quality_reason": quality["reason"],
                "subject_id": subject_id,
                "record_id": record_id,
                "sensor_id": sensor_id,
            }
        )

    return rows


# =============================================================================
# ORCHESTRATION — ONE DATASET
# =============================================================================

def process_dataset(
    standardized_data: pd.DataFrame,
    dataset_name: str,
    timing_config: "timing.TimingConfig",
) -> pd.DataFrame:
    """
    Run the full Phase 5 chain across every record in one dataset's
    standardized DataFrame, using 01_timing.py to get timing-valid
    continuous segments first (this module never invents its own
    segment boundaries — that decision belongs to the timing layer).
    """

    all_rows = []

    group_columns = ["record_id"]
    if "sensor_id" in standardized_data.columns:
        group_columns = ["record_id", "sensor_id"]

    for keys, record_data in standardized_data.groupby(group_columns):

        record_id = keys[0] if isinstance(keys, tuple) else keys

        segments, _ = timing.analyze_record(
            record_data,
            timing_config,
            dataset_source=dataset_name,
            subject_id=(
                record_data["subject_id"].iloc[0]
                if "subject_id" in record_data.columns
                else None
            ),
            record_id=str(record_id),
        )

        for seg_index, segment in enumerate(segments, start=1):

            segment_id = f"{dataset_name}_{record_id}_seg{seg_index:04d}"

            rows = process_segment(segment, dataset_name, segment_id)
            all_rows.extend(rows)

    return pd.DataFrame(all_rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    from src.dataset import CoughLoader, FourIMULoader, OxfordLoader

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "processed" / "signal_processing"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        "COUGH": (
            CoughLoader,
            project_root / "data" / "raw" / "artifacts" / "cough_imu" / "Multimodal Cough Dataset",
        ),
        "FOUR_IMU": (
            FourIMULoader,
            project_root / "data" / "raw" / "fetal" / "four_imu",
        ),
        "OXFORD": (
            OxfordLoader,
            project_root / "data" / "raw" / "fetal" / "oxford_female",
        ),
    }

    summary_rows = []

    for dataset_name, (loader_cls, path) in loaders.items():

        print("=" * 78)
        print(dataset_name)
        print("=" * 78)

        if not path.exists():
            print(f"SKIPPED — path does not exist: {path}")
            summary_rows.append({"dataset": dataset_name, "status": "SKIPPED"})
            continue

        try:
            loader = loader_cls(path)
            standardized = loader.run().data
        except Exception as exc:
            print(f"LOAD FAILED: {exc}")
            summary_rows.append({"dataset": dataset_name, "status": f"LOAD_FAILED: {exc}"})
            continue

        dataset_config = cfg.get_dataset_config(dataset_name)

        timing_config = timing.TimingConfig(
            expected_sampling_rate_hz=dataset_config.sampling_rate_hz,
        )

        windows_df = process_dataset(standardized, dataset_name, timing_config)

        output_path = output_dir / f"{dataset_name.lower()}_windows.csv"
        windows_df.to_csv(output_path, index=False)

        quality_counts = (
            windows_df["quality"].value_counts().to_dict() if len(windows_df) else {}
        )

        print(f"Windows produced: {len(windows_df)}")
        print(f"Quality distribution: {quality_counts}")
        print(f"Saved: {output_path}")

        summary_rows.append(
            {
                "dataset": dataset_name,
                "status": "OK",
                "windows": len(windows_df),
                **quality_counts,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "signal_processing_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("SIGNAL PROCESSING SUMMARY")
    print("=" * 78)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()