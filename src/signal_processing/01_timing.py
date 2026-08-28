"""
01_timing.py

Timing and continuity analysis for the fetal assessment project.

Responsibilities
----------------
This module is responsible ONLY for:

1. Detecting timestamp gaps.
2. Detecting timestamp discontinuities.
3. Splitting recordings into continuous segments.
4. Validating continuous segments.
5. Preserving dataset / subject / record / sensor identity.

This module does NOT:
- filter signals
- resample signals
- interpolate missing samples
- remove artifacts
- extract features
- perform windowing

The output of this module becomes the input to later
resampling / artifact handling / filtering stages.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TimingConfig:
    """
    Configuration controlling timestamp continuity detection.

    Parameters
    ----------
    expected_sampling_rate_hz:
        Nominal sampling frequency.

    gap_factor:
        A timestamp interval is considered a gap when:

            dt > expected_dt * gap_factor

        Example:
            30 Hz -> expected dt = 0.033333 s

            gap_factor = 1.5

            threshold = 0.05 s

    jitter_tolerance_fraction:
        Relative tolerance used when evaluating normal timing jitter.

        This does NOT create segments.

    minimum_segment_samples:
        Minimum number of samples required for a segment to be considered
        usable.

    allow_single_sample_segments:
        Whether one-sample segments are allowed during splitting.
        Normally False.
    """

    expected_sampling_rate_hz: Optional[float] = None

    gap_factor: float = 1.5

    jitter_tolerance_fraction: float = 0.20

    minimum_segment_samples: int = 10

    allow_single_sample_segments: bool = False


@dataclass
class Gap:
    """
    Represents one detected timestamp gap.
    """

    index_before: int
    index_after: int

    timestamp_before: float
    timestamp_after: float

    delta_seconds: float

    expected_delta_seconds: Optional[float]

    gap_ratio: Optional[float]


@dataclass
class SegmentInfo:
    """
    Metadata describing one continuous signal segment.
    """

    dataset_source: str
    subject_id: str
    record_id: str

    segment_id: str

    sensor_id: Optional[str]

    start_timestamp: float
    end_timestamp: float

    sample_count: int

    duration_seconds: float

    sampling_rate_hz: Optional[float]

    gap_count_inside: int

    negative_interval_count: int

    jitter_cv: Optional[float]

    timing_quality: str

    valid: bool

    rejection_reason: Optional[str] = None


# ============================================================================
# BASIC TIMING UTILITIES
# ============================================================================

def calculate_expected_interval(
    sampling_rate_hz: Optional[float],
) -> Optional[float]:
    """
    Calculate expected sample interval from sampling frequency.

    Example
    -------
    100 Hz -> 0.01 s
    30 Hz  -> 0.033333... s
    500 Hz -> 0.002 s
    """

    if sampling_rate_hz is None:
        return None

    if sampling_rate_hz <= 0:
        raise ValueError(
            "sampling_rate_hz must be greater than zero."
        )

    return 1.0 / float(sampling_rate_hz)


def calculate_timestamp_deltas(
    timestamps: Sequence[float],
) -> np.ndarray:
    """
    Calculate consecutive timestamp differences.
    """

    ts = np.asarray(timestamps, dtype=float)

    if ts.ndim != 1:
        raise ValueError("timestamps must be one-dimensional.")

    if len(ts) < 2:
        return np.array([], dtype=float)

    return np.diff(ts)


def calculate_jitter_metrics(
    timestamps: Sequence[float],
    config: TimingConfig,
) -> dict:
    """
    Report how much the sampling interval varies, as an explicit,
    reportable metric — not just the internal threshold already
    used inside detect_discontinuities().

    Gaps and negative/duplicate intervals are excluded first, so
    this measures ordinary timing jitter within otherwise-normal
    sampling, not the effect of discontinuities already handled
    elsewhere.

    Returns
    -------
    dict with:
        interval_count            number of Δt values considered
        mean_dt_seconds
        std_dt_seconds
        jitter_cv                 std/mean of Δt (coefficient of
                                   variation) — the standard way to
                                   report jitter independent of the
                                   sampling rate's absolute scale.
                                   0.0 = perfectly regular sampling.
        within_tolerance          whether jitter_cv is within
                                   config.jitter_tolerance_fraction
    """

    timestamps_array = np.asarray(timestamps, dtype=float)

    if len(timestamps_array) < 2:

        return {
            "interval_count": 0,
            "mean_dt_seconds": None,
            "std_dt_seconds": None,
            "jitter_cv": None,
            "within_tolerance": None,
        }

    deltas = calculate_timestamp_deltas(timestamps_array)

    expected_dt = calculate_expected_interval(
        config.expected_sampling_rate_hz
    )

    if expected_dt is None:
        positive = deltas[deltas > 0]
        expected_dt = (
            float(np.median(positive)) if len(positive) > 0 else None
        )

    ordinary_deltas = deltas[
        (deltas > 0)
        & (deltas <= (expected_dt * config.gap_factor if expected_dt else np.inf))
    ]

    if len(ordinary_deltas) == 0:

        return {
            "interval_count": 0,
            "mean_dt_seconds": None,
            "std_dt_seconds": None,
            "jitter_cv": None,
            "within_tolerance": None,
        }

    mean_dt = float(np.mean(ordinary_deltas))
    std_dt = float(np.std(ordinary_deltas))
    jitter_cv = (std_dt / mean_dt) if mean_dt > 0 else None

    within_tolerance = (
        jitter_cv is not None
        and jitter_cv <= config.jitter_tolerance_fraction
    )

    return {
        "interval_count": int(len(ordinary_deltas)),
        "mean_dt_seconds": mean_dt,
        "std_dt_seconds": std_dt,
        "jitter_cv": jitter_cv,
        "within_tolerance": within_tolerance,
    }


# ============================================================================
# GAP DETECTION
# ============================================================================

def detect_gaps(
    timestamps: Sequence[float],
    config: TimingConfig,
) -> list[Gap]:
    """
    Detect genuine timing gaps.

    A gap is NOT defined as every timestamp deviation.

    Instead:

        dt > expected_dt * gap_factor

    is treated as a gap.

    This is important because real recordings may have small timing jitter.

    Parameters
    ----------
    timestamps:
        Timestamp sequence.

    config:
        Timing configuration.

    Returns
    -------
    list[Gap]
        Detected gaps.
    """

    timestamps_array = np.asarray(timestamps, dtype=float)

    if len(timestamps_array) < 2:
        return []

    deltas = calculate_timestamp_deltas(timestamps_array)

    expected_dt = calculate_expected_interval(
        config.expected_sampling_rate_hz
    )

    # If no expected sampling frequency is known,
    # we cannot use a frequency-based absolute threshold.
    #
    # In that case, estimate the normal interval from the median.
    if expected_dt is None:

        positive_deltas = deltas[deltas > 0]

        if len(positive_deltas) == 0:
            return []

        expected_dt = float(np.median(positive_deltas))

    threshold = expected_dt * config.gap_factor

    gaps: list[Gap] = []

    for i, delta in enumerate(deltas):

        # Negative timestamp movement is a discontinuity,
        # not a normal gap.
        if delta <= 0:
            continue

        if delta > threshold:

            ratio = (
                delta / expected_dt
                if expected_dt > 0
                else None
            )

            gaps.append(
                Gap(
                    index_before=i,
                    index_after=i + 1,
                    timestamp_before=float(
                        timestamps_array[i]
                    ),
                    timestamp_after=float(
                        timestamps_array[i + 1]
                    ),
                    delta_seconds=float(delta),
                    expected_delta_seconds=float(
                        expected_dt
                    ),
                    gap_ratio=float(ratio)
                    if ratio is not None
                    else None,
                )
            )

    return gaps


# ============================================================================
# DISCONTINUITY DETECTION
# ============================================================================

def detect_discontinuities(
    timestamps: Sequence[float],
    config: TimingConfig,
) -> dict:
    """
    Detect all important timestamp discontinuities.

    Returns
    -------
    dict
        Contains:

        gaps
        negative_intervals
        zero_intervals
        irregular_intervals
        statistics
    """

    timestamps_array = np.asarray(
        timestamps,
        dtype=float,
    )

    if len(timestamps_array) < 2:

        return {
            "gaps": [],
            "negative_intervals": [],
            "zero_intervals": [],
            "irregular_intervals": [],
            "statistics": {
                "sample_count": len(timestamps_array),
                "interval_count": 0,
            },
        }

    deltas = calculate_timestamp_deltas(
        timestamps_array
    )

    expected_dt = calculate_expected_interval(
        config.expected_sampling_rate_hz
    )

    if expected_dt is None:

        positive = deltas[deltas > 0]

        expected_dt = (
            float(np.median(positive))
            if len(positive) > 0
            else None
        )

    gaps = detect_gaps(
        timestamps_array,
        config,
    )

    negative_intervals = [
        {
            "index_before": int(i),
            "index_after": int(i + 1),
            "timestamp_before": float(
                timestamps_array[i]
            ),
            "timestamp_after": float(
                timestamps_array[i + 1]
            ),
            "delta_seconds": float(delta),
        }
        for i, delta in enumerate(deltas)
        if delta < 0
    ]

    zero_intervals = [
        {
            "index_before": int(i),
            "index_after": int(i + 1),
            "timestamp": float(
                timestamps_array[i]
            ),
        }
        for i, delta in enumerate(deltas)
        if delta == 0
    ]

    irregular_intervals = []

    if expected_dt is not None:

        lower = expected_dt * (
            1.0 - config.jitter_tolerance_fraction
        )

        upper = expected_dt * (
            1.0 + config.jitter_tolerance_fraction
        )

        for i, delta in enumerate(deltas):

            if delta <= 0:
                continue

            # Ignore genuine gaps here.
            if delta > expected_dt * config.gap_factor:
                continue

            if delta < lower or delta > upper:

                irregular_intervals.append(
                    {
                        "index_before": int(i),
                        "index_after": int(i + 1),
                        "delta_seconds": float(delta),
                        "expected_delta_seconds": float(
                            expected_dt
                        ),
                    }
                )

    return {
        "gaps": gaps,
        "negative_intervals": negative_intervals,
        "zero_intervals": zero_intervals,
        "irregular_intervals": irregular_intervals,
        "statistics": {
            "sample_count": len(timestamps_array),
            "interval_count": len(deltas),
            "expected_delta_seconds": expected_dt,
            "gap_count": len(gaps),
            "negative_interval_count": len(
                negative_intervals
            ),
            "zero_interval_count": len(
                zero_intervals
            ),
            "irregular_interval_count": len(
                irregular_intervals
            ),
        },
    }


# ============================================================================
# SEGMENT SPLITTING
# ============================================================================

def split_into_segments(
    data: pd.DataFrame,
    config: TimingConfig,
    timestamp_column: str = "timestamp",
) -> list[pd.DataFrame]:
    """
    Split a recording into continuous timestamp segments.

    IMPORTANT
    ---------
    This function preserves the original samples.

    It does NOT:
    - interpolate
    - resample
    - filter
    - remove samples

    A new segment begins immediately after a detected gap
    or timestamp discontinuity.

    Parameters
    ----------
    data:
        One recording.

    config:
        Timing configuration.

    timestamp_column:
        Timestamp column name.

    Returns
    -------
    list[pd.DataFrame]
        Continuous segments.
    """

    if data.empty:
        return []

    if timestamp_column not in data.columns:
        raise ValueError(
            f"Timestamp column '{timestamp_column}' "
            "not found."
        )

    df = data.copy()

    timestamps = pd.to_numeric(
        df[timestamp_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid_timestamp_mask = np.isfinite(
        timestamps
    )

    if not valid_timestamp_mask.all():

        df = df.loc[
            valid_timestamp_mask
        ].copy()

        timestamps = timestamps[
            valid_timestamp_mask
        ]

    if len(df) == 0:
        return []

    # Reset index so positional indices correspond exactly
    # to the timestamp array.
    df = df.reset_index(drop=True)

    discontinuity_info = detect_discontinuities(
        timestamps,
        config,
    )

    boundaries = {
        0,
        len(df),
    }

    # A gap between i and i+1 means:
    #
    # segment ends at i
    # next segment begins at i+1
    #
    for gap in discontinuity_info["gaps"]:

        boundaries.add(
            gap.index_after
        )

    # Negative timestamps indicate a discontinuity.
    for item in discontinuity_info[
        "negative_intervals"
    ]:

        boundaries.add(
            item["index_after"]
        )

    boundaries = sorted(boundaries)

    segments: list[pd.DataFrame] = []

    for start, end in zip(
        boundaries[:-1],
        boundaries[1:],
    ):

        segment = df.iloc[
            start:end
        ].copy()

        if segment.empty:
            continue

        segment = segment.reset_index(
            drop=True
        )

        segments.append(segment)

    return segments


# ============================================================================
# SEGMENT VALIDATION
# ============================================================================

def validate_segment(
    segment: pd.DataFrame,
    config: TimingConfig,
    timestamp_column: str = "timestamp",
) -> dict:
    """
    Validate one continuous segment.

    Returns a dictionary describing whether the segment
    is suitable for downstream processing.

    A segment can be rejected for:

    - too few samples
    - invalid timestamps
    - negative timestamp intervals
    - repeated timestamps
    """

    if segment.empty:

        return {
            "valid": False,
            "reason": "empty_segment",
            "sample_count": 0,
        }

    if timestamp_column not in segment.columns:

        return {
            "valid": False,
            "reason": "missing_timestamp_column",
            "sample_count": len(segment),
        }

    timestamps = pd.to_numeric(
        segment[timestamp_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.all(
        np.isfinite(timestamps)
    ):

        return {
            "valid": False,
            "reason": "invalid_timestamp_values",
            "sample_count": len(segment),
        }

    sample_count = len(segment)

    if (
        sample_count < config.minimum_segment_samples
        and not config.allow_single_sample_segments
    ):

        return {
            "valid": False,
            "reason": "segment_too_short",
            "sample_count": sample_count,
        }

    deltas = np.diff(timestamps)

    negative_count = int(
        np.sum(deltas < 0)
    )

    zero_count = int(
        np.sum(deltas == 0)
    )

    if negative_count > 0:

        return {
            "valid": False,
            "reason": "negative_timestamp_interval",
            "sample_count": sample_count,
            "negative_interval_count": negative_count,
        }

    if zero_count > 0:

        return {
            "valid": False,
            "reason": "duplicate_timestamp",
            "sample_count": sample_count,
            "zero_interval_count": zero_count,
        }

    return {
        "valid": True,
        "reason": None,
        "sample_count": sample_count,
        "negative_interval_count": 0,
        "zero_interval_count": 0,
    }


# ============================================================================
# SEGMENT METADATA
# ============================================================================

def summarize_segment(
    segment: pd.DataFrame,
    dataset_source: str,
    subject_id: str,
    record_id: str,
    segment_id: str,
    config: TimingConfig,
    timestamp_column: str = "timestamp",
) -> SegmentInfo:
    """
    Generate standardized metadata for one segment.
    """

    timestamps = pd.to_numeric(
        segment[timestamp_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    sample_count = len(segment)

    start_timestamp = float(
        timestamps[0]
    )

    end_timestamp = float(
        timestamps[-1]
    )

    duration_seconds = (
        end_timestamp
        - start_timestamp
    )

    deltas = np.diff(timestamps)

    negative_count = int(
        np.sum(deltas < 0)
    )

    gaps = detect_gaps(
        timestamps,
        config,
    )

    jitter = calculate_jitter_metrics(
        timestamps,
        config,
    )

    # Estimate actual segment sampling rate
    # using the median positive interval.
    positive_deltas = deltas[
        deltas > 0
    ]

    if len(positive_deltas) > 0:

        median_dt = float(
            np.median(
                positive_deltas
            )
        )

        if median_dt > 0:

            sampling_rate = (
                1.0 / median_dt
            )

        else:

            sampling_rate = None

    else:

        sampling_rate = (
            config.expected_sampling_rate_hz
        )

    # Determine quality.
    if negative_count > 0:

        timing_quality = "invalid"

    elif len(gaps) > 0:

        timing_quality = "contains_gaps"

    else:

        timing_quality = "continuous"

    sensor_id = None

    if "sensor_id" in segment.columns:

        unique_sensors = (
            segment["sensor_id"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(unique_sensors) == 1:

            sensor_id = unique_sensors[0]

        elif len(unique_sensors) > 1:

            sensor_id = "MULTIPLE"

    validation = validate_segment(
        segment,
        config,
        timestamp_column,
    )

    return SegmentInfo(
        dataset_source=str(
            dataset_source
        ),
        subject_id=str(
            subject_id
        ),
        record_id=str(
            record_id
        ),
        segment_id=str(
            segment_id
        ),
        sensor_id=sensor_id,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        sample_count=sample_count,
        duration_seconds=duration_seconds,
        sampling_rate_hz=sampling_rate,
        gap_count_inside=len(gaps),
        negative_interval_count=negative_count,
        jitter_cv=jitter["jitter_cv"],
        timing_quality=timing_quality,
        valid=validation["valid"],
        rejection_reason=validation[
            "reason"
        ],
    )


# ============================================================================
# COMPLETE RECORD PROCESSING
# ============================================================================

def analyze_record(
    data: pd.DataFrame,
    config: TimingConfig,
    timestamp_column: str = "timestamp",
    dataset_source: Optional[str] = None,
    subject_id: Optional[str] = None,
    record_id: Optional[str] = None,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """
    Analyze one complete recording.

    Returns
    -------
    segments:
        List of continuous signal DataFrames.

    segment_metadata:
        DataFrame containing one row per segment.
    """

    if data.empty:

        return [], pd.DataFrame()

    # ------------------------------------------------------------------
    # Metadata fallback from dataframe
    # ------------------------------------------------------------------

    if dataset_source is None:

        if "dataset_source" in data.columns:

            dataset_source = str(
                data["dataset_source"]
                .iloc[0]
            )

        else:

            dataset_source = "UNKNOWN"

    if subject_id is None:

        if "subject_id" in data.columns:

            subject_id = str(
                data["subject_id"]
                .iloc[0]
            )

        else:

            subject_id = "UNKNOWN"

    if record_id is None:

        if "record_id" in data.columns:

            record_id = str(
                data["record_id"]
                .iloc[0]
            )

        else:

            record_id = "UNKNOWN"

    segments = split_into_segments(
        data,
        config,
        timestamp_column,
    )

    metadata = []

    for index, segment in enumerate(
        segments,
        start=1,
    ):

        segment_id = (
            f"{record_id}_segment_{index:04d}"
        )

        info = summarize_segment(
            segment=segment,
            dataset_source=dataset_source,
            subject_id=subject_id,
            record_id=record_id,
            segment_id=segment_id,
            config=config,
            timestamp_column=timestamp_column,
        )

        metadata.append(
            asdict(info)
        )

    metadata_df = pd.DataFrame(
        metadata
    )

    return segments, metadata_df


# ============================================================================
# MULTI-SENSOR SAFETY
# ============================================================================

def split_by_sensor(
    data: pd.DataFrame,
    sensor_column: str = "sensor_id",
) -> dict[str, pd.DataFrame]:
    """
    Separate temporal streams by sensor.

    This is particularly important for FOUR-IMU.

    Example
    -------
    IMU1
    IMU2
    IMU3
    IMU4

    are returned as four independent DataFrames.

    No sensor streams are mixed.
    """

    if sensor_column not in data.columns:

        raise ValueError(
            f"Sensor column '{sensor_column}' "
            "not found."
        )

    result = {}

    for sensor_id, sensor_data in data.groupby(
        sensor_column,
        sort=False,
    ):

        result[str(sensor_id)] = (
            sensor_data
            .copy()
            .reset_index(drop=True)
        )

    return result


def analyze_record_by_sensor(
    data: pd.DataFrame,
    config: TimingConfig,
    timestamp_column: str = "timestamp",
    sensor_column: str = "sensor_id",
) -> tuple[dict[str, list[pd.DataFrame]], pd.DataFrame]:
    """
    Analyze a multi-sensor recording without mixing sensors.

    Intended primarily for FOUR-IMU.

    Each sensor is independently segmented.
    """

    sensor_groups = split_by_sensor(
        data,
        sensor_column,
    )

    all_segments = {}

    metadata_frames = []

    for sensor_id, sensor_data in (
        sensor_groups.items()
    ):

        segments, metadata = analyze_record(
            data=sensor_data,
            config=config,
            timestamp_column=timestamp_column,
        )

        all_segments[sensor_id] = segments

        if not metadata.empty:

            metadata.insert(
                4,
                "sensor_id",
                sensor_id,
            )

            metadata_frames.append(
                metadata
            )

    if metadata_frames:

        metadata_df = pd.concat(
            metadata_frames,
            ignore_index=True,
        )

    else:

        metadata_df = pd.DataFrame()

    return (
        all_segments,
        metadata_df,
    )


# ============================================================================
# DATASET-SPECIFIC CONFIGURATIONS
# ============================================================================

COUGH_TIMING_CONFIG = TimingConfig(
    expected_sampling_rate_hz=100.0,

    # We do not want ordinary jitter to create thousands
    # of tiny segments.
    gap_factor=1.5,

    jitter_tolerance_fraction=0.20,

    minimum_segment_samples=100,
)


FOUR_IMU_TIMING_CONFIG = TimingConfig(
    expected_sampling_rate_hz=30.0,

    gap_factor=1.5,

    jitter_tolerance_fraction=0.20,

    minimum_segment_samples=30,
)


OXFORD_TIMING_CONFIG = TimingConfig(
    expected_sampling_rate_hz=500.0,

    gap_factor=1.5,

    jitter_tolerance_fraction=0.05,

    minimum_segment_samples=500,
)


# ============================================================================
# HUMAN-READABLE SUMMARY
# ============================================================================

def print_timing_summary(
    metadata: pd.DataFrame,
) -> None:
    """
    Print a compact summary of segment analysis.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TIMING / CONTINUITY SUMMARY"
    )

    print(
        "=" * 70
    )

    if metadata.empty:

        print(
            "No segments generated."
        )

        return

    print(
        f"\nTotal segments: "
        f"{len(metadata)}"
    )

    print(
        f"Valid segments: "
        f"{metadata['valid'].sum()}"
    )

    print(
        f"Rejected segments: "
        f"{(~metadata['valid']).sum()}"
    )

    print(
        "\nTiming quality:"
    )

    print(
        metadata[
            "timing_quality"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\nSegment duration:"
    )

    print(
        metadata[
            "duration_seconds"
        ].describe()
    )

    print(
        "\nSegment sample count:"
    )

    print(
        metadata[
            "sample_count"
        ].describe()
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "TimingConfig",
    "Gap",
    "SegmentInfo",
    "calculate_expected_interval",
    "calculate_timestamp_deltas",
    "detect_gaps",
    "detect_discontinuities",
    "split_into_segments",
    "validate_segment",
    "summarize_segment",
    "analyze_record",
    "split_by_sensor",
    "analyze_record_by_sensor",
    "COUGH_TIMING_CONFIG",
    "FOUR_IMU_TIMING_CONFIG",
    "OXFORD_TIMING_CONFIG",
    "print_timing_summary",
]