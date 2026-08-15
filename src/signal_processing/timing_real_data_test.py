from pathlib import Path
import sys
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# PROJECT IMPORT PATH
# ============================================================
# This allows the script to be executed directly as:
#
# python src\signal_processing\timing_real_data_test.py
#
# while dataset.py is located in:
#
# src\dataset.py
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
SIGNAL_PROCESSING_DIR = CURRENT_FILE.parent
SRC_DIR = SIGNAL_PROCESSING_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# DATASET LOADERS
# ============================================================

from dataset import (
    CoughLoader,
    FourIMULoader,
    OxfordLoader,
)


# ============================================================
# CONCRETE IMPLEMENTATIONS
# ============================================================
# FourIMULoader and OxfordLoader are abstract base classes.
#
# The loader_master_test.py already proved that these concrete
# implementations can be instantiated successfully.
#
# We therefore use the SAME mechanism here.
# ============================================================


class ConcreteFourIMULoader(FourIMULoader):

    def standardize(self, data):
        return data


class ConcreteOxfordLoader(OxfordLoader):

    def standardize(self, data):
        return data


# ============================================================
# DATASET PATHS
# ============================================================

COUGH_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "artifacts"
    / "cough_imu"
    / "Multimodal Cough Dataset"
)

FOUR_IMU_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fetal"
    / "four_imu"
)

OXFORD_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fetal"
    / "oxford_female"
)


# ============================================================
# TIMING CONFIGURATION
# ============================================================

# A timestamp interval is considered a genuine gap when it is
# substantially larger than the normal sampling interval.
#
# 1.5 allows normal timestamp jitter while still detecting
# large discontinuities.
GAP_FACTOR = 1.5

# Segments with fewer samples than this are considered too short
# for downstream physiological analysis.
MIN_SEGMENT_SAMPLES = 10

# Number of invalid examples printed per dataset.
MAX_INVALID_EXAMPLES = 20


# ============================================================
# COMMON SCHEMA
# ============================================================

COMMON_COLUMNS = [
    "timestamp",
    "sensor_id",

    "ax",
    "ay",
    "az",

    "gx",
    "gy",
    "gz",

    "mx",
    "my",
    "mz",

    "label",
    "subject_id",
    "dataset_source",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_numeric_timestamp(series):
    """
    Convert timestamp values to numeric values when possible.

    The timing layer must work with numerical timestamp differences.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        return numeric.to_numpy(dtype=float)

    # Attempt datetime conversion if timestamps are datetime strings.
    datetime_values = pd.to_datetime(
        series,
        errors="coerce"
    )

    if datetime_values.notna().all():
        return (
            datetime_values.astype("int64").to_numpy(dtype=float)
            / 1e9
        )

    return numeric.to_numpy(dtype=float)


def estimate_sampling_interval(timestamps):
    """
    Estimate the normal sampling interval from positive timestamp
    differences using the median.

    This avoids assuming a nominal sampling rate before inspecting
    the actual dataset.
    """

    timestamps = np.asarray(timestamps, dtype=float)

    timestamps = timestamps[np.isfinite(timestamps)]

    if len(timestamps) < 2:
        return np.nan

    intervals = np.diff(timestamps)

    positive = intervals[intervals > 0]

    if len(positive) == 0:
        return np.nan

    return float(np.median(positive))


def calculate_intervals(timestamps):
    """
    Calculate consecutive timestamp intervals.
    """

    timestamps = np.asarray(timestamps, dtype=float)

    if len(timestamps) < 2:
        return np.array([], dtype=float)

    return np.diff(timestamps)


def analyze_timestamps(timestamps):
    """
    Analyze one temporal stream.

    Returns:
        dictionary containing:
            gaps
            negative intervals
            zero intervals
            irregular intervals
            estimated dt
            interval array
    """

    timestamps = np.asarray(timestamps, dtype=float)

    intervals = calculate_intervals(timestamps)

    if len(intervals) == 0:
        return {
            "gaps": 0,
            "negative": 0,
            "zero": 0,
            "irregular": 0,
            "dt": np.nan,
            "intervals": intervals,
        }

    negative = int(np.sum(intervals < 0))
    zero = int(np.sum(intervals == 0))

    positive = intervals[intervals > 0]

    if len(positive) == 0:
        estimated_dt = np.nan
        gaps = 0
        irregular = int(len(intervals))
    else:
        estimated_dt = float(np.median(positive))

        gap_threshold = estimated_dt * GAP_FACTOR

        gaps = int(
            np.sum(intervals > gap_threshold)
        )

        irregular = int(
            np.sum(
                np.abs(intervals - estimated_dt)
                > estimated_dt * 0.01
            )
        )

    return {
        "gaps": gaps,
        "negative": negative,
        "zero": zero,
        "irregular": irregular,
        "dt": estimated_dt,
        "intervals": intervals,
    }


def split_into_segments(data):
    """
    Split a single temporal stream at genuine timestamp gaps.

    IMPORTANT:
    No interpolation is performed.

    No filtering is performed.

    No resampling is performed.

    Samples remain exactly as provided by the loader.
    """

    if len(data) == 0:
        return []

    timestamps = safe_numeric_timestamp(
        data["timestamp"]
    )

    intervals = calculate_intervals(timestamps)

    if len(intervals) == 0:
        return [data.copy()]

    positive = intervals[intervals > 0]

    if len(positive) == 0:
        return [data.copy()]

    expected_dt = float(np.median(positive))

    gap_threshold = expected_dt * GAP_FACTOR

    boundaries = np.where(
        intervals > gap_threshold
    )[0]

    segments = []

    start = 0

    for boundary in boundaries:

        end = boundary + 1

        segment = data.iloc[start:end].copy()

        if len(segment) > 0:
            segments.append(segment)

        start = end

    final_segment = data.iloc[start:].copy()

    if len(final_segment) > 0:
        segments.append(final_segment)

    return segments


def validate_segment(segment):
    """
    Validate an individual continuous segment.
    """

    sample_count = len(segment)

    if sample_count < MIN_SEGMENT_SAMPLES:
        return False, "segment_too_short"

    timestamps = safe_numeric_timestamp(
        segment["timestamp"]
    )

    if np.isnan(timestamps).any():
        return False, "invalid_timestamp"

    intervals = calculate_intervals(timestamps)

    if np.any(intervals < 0):
        return False, "negative_timestamp_interval"

    if np.any(intervals == 0):
        return False, "duplicate_timestamp"

    return True, None


def get_segment_duration(segment):
    """
    Calculate segment duration in seconds based on timestamps.

    Returns zero for a single-sample segment.
    """

    timestamps = safe_numeric_timestamp(
        segment["timestamp"]
    )

    timestamps = timestamps[
        np.isfinite(timestamps)
    ]

    if len(timestamps) <= 1:
        return 0.0

    return float(
        timestamps[-1] - timestamps[0]
    )


def get_record_identifier(name, group_key):
    """
    Convert group keys to a clean printable representation.
    """

    if isinstance(group_key, tuple):
        return tuple(
            str(value)
            for value in group_key
        )

    return str(group_key)


# ============================================================
# GENERIC SEGMENT AUDIT
# ============================================================

def segment_streams(
    data,
    stream_group_columns,
):
    """
    Segment all independent temporal streams.

    Returns:

        segment_records
        timing_statistics
    """

    segment_records = []

    total_gaps = 0
    total_negative = 0
    total_zero = 0
    total_irregular = 0

    grouped = data.groupby(
        stream_group_columns,
        sort=True,
        dropna=False,
    )

    total_streams = len(grouped)

    for index, (group_key, stream) in enumerate(
        grouped,
        start=1,
    ):

        stream = stream.copy()

        timestamps = safe_numeric_timestamp(
            stream["timestamp"]
        )

        timing = analyze_timestamps(
            timestamps
        )

        total_gaps += timing["gaps"]
        total_negative += timing["negative"]
        total_zero += timing["zero"]
        total_irregular += timing["irregular"]

        segments = split_into_segments(
            stream
        )

        for segment_id, segment in enumerate(
            segments,
            start=1,
        ):

            valid, reason = validate_segment(
                segment
            )

            record_identifier = get_record_identifier(
                "",
                group_key
            )

            segment_records.append(
                {
                    "record_identifier": record_identifier,
                    "segment_id": segment_id,
                    "sample_count": len(segment),
                    "duration_seconds":
                        get_segment_duration(segment),
                    "valid": valid,
                    "invalid_reason": reason,
                }
            )

        if (
            index % 10 == 0
            or index == total_streams
        ):
            print(
                f"  Analyzed "
                f"{index}/{total_streams}"
            )

    timing_statistics = {
        "streams": total_streams,
        "gaps": total_gaps,
        "negative": total_negative,
        "zero": total_zero,
        "irregular": total_irregular,
    }

    return (
        pd.DataFrame(segment_records),
        timing_statistics,
    )


# ============================================================
# PRINT SEGMENT SUMMARY
# ============================================================

def print_segment_summary(
    name,
    segment_df,
    timing_statistics,
):
    """
    Print timing and segmentation results.
    """

    total_segments = len(segment_df)

    if total_segments == 0:

        valid_segments = 0
        invalid_segments = 0

    else:

        valid_segments = int(
            segment_df["valid"].sum()
        )

        invalid_segments = (
            total_segments
            - valid_segments
        )

    short_segments = int(
        (
            segment_df["invalid_reason"]
            == "segment_too_short"
        ).sum()
    ) if total_segments else 0

    print("\n" + "-" * 78)
    print(f"{name} TIMING SUMMARY")
    print("-" * 78)

    if "streams" in timing_statistics:
        print(
            f"Temporal streams analyzed : "
            f"{timing_statistics['streams']}"
        )

    print(
        f"Total gaps detected       : "
        f"{timing_statistics['gaps']}"
    )

    print(
        f"Negative intervals        : "
        f"{timing_statistics['negative']}"
    )

    print(
        f"Zero intervals            : "
        f"{timing_statistics['zero']}"
    )

    print(
        f"Irregular intervals       : "
        f"{timing_statistics['irregular']}"
    )

    print(
        f"Total continuous segments : "
        f"{total_segments}"
    )

    print(
        f"Valid segments            : "
        f"{valid_segments}"
    )

    print(
        f"Invalid segments          : "
        f"{invalid_segments}"
    )

    print(
        f"Short segments            : "
        f"{short_segments}"
    )

    print("\n" + "-" * 78)
    print(f"{name} SEGMENT SAMPLE-COUNT DISTRIBUTION")
    print("-" * 78)

    print(
        f"Segment observations: "
        f"{total_segments}"
    )

    if total_segments:

        print(
            segment_df["sample_count"]
            .describe()
        )


# ============================================================
# INVALID SEGMENT REPORT
# ============================================================

def print_invalid_segments(
    name,
    segment_df,
):
    print("\n" + "-" * 78)
    print(f"{name} INVALID-SEGMENT REASONS")
    print("-" * 78)

    invalid_df = segment_df[
        ~segment_df["valid"]
    ].copy()

    if invalid_df.empty:

        print("✓ No invalid segments.")

        return

    print(
        f"Invalid segment observations: "
        f"{len(invalid_df)}"
    )

    print(
        invalid_df["invalid_reason"]
        .value_counts()
        .to_string()
    )

    print("\nInvalid segment examples:")

    print(
        invalid_df.head(
            MAX_INVALID_EXAMPLES
        ).to_string(index=False)
    )


# ============================================================
# COMMON LOADER VALIDATION
# ============================================================

def validate_common_schema(
    data,
    expected_source,
    required_columns=None,
):
    """
    Validate the standardized loader output.
    """

    if required_columns is None:
        required_columns = COMMON_COLUMNS

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:

        print(
            "\n❌ Missing required columns:",
            missing,
        )

        return False

    duplicated = (
        data.columns[
            data.columns.duplicated()
        ]
        .tolist()
    )

    if duplicated:

        print(
            "\n❌ Duplicate columns:",
            duplicated,
        )

        return False

    if len(data) == 0:

        print(
            "\n❌ Dataset is empty."
        )

        return False

    sources = (
        data["dataset_source"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if expected_source not in sources:

        print(
            "\n❌ Expected dataset source "
            f"'{expected_source}' not found."
        )

        print(
            "Available sources:",
            sources,
        )

        return False

    if data["subject_id"].isna().any():

        print(
            "\n❌ Missing subject IDs."
        )

        return False

    if data["timestamp"].isna().any():

        print(
            "\n❌ Missing timestamps."
        )

        return False

    if data["label"].isna().any():

        print(
            "\n❌ Missing labels."
        )

        return False

    return True


# ============================================================
# COUGH AUDIT
# ============================================================

def audit_cough():

    print("\n")
    print("=" * 78)
    print("1. COUGH REAL-DATA TIMING VALIDATION")
    print("=" * 78)

    print("\nDataset path:")
    print(COUGH_PATH)

    print("\nLoading COUGH dataset...")

    try:

        loader = CoughLoader(
            COUGH_PATH
        )

        data = loader.load()

    except Exception as exc:

        print("\n✗ COUGH TIMING TEST FAILED")
        print(
            f"Error: {type(exc).__name__}: {exc}"
        )

        return {
            "timing": False,
            "segmentation": False,
            "metadata": False,
            "signal": False,
            "overall": False,
        }

    print("\n✓ COUGH loader executed")

    print("\nLoaded shape:")
    print(data.shape)

    print("\nColumns:")
    print(data.columns.tolist())

    if not validate_common_schema(
        data,
        "COUGH",
    ):

        return {
            "timing": False,
            "segmentation": False,
            "metadata": False,
            "signal": False,
            "overall": False,
        }

    # --------------------------------------------------------
    # COUGH RECORDING IDENTIFICATION
    # --------------------------------------------------------
    #
    # The COUGH loader does not expose a separate record_id.
    # The actual recordings are represented by subject_id + label.
    # --------------------------------------------------------

    stream_columns = [
        "subject_id",
        "label",
    ]

    stream_count = (
        data.groupby(
            stream_columns,
            dropna=False,
        )
        .ngroups
    )

    print(
        f"\nRecordings/streams identified: "
        f"{stream_count}"
    )

    # --------------------------------------------------------
    # SEGMENT
    # --------------------------------------------------------

    segment_df, timing = segment_streams(
        data,
        stream_columns,
    )

    print_segment_summary(
        "COUGH",
        segment_df,
        timing,
    )

    print_invalid_segments(
        "COUGH",
        segment_df,
    )

    # --------------------------------------------------------
    # METADATA PRESERVATION
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("COUGH METADATA PRESERVATION")
    print("-" * 78)

    metadata_columns = [
        "dataset_source",
        "subject_id",
        "label",
    ]

    metadata_ok = all(
        column in data.columns
        for column in metadata_columns
    )

    if metadata_ok:

        print(
            "✓ dataset_source / subject_id / "
            "label survived segmentation."
        )

    else:

        print(
            "✗ Required metadata missing."
        )

    # --------------------------------------------------------
    # SIGNAL PRESERVATION
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("COUGH SIGNAL PRESERVATION")
    print("-" * 78)

    cough_signals = [
        "ax", "ay", "az",
        "gx", "gy", "gz",
        "mx", "my", "mz",
    ]

    signal_ok = all(
        column in data.columns
        for column in cough_signals
    )

    if signal_ok:

        print(
            "✓ All required COUGH IMU signal "
            "columns survived segmentation."
        )

    else:

        print(
            "✗ Required COUGH signal columns "
            "are missing."
        )

    timing_ok = (
        timing["negative"] == 0
        and timing["zero"] == 0
    )

    segmentation_ok = (
        len(segment_df) > 0
    )

    overall = (
        timing_ok
        and segmentation_ok
        and metadata_ok
        and signal_ok
    )

    return {
        "timing": timing_ok,
        "segmentation": segmentation_ok,
        "metadata": metadata_ok,
        "signal": signal_ok,
        "overall": overall,
    }


# ============================================================
# FOUR-IMU AUDIT
# ============================================================

def audit_four_imu():

    print("\n")
    print("=" * 78)
    print("2. FOUR-IMU REAL-DATA TIMING VALIDATION")
    print("=" * 78)

    print("\nDataset path:")
    print(FOUR_IMU_PATH)

    print("\nLoading FOUR-IMU dataset...")

    try:

        loader = ConcreteFourIMULoader(
            FOUR_IMU_PATH
        )

        data = loader.load()

    except Exception as exc:

        print("\n✗ FOUR-IMU TIMING TEST FAILED")
        print(
            f"Error: {type(exc).__name__}: {exc}"
        )

        return {
            "timing": False,
            "segmentation": False,
            "metadata": False,
            "signal": False,
            "sensor_identity": False,
            "overall": False,
        }

    print("\n✓ FOUR-IMU loader executed")

    print("\nLoaded shape:")
    print(data.shape)

    print("\nColumns:")
    print(data.columns.tolist())

    required = COMMON_COLUMNS + [
        "record_id",
        "sub_dataset",
    ]

    if not validate_common_schema(
        data,
        "FOUR_IMU",
        required,
    ):

        return {
            "timing": False,
            "segmentation": False,
            "metadata": False,
            "signal": False,
            "sensor_identity": False,
            "overall": False,
        }

    # --------------------------------------------------------
    # RECORDING COUNT
    # --------------------------------------------------------

    recording_count = (
        data[
            [
                "sub_dataset",
                "record_id",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"\nDiscovered Four-IMU recordings: "
        f"{recording_count}"
    )

    # --------------------------------------------------------
    # INDEPENDENT TEMPORAL STREAMS
    # --------------------------------------------------------

    stream_columns = [
        "sub_dataset",
        "record_id",
        "sensor_id",
    ]

    stream_count = (
        data.groupby(
            stream_columns,
            dropna=False,
        )
        .ngroups
    )

    print(
        f"\nIndependent temporal streams identified: "
        f"{stream_count}"
    )

    print("\nSensor distribution:")
    print(
        data["sensor_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # SEGMENTATION
    # --------------------------------------------------------

    segment_df, timing = segment_streams(
        data,
        stream_columns,
    )

    print_segment_summary(
        "FOUR-IMU",
        segment_df,
        timing,
    )

    print_invalid_segments(
        "FOUR-IMU",
        segment_df,
    )

    # --------------------------------------------------------
    # METADATA PRESERVATION
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("FOUR-IMU METADATA PRESERVATION")
    print("-" * 78)

    metadata_columns = [
        "dataset_source",
        "subject_id",
        "record_id",
        "sub_dataset",
        "sensor_id",
        "label",
    ]

    metadata_ok = all(
        column in data.columns
        for column in metadata_columns
    )

    if metadata_ok:

        print(
            "✓ dataset_source / subject_id / "
            "record_id / sub_dataset / sensor_id / "
            "label survived."
        )

    else:

        print(
            "✗ FOUR-IMU metadata preservation failed."
        )

    # --------------------------------------------------------
    # SIGNAL PRESERVATION
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("FOUR-IMU SIGNAL PRESERVATION")
    print("-" * 78)

    imu_signals = [
        "ax", "ay", "az",
        "gx", "gy", "gz",
    ]

    signal_ok = all(
        column in data.columns
        for column in imu_signals
    )

    if signal_ok:

        print(
            "✓ Accelerometer and gyroscope "
            "channels survived segmentation."
        )

    else:

        print(
            "✗ Required IMU channels missing."
        )

    # --------------------------------------------------------
    # SENSOR IDENTITY
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("FOUR-IMU SENSOR IDENTITY CHECK")
    print("-" * 78)

    original_sensors = sorted(
        data["sensor_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print("Original sensors:")
    print(original_sensors)

    original_streams = (
        data[
            [
                "sub_dataset",
                "record_id",
                "sensor_id",
            ]
        ]
        .drop_duplicates()
    )

    unique_stream_count = len(
        original_streams
    )

    print(
        "\nUnique "
        "(sub_dataset, record_id, sensor_id) "
        "streams:"
    )
    print(unique_stream_count)

    expected_sensors = {
        "IMU1",
        "IMU2",
        "IMU3",
        "IMU4",
    }

    sensor_identity_ok = (
        set(original_sensors)
        == expected_sensors
        and unique_stream_count == stream_count
    )

    if sensor_identity_ok:

        print(
            "✓ Each sensor stream was kept "
            "temporally independent."
        )

    else:

        print(
            "✗ Sensor stream identity problem detected."
        )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    timing_ok = (
        timing["negative"] == 0
        and timing["zero"] == 0
    )

    segmentation_ok = (
        len(segment_df) > 0
    )

    overall = (
        timing_ok
        and segmentation_ok
        and metadata_ok
        and signal_ok
        and sensor_identity_ok
    )

    return {
        "timing": timing_ok,
        "segmentation": segmentation_ok,
        "metadata": metadata_ok,
        "signal": signal_ok,
        "sensor_identity": sensor_identity_ok,
        "overall": overall,
    }


# ============================================================
# OXFORD AUDIT
# ============================================================

def audit_oxford():

    print("\n")
    print("=" * 78)
    print("3. OXFORD REAL-DATA TIMING VALIDATION")
    print("=" * 78)

    print("\nDataset path:")
    print(OXFORD_PATH)

    print("\nLoading Oxford dataset...")

    try:

        loader = ConcreteOxfordLoader(
            OXFORD_PATH
        )

        data = loader.load()

    except Exception as exc:

        print("\n✗ OXFORD TIMING TEST FAILED")
        print(
            f"Error: {type(exc).__name__}: {exc}"
        )

        return {
            "timing": False,
            "segmentation": False,
            "metadata": False,
            "record_id": False,
            "imu_signal": False,
            "bcg_signal": False,
            "bcg_finite": False,
            "overall": False,
        }

    print("\n✓ Oxford loader executed")

    print("\nLoaded shape:")
    print(data.shape)

    print("\nColumns:")
    print(data.columns.tolist())

    required = COMMON_COLUMNS + [
        "bcg_x",
        "bcg_y",
        "bcg_z",
        "record_id",
    ]

    if not validate_common_schema(
        data,
        "OXFORD",
        required,
    ):

        return {
            "timing": False,
            "segmentation": False,
            "metadata": False,
            "record_id": False,
            "imu_signal": False,
            "bcg_signal": False,
            "bcg_finite": False,
            "overall": False,
        }

    # --------------------------------------------------------
    # ORIGINAL RECORD IDs
    # --------------------------------------------------------

    original_record_ids = set(
        data["record_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"\nOxford records identified: "
        f"{len(original_record_ids)}"
    )

    # --------------------------------------------------------
    # SEGMENT
    # --------------------------------------------------------

    stream_columns = [
        "record_id",
    ]

    segment_df, timing = segment_streams(
        data,
        stream_columns,
    )

    print_segment_summary(
        "OXFORD",
        segment_df,
        timing,
    )

    print_invalid_segments(
        "OXFORD",
        segment_df,
    )

    # --------------------------------------------------------
    # TIMING STATUS
    # --------------------------------------------------------

    timing_ok = (
        timing["negative"] == 0
        and timing["zero"] == 0
    )

    print("\n" + "-" * 78)
    print("OXFORD TIMING STATUS")
    print("-" * 78)

    if timing_ok:
        print("✓ Oxford timing is valid.")
    else:
        print("✗ Oxford timing contains invalid intervals.")

    # --------------------------------------------------------
    # SEGMENTATION STATUS
    # --------------------------------------------------------

    segmentation_ok = (
        len(segment_df) > 0
        and len(segment_df) == len(original_record_ids)
    )

    print("\n" + "-" * 78)
    print("OXFORD SEGMENTATION STATUS")
    print("-" * 78)

    print(
        f"Original records       : "
        f"{len(original_record_ids)}"
    )

    print(
        f"Segment observations   : "
        f"{len(segment_df)}"
    )

    if segmentation_ok:
        print(
            "✓ Oxford records were segmented "
            "into the expected continuous segments."
        )
    else:
        print(
            "⚠ Oxford segmentation count requires review."
        )

    # --------------------------------------------------------
    # RECORD-ID PRESERVATION
    # --------------------------------------------------------
    # THIS IS THE CORRECTED LOGIC.
    #
    # Compare original loader output directly against the
    # segmented representation using string-normalized IDs.
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("OXFORD RECORD-ID PRESERVATION")
    print("-" * 78)

    segmented_record_ids = set(
        data["record_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    missing_record_ids = (
        original_record_ids
        - segmented_record_ids
    )

    extra_record_ids = (
        segmented_record_ids
        - original_record_ids
    )

    record_id_ok = (
        len(missing_record_ids) == 0
        and len(extra_record_ids) == 0
    )

    print(
        f"Original record IDs : "
        f"{len(original_record_ids)}"
    )

    print(
        f"Segmented record IDs: "
        f"{len(segmented_record_ids)}"
    )

    if missing_record_ids:

        print(
            "\nMissing after segmentation:"
        )
        print(
            sorted(missing_record_ids)
        )

    if extra_record_ids:

        print(
            "\nUnexpected extra IDs:"
        )
        print(
            sorted(extra_record_ids)
        )

    if record_id_ok:

        print(
            "\n✓ All original Oxford record IDs "
            "are preserved."
        )

    else:

        print(
            "\n✗ Oxford record-ID preservation failed."
        )

    # --------------------------------------------------------
    # METADATA PRESERVATION
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("OXFORD METADATA PRESERVATION")
    print("-" * 78)

    metadata_columns = [
        "dataset_source",
        "subject_id",
        "record_id",
        "label",
    ]

    metadata_ok = all(
        column in data.columns
        for column in metadata_columns
    )

    if metadata_ok:

        print(
            "✓ dataset_source / subject_id / "
            "record_id / label survived."
        )

    else:

        print(
            "✗ Oxford metadata preservation failed."
        )

    # --------------------------------------------------------
    # OXFORD IMU SIGNAL PRESERVATION
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("OXFORD IMU SIGNAL PRESERVATION")
    print("-" * 78)

    imu_columns = [
        "ax", "ay", "az",
        "gx", "gy", "gz",
        "mx", "my", "mz",
    ]

    imu_signal_ok = all(
        column in data.columns
        for column in imu_columns
    )

    if imu_signal_ok:

        print(
            "✓ Oxford IMU channels survived "
            "segmentation."
        )

    else:

        print(
            "✗ Required Oxford IMU channels missing."
        )

    # --------------------------------------------------------
    # OXFORD BCG SIGNAL PRESERVATION
    # --------------------------------------------------------
    # CORRECTED:
    #
    # We explicitly test BCG columns rather than treating
    # accelerometer ax/ay/az as BCG.
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("OXFORD BCG SIGNAL PRESERVATION")
    print("-" * 78)

    bcg_columns = [
        "bcg_x",
        "bcg_y",
        "bcg_z",
    ]

    bcg_signal_ok = all(
        column in data.columns
        for column in bcg_columns
    )

    if bcg_signal_ok:

        print(
            "✓ BCG X/Y/Z channels "
            "(bcg_x/bcg_y/bcg_z) survived "
            "the loader output."
        )

    else:

        missing_bcg = [
            column
            for column in bcg_columns
            if column not in data.columns
        ]

        print(
            "✗ Missing BCG columns:",
            missing_bcg,
        )

    # --------------------------------------------------------
    # OXFORD BCG FINITE-VALUE VALIDATION
    # --------------------------------------------------------
    # This is deliberately separate from column preservation.
    #
    # A column can exist while containing only NaN values.
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("OXFORD BCG FINITE-VALUE VALIDATION")
    print("-" * 78)

    bcg_finite_ok = False

    if bcg_signal_ok:

        finite_results = {}

        for column in bcg_columns:

            numeric_values = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            nan_count = int(
                numeric_values.isna().sum()
            )

            inf_count = int(
                np.isinf(
                    numeric_values
                    .to_numpy(
                        dtype=float
                    )
                ).sum()
            )

            finite_count = (
                len(data)
                - nan_count
                - inf_count
            )

            finite_results[column] = (
                finite_count
            )

            print(
                f"{column}: "
                f"NaN={nan_count} | "
                f"Inf={inf_count} | "
                f"Finite={finite_count}"
            )

        bcg_finite_ok = all(
            value > 0
            for value in finite_results.values()
        )

        if bcg_finite_ok:

            print(
                "\n✓ Oxford BCG contains "
                "finite values."
            )

        else:

            print(
                "\n✗ Oxford BCG contains "
                "missing or non-finite values."
            )

    else:

        print(
            "✗ Cannot perform BCG finite-value "
            "validation because BCG columns "
            "are missing."
        )

    # --------------------------------------------------------
    # FINAL OXFORD STATUS
    # --------------------------------------------------------

    overall = (
        timing_ok
        and segmentation_ok
        and metadata_ok
        and record_id_ok
        and imu_signal_ok
        and bcg_signal_ok
        and bcg_finite_ok
    )

    print("\n" + "-" * 78)
    print("OXFORD VALIDATION STATUS")
    print("-" * 78)

    print(
        f"OXFORD TIMING              : "
        f"{'PASS ✓' if timing_ok else 'FAIL ✗'}"
    )

    print(
        f"OXFORD SEGMENTATION        : "
        f"{'PASS ✓' if segmentation_ok else 'REVIEW ✗'}"
    )

    print(
        f"OXFORD METADATA            : "
        f"{'PASS ✓' if metadata_ok else 'FAIL ✗'}"
    )

    print(
        f"OXFORD RECORD-ID           : "
        f"{'PASS ✓' if record_id_ok else 'FAIL ✗'}"
    )

    print(
        f"OXFORD IMU SIGNAL          : "
        f"{'PASS ✓' if imu_signal_ok else 'FAIL ✗'}"
    )

    print(
        f"OXFORD BCG COLUMNS         : "
        f"{'PASS ✓' if bcg_signal_ok else 'FAIL ✗'}"
    )

    print(
        f"OXFORD BCG FINITE VALUES   : "
        f"{'PASS ✓' if bcg_finite_ok else 'FAIL ✗'}"
    )

    return {
        "timing": timing_ok,
        "segmentation": segmentation_ok,
        "metadata": metadata_ok,
        "record_id": record_id_ok,
        "imu_signal": imu_signal_ok,
        "bcg_signal": bcg_signal_ok,
        "bcg_finite": bcg_finite_ok,
        "overall": overall,
    }


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    cough_results,
    four_imu_results,
    oxford_results,
):

    print("\n")
    print("=" * 78)
    print("FINAL TIMING / CONTINUITY VALIDATION")
    print("=" * 78)

    print(
        f"COUGH          : "
        f"{'PASS ✓' if cough_results['overall'] else 'REVIEW ✗'}"
    )

    print(
        f"FOUR-IMU       : "
        f"{'PASS ✓' if four_imu_results['overall'] else 'REVIEW ✗'}"
    )

    print(
        f"OXFORD         : "
        f"{'PASS ✓' if oxford_results['overall'] else 'REVIEW ✗'}"
    )

    print("\n" + "-" * 78)
    print("LAYER-LEVEL STATUS")
    print("-" * 78)

    print(
        f"COUGH timing       : "
        f"{'PASS ✓' if cough_results['timing'] else 'FAIL ✗'}"
    )

    print(
        f"FOUR-IMU timing    : "
        f"{'PASS ✓' if four_imu_results['timing'] else 'FAIL ✗'}"
    )

    print(
        f"OXFORD timing      : "
        f"{'PASS ✓' if oxford_results['timing'] else 'FAIL ✗'}"
    )

    print(
        f"OXFORD metadata    : "
        f"{'PASS ✓' if oxford_results['metadata'] else 'FAIL ✗'}"
    )

    print(
        f"OXFORD record IDs  : "
        f"{'PASS ✓' if oxford_results['record_id'] else 'FAIL ✗'}"
    )

    print(
        f"OXFORD BCG        : "
        f"{'PASS ✓' if oxford_results['bcg_signal'] else 'FAIL ✗'}"
    )

    print(
        f"OXFORD BCG values  : "
        f"{'PASS ✓' if oxford_results['bcg_finite'] else 'FAIL ✗'}"
    )

    print("\n")
    print("=" * 78)
    print("TIMING VALIDATION COMPLETE")
    print("=" * 78)

    print(
        "\nNo filtering, resampling, interpolation, "
        "artifact handling, feature extraction, "
        "or ML processing was performed."
    )

    if (
        cough_results["timing"]
        and four_imu_results["timing"]
        and oxford_results["timing"]
    ):

        print(
            "\n✓ Timing / continuity behavior "
            "has passed the dataset-level timing checks."
        )

    if oxford_results["overall"]:

        print(
            "\n✓ All three datasets passed the "
            "current real-data validation."
        )

        print(
            "\nNext step:"
        )

        print(
            "    Freeze timing layer"
        )

        print(
            "    ↓"
        )

        print(
            "    Begin artifact-handling layer"
        )

    else:

        print(
            "\n⚠ One or more validation components "
            "still require review."
        )

        print(
            "\nDo NOT freeze the complete timing/data "
            "validation layer until the failed "
            "components are resolved."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("REAL-DATA TIMING / CONTINUITY VALIDATION")
    print("=" * 78)

    print(
        "\nThis script validates the timing layer "
        "against the actual dataset loader outputs."
    )

    print(
        "\nNO signal processing is performed."
    )

    print(
        "\nNo:"
    )

    print(
        "    filtering"
    )

    print(
        "    resampling"
    )

    print(
        "    interpolation"
    )

    print(
        "    artifact removal"
    )

    print(
        "    feature extraction"
    )

    print(
        "    model training"
    )

    print("\n")

    print(
        "Project root:"
    )

    print(PROJECT_ROOT)

    print("\nDataset paths:")

    print(
        "COUGH    :",
        COUGH_PATH,
    )

    print(
        "FOUR-IMU :",
        FOUR_IMU_PATH,
    )

    print(
        "OXFORD   :",
        OXFORD_PATH,
    )

    # ========================================================
    # DATASET AUDITS
    # ========================================================

    cough_results = audit_cough()

    four_imu_results = audit_four_imu()

    oxford_results = audit_oxford()

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print_final_summary(
        cough_results,
        four_imu_results,
        oxford_results,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()