"""
Temporal / Sampling Audit
=========================

Purpose
-------
Establish the empirical temporal characteristics of:

1. Multimodal Cough Dataset
2. Fetal Movement Dataset Recorded Using Four IMUs
3. Oxford Female Fetal Dataset

This script does NOT:
- filter signals
- resample signals
- interpolate signals
- remove artifacts
- modify labels
- modify raw files

It only measures:
- sampling interval
- estimated sampling frequency
- sampling-rate variability
- duration
- timestamp irregularity
- large gaps / discontinuities
- signal/label alignment for Oxford

The results from this audit will be used to finalize config.py
before implementing actual signal processing.
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ============================================================
# DATASET PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Asus\OneDrive\Desktop\fetal_assesment_project"
)

COUGH_ROOT = PROJECT_ROOT / (
    r"data\raw\artifacts\cough_imu"
    r"Multimodal Cough Dataset"
)

FOUR_IMU_ROOT = PROJECT_ROOT / (
    r"data\raw\fetal\four_imu"
)

OXFORD_ROOT = PROJECT_ROOT / (
    r"data\raw\fetal\oxford_female"
)


# ============================================================
# AUDIT PARAMETERS
# ============================================================

# A timestamp interval larger than this multiple of the
# median interval will be reported as a potential gap.
GAP_MULTIPLIER = 2.0

# Relative tolerance used for judging timestamp regularity.
#
# Example:
# median dt = 0.010 s
# tolerance = 0.001 s
#
# Anything outside this range is considered irregular.
IRREGULARITY_TOLERANCE = 0.05


# ============================================================
# GENERIC TEMPORAL ANALYSIS
# ============================================================

def analyze_timestamps(
    timestamps,
    gap_multiplier=GAP_MULTIPLIER,
    irregularity_tolerance=IRREGULARITY_TOLERANCE,
):
    """
    Analyze timestamp sequence.

    Returns a dictionary containing:

    - number of samples
    - duration
    - median dt
    - mean dt
    - std dt
    - min dt
    - max dt
    - estimated sampling rate
    - irregular sample count
    - irregular percentage
    - large gap count
    - largest gap
    - gap locations
    """

    ts = np.asarray(timestamps, dtype=float)

    ts = ts[np.isfinite(ts)]

    if len(ts) < 2:
        return {
            "samples": len(ts),
            "duration_s": np.nan,
            "median_dt_s": np.nan,
            "mean_dt_s": np.nan,
            "std_dt_s": np.nan,
            "min_dt_s": np.nan,
            "max_dt_s": np.nan,
            "estimated_fs_hz": np.nan,
            "irregular_count": 0,
            "irregular_percent": 0.0,
            "large_gap_count": 0,
            "largest_gap_s": np.nan,
            "gap_indices": [],
        }

    # --------------------------------------------------------
    # Sort timestamps
    # --------------------------------------------------------

    ts = np.sort(ts)

    # --------------------------------------------------------
    # Timestamp differences
    # --------------------------------------------------------

    dt = np.diff(ts)

    # Ignore zero/negative intervals when estimating normal
    # sampling interval.
    positive_dt = dt[dt > 0]

    if len(positive_dt) == 0:
        return {
            "samples": len(ts),
            "duration_s": ts[-1] - ts[0],
            "median_dt_s": np.nan,
            "mean_dt_s": np.nan,
            "std_dt_s": np.nan,
            "min_dt_s": np.nan,
            "max_dt_s": np.nan,
            "estimated_fs_hz": np.nan,
            "irregular_count": 0,
            "irregular_percent": 0.0,
            "large_gap_count": 0,
            "largest_gap_s": np.nan,
            "gap_indices": [],
        }

    median_dt = np.median(positive_dt)

    mean_dt = np.mean(positive_dt)

    std_dt = np.std(positive_dt)

    min_dt = np.min(positive_dt)

    max_dt = np.max(positive_dt)

    estimated_fs = 1.0 / median_dt

    # --------------------------------------------------------
    # Irregular timestamps
    # --------------------------------------------------------

    lower = median_dt * (1.0 - irregularity_tolerance)

    upper = median_dt * (1.0 + irregularity_tolerance)

    irregular_mask = (
        (positive_dt < lower)
        | (positive_dt > upper)
    )

    irregular_count = int(np.sum(irregular_mask))

    irregular_percent = (
        irregular_count / len(positive_dt)
    ) * 100.0

    # --------------------------------------------------------
    # Large gaps
    # --------------------------------------------------------

    gap_threshold = median_dt * gap_multiplier

    gap_indices = np.where(
        dt > gap_threshold
    )[0]

    large_gap_count = len(gap_indices)

    if large_gap_count > 0:
        largest_gap = float(
            np.max(dt[gap_indices])
        )
    else:
        largest_gap = float(np.max(dt))

    return {
        "samples": len(ts),
        "duration_s": float(ts[-1] - ts[0]),
        "median_dt_s": float(median_dt),
        "mean_dt_s": float(mean_dt),
        "std_dt_s": float(std_dt),
        "min_dt_s": float(min_dt),
        "max_dt_s": float(max_dt),
        "estimated_fs_hz": float(estimated_fs),
        "irregular_count": irregular_count,
        "irregular_percent": float(irregular_percent),
        "large_gap_count": large_gap_count,
        "largest_gap_s": largest_gap,
        "gap_indices": gap_indices.tolist(),
    }


# ============================================================
# FORMAT HELPER
# ============================================================

def print_result(result):
    """
    Pretty-print one temporal audit result.
    """

    print(f"Samples              : {result['samples']:,}")
    print(
        f"Duration              : "
        f"{result['duration_s']:.3f} s "
        f"({result['duration_s'] / 60:.3f} min)"
    )

    print(
        f"Median dt             : "
        f"{result['median_dt_s']:.9f} s"
    )

    print(
        f"Mean dt               : "
        f"{result['mean_dt_s']:.9f} s"
    )

    print(
        f"Std dt                : "
        f"{result['std_dt_s']:.9f} s"
    )

    print(
        f"Min dt                : "
        f"{result['min_dt_s']:.9f} s"
    )

    print(
        f"Max dt                : "
        f"{result['max_dt_s']:.9f} s"
    )

    print(
        f"Estimated sampling Hz : "
        f"{result['estimated_fs_hz']:.6f}"
    )

    print(
        f"Irregular intervals   : "
        f"{result['irregular_count']:,} "
        f"({result['irregular_percent']:.4f}%)"
    )

    print(
        f"Large gaps            : "
        f"{result['large_gap_count']:,}"
    )

    print(
        f"Largest interval      : "
        f"{result['largest_gap_s']:.9f} s"
    )


# ============================================================
# COUGH DATASET
# ============================================================

def audit_cough():
    """
    Audit all cough accelerometer recordings.

    The accelerometer timestamp is used as the master temporal
    reference because all three IMU modalities are aligned to it
    by the loader.
    """

    print("\n")
    print("=" * 70)
    print("COUGH DATASET TEMPORAL AUDIT")
    print("=" * 70)

    files = sorted(
        COUGH_ROOT.rglob("Accelerometer.csv")
    )

    print(f"\nAccelerometer recordings found: {len(files)}")

    if not files:
        print("ERROR: No cough recordings found.")
        return

    results = []

    for file in files:

        try:

            df = pd.read_csv(
                file,
                usecols=["elapsed (s)"]
            )

            timestamps = df["elapsed (s)"].to_numpy()

            result = analyze_timestamps(
                timestamps
            )

            subject_id = file.parent.parent.name

            trial = file.parent.name

            result["subject_id"] = subject_id

            result["trial"] = trial

            result["file"] = str(file)

            results.append(result)

        except Exception as e:

            print(
                f"\nWARNING: Failed to analyze "
                f"{file.name}: {e}"
            )

    if not results:
        print("No cough recordings could be analyzed.")
        return

    result_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Recording-level results
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("COUGH RECORDING-LEVEL RESULTS")
    print("-" * 70)

    display_columns = [
        "subject_id",
        "trial",
        "samples",
        "duration_s",
        "estimated_fs_hz",
        "irregular_percent",
        "large_gap_count",
        "largest_gap_s",
    ]

    print(
        result_df[
            display_columns
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Sampling-rate distribution
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("COUGH SAMPLING-RATE DISTRIBUTION")
    print("-" * 70)

    fs = result_df["estimated_fs_hz"]

    print(
        f"Minimum : {fs.min():.6f} Hz"
    )

    print(
        f"Maximum : {fs.max():.6f} Hz"
    )

    print(
        f"Mean    : {fs.mean():.6f} Hz"
    )

    print(
        f"Median  : {fs.median():.6f} Hz"
    )

    print(
        f"Std     : {fs.std():.6f} Hz"
    )

    # --------------------------------------------------------
    # Rounded frequency groups
    # --------------------------------------------------------

    print("\nSampling-rate groups:")

    print(
        fs.round(3)
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Duration distribution
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("COUGH DURATION DISTRIBUTION")
    print("-" * 70)

    durations = result_df["duration_s"]

    print(
        durations.describe().to_string()
    )

    # --------------------------------------------------------
    # Irregularity
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("COUGH TIMESTAMP REGULARITY")
    print("-" * 70)

    print(
        f"Recordings with irregular intervals: "
        f"{(result_df['irregular_count'] > 0).sum()} "
        f"/ {len(result_df)}"
    )

    print(
        f"Recordings with large gaps: "
        f"{(result_df['large_gap_count'] > 0).sum()} "
        f"/ {len(result_df)}"
    )

    print(
        f"Total large gaps: "
        f"{result_df['large_gap_count'].sum():,}"
    )

    print(
        f"Largest observed gap: "
        f"{result_df['largest_gap_s'].max():.6f} s"
    )

    return result_df


# ============================================================
# FOUR-IMU DATASET
# ============================================================

def find_four_imu_timestamp_column(columns):
    """
    Identify timestamp column in Four-IMU CSV files.
    """

    normalized = {
        str(col).strip().lower(): col
        for col in columns
    }

    candidates = [
        "timestamp",
        "time",
        "elapsed (s)",
        "elapsed",
        "time (s)",
        "timestamp (s)",
    ]

    for candidate in candidates:

        if candidate in normalized:
            return normalized[candidate]

    return None


def audit_four_imu():
    """
    Audit temporal characteristics of every Four-IMU
    recording file.

    Each physical recording is expected to contain four IMUs.
    """

    print("\n")
    print("=" * 70)
    print("FOUR-IMU DATASET TEMPORAL AUDIT")
    print("=" * 70)

    files = sorted(
        FOUR_IMU_ROOT.rglob("*.csv")
    )

    print(f"\nCSV recordings found: {len(files)}")

    if not files:
        print("ERROR: No Four-IMU CSV files found.")
        return

    results = []

    for file in files:

        try:

            # Read only the header first.
            header = pd.read_csv(
                file,
                nrows=0
            )

            timestamp_column = (
                find_four_imu_timestamp_column(
                    header.columns
                )
            )

            if timestamp_column is None:

                print(
                    f"\nWARNING: No timestamp column "
                    f"found in {file.name}"
                )

                continue

            df = pd.read_csv(
                file,
                usecols=[timestamp_column]
            )

            timestamps = pd.to_numeric(
                df[timestamp_column],
                errors="coerce"
            ).dropna().to_numpy()

            # ------------------------------------------------
            # Determine units
            #
            # Current Four-IMU loader uses seconds.
            # Therefore timestamp is expected to be seconds.
            # ------------------------------------------------

            result = analyze_timestamps(
                timestamps
            )

            result["file"] = str(file)

            result["sub_dataset"] = file.parent.name

            result["record_id"] = file.stem

            results.append(result)

        except Exception as e:

            print(
                f"\nWARNING: Failed to analyze "
                f"{file.name}: {e}"
            )

    if not results:
        print("No Four-IMU recordings could be analyzed.")
        return

    result_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Recording-level results
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("FOUR-IMU RECORDING-LEVEL RESULTS")
    print("-" * 70)

    display_columns = [
        "sub_dataset",
        "record_id",
        "samples",
        "duration_s",
        "estimated_fs_hz",
        "irregular_percent",
        "large_gap_count",
        "largest_gap_s",
    ]

    print(
        result_df[
            display_columns
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Sampling rate
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("FOUR-IMU SAMPLING-RATE DISTRIBUTION")
    print("-" * 70)

    fs = result_df["estimated_fs_hz"]

    print(
        f"Minimum : {fs.min():.6f} Hz"
    )

    print(
        f"Maximum : {fs.max():.6f} Hz"
    )

    print(
        f"Mean    : {fs.mean():.6f} Hz"
    )

    print(
        f"Median  : {fs.median():.6f} Hz"
    )

    print(
        f"Std     : {fs.std():.6f} Hz"
    )

    print("\nSampling-rate groups:")

    print(
        fs.round(3)
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("FOUR-IMU DURATION DISTRIBUTION")
    print("-" * 70)

    print(
        result_df["duration_s"]
        .describe()
        .to_string()
    )

    # --------------------------------------------------------
    # Regularity
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("FOUR-IMU TIMESTAMP REGULARITY")
    print("-" * 70)

    print(
        f"Recordings with irregular intervals: "
        f"{(result_df['irregular_count'] > 0).sum()} "
        f"/ {len(result_df)}"
    )

    print(
        f"Recordings with large gaps: "
        f"{(result_df['large_gap_count'] > 0).sum()} "
        f"/ {len(result_df)}"
    )

    print(
        f"Total large gaps: "
        f"{result_df['large_gap_count'].sum():,}"
    )

    print(
        f"Largest observed gap: "
        f"{result_df['largest_gap_s'].max():.6f} s"
    )

    return result_df


"""
sampling_audit.py

Signal-processing sampling audit for:

    1. Multimodal Cough Dataset
    2. Fetal Movement Dataset Recorded Using Four IMUs
    3. Oxford Female Fetal Dataset

Purpose
-------
Before implementing signal processing, establish:

    - Sampling-rate characteristics
    - Timestamp regularity
    - Duration distributions
    - Timestamp gaps
    - Recording discontinuities
    - Signal integrity
    - Dataset-specific timing assumptions

Important
---------
This script performs AUDITING ONLY.

It does NOT:
    - filter signals
    - resample signals
    - interpolate missing samples
    - modify raw data
    - remove artifacts

Oxford sampling rate
--------------------
The Oxford dataset README specifies:

    Sampling frequency = 500 Hz

Therefore Oxford sampling frequency is treated as known ground truth.

Oxford files contain sample-indexed BCG data rather than an explicit
timestamp vector. Therefore:

    sample_index -> time = sample_index / 500 Hz

The audit must NOT estimate Oxford sampling frequency from signal values.
"""


# ==============================================================
# Imports
# ==============================================================

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ==============================================================
# DATASET PATHS
# ==============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Asus\OneDrive\Desktop\fetal_assesment_project"
)


# --------------------------------------------------------------
# COUGH
# --------------------------------------------------------------

COUGH_PATH = PROJECT_ROOT / "data" / "raw" / "artifacts" / "cough_imu" / (
    "Multimodal Cough Dataset"
)


# --------------------------------------------------------------
# FOUR-IMU
# --------------------------------------------------------------

FOUR_IMU_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fetal"
    / "four_imu"
)


# --------------------------------------------------------------
# OXFORD
# --------------------------------------------------------------

OXFORD_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fetal"
    / "oxford_female"
)


# ==============================================================
# KNOWN DATASET CONFIGURATION
# ==============================================================

FOUR_IMU_FS = 30.0

OXFORD_FS = 500.0


# ==============================================================
# GENERAL AUDIT SETTINGS
# ==============================================================

# A timestamp interval is considered suspicious when it differs
# substantially from the expected interval.
#
# This is deliberately conservative because real sensor timestamps
# can contain small amounts of jitter.

RELATIVE_INTERVAL_TOLERANCE = 0.10


# Number of largest gaps to display per recording.
TOP_GAPS_TO_SHOW = 10


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================


def print_header(title):
    """Print a large section header."""

    print("\n")
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_subheader(title):
    """Print a subsection header."""

    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def safe_float_array(values):
    """
    Convert values to a clean float64 NumPy array.
    """

    return np.asarray(values, dtype=np.float64).reshape(-1)


def finite_values(values):
    """
    Return only finite values.
    """

    values = safe_float_array(values)

    return values[np.isfinite(values)]


def describe_intervals(intervals):
    """
    Calculate descriptive statistics for timestamp intervals.
    """

    intervals = finite_values(intervals)

    if len(intervals) == 0:

        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }

    return {
        "count": len(intervals),
        "mean": float(np.mean(intervals)),
        "median": float(np.median(intervals)),
        "std": float(np.std(intervals)),
        "min": float(np.min(intervals)),
        "max": float(np.max(intervals)),
        "p01": float(np.percentile(intervals, 1)),
        "p05": float(np.percentile(intervals, 5)),
        "p95": float(np.percentile(intervals, 95)),
        "p99": float(np.percentile(intervals, 99)),
    }


def expected_interval(fs):
    """
    Return expected sample interval for a sampling rate.
    """

    if fs is None or fs <= 0:
        return None

    return 1.0 / fs


def detect_timestamp_gaps(
    timestamps,
    expected_fs=None,
    tolerance=RELATIVE_INTERVAL_TOLERANCE,
):
    """
    Analyze timestamp intervals.

    Returns
    -------
    dict
        Detailed timing diagnostics.
    """

    timestamps = safe_float_array(timestamps)

    result = {
        "samples": len(timestamps),
        "duration": np.nan,
        "negative_intervals": 0,
        "zero_intervals": 0,
        "irregular_intervals": 0,
        "large_gaps": 0,
        "interval_stats": None,
        "largest_gaps": [],
    }

    if len(timestamps) < 2:
        return result

    intervals = np.diff(timestamps)

    finite_mask = np.isfinite(intervals)

    intervals = intervals[finite_mask]

    if len(intervals) == 0:
        return result

    result["duration"] = float(
        timestamps[-1] - timestamps[0]
    )

    result["negative_intervals"] = int(
        np.sum(intervals < 0)
    )

    result["zero_intervals"] = int(
        np.sum(intervals == 0)
    )

    result["interval_stats"] = describe_intervals(intervals)

    if expected_fs is not None:

        expected_dt = 1.0 / expected_fs

        lower = expected_dt * (1.0 - tolerance)
        upper = expected_dt * (1.0 + tolerance)

        irregular_mask = (
            (intervals < lower)
            | (intervals > upper)
        )

        result["irregular_intervals"] = int(
            np.sum(irregular_mask)
        )

        # A large gap means the interval is substantially greater
        # than the expected sampling interval.
        large_gap_mask = intervals > upper

        result["large_gaps"] = int(
            np.sum(large_gap_mask)
        )

        gap_indices = np.where(large_gap_mask)[0]

        if len(gap_indices) > 0:

            gap_records = []

            for idx in gap_indices:

                gap_records.append(
                    {
                        "index": int(idx),
                        "timestamp_before": float(
                            timestamps[idx]
                        ),
                        "timestamp_after": float(
                            timestamps[idx + 1]
                        ),
                        "interval": float(
                            intervals[idx]
                        ),
                        "expected_interval": float(
                            expected_dt
                        ),
                        "ratio": float(
                            intervals[idx] / expected_dt
                        ),
                    }
                )

            gap_records.sort(
                key=lambda x: x["interval"],
                reverse=True,
            )

            result["largest_gaps"] = (
                gap_records[:TOP_GAPS_TO_SHOW]
            )

    return result


def estimate_sampling_rate_from_timestamps(timestamps):
    """
    Estimate sampling rate from timestamp differences.

    IMPORTANT:
    This function is intended for datasets where timestamps are
    actually available.

    It is NOT used for Oxford because Oxford provides sample-indexed
    data and the README specifies 500 Hz.
    """

    timestamps = safe_float_array(timestamps)

    if len(timestamps) < 2:
        return np.nan

    intervals = np.diff(timestamps)

    intervals = intervals[
        np.isfinite(intervals)
        & (intervals > 0)
    ]

    if len(intervals) == 0:
        return np.nan

    median_dt = np.median(intervals)

    if median_dt <= 0:
        return np.nan

    return float(1.0 / median_dt)


def signal_integrity(values):
    """
    Check NaN / Inf / finite samples.
    """

    values = safe_float_array(values)

    return {
        "samples": len(values),
        "nan": int(np.isnan(values).sum()),
        "inf": int(np.isinf(values).sum()),
        "finite": int(np.isfinite(values).sum()),
    }


def print_interval_stats(stats):
    """Pretty-print interval statistics."""

    if stats is None:
        print("Interval statistics: unavailable")
        return

    print(
        f"Count      : {stats['count']:,}"
    )

    print(
        f"Mean       : {stats['mean']:.9f}"
    )

    print(
        f"Median     : {stats['median']:.9f}"
    )

    print(
        f"Std        : {stats['std']:.9f}"
    )

    print(
        f"Min        : {stats['min']:.9f}"
    )

    print(
        f"Max        : {stats['max']:.9f}"
    )

    print(
        f"P01        : {stats['p01']:.9f}"
    )

    print(
        f"P05        : {stats['p05']:.9f}"
    )

    print(
        f"P95        : {stats['p95']:.9f}"
    )

    print(
        f"P99        : {stats['p99']:.9f}"
    )


def print_largest_gaps(gaps):
    """Print largest timestamp gaps."""

    if not gaps:

        print("No large gaps detected.")

        return

    print(
        f"Largest gaps ({len(gaps)} shown):"
    )

    for gap in gaps:

        print(
            f"  index={gap['index']:,} | "
            f"interval={gap['interval']:.9f}s | "
            f"expected={gap['expected_interval']:.9f}s | "
            f"ratio={gap['ratio']:.2f}x"
        )


# ==============================================================
# COUGH AUDIT
# ==============================================================


def discover_cough_trials():
    """
    Discover all Cough dataset trials.

    Expected structure:

        subject/
            Trial_x/
                Accelerometer.csv
                Gyroscope.csv
                Magnetometer.csv
                ...
    """

    trials = []

    if not COUGH_PATH.exists():

        raise FileNotFoundError(
            f"COUGH dataset path does not exist:\n"
            f"{COUGH_PATH}"
        )

    for subject_dir in sorted(COUGH_PATH.iterdir()):

        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name

        for trial_dir in sorted(subject_dir.iterdir()):

            if not trial_dir.is_dir():
                continue

            accel = trial_dir / "Accelerometer.csv"

            gyro = trial_dir / "Gyroscope.csv"

            mag = trial_dir / "Magnetometer.csv"

            if not accel.exists():
                continue

            trials.append(
                {
                    "subject_id": subject_id,
                    "trial": trial_dir.name,
                    "accelerometer": accel,
                    "gyroscope": gyro
                    if gyro.exists()
                    else None,
                    "magnetometer": mag
                    if mag.exists()
                    else None,
                }
            )

    return trials


def load_cough_timestamp(path):
    """
    Read timestamp from a Cough accelerometer file.

    The common Cough schema uses:

        elapsed (s)
    """

    df = pd.read_csv(path)

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    if "elapsed (s)" not in df.columns:

        raise ValueError(
            f"Missing 'elapsed (s)' in:\n{path}"
        )

    return safe_float_array(
        df["elapsed (s)"].values
    )


def audit_cough():
    """
    Audit Cough dataset sampling characteristics.
    """

    print_header(
        "1. COUGH DATASET SAMPLING AUDIT"
    )

    print(
        f"Dataset path:\n{COUGH_PATH}"
    )

    trials = discover_cough_trials()

    print(
        f"\nTotal trials discovered: {len(trials)}"
    )

    if not trials:

        raise RuntimeError(
            "No Cough trials discovered."
        )

    records = []

    sampling_rates = []

    durations = []

    all_gap_counts = []

    all_irregular_counts = []

    for i, trial in enumerate(trials, start=1):

        try:

            timestamps = load_cough_timestamp(
                trial["accelerometer"]
            )

            fs = estimate_sampling_rate_from_timestamps(
                timestamps
            )

            timing = detect_timestamp_gaps(
                timestamps,
                expected_fs=fs,
            )

            duration = (
                float(timestamps[-1] - timestamps[0])
                if len(timestamps) > 1
                else 0.0
            )

            sampling_rates.append(fs)

            durations.append(duration)

            all_gap_counts.append(
                timing["large_gaps"]
            )

            all_irregular_counts.append(
                timing["irregular_intervals"]
            )

            records.append(
                {
                    "subject_id": trial["subject_id"],
                    "trial": trial["trial"],
                    "samples": len(timestamps),
                    "sampling_rate_hz": fs,
                    "duration_seconds": duration,
                    "irregular_intervals":
                        timing["irregular_intervals"],
                    "large_gaps":
                        timing["large_gaps"],
                }
            )

        except Exception as e:

            print(
                f"WARNING: Could not audit "
                f"{trial['subject_id']} / "
                f"{trial['trial']}: {e}"
            )

    audit_df = pd.DataFrame(records)

    print_subheader(
        "COUGH SAMPLING-RATE DISTRIBUTION"
    )

    if not audit_df.empty:

        print(
            audit_df[
                "sampling_rate_hz"
            ].describe().to_string()
        )

        print(
            "\nMedian sampling rate:"
        )

        print(
            f"{audit_df['sampling_rate_hz'].median():.6f} Hz"
        )

        print(
            "\nMinimum sampling rate:"
        )

        print(
            f"{audit_df['sampling_rate_hz'].min():.6f} Hz"
        )

        print(
            "\nMaximum sampling rate:"
        )

        print(
            f"{audit_df['sampling_rate_hz'].max():.6f} Hz"
        )

        print(
            "\nUnique rounded sampling rates:"
        )

        rounded_rates = (
            audit_df[
                "sampling_rate_hz"
            ]
            .round(3)
            .value_counts()
            .sort_index()
        )

        print(rounded_rates)

    print_subheader(
        "COUGH DURATION DISTRIBUTION"
    )

    if durations:

        duration_series = pd.Series(
            durations,
            name="duration_seconds",
        )

        print(
            duration_series.describe().to_string()
        )

    print_subheader(
        "COUGH TIMESTAMP IRREGULARITY"
    )

    print(
        f"Trials with large gaps: "
        f"{sum(x > 0 for x in all_gap_counts)} / "
        f"{len(all_gap_counts)}"
    )

    print(
        f"Trials with irregular intervals: "
        f"{sum(x > 0 for x in all_irregular_counts)} / "
        f"{len(all_irregular_counts)}"
    )

    print_subheader(
        "COUGH PER-TRIAL SUMMARY"
    )

    if not audit_df.empty:

        print(
            audit_df.to_string(index=False)
        )

    return audit_df


# ==============================================================
# FOUR-IMU AUDIT
# ==============================================================


def discover_four_imu_files():
    """
    Discover Four-IMU recording CSV files.
    """

    if not FOUR_IMU_PATH.exists():

        raise FileNotFoundError(
            f"FOUR-IMU dataset path does not exist:\n"
            f"{FOUR_IMU_PATH}"
        )

    files = []

    for csv_file in sorted(
        FOUR_IMU_PATH.rglob("*.csv")
    ):

        files.append(csv_file)

    return files


def load_four_imu_timestamp(path):
    """
    Read timestamps from a Four-IMU CSV.
    """

    df = pd.read_csv(path)

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    possible_columns = [
        "timestamp",
        "time",
        "elapsed (s)",
    ]

    timestamp_column = None

    for column in possible_columns:

        if column in df.columns:

            timestamp_column = column

            break

    if timestamp_column is None:

        raise ValueError(
            f"No recognized timestamp column "
            f"found in:\n{path}\n"
            f"Columns: {list(df.columns)}"
        )

    return safe_float_array(
        df[timestamp_column].values
    )


def audit_four_imu():
    """
    Audit Four-IMU timing.

    The dataset metadata specifies 30 Hz.

    Therefore timestamp regularity is evaluated against:

        dt = 1 / 30 = 0.033333... seconds
    """

    print_header(
        "2. FOUR-IMU TIMESTAMP REGULARITY AUDIT"
    )

    print(
        f"Dataset path:\n{FOUR_IMU_PATH}"
    )

    print(
        f"\nConfigured sampling rate: "
        f"{FOUR_IMU_FS} Hz"
    )

    expected_dt = 1.0 / FOUR_IMU_FS

    print(
        f"Expected interval: "
        f"{expected_dt:.9f} seconds"
    )

    files = discover_four_imu_files()

    print(
        f"\nRecording CSV files discovered: "
        f"{len(files)}"
    )

    records = []

    durations = []

    for path in files:

        try:

            timestamps = load_four_imu_timestamp(
                path
            )

            timing = detect_timestamp_gaps(
                timestamps,
                expected_fs=FOUR_IMU_FS,
            )

            duration = (
                float(timestamps[-1] - timestamps[0])
                if len(timestamps) > 1
                else 0.0
            )

            durations.append(duration)

            records.append(
                {
                    "file": path.name,
                    "samples": len(timestamps),
                    "duration_seconds": duration,
                    "median_dt": (
                        timing["interval_stats"]["median"]
                        if timing["interval_stats"]
                        else np.nan
                    ),
                    "irregular_intervals":
                        timing["irregular_intervals"],
                    "large_gaps":
                        timing["large_gaps"],
                    "negative_intervals":
                        timing["negative_intervals"],
                }
            )

        except Exception as e:

            print(
                f"WARNING: Could not audit "
                f"{path.name}: {e}"
            )

    audit_df = pd.DataFrame(records)

    print_subheader(
        "FOUR-IMU TIMESTAMP STATISTICS"
    )

    if not audit_df.empty:

        print(
            audit_df[
                [
                    "samples",
                    "duration_seconds",
                    "median_dt",
                    "irregular_intervals",
                    "large_gaps",
                    "negative_intervals",
                ]
            ].describe().to_string()
        )

    print_subheader(
        "FOUR-IMU REGULARITY"
    )

    if not audit_df.empty:

        print(
            f"Files with irregular intervals: "
            f"{(
                audit_df['irregular_intervals'] > 0
            ).sum()} / {len(audit_df)}"
        )

        print(
            f"Files with large gaps: "
            f"{(
                audit_df['large_gaps'] > 0
            ).sum()} / {len(audit_df)}"
        )

        print(
            f"Files with negative intervals: "
            f"{(
                audit_df['negative_intervals'] > 0
            ).sum()} / {len(audit_df)}"
        )

    print_subheader(
        "FOUR-IMU DURATION DISTRIBUTION"
    )

    if durations:

        duration_series = pd.Series(
            durations,
            name="duration_seconds",
        )

        print(
            duration_series.describe().to_string()
        )

    print_subheader(
        "FOUR-IMU PER-FILE SUMMARY"
    )

    if not audit_df.empty:

        print(
            audit_df.to_string(index=False)
        )

    return audit_df


# ==============================================================
# OXFORD AUDIT
# ==============================================================


def discover_oxford_records():
    """
    Discover Oxford signal/BP pairs.

    Expected:

        <record_id>_signal.mat
        <record_id>_bp.mat
    """

    if not OXFORD_PATH.exists():

        raise FileNotFoundError(
            f"OXFORD dataset path does not exist:\n"
            f"{OXFORD_PATH}"
        )

    signal_files = {
        path.name.replace(
            "_signal.mat",
            ""
        ): path

        for path in OXFORD_PATH.glob(
            "*_signal.mat"
        )
    }

    bp_files = {
        path.name.replace(
            "_bp.mat",
            ""
        ): path

        for path in OXFORD_PATH.glob(
            "*_bp.mat"
        )
    }

    record_ids = sorted(
        set(signal_files)
        | set(bp_files)
    )

    records = []

    for record_id in record_ids:

        records.append(
            {
                "record_id": record_id,
                "signal_file":
                    signal_files.get(record_id),
                "bp_file":
                    bp_files.get(record_id),
            }
        )

    return records


def extract_oxford_signal(mat):
    """
    Extract Oxford BCG signal.

    Expected variable:

        BCG_PREPROC_3AXIS

    Expected shape:

        N x 3
    """

    variable = "BCG_PREPROC_3AXIS"

    if variable not in mat:

        raise KeyError(
            f"Oxford signal variable "
            f"'{variable}' not found.\n"
            f"Available variables: "
            f"{list(mat.keys())}"
        )

    signal = np.asarray(
        mat[variable],
        dtype=np.float64,
    )

    if signal.ndim != 2:

        raise ValueError(
            f"Oxford signal must be 2-D. "
            f"Got shape {signal.shape}"
        )

    if signal.shape[1] != 3:

        raise ValueError(
            f"Oxford signal must have 3 axes. "
            f"Got shape {signal.shape}"
        )

    return signal


def extract_oxford_labels(mat):
    """
    Extract Oxford movement labels.

    Expected variable:

        BP_MOUV_FILES
    """

    variable = "BP_MOUV_FILES"

    if variable not in mat:

        raise KeyError(
            f"Oxford label variable "
            f"'{variable}' not found.\n"
            f"Available variables: "
            f"{list(mat.keys())}"
        )

    labels = safe_float_array(
        mat[variable]
    )

    return labels


def audit_oxford():
    """
    Audit Oxford BCG sampling and integrity.

    Oxford has no explicit timestamp vector.

    The README specifies 500 Hz.

    Therefore:

        expected_dt = 1 / 500
                    = 0.002 seconds

    Duration is computed as:

        number_of_aligned_samples / 500
    """

    print_header(
        "3. OXFORD BCG SAMPLING AUDIT"
    )

    print(
        f"Dataset path:\n{OXFORD_PATH}"
    )

    print(
        "\nSampling frequency source:"
    )

    print(
        "Oxford dataset README"
    )

    print(
        f"Configured sampling frequency: "
        f"{OXFORD_FS} Hz"
    )

    expected_dt = 1.0 / OXFORD_FS

    print(
        f"Expected sample interval: "
        f"{expected_dt:.6f} seconds"
    )

    records = discover_oxford_records()

    print(
        f"\nRecords discovered: {len(records)}"
    )

    paired = 0
    signal_only = 0
    bp_only = 0

    audit_records = []

    total_signal_samples = 0
    total_label_samples = 0
    total_aligned_samples = 0

    truncated_records = []

    for record in records:

        record_id = record["record_id"]

        signal_file = record["signal_file"]

        bp_file = record["bp_file"]

        if signal_file is None:

            bp_only += 1

            continue

        if bp_file is None:

            signal_only += 1

            continue

        paired += 1

        try:

            signal_mat = loadmat(
                signal_file
            )

            bp_mat = loadmat(
                bp_file
            )

            signal = extract_oxford_signal(
                signal_mat
            )

            labels = extract_oxford_labels(
                bp_mat
            )

            signal_length = signal.shape[0]

            label_length = len(labels)

            aligned_length = min(
                signal_length,
                label_length,
            )

            total_signal_samples += (
                signal_length
            )

            total_label_samples += (
                label_length
            )

            total_aligned_samples += (
                aligned_length
            )

            if signal_length != label_length:

                truncated_records.append(
                    {
                        "record_id": record_id,
                        "signal_length":
                            signal_length,
                        "label_length":
                            label_length,
                        "aligned_length":
                            aligned_length,
                    }
                )

            duration_seconds = (
                aligned_length / OXFORD_FS
            )

            # --------------------------------------------------
            # Sample-index continuity
            # --------------------------------------------------

            # There is no explicit timestamp vector in the
            # Oxford signal files. Samples are assumed to be
            # sequentially indexed.

            sample_indices = np.arange(
                aligned_length
            )

            if len(sample_indices) > 1:

                index_intervals = np.diff(
                    sample_indices
                )

                index_continuous = bool(
                    np.all(
                        index_intervals == 1
                    )
                )

            else:

                index_continuous = True

            # --------------------------------------------------
            # Signal integrity
            # --------------------------------------------------

            bcg_integrity = {}

            for axis_index, axis_name in enumerate(
                [
                    "bcg_x",
                    "bcg_y",
                    "bcg_z",
                ]
            ):

                bcg_integrity[axis_name] = (
                    signal_integrity(
                        signal[
                            :aligned_length,
                            axis_index
                        ]
                    )
                )

            # --------------------------------------------------
            # Label integrity
            # --------------------------------------------------

            label_integrity = signal_integrity(
                labels[
                    :aligned_length
                ]
            )

            # --------------------------------------------------
            # Label distribution
            # --------------------------------------------------

            label_values = labels[
                :aligned_length
            ]

            unique_labels, counts = np.unique(
                label_values,
                return_counts=True,
            )

            label_distribution = {
                str(label): int(count)
                for label, count
                in zip(
                    unique_labels,
                    counts,
                )
            }

            audit_records.append(
                {
                    "record_id": record_id,
                    "signal_samples":
                        signal_length,
                    "label_samples":
                        label_length,
                    "aligned_samples":
                        aligned_length,
                    "duration_seconds":
                        duration_seconds,
                    "sampling_rate_hz":
                        OXFORD_FS,
                    "expected_dt_seconds":
                        expected_dt,
                    "sample_index_continuous":
                        index_continuous,
                    "bcg_x_nan":
                        bcg_integrity[
                            "bcg_x"
                        ]["nan"],
                    "bcg_y_nan":
                        bcg_integrity[
                            "bcg_y"
                        ]["nan"],
                    "bcg_z_nan":
                        bcg_integrity[
                            "bcg_z"
                        ]["nan"],
                    "bcg_x_inf":
                        bcg_integrity[
                            "bcg_x"
                        ]["inf"],
                    "bcg_y_inf":
                        bcg_integrity[
                            "bcg_y"
                        ]["inf"],
                    "bcg_z_inf":
                        bcg_integrity[
                            "bcg_z"
                        ]["inf"],
                    "label_nan":
                        label_integrity["nan"],
                    "label_inf":
                        label_integrity["inf"],
                    "label_distribution":
                        label_distribution,
                }
            )

        except Exception as e:

            print(
                f"WARNING: Could not audit Oxford "
                f"record {record_id}: {e}"
            )

    audit_df = pd.DataFrame(
        audit_records
    )

    print_subheader(
        "OXFORD FILE PAIRING"
    )

    print(
        f"Paired       : {paired}"
    )

    print(
        f"Signal only  : {signal_only}"
    )

    print(
        f"BP only      : {bp_only}"
    )

    print_subheader(
        "OXFORD SAMPLING SPECIFICATION"
    )

    print(
        f"Sampling rate : {OXFORD_FS} Hz"
    )

    print(
        f"Sample period : {expected_dt:.6f} seconds"
    )

    print(
        "Timestamp source: sample index"
    )

    print(
        "Sampling-rate inference from timestamps: "
        "NOT APPLICABLE"
    )

    print_subheader(
        "OXFORD DURATION DISTRIBUTION"
    )

    if not audit_df.empty:

        print(
            audit_df[
                "duration_seconds"
            ].describe().to_string()
        )

    print_subheader(
        "OXFORD LENGTH ALIGNMENT"
    )

    print(
        f"Total original signal samples: "
        f"{total_signal_samples:,}"
    )

    print(
        f"Total original label samples: "
        f"{total_label_samples:,}"
    )

    print(
        f"Total aligned samples: "
        f"{total_aligned_samples:,}"
    )

    print(
        f"Records requiring truncation: "
        f"{len(truncated_records)}"
    )

    if truncated_records:

        for item in truncated_records:

            print(
                f"  {item['record_id']} | "
                f"signal={item['signal_length']:,} | "
                f"labels={item['label_length']:,} | "
                f"aligned={item['aligned_length']:,}"
            )

    print_subheader(
        "OXFORD SAMPLE-INDEX CONTINUITY"
    )

    if not audit_df.empty:

        discontinuous = (
            ~audit_df[
                "sample_index_continuous"
            ]
        ).sum()

        print(
            f"Records with discontinuous "
            f"sample indices: "
            f"{discontinuous} / "
            f"{len(audit_df)}"
        )

    print_subheader(
        "OXFORD BCG SIGNAL INTEGRITY"
    )

    if not audit_df.empty:

        for axis in [
            "bcg_x",
            "bcg_y",
            "bcg_z",
        ]:

            nan_count = int(
                audit_df[
                    f"{axis}_nan"
                ].sum()
            )

            inf_count = int(
                audit_df[
                    f"{axis}_inf"
                ].sum()
            )

            print(
                f"{axis}: "
                f"NaN={nan_count:,} | "
                f"Inf={inf_count:,}"
            )

    print_subheader(
        "OXFORD LABEL DISTRIBUTION"
    )

    if not audit_df.empty:

        counter = Counter()

        for distribution in audit_df[
            "label_distribution"
        ]:

            for label, count in distribution.items():

                counter[label] += count

        for label in sorted(
            counter,
            key=lambda x: float(x),
        ):

            print(
                f"Label {label}: "
                f"{counter[label]:,}"
            )

    print_subheader(
        "OXFORD PER-RECORD SUMMARY"
    )

    if not audit_df.empty:

        print(
            audit_df[
                [
                    "record_id",
                    "signal_samples",
                    "label_samples",
                    "aligned_samples",
                    "duration_seconds",
                    "sampling_rate_hz",
                    "sample_index_continuous",
                ]
            ].to_string(index=False)
        )

    return audit_df


# ==============================================================
# FINAL SUMMARY
# ==============================================================


def print_final_summary(
    cough_df,
    four_imu_df,
    oxford_df,
):
    """
    Print final audit conclusions.
    """

    print_header(
        "FINAL SAMPLING AUDIT SUMMARY"
    )

    # ----------------------------------------------------------
    # Cough
    # ----------------------------------------------------------

    print(
        "\nCOUGH"
    )

    if cough_df is not None and not cough_df.empty:

        print(
            f"  Recordings audited : "
            f"{len(cough_df)}"
        )

        print(
            f"  Median estimated fs: "
            f"{cough_df['sampling_rate_hz'].median():.6f} Hz"
        )

        print(
            f"  Min estimated fs   : "
            f"{cough_df['sampling_rate_hz'].min():.6f} Hz"
        )

        print(
            f"  Max estimated fs   : "
            f"{cough_df['sampling_rate_hz'].max():.6f} Hz"
        )

        print(
            f"  Trials with gaps   : "
            f"{(
                cough_df['large_gaps'] > 0
            ).sum()}"
        )

    # ----------------------------------------------------------
    # Four IMU
    # ----------------------------------------------------------

    print(
        "\nFOUR-IMU"
    )

    if four_imu_df is not None and not four_imu_df.empty:

        print(
            f"  Recordings audited : "
            f"{len(four_imu_df)}"
        )

        print(
            f"  Configured fs      : "
            f"{FOUR_IMU_FS:.1f} Hz"
        )

        print(
            f"  Expected dt        : "
            f"{1.0 / FOUR_IMU_FS:.9f} s"
        )

        print(
            f"  Files with gaps    : "
            f"{(
                four_imu_df['large_gaps'] > 0
            ).sum()}"
        )

        print(
            f"  Files with negative "
            f"timestamps          : "
            f"{(
                four_imu_df['negative_intervals'] > 0
            ).sum()}"
        )

    # ----------------------------------------------------------
    # Oxford
    # ----------------------------------------------------------

    print(
        "\nOXFORD"
    )

    if oxford_df is not None and not oxford_df.empty:

        print(
            f"  Records audited    : "
            f"{len(oxford_df)}"
        )

        print(
            f"  Sampling rate      : "
            f"{OXFORD_FS:.1f} Hz "
            f"(README)"
        )

        print(
            f"  Expected dt        : "
            f"{1.0 / OXFORD_FS:.6f} s"
        )

        print(
            f"  BCG axes           : "
            f"3"
        )

        print(
            f"  Records truncated  : "
            f"{(
                oxford_df['signal_samples']
                != oxford_df['label_samples']
            ).sum()}"
        )

        print(
            f"  Index discontinuity : "
            f"{(
                ~oxford_df[
                    'sample_index_continuous'
                ]
            ).sum()}"
        )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "SAMPLING AUDIT COMPLETE"
    )

    print(
        "=" * 78
    )


# ==============================================================
# MAIN
# ==============================================================


def main():

    print_header(
        "SIGNAL PROCESSING SAMPLING AUDIT"
    )

    print(
        "This script audits timing characteristics only."
    )

    print(
        "No filtering, resampling, interpolation, "
        "or artifact removal is performed."
    )

    print(
        "\nDataset paths:"
    )

    print(
        f"\nCOUGH:\n{COUGH_PATH}"
    )

    print(
        f"\nFOUR-IMU:\n{FOUR_IMU_PATH}"
    )

    print(
        f"\nOXFORD:\n{OXFORD_PATH}"
    )

    # ----------------------------------------------------------
    # Cough
    # ----------------------------------------------------------

    cough_df = audit_cough()

    # ----------------------------------------------------------
    # Four IMU
    # ----------------------------------------------------------

    four_imu_df = audit_four_imu()

    # ----------------------------------------------------------
    # Oxford
    # ----------------------------------------------------------

    oxford_df = audit_oxford()

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------

    print_final_summary(
        cough_df,
        four_imu_df,
        oxford_df,
    )


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    main()