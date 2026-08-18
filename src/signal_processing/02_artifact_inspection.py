"""
02_artifact_inspection.py

ARTIFACT INSPECTION / CANDIDATE EVENT GENERATION
================================================

Purpose
-------
This is the FIRST artifact-handling stage of the fetal movement
analysis pipeline.

It operates AFTER the validated timing / continuity layer.

Pipeline:

    Raw dataset
        ↓
    Dataset loader
        ↓
    Standardized DataFrame
        ↓
    Timing / continuity layer
        ↓
    Continuous segments
        ↓
    Artifact inspection
        ↓
    Quantitative signal characteristics
        ↓
    Candidate artifact events
        ↓
    Manual validation
        ↓
    Future artifact ML dataset


IMPORTANT
---------
This script DOES NOT:

    - interpolate across gaps
    - perform artifact removal
    - train an ML model
    - classify artifacts with a learned model
    - classify fetal movement
    - extract final fetal behavioral features

The candidate labels generated here are ONLY:

    "candidate"

They are NOT ground-truth artifact labels.

The purpose is to understand the real signal before building
the artifact classifier.


OUTPUTS
-------
Results are written to:

    data/processed/artifact_inspection/

including:

    segment_features.csv
    candidate_events.csv
    manual_label_template.csv
    inspection_summary.csv

and representative plots under:

    plots/
"""


# ======================================================================
# STANDARD LIBRARY
# ======================================================================

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


# ======================================================================
# NUMPY / PANDAS
# ======================================================================

import numpy as np
import pandas as pd


# ======================================================================
# SCIPY
# ======================================================================

from scipy.signal import welch


# ======================================================================
# MATPLOTLIB
# ======================================================================

import matplotlib.pyplot as plt


# ======================================================================
# PROJECT PATHS
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()

SIGNAL_PROCESSING_DIR = CURRENT_FILE.parent

SRC_DIR = SIGNAL_PROCESSING_DIR.parent

PROJECT_ROOT = SRC_DIR.parent

DATA_ROOT = PROJECT_ROOT / "data" / "raw"


# ======================================================================
# OUTPUT DIRECTORY
# ======================================================================

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artifact_inspection"
)

PLOTS_ROOT = OUTPUT_ROOT / "plots"

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

PLOTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# MAKE SRC IMPORTABLE
# ======================================================================

if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR),
    )


# ======================================================================
# DATASET LOADERS
# ======================================================================

from dataset import (
    CoughLoader,
    FourIMULoader,
    OxfordLoader,
)


# ======================================================================
# CONCRETE LOADER IMPLEMENTATIONS
# ======================================================================

class ConcreteFourIMULoader(FourIMULoader):

    def standardize(self, data):

        return data


class ConcreteOxfordLoader(OxfordLoader):

    def standardize(self, data):

        return data


# ======================================================================
# DATASET PATHS
# ======================================================================

COUGH_PATH = (
    DATA_ROOT
    / "artifacts"
    / "cough_imu"
    / "Multimodal Cough Dataset"
)

FOUR_IMU_PATH = (
    DATA_ROOT
    / "fetal"
    / "four_imu"
)

OXFORD_PATH = (
    DATA_ROOT
    / "fetal"
    / "oxford_female"
)


# ======================================================================
# LOAD TIMING MODULE
# ======================================================================

def load_module_from_file(
    module_name: str,
    file_path: Path,
):

    if not file_path.exists():

        raise FileNotFoundError(
            f"Timing module not found:\n{file_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(file_path),
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            f"Could not load timing module:\n{file_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    # Required for dataclass resolution.
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


TIMING_PATH = (
    SIGNAL_PROCESSING_DIR
    / "01_timing.py"
)


timing = load_module_from_file(
    "signal_processing_01_timing",
    TIMING_PATH,
)


TimingConfig = timing.TimingConfig

split_into_segments = timing.split_into_segments

validate_segment = timing.validate_segment


# ======================================================================
# CONFIGURATION
# ======================================================================

# Expected / nominal sampling rates.
#
# These are used ONLY for feature interpretation.
# Timing segmentation itself continues to use 01_timing.py.

SAMPLING_RATES = {

    "COUGH": 100.0,

    "FOUR-IMU": 30.0,

    "OXFORD": 500.0,
}


# Minimum useful segment duration for artifact inspection.

MIN_ANALYSIS_DURATION_SECONDS = 0.5


# Number of representative segments to plot per dataset.

MAX_PLOTS_PER_DATASET = 12


# Maximum number of points displayed in a plot.
#
# This is ONLY visualization downsampling.
# Original data are NOT modified.

MAX_PLOT_POINTS = 5000


# Candidate threshold configuration.
#
# These are intentionally conservative.
#
# They are NOT final artifact thresholds.
#
# Their purpose is to identify interesting signal regions
# for manual inspection.

CANDIDATE_ZSCORE_THRESHOLD = 3.0


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

def print_header(title: str):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_section(title: str):

    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


# ======================================================================
# LOADER HELPER
# ======================================================================

def load_dataset(
    dataset_name: str,
):

    if dataset_name == "COUGH":

        print("Loading COUGH dataset...")

        loader = CoughLoader(
            COUGH_PATH
        )

        data = loader.load()

        return data


    if dataset_name == "FOUR-IMU":

        print("Loading FOUR-IMU dataset...")

        loader = ConcreteFourIMULoader(
            FOUR_IMU_PATH
        )

        data = loader.load()

        return data


    if dataset_name == "OXFORD":

        print("Loading Oxford dataset...")

        loader = ConcreteOxfordLoader(
            OXFORD_PATH
        )

        data = loader.load()

        return data


    raise ValueError(
        f"Unknown dataset: {dataset_name}"
    )


# ======================================================================
# STREAM DEFINITIONS
# ======================================================================

def get_stream_columns(
    dataset_name: str,
):

    if dataset_name == "COUGH":

        return [
            "subject_id",
            "sensor_id",
        ]


    if dataset_name == "FOUR-IMU":

        return [
            "sub_dataset",
            "record_id",
            "sensor_id",
        ]


    if dataset_name == "OXFORD":

        return [
            "record_id",
        ]


    raise ValueError(
        f"Unknown dataset: {dataset_name}"
    )


# ======================================================================
# SIGNAL DEFINITIONS
# ======================================================================

def get_signal_columns(
    dataset_name: str,
):

    if dataset_name == "COUGH":

        return [
            "ax",
            "ay",
            "az",
            "gx",
            "gy",
            "gz",
        ]


    if dataset_name == "FOUR-IMU":

        return [
            "ax",
            "ay",
            "az",
            "gx",
            "gy",
            "gz",
        ]


    if dataset_name == "OXFORD":

        return [
            "ax",
            "ay",
            "az",
            "gx",
            "gy",
            "gz",
            "bcg_x",
            "bcg_y",
            "bcg_z",
        ]


    raise ValueError(
        f"Unknown dataset: {dataset_name}"
    )


# ======================================================================
# GROUP ITERATOR
# ======================================================================

def iterate_streams(
    data: pd.DataFrame,
    key_columns: list[str],
):

    grouped = data.groupby(
        key_columns,
        sort=False,
        dropna=False,
    )

    for key, group in grouped:

        if not isinstance(key, tuple):

            key = (key,)

        yield key, group.copy()


# ======================================================================
# TIMESTAMP CLEANING
# ======================================================================

def prepare_timestamps(
    segment: pd.DataFrame,
):

    timestamps = pd.to_numeric(
        segment["timestamp"],
        errors="coerce",
    )

    mask = timestamps.notna()

    clean_segment = segment.loc[
        mask
    ].copy()

    timestamps = timestamps.loc[
        mask
    ].to_numpy(
        dtype=float
    )

    return (
        clean_segment,
        timestamps,
    )


# ======================================================================
# BASIC STATISTICAL FEATURES
# ======================================================================

def safe_std(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return np.nan

    return float(
        np.std(
            values
        )
    )


def safe_mean(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return np.nan

    return float(
        np.mean(
            values
        )
    )


def safe_rms(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return np.nan

    return float(
        np.sqrt(
            np.mean(
                values ** 2
            )
        )
    )


def safe_peak_to_peak(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return np.nan

    return float(
        np.max(values)
        - np.min(values)
    )


# ======================================================================
# SIGNAL MAGNITUDE
# ======================================================================

def vector_magnitude(
    x,
    y,
    z,
):

    x = pd.to_numeric(
        x,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    y = pd.to_numeric(
        y,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    z = pd.to_numeric(
        z,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    return np.sqrt(
        x ** 2
        + y ** 2
        + z ** 2
    )


# ======================================================================
# JERK
# ======================================================================

def calculate_jerk(
    magnitude,
    timestamps,
):

    magnitude = np.asarray(
        magnitude,
        dtype=float,
    )

    timestamps = np.asarray(
        timestamps,
        dtype=float,
    )

    if len(magnitude) < 2:

        return np.array([])

    dt = np.diff(
        timestamps
    )

    dm = np.diff(
        magnitude
    )

    valid = (
        np.isfinite(dt)
        &
        np.isfinite(dm)
        &
        (dt > 0)
    )

    jerk = np.full(
        len(dm),
        np.nan,
    )

    jerk[valid] = (
        dm[valid]
        /
        dt[valid]
    )

    return jerk


# ======================================================================
# FREQUENCY FEATURES
# ======================================================================

def calculate_frequency_features(
    signal,
    sampling_rate_hz,
):

    signal = np.asarray(
        signal,
        dtype=float,
    )

    signal = signal[
        np.isfinite(signal)
    ]

    result = {

        "dominant_frequency_hz":
            np.nan,

        "spectral_centroid_hz":
            np.nan,

        "spectral_bandwidth_hz":
            np.nan,

        "spectral_energy":
            np.nan,

    }

    if len(signal) < 16:

        return result

    signal = (
        signal
        - np.mean(signal)
    )

    nperseg = min(
        2048,
        len(signal),
    )

    try:

        frequencies, power = welch(
            signal,
            fs=sampling_rate_hz,
            nperseg=nperseg,
        )

    except Exception:

        return result


    if len(power) == 0:

        return result


    total_power = np.sum(
        power
    )


    if total_power <= 0:

        return result


    dominant_index = int(
        np.argmax(power)
    )


    dominant_frequency = (
        frequencies[
            dominant_index
        ]
    )


    centroid = (
        np.sum(
            frequencies
            * power
        )
        /
        total_power
    )


    bandwidth = np.sqrt(
        np.sum(
            (
                frequencies
                - centroid
            ) ** 2
            * power
        )
        /
        total_power
    )


    result[
        "dominant_frequency_hz"
    ] = float(
        dominant_frequency
    )

    result[
        "spectral_centroid_hz"
    ] = float(
        centroid
    )

    result[
        "spectral_bandwidth_hz"
    ] = float(
        bandwidth
    )

    result[
        "spectral_energy"
    ] = float(
        total_power
    )


    return result


# ======================================================================
# Z-SCORE HELPER
# ======================================================================

def robust_zscore(
    value,
    median,
    mad,
):

    if not np.isfinite(value):

        return np.nan

    if not np.isfinite(
        median
    ):

        return np.nan

    if not np.isfinite(
        mad
    ):

        return np.nan

    if mad < 1e-12:

        return 0.0

    return float(
        0.6745
        *
        (
            value
            - median
        )
        /
        mad
    )


# ======================================================================
# FEATURE EXTRACTION FOR ONE SEGMENT
# ======================================================================

def extract_segment_features(
    segment: pd.DataFrame,
    dataset_name: str,
    record_identifier,
    segment_id: int,
):

    clean_segment, timestamps = (
        prepare_timestamps(
            segment
        )
    )


    if len(clean_segment) == 0:

        return None


    sampling_rate = (
        SAMPLING_RATES[
            dataset_name
        ]
    )


    sample_count = len(
        clean_segment
    )


    start_timestamp = float(
        timestamps[0]
    )


    end_timestamp = float(
        timestamps[-1]
    )


    duration = (
        end_timestamp
        - start_timestamp
    )


    row = {

        "dataset":
            dataset_name,

        "record_identifier":
            str(record_identifier),

        "segment_id":
            segment_id,

        "sample_count":
            sample_count,

        "duration_seconds":
            duration,

        "sampling_rate_hz":
            sampling_rate,

    }


    # --------------------------------------------------------------
    # ACCELEROMETER
    # --------------------------------------------------------------

    if all(
        column in clean_segment.columns
        for column in [
            "ax",
            "ay",
            "az",
        ]
    ):

        acceleration = (
            vector_magnitude(
                clean_segment["ax"],
                clean_segment["ay"],
                clean_segment["az"],
            )
        )

        valid_acc = acceleration[
            np.isfinite(acceleration)
        ]


        row[
            "acc_mean"
        ] = safe_mean(
            valid_acc
        )

        row[
            "acc_std"
        ] = safe_std(
            valid_acc
        )

        row[
            "acc_rms"
        ] = safe_rms(
            valid_acc
        )

        row[
            "acc_peak_to_peak"
        ] = safe_peak_to_peak(
            valid_acc
        )


        jerk = calculate_jerk(
            acceleration,
            timestamps,
        )

        row[
            "jerk_rms"
        ] = safe_rms(
            jerk
        )

        row[
            "jerk_peak_to_peak"
        ] = safe_peak_to_peak(
            jerk
        )


        frequency_features = (
            calculate_frequency_features(
                valid_acc,
                sampling_rate,
            )
        )


        for key, value in (
            frequency_features.items()
        ):

            row[
                f"acc_{key}"
            ] = value


    # --------------------------------------------------------------
    # GYROSCOPE
    # --------------------------------------------------------------

    if all(
        column in clean_segment.columns
        for column in [
            "gx",
            "gy",
            "gz",
        ]
    ):

        gyroscope = (
            vector_magnitude(
                clean_segment["gx"],
                clean_segment["gy"],
                clean_segment["gz"],
            )
        )

        valid_gyro = gyroscope[
            np.isfinite(gyroscope)
        ]


        row[
            "gyro_mean"
        ] = safe_mean(
            valid_gyro
        )

        row[
            "gyro_std"
        ] = safe_std(
            valid_gyro
        )

        row[
            "gyro_rms"
        ] = safe_rms(
            valid_gyro
        )

        row[
            "gyro_peak_to_peak"
        ] = safe_peak_to_peak(
            valid_gyro
        )


        gyro_frequency_features = (
            calculate_frequency_features(
                valid_gyro,
                sampling_rate,
            )
        )


        for key, value in (
            gyro_frequency_features.items()
        ):

            row[
                f"gyro_{key}"
            ] = value


    # --------------------------------------------------------------
    # OXFORD BCG
    # --------------------------------------------------------------

    if all(
        column in clean_segment.columns
        for column in [
            "bcg_x",
            "bcg_y",
            "bcg_z",
        ]
    ):

        bcg = vector_magnitude(
            clean_segment["bcg_x"],
            clean_segment["bcg_y"],
            clean_segment["bcg_z"],
        )

        valid_bcg = bcg[
            np.isfinite(bcg)
        ]


        row[
            "bcg_mean"
        ] = safe_mean(
            valid_bcg
        )

        row[
            "bcg_std"
        ] = safe_std(
            valid_bcg
        )

        row[
            "bcg_rms"
        ] = safe_rms(
            valid_bcg
        )

        row[
            "bcg_peak_to_peak"
        ] = safe_peak_to_peak(
            valid_bcg
        )


        bcg_frequency_features = (
            calculate_frequency_features(
                valid_bcg,
                sampling_rate,
            )
        )


        for key, value in (
            bcg_frequency_features.items()
        ):

            row[
                f"bcg_{key}"
            ] = value


    return row


# ======================================================================
# CANDIDATE EVENT GENERATION
# ======================================================================

def generate_candidate_reason(
    feature_row,
    dataset_feature_medians,
):

    reasons = []


    # --------------------------------------------------------------
    # HIGH ACCELERATION VARIABILITY
    # --------------------------------------------------------------

    acc_std = feature_row.get(
        "acc_std",
        np.nan,
    )

    acc_std_median = (
        dataset_feature_medians.get(
            "acc_std_median",
            np.nan,
        )
    )

    acc_std_mad = (
        dataset_feature_medians.get(
            "acc_std_mad",
            np.nan,
        )
    )


    z_acc = robust_zscore(
        acc_std,
        acc_std_median,
        acc_std_mad,
    )


    if (
        np.isfinite(z_acc)
        and
        abs(z_acc)
        >= CANDIDATE_ZSCORE_THRESHOLD
    ):

        reasons.append(
            "unusually_high_acceleration_variability"
        )


    # --------------------------------------------------------------
    # HIGH JERK
    # --------------------------------------------------------------

    jerk_rms = feature_row.get(
        "jerk_rms",
        np.nan,
    )

    jerk_median = (
        dataset_feature_medians.get(
            "jerk_rms_median",
            np.nan,
        )
    )

    jerk_mad = (
        dataset_feature_medians.get(
            "jerk_rms_mad",
            np.nan,
        )
    )


    z_jerk = robust_zscore(
        jerk_rms,
        jerk_median,
        jerk_mad,
    )


    if (
        np.isfinite(z_jerk)
        and
        abs(z_jerk)
        >= CANDIDATE_ZSCORE_THRESHOLD
    ):

        reasons.append(
            "unusually_high_jerk"
        )


    # --------------------------------------------------------------
    # HIGH GYROSCOPE ACTIVITY
    # --------------------------------------------------------------

    gyro_std = feature_row.get(
        "gyro_std",
        np.nan,
    )

    gyro_median = (
        dataset_feature_medians.get(
            "gyro_std_median",
            np.nan,
        )
    )

    gyro_mad = (
        dataset_feature_medians.get(
            "gyro_std_mad",
            np.nan,
        )
    )


    z_gyro = robust_zscore(
        gyro_std,
        gyro_median,
        gyro_mad,
    )


    if (
        np.isfinite(z_gyro)
        and
        abs(z_gyro)
        >= CANDIDATE_ZSCORE_THRESHOLD
    ):

        reasons.append(
            "unusually_high_gyroscope_variability"
        )


    # --------------------------------------------------------------
    # HIGH ACCELERATION RANGE
    # --------------------------------------------------------------

    acc_range = feature_row.get(
        "acc_peak_to_peak",
        np.nan,
    )

    acc_range_median = (
        dataset_feature_medians.get(
            "acc_peak_to_peak_median",
            np.nan,
        )
    )

    acc_range_mad = (
        dataset_feature_medians.get(
            "acc_peak_to_peak_mad",
            np.nan,
        )
    )


    z_range = robust_zscore(
        acc_range,
        acc_range_median,
        acc_range_mad,
    )


    if (
        np.isfinite(z_range)
        and
        abs(z_range)
        >= CANDIDATE_ZSCORE_THRESHOLD
    ):

        reasons.append(
            "unusually_large_acceleration_range"
        )


    return reasons


# ======================================================================
# DATASET FEATURE ROBUST STATISTICS
# ======================================================================

def calculate_dataset_statistics(
    feature_df: pd.DataFrame,
):

    statistics = {}


    numeric_columns = [
        "acc_std",
        "jerk_rms",
        "gyro_std",
        "acc_peak_to_peak",
    ]


    for column in numeric_columns:

        if column not in feature_df.columns:

            continue


        values = pd.to_numeric(
            feature_df[column],
            errors="coerce",
        )

        values = values[
            np.isfinite(values)
        ]


        if len(values) == 0:

            continue


        median = float(
            np.median(
                values
            )
        )


        mad = float(
            np.median(
                np.abs(
                    values
                    - median
                )
            )
        )


        statistics[
            f"{column}_median"
        ] = median

        statistics[
            f"{column}_mad"
        ] = mad


    return statistics


# ======================================================================
# PLOT DOWNSAMPLING
# ======================================================================

def downsample_for_plot(
    timestamps,
    values,
    max_points=MAX_PLOT_POINTS,
):

    timestamps = np.asarray(
        timestamps
    )

    values = np.asarray(
        values
    )


    if len(timestamps) <= max_points:

        return (
            timestamps,
            values,
        )


    indices = np.linspace(
        0,
        len(timestamps) - 1,
        max_points,
    ).astype(int)


    return (
        timestamps[indices],
        values[indices],
    )


# ======================================================================
# PLOT REPRESENTATIVE SEGMENT
# ======================================================================

def plot_segment(
    segment: pd.DataFrame,
    dataset_name: str,
    record_identifier,
    segment_id: int,
    output_path: Path,
):

    clean_segment, timestamps = (
        prepare_timestamps(
            segment
        )
    )


    if len(clean_segment) == 0:

        return


    timestamps = (
        timestamps
        - timestamps[0]
    )


    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        sharex=True,
    )


    # --------------------------------------------------------------
    # ACCELERATION
    # --------------------------------------------------------------

    if all(
        c in clean_segment.columns
        for c in [
            "ax",
            "ay",
            "az",
        ]
    ):

        for column in [
            "ax",
            "ay",
            "az",
        ]:

            t, y = downsample_for_plot(
                timestamps,
                pd.to_numeric(
                    clean_segment[column],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                ),
            )

            axes[0].plot(
                t,
                y,
                label=column,
            )


        magnitude = vector_magnitude(
            clean_segment["ax"],
            clean_segment["ay"],
            clean_segment["az"],
        )


        t, y = downsample_for_plot(
            timestamps,
            magnitude,
        )


        axes[0].plot(
            t,
            y,
            label="acc_magnitude",
            linewidth=1.5,
        )


    axes[0].set_ylabel(
        "Acceleration"
    )

    axes[0].set_title(
        "Accelerometer"
    )

    axes[0].legend(
        loc="upper right"
    )

    axes[0].grid(
        alpha=0.25
    )


    # --------------------------------------------------------------
    # GYROSCOPE
    # --------------------------------------------------------------

    if all(
        c in clean_segment.columns
        for c in [
            "gx",
            "gy",
            "gz",
        ]
    ):

        for column in [
            "gx",
            "gy",
            "gz",
        ]:

            t, y = downsample_for_plot(
                timestamps,
                pd.to_numeric(
                    clean_segment[column],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                ),
            )

            axes[1].plot(
                t,
                y,
                label=column,
            )


        axes[1].set_ylabel(
            "Gyroscope"
        )

        axes[1].set_title(
            "Gyroscope"
        )

        axes[1].legend(
            loc="upper right"
        )

        axes[1].grid(
            alpha=0.25
        )


    # --------------------------------------------------------------
    # OXFORD BCG
    # --------------------------------------------------------------

    if all(
        c in clean_segment.columns
        for c in [
            "bcg_x",
            "bcg_y",
            "bcg_z",
        ]
    ):

        for column in [
            "bcg_x",
            "bcg_y",
            "bcg_z",
        ]:

            t, y = downsample_for_plot(
                timestamps,
                pd.to_numeric(
                    clean_segment[column],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                ),
            )

            axes[2].plot(
                t,
                y,
                label=column,
            )


        axes[2].set_title(
            "Oxford BCG"
        )

        axes[2].set_ylabel(
            "BCG"
        )

        axes[2].legend(
            loc="upper right"
        )

        axes[2].grid(
            alpha=0.25
        )


    else:

        axes[2].axis(
            "off"
        )


    axes[-1].set_xlabel(
        "Time within segment (s)"
    )


    fig.suptitle(
        (
            f"{dataset_name} | "
            f"record={record_identifier} | "
            f"segment={segment_id}"
        ),
        fontsize=13,
    )


    fig.tight_layout()


    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ======================================================================
# REPRESENTATIVE SEGMENT SELECTION
# ======================================================================

def select_representative_segments(
    feature_df: pd.DataFrame,
    max_segments: int,
):

    if feature_df.empty:

        return feature_df


    # Prefer a spread across the dataset rather than
    # selecting only the largest signals.

    feature_df = feature_df.sort_values(
        by=[
            "sample_count",
            "duration_seconds",
        ],
        kind="stable",
    )


    if len(feature_df) <= max_segments:

        return feature_df


    indices = np.linspace(
        0,
        len(feature_df) - 1,
        max_segments,
    ).astype(int)


    return feature_df.iloc[
        indices
    ].copy()


# ======================================================================
# MAIN DATASET PROCESSOR
# ======================================================================

def process_dataset(
    dataset_name: str,
    config,
):

    print_header(
        f"{dataset_name} ARTIFACT INSPECTION"
    )


    # --------------------------------------------------------------
    # LOAD
    # --------------------------------------------------------------

    data = load_dataset(
        dataset_name
    )


    print(
        f"Loaded shape: {data.shape}"
    )


    print(
        f"Columns: {list(data.columns)}"
    )


    # --------------------------------------------------------------
    # STREAMS
    # --------------------------------------------------------------

    stream_columns = (
        get_stream_columns(
            dataset_name
        )
    )


    signal_columns = (
        get_signal_columns(
            dataset_name
        )
    )


    print(
        f"Temporal stream keys: {stream_columns}"
    )


    # --------------------------------------------------------------
    # FIRST PASS
    #
    # Generate continuous segments and
    # calculate features.
    # --------------------------------------------------------------

    feature_rows = []

    segment_cache = []


    streams = iterate_streams(
        data,
        stream_columns,
    )


    stream_count = 0


    for record_identifier, recording in streams:

        stream_count += 1


        if (
            stream_count % 20 == 0
        ):

            print(
                f"  Processed "
                f"{stream_count} streams"
            )


        if recording.empty:

            continue


        segments = (
            split_into_segments(
                recording,
                config,
                timestamp_column="timestamp",
            )
        )


        for segment_id, segment in enumerate(
            segments,
            start=1,
        ):

            if segment.empty:

                continue


            validation = (
                validate_segment(
                    segment,
                    config,
                    timestamp_column="timestamp",
                )
            )


            if not validation.get(
                "valid",
                False,
            ):

                continue


            clean_segment, timestamps = (
                prepare_timestamps(
                    segment
                )
            )


            if len(timestamps) < 2:

                continue


            duration = (
                timestamps[-1]
                -
                timestamps[0]
            )


            # ------------------------------------------------------
            # Do not perform artifact analysis on extremely
            # short timing fragments.
            # ------------------------------------------------------

            if (
                duration
                <
                MIN_ANALYSIS_DURATION_SECONDS
            ):

                continue


            row = extract_segment_features(
                clean_segment,
                dataset_name,
                record_identifier,
                segment_id,
            )


            if row is None:

                continue


            # ------------------------------------------------------
            # Preserve important metadata.
            # ------------------------------------------------------

            if "subject_id" in clean_segment.columns:

                row[
                    "subject_id"
                ] = str(
                    clean_segment[
                        "subject_id"
                    ].iloc[0]
                )


            if "record_id" in clean_segment.columns:

                row[
                    "record_id"
                ] = str(
                    clean_segment[
                        "record_id"
                    ].iloc[0]
                )


            if "sub_dataset" in clean_segment.columns:

                row[
                    "sub_dataset"
                ] = str(
                    clean_segment[
                        "sub_dataset"
                    ].iloc[0]
                )


            if "sensor_id" in clean_segment.columns:

                row[
                    "sensor_id"
                ] = str(
                    clean_segment[
                        "sensor_id"
                    ].iloc[0]
                )


            feature_rows.append(
                row
            )


            # Keep only representative segments
            # in memory for plotting later.

            segment_cache.append(
                (
                    row,
                    clean_segment,
                )
            )


    print(
        f"Temporal streams processed: {stream_count}"
    )


    feature_df = pd.DataFrame(
        feature_rows
    )


    if feature_df.empty:

        print(
            "No valid analysis segments were produced."
        )

        return (
            feature_df,
            pd.DataFrame(),
        )


    # --------------------------------------------------------------
    # DATASET STATISTICS
    # --------------------------------------------------------------

    dataset_statistics = (
        calculate_dataset_statistics(
            feature_df
        )
    )


    # --------------------------------------------------------------
    # CANDIDATE GENERATION
    # --------------------------------------------------------------

    candidate_rows = []


    for _, row in feature_df.iterrows():

        reasons = (
            generate_candidate_reason(
                row.to_dict(),
                dataset_statistics,
            )
        )


        is_candidate = (
            len(reasons) > 0
        )


        if is_candidate:

            candidate = (
                row.to_dict()
            )


            candidate[
                "candidate_artifact"
            ] = True


            candidate[
                "candidate_reason"
            ] = ";".join(
                reasons
            )


            # IMPORTANT:
            #
            # This is deliberately NOT:
            #
            # artifact_label = 1
            #
            # because we don't yet know whether
            # the event is actually an artifact.

            candidate[
                "manual_label"
            ] = ""


            candidate_rows.append(
                candidate
            )


    candidate_df = pd.DataFrame(
        candidate_rows
    )


    # --------------------------------------------------------------
    # SAVE SEGMENT FEATURES
    # --------------------------------------------------------------

    segment_output = (
        OUTPUT_ROOT
        / f"{dataset_name.lower().replace('-', '_')}"
        "_segment_features.csv"
    )


    feature_df.to_csv(
        segment_output,
        index=False,
    )


    # --------------------------------------------------------------
    # SAVE CANDIDATES
    # --------------------------------------------------------------

    candidate_output = (
        OUTPUT_ROOT
        / f"{dataset_name.lower().replace('-', '_')}"
        "_candidate_events.csv"
    )


    candidate_df.to_csv(
        candidate_output,
        index=False,
    )


    # --------------------------------------------------------------
    # PLOT REPRESENTATIVE SEGMENTS
    # --------------------------------------------------------------

    selected = select_representative_segments(
        feature_df,
        MAX_PLOTS_PER_DATASET,
    )


    selected_keys = set()


    for _, row in selected.iterrows():

        selected_keys.add(
            (
                str(
                    row[
                        "record_identifier"
                    ]
                ),
                int(
                    row[
                        "segment_id"
                    ]
                ),
            )
        )


    plot_count = 0


    for row, segment in segment_cache:

        key = (
            str(
                row[
                    "record_identifier"
                ]
            ),
            int(
                row[
                    "segment_id"
                ]
            ),
        )


        if key not in selected_keys:

            continue


        filename = (
            f"{dataset_name.lower().replace('-', '_')}"
            f"_record_{plot_count + 1:02d}"
            f"_segment_{int(row['segment_id'])}"
            ".png"
        )


        plot_path = (
            PLOTS_ROOT
            / filename
        )


        plot_segment(
            segment,
            dataset_name,
            row[
                "record_identifier"
            ],
            int(
                row[
                    "segment_id"
                ]
            ),
            plot_path,
        )


        plot_count += 1


    print_section(
        f"{dataset_name} INSPECTION SUMMARY"
    )


    print(
        f"Valid analysis segments : "
        f"{len(feature_df)}"
    )


    print(
        f"Candidate events         : "
        f"{len(candidate_df)}"
    )


    print(
        f"Representative plots     : "
        f"{plot_count}"
    )


    print(
        f"Segment features saved   : "
        f"{segment_output}"
    )


    print(
        f"Candidate events saved   : "
        f"{candidate_output}"
    )


    return (
        feature_df,
        candidate_df,
    )


# ======================================================================
# MANUAL LABEL TEMPLATE
# ======================================================================

def create_manual_label_template(
    candidate_tables,
):

    print_header(
        "CREATING MANUAL ARTIFACT-LABEL TEMPLATE"
    )


    all_candidates = []


    for dataset_name, candidate_df in (
        candidate_tables.items()
    ):

        if candidate_df.empty:

            continue


        candidate_copy = (
            candidate_df.copy()
        )


        candidate_copy[
            "dataset"
        ] = dataset_name


        all_candidates.append(
            candidate_copy
        )


    if not all_candidates:

        print(
            "No candidate events found."
        )

        return


    combined = pd.concat(
        all_candidates,
        ignore_index=True,
    )


    # --------------------------------------------------------------
    # Human annotation fields
    # --------------------------------------------------------------

    combined[
        "manual_label"
    ] = ""


    combined[
        "confidence"
    ] = ""


    combined[
        "reviewer_notes"
    ] = ""


    combined[
        "artifact_type"
    ] = ""


    # --------------------------------------------------------------
    # Recommended labels
    #
    # Keep them simple during the first
    # annotation round.
    # --------------------------------------------------------------

    combined[
        "allowed_label_values"
    ] = (
        "artifact / likely_fetal / "
        "uncertain"
    )


    output_path = (
        OUTPUT_ROOT
        / "manual_label_template.csv"
    )


    combined.to_csv(
        output_path,
        index=False,
    )


    print(
        f"Manual labeling template saved:\n"
        f"{output_path}"
    )


    print()
    print(
        "Recommended annotation labels:"
    )

    print(
        "    artifact"
    )

    print(
        "    likely_fetal"
    )

    print(
        "    uncertain"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print_header(
        "ARTIFACT INSPECTION / CANDIDATE GENERATION"
    )


    print(
        """
This stage begins AFTER the validated timing layer.

The purpose is to understand real signal behavior
before building an artifact ML model.

NO artifact removal is performed.
NO ML model is trained.
NO fetal movement classifier is trained.
"""
    )


    print(
        f"Project root:\n{PROJECT_ROOT}"
    )


    print(
        f"\nOutput directory:\n{OUTPUT_ROOT}"
    )


    # --------------------------------------------------------------
    # TIMING CONFIG
    # --------------------------------------------------------------

    config = TimingConfig()


    # --------------------------------------------------------------
    # PROCESS DATASETS
    # --------------------------------------------------------------

    datasets = [
        "COUGH",
        "FOUR-IMU",
        "OXFORD",
    ]


    candidate_tables = {}


    summary_rows = []


    for dataset_name in datasets:

        try:

            feature_df, candidate_df = (
                process_dataset(
                    dataset_name,
                    config,
                )
            )


            candidate_tables[
                dataset_name
            ] = candidate_df


            summary_rows.append(
                {
                    "dataset":
                        dataset_name,

                    "analysis_segments":
                        len(feature_df),

                    "candidate_events":
                        len(candidate_df),

                    "status":
                        "completed",
                }
            )


        except Exception as exc:

            print()
            print(
                f"ERROR processing "
                f"{dataset_name}:"
            )

            print(
                repr(exc)
            )


            summary_rows.append(
                {
                    "dataset":
                        dataset_name,

                    "analysis_segments":
                        0,

                    "candidate_events":
                        0,

                    "status":
                        f"ERROR: {exc}",
                }
            )


    # --------------------------------------------------------------
    # MANUAL LABEL FILE
    # --------------------------------------------------------------

    create_manual_label_template(
        candidate_tables
    )


    # --------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_path = (
        OUTPUT_ROOT
        / "inspection_summary.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False,
    )


    # --------------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------------

    print_header(
        "ARTIFACT INSPECTION COMPLETE"
    )


    print(
        "Outputs:"
    )


    print(
        f"    {OUTPUT_ROOT}"
    )


    print()
    print(
        "Next scientific step:"
    )


    print(
        """
1. Open the representative plots.
2. Examine high-activity and normal segments.
3. Examine candidate events.
4. Manually label candidates as:
       artifact
       likely_fetal
       uncertain
5. Review the artifact patterns.
6. Build a clean artifact-event dataset.
7. ONLY THEN design and train the artifact ML detector.
"""
    )


    print()
    print(
        "IMPORTANT:"
    )


    print(
        """
The candidate-generation rules are NOT the final
artifact detector.

They are only a transparent mechanism for finding
interesting regions for human inspection.

The final artifact classifier will be developed
after the labels have been validated.
"""
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()