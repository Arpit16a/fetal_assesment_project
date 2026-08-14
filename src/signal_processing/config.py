"""
Signal Processing Configuration
================================

Central configuration for the signal-processing layer.

IMPORTANT
---------
This file contains:
1. Dataset-specific signal facts
2. Processing parameters
3. Validation constraints

No actual signal processing should happen here.

Dataset facts are based on the current loader validation.
Processing parameters should only be enabled once they have
been scientifically justified and verified.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


# ============================================================
# COMMON SIGNAL DEFINITIONS
# ============================================================

COMMON_IMU_AXES = (
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "mx",
    "my",
    "mz",
)

ACCELEROMETER_AXES = (
    "ax",
    "ay",
    "az",
)

GYROSCOPE_AXES = (
    "gx",
    "gy",
    "gz",
)

MAGNETOMETER_AXES = (
    "mx",
    "my",
    "mz",
)

BCG_AXES = (
    "bcg_x",
    "bcg_y",
    "bcg_z",
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuration describing one dataset.

    This object describes both known properties of the raw
    dataset and processing decisions that will be applied later.
    """

    name: str

    # --------------------------------------------------------
    # Dataset properties
    # --------------------------------------------------------

    sampling_rate_hz: Optional[float]

    timestamp_unit: str

    signal_type: str

    signal_axes: Tuple[str, ...]

    # --------------------------------------------------------
    # Available modalities
    # --------------------------------------------------------

    accelerometer: bool = False
    gyroscope: bool = False
    magnetometer: bool = False
    bcg: bool = False

    # --------------------------------------------------------
    # Dataset-specific structure
    # --------------------------------------------------------

    multiple_sensors: bool = False
    sensor_count: int = 1

    # --------------------------------------------------------
    # Processing configuration
    #
    # These remain None until experimentally/scientifically
    # established.
    # --------------------------------------------------------

    target_sampling_rate_hz: Optional[float] = None

    bandpass_hz: Optional[Tuple[float, float]] = None

    lowpass_hz: Optional[float] = None

    highpass_hz: Optional[float] = None

    filter_order: Optional[int] = None

    window_seconds: Optional[float] = None

    overlap_fraction: Optional[float] = None


# ============================================================
# COUGH DATASET
# ============================================================

COUGH_CONFIG = DatasetConfig(

    name="Multimodal Cough Dataset",

    # Loader discovery showed timestamps in seconds.
    #
    # We are NOT hard-coding the exact sampling rate yet.
    # It should be measured from the timestamps.
    sampling_rate_hz=100.0,

    timestamp_unit="seconds",

    signal_type="IMU",

    signal_axes=COMMON_IMU_AXES,

    accelerometer=True,

    gyroscope=True,

    magnetometer=True,

    bcg=False,

    multiple_sensors=False,

    sensor_count=1,

    # Processing parameters intentionally not fixed yet.
    target_sampling_rate_hz=None,

    bandpass_hz=None,

    lowpass_hz=None,

    highpass_hz=None,

    filter_order=None,

    window_seconds=None,

    overlap_fraction=None,
)


# ============================================================
# FOUR-IMU DATASET
# ============================================================

FOUR_IMU_CONFIG = DatasetConfig(

    name="Fetal Movement Dataset Recorded Using Four IMUs",

    # Current loader metadata reports approximately 30 Hz.
    sampling_rate_hz=30.0,

    timestamp_unit="seconds",

    signal_type="IMU",

    signal_axes=COMMON_IMU_AXES,

    accelerometer=True,

    gyroscope=True,

    magnetometer=False,

    bcg=False,

    multiple_sensors=True,

    sensor_count=4,

    # Do NOT resample automatically at this stage.
    target_sampling_rate_hz=None,

    bandpass_hz=None,

    lowpass_hz=None,

    highpass_hz=None,

    filter_order=None,

    window_seconds=None,

    overlap_fraction=None,
)


# ============================================================
# OXFORD DATASET
# ============================================================

OXFORD_CONFIG = DatasetConfig(

    name="Oxford Female Fetal Dataset",

    # The loader currently reports sampling_rate_hz=None.
    #
    # The signal is represented using sample indices.
    # We must establish the true sampling frequency before
    # designing any frequency-domain filter.
    sampling_rate_hz=500.0,

    timestamp_unit="sample_index",

    signal_type="BCG",

    signal_axes=BCG_AXES,

    accelerometer=False,

    gyroscope=False,

    magnetometer=False,

    bcg=True,

    multiple_sensors=False,

    sensor_count=1,

    target_sampling_rate_hz=None,

    bandpass_hz=None,

    lowpass_hz=None,

    highpass_hz=None,

    filter_order=None,

    window_seconds=None,

    overlap_fraction=None,
)


# ============================================================
# MASTER CONFIGURATION REGISTRY
# ============================================================

DATASET_CONFIGS = {
    "COUGH": COUGH_CONFIG,
    "FOUR_IMU": FOUR_IMU_CONFIG,
    "OXFORD": OXFORD_CONFIG,
}


# ============================================================
# COMMON PROCESSING POLICY
# ============================================================

@dataclass(frozen=True)
class ProcessingPolicy:
    """
    Global rules for signal processing.

    These are methodological safeguards rather than
    dataset-specific filter parameters.
    """

    # --------------------------------------------------------
    # Resampling
    # --------------------------------------------------------

    allow_resampling: bool = True

    # Never resample across independent recordings.
    preserve_record_boundaries: bool = True

    # --------------------------------------------------------
    # Filtering
    # --------------------------------------------------------

    zero_phase_filtering: bool = True

    # --------------------------------------------------------
    # Missing data
    # --------------------------------------------------------

    allow_missing_signal_values: bool = False

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    preserve_labels: bool = True

    preserve_subject_ids: bool = True

    preserve_record_ids: bool = True

    # --------------------------------------------------------
    # Leakage prevention
    # --------------------------------------------------------

    fit_transform_parameters_on_training_only: bool = True


PROCESSING_POLICY = ProcessingPolicy()


# ============================================================
# VALIDATION HELPERS
# ============================================================

def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """
    Return configuration for a dataset.

    Parameters
    ----------
    dataset_name:
        Dataset identifier:
        COUGH
        FOUR_IMU
        OXFORD

    Returns
    -------
    DatasetConfig
    """

    key = dataset_name.upper()

    if key not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {list(DATASET_CONFIGS)}"
        )

    return DATASET_CONFIGS[key]


def validate_config(config: DatasetConfig) -> None:
    """
    Validate basic configuration consistency.
    """

    if config.sampling_rate_hz is not None:
        if config.sampling_rate_hz <= 0:
            raise ValueError(
                "Sampling rate must be greater than zero."
            )

    if config.target_sampling_rate_hz is not None:
        if config.target_sampling_rate_hz <= 0:
            raise ValueError(
                "Target sampling rate must be greater than zero."
            )

    if config.bandpass_hz is not None:

        low, high = config.bandpass_hz

        if low <= 0:
            raise ValueError(
                "Band-pass lower frequency must be > 0."
            )

        if high <= low:
            raise ValueError(
                "Band-pass upper frequency must be greater "
                "than the lower frequency."
            )

        if config.sampling_rate_hz is not None:

            nyquist = config.sampling_rate_hz / 2

            if high >= nyquist:
                raise ValueError(
                    f"Band-pass upper frequency ({high} Hz) "
                    f"must be below Nyquist ({nyquist} Hz)."
                )

    if config.window_seconds is not None:

        if config.window_seconds <= 0:
            raise ValueError(
                "Window duration must be greater than zero."
            )

    if config.overlap_fraction is not None:

        if not 0 <= config.overlap_fraction < 1:
            raise ValueError(
                "Overlap fraction must satisfy "
                "0 <= overlap < 1."
            )


# ============================================================
# VALIDATE ALL CONFIGURATIONS ON IMPORT
# ============================================================

for _config in DATASET_CONFIGS.values():
    validate_config(_config)