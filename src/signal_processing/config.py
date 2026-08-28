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

    # Nyquist at 100 Hz is 50 Hz, so this band has plenty of
    # headroom. 1-20 Hz matches the same general-body-motion band
    # used for the fetal datasets (see FOUR_IMU_CONFIG /
    # OXFORD_CONFIG below) — appropriate here because COUGH is used
    # as a non-target-motion reference, not because it needs a
    # fetal-specific band. Low cut removes slow drift/gravity
    # offset; high cut removes high-frequency sensor noise while
    # keeping the general human-motion band.
    bandpass_hz=(1.0, 20.0),

    lowpass_hz=None,

    highpass_hz=None,

    # 4th-order Butterworth is the standard choice in the
    # accelerometer-based movement-detection literature (e.g.
    # Altini et al.) and is what PROCESSING_POLICY.zero_phase_filtering
    # below assumes (filtfilt doubles the effective order to 8th).
    filter_order=4,

    # 2-second windows, 50% overlap: short enough to localize
    # discrete motion events, long enough to compute a stable FFT
    # at 100 Hz (200 samples/window).
    window_seconds=2.0,

    overlap_fraction=0.5,
)


# ============================================================
# FOUR-IMU DATASET
# ============================================================

FOUR_IMU_CONFIG = DatasetConfig(

    name="Fetal Movement Dataset Recorded Using Four IMUs",

    # Current loader metadata reports approximately 30 Hz.
    # CONFIRM against sampling_audit.py's actual measured output
    # before trusting this for anything beyond the current estimate
    # — every value below is directly derived from it and must be
    # re-derived if the real rate differs.
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

    # IMPORTANT: Nyquist at 30 Hz is only 15 Hz. A 1-20 Hz band
    # (as literature commonly uses for higher-rate accelerometer
    # data) is NOT valid here — it would exceed Nyquist and
    # validate_config() below would reject it. 1-10 Hz keeps
    # meaningful headroom under the 15 Hz ceiling while still
    # covering the fetal-movement-relevant band reported in the
    # literature (e.g. Altini et al., 1-10 Hz for IMU-based fetal
    # movement). This is the dataset-specific reason this band
    # differs from COUGH/OXFORD — not an oversight.
    bandpass_hz=(1.0, 10.0),

    lowpass_hz=None,

    highpass_hz=None,

    filter_order=4,

    # 2-second windows at 30 Hz = 60 samples/window, still enough
    # for the 1-10 Hz band above. 50% overlap avoids missing events
    # that straddle a window boundary.
    window_seconds=2.0,

    overlap_fraction=0.5,
)


# ============================================================
# OXFORD DATASET
# ============================================================

OXFORD_CONFIG = DatasetConfig(

    name="Oxford Female Fetal Dataset",

    # Confirmed against the dataset's published documentation
    # (ADXL355 accelerometer, 500 Hz) and now set at the loader
    # level too (dataset.py OxfordLoader) — this is no longer a
    # workaround value living only in this file.
    sampling_rate_hz=500.0,

    timestamp_unit="seconds",

    signal_type="BCG",

    signal_axes=BCG_AXES,

    accelerometer=False,

    gyroscope=False,

    magnetometer=False,

    bcg=True,

    multiple_sensors=False,

    sensor_count=1,

    target_sampling_rate_hz=None,

    # Nyquist at 500 Hz is 250 Hz, so 1-20 Hz has very wide
    # headroom. Kept identical to the fetal-movement band literature
    # uses at comparable sampling rates, not widened just because
    # Nyquist allows it — a wider band would just reintroduce the
    # high-frequency noise this filter exists to remove.
    bandpass_hz=(1.0, 20.0),

    lowpass_hz=None,

    highpass_hz=None,

    filter_order=4,

    # 2-second windows at 500 Hz = 1000 samples/window.
    window_seconds=2.0,

    overlap_fraction=0.5,
)


# ============================================================
# MATERNAL MPU6050 DATASET (REAL DEPLOYMENT DATA)
# ============================================================
#
# Real maternal data is not collected yet. Every processing
# parameter below is intentionally left None/unset — this dataset's
# sampling rate is ESTIMATED PER RECORDING by MaternalMPU6050Loader
# (see dataset.py), not assumed, because real BLE/serial capture
# commonly drifts from a nominal rate. Do not hardcode a bandpass
# or window size here until real recordings exist to derive
# Nyquist-safe values from, the same way FOUR_IMU's band above was
# derived from its own confirmed sampling rate.

MATERNAL_MPU6050_CONFIG = DatasetConfig(

    name="Maternal MPU6050 Wearable Dataset",

    sampling_rate_hz=None,

    timestamp_unit="seconds",

    signal_type="IMU",

    signal_axes=COMMON_IMU_AXES,

    accelerometer=True,

    gyroscope=True,

    magnetometer=False,

    bcg=False,

    multiple_sensors=False,

    sensor_count=1,

    target_sampling_rate_hz=None,

    bandpass_hz=None,

    lowpass_hz=None,

    highpass_hz=None,

    filter_order=4,

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
    "MATERNAL_MPU6050": MATERNAL_MPU6050_CONFIG,
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
# SIGNAL QUALITY THRESHOLDS
# ============================================================

@dataclass(frozen=True)
class QualityThresholds:
    """
    Window-level signal quality thresholds, used by
    02_signal_processing.py to flag GOOD / QUESTIONABLE / INVALID
    windows before they ever reach the artifact layer.

    These catch SENSOR-level faults (dead sensor, clipping,
    implausible amplitude). They do NOT catch motion-SOURCE
    contamination (maternal movement, belt shift, etc.) — that
    distinction matters: sensor faults belong here, motion-source
    questions belong to the artifact layer that runs after this one.
    """

    # A window with more than this fraction of NaN/missing samples
    # in any signal channel is INVALID, not just QUESTIONABLE.
    max_missing_fraction: float = 0.05

    # A flat-lined channel (std below this, in the sensor's own
    # units) for an entire window suggests a dead/disconnected
    # sensor rather than a genuinely still fetus/mother.
    min_channel_std: float = 1e-6

    # Consecutive identical samples beyond this count is treated as
    # the same flat-line/dead-sensor signal, even if overall std
    # isn't exactly zero (e.g. quantization noise).
    max_consecutive_identical_samples: int = 20

    # Clipping: true ADC saturation repeats the exact same value
    # many times within a window — see assess_window_quality() for
    # why this uses exact ties rather than proximity to the window's
    # min/max (a smooth peak is close to the extreme but not equal
    # to it repeatedly). 5% is deliberately more lenient than a
    # missingness threshold — a couple of coincidental exact ties in
    # a real signal (e.g. a flat few-sample peak) shouldn't fail a
    # whole window on their own.
    max_clipped_fraction: float = 0.05

    # Amplitude sanity bounds are intentionally NOT set to one
    # number for all datasets — accelerometer units/ranges differ
    # (g vs m/s^2, IMU vs BCG). Leave None until each dataset's
    # actual observed amplitude range has been checked; a wrong
    # guessed bound is worse than no bound because it silently
    # discards real data.
    max_abs_amplitude: Optional[float] = None


QUALITY_THRESHOLDS = QualityThresholds()


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