"""
02a_oxford_artifact_investigation.py

STEP 5A–5D
Oxford Artifact Candidate Investigation

Purpose
-------
Investigate why the current artifact-inspection stage generated
zero Oxford candidate events.

This script DOES NOT:
    - modify the artifact detector
    - change artifact thresholds
    - remove artifacts
    - train ML models
    - classify fetal movements
    - alter the timing layer
    - alter the Oxford dataset loader

It investigates:

    5A. Oxford feature availability
    5B. Oxford feature distributions
    5C. Representative Oxford signal characteristics
    5D. Current candidate-generation applicability

The purpose is diagnostic only.

Important
---------
If a signal channel is absent/NaN in the loader output, the script
must NOT interpret that as "no activity".

It must report the channel as unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =============================================================================
# PROJECT PATHS
# =============================================================================

# Script location:
#     project_root/
#         src/
#             signal_processing/
#                 02a_oxford_artifact_investigation.py
#
# parents[0] = signal_processing
# parents[1] = src
# parents[2] = project_root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_INSPECTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artifact_inspection"
)

OUTPUT_DIR = (
    ARTIFACT_INSPECTION_DIR
    / "oxford_5a_5d_investigation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEGMENT_FEATURE_FILE = (
    ARTIFACT_INSPECTION_DIR
    / "oxford_segment_features.csv"
)

CANDIDATE_FILE = (
    ARTIFACT_INSPECTION_DIR
    / "oxford_candidate_events.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# IMPORTANT:
#
# This is the CURRENT candidate threshold being investigated.
#
# It is NOT being declared as the final threshold.
#
# We are checking whether the current rule can operate on Oxford.
#
CURRENT_Z_THRESHOLD = 3.0


# =============================================================================
# EXPECTED FEATURE GROUPS
# =============================================================================

ACC_FEATURES = [
    "acc_mean",
    "acc_std",
    "acc_rms",
    "acc_peak_to_peak",
]

JERK_FEATURES = [
    "jerk_rms",
    "jerk_peak_to_peak",
]

GYRO_FEATURES = [
    "gyro_mean",
    "gyro_std",
    "gyro_rms",
    "gyro_peak_to_peak",
]

ACC_FREQ_FEATURES = [
    "acc_dominant_frequency_hz",
    "acc_spectral_centroid_hz",
    "acc_spectral_bandwidth_hz",
    "acc_spectral_energy",
]

GYRO_FREQ_FEATURES = [
    "gyro_dominant_frequency_hz",
    "gyro_spectral_centroid_hz",
    "gyro_spectral_bandwidth_hz",
    "gyro_spectral_energy",
]

BCG_FEATURES = [
    "bcg_mean",
    "bcg_std",
    "bcg_rms",
    "bcg_peak_to_peak",
    "bcg_dominant_frequency_hz",
    "bcg_spectral_centroid_hz",
    "bcg_spectral_bandwidth_hz",
    "bcg_spectral_energy",
]


# =============================================================================
# PRINT HELPERS
# =============================================================================

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def subsection(title: str) -> None:
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


# =============================================================================
# NUMERICAL HELPERS
# =============================================================================

def numeric_series(
    df: pd.DataFrame,
    feature: str,
) -> pd.Series:
    """
    Convert a dataframe column to numeric safely.

    Missing/non-numeric values become NaN.
    """
    if feature not in df.columns:
        return pd.Series(
            dtype=float,
            index=df.index,
        )

    return pd.to_numeric(
        df[feature],
        errors="coerce",
    )


def finite_values(
    df: pd.DataFrame,
    feature: str,
) -> pd.Series:
    """
    Return only finite numerical values for a feature.
    """
    values = numeric_series(
        df,
        feature,
    )

    if len(values) == 0:
        return values

    values = values[
        np.isfinite(values.to_numpy())
    ]

    return values


def feature_available(
    df: pd.DataFrame,
    feature: str,
) -> bool:
    """
    Return True if at least one finite value exists.
    """
    if feature not in df.columns:
        return False

    values = finite_values(
        df,
        feature,
    )

    return len(values) > 0


# =============================================================================
# FILE VALIDATION
# =============================================================================

def validate_input_files() -> None:

    section("INPUT FILE VALIDATION")

    print("Project root:")
    print(f"    {PROJECT_ROOT}")

    print()
    print("Expected Oxford segment feature file:")
    print(f"    {SEGMENT_FEATURE_FILE}")

    if not SEGMENT_FEATURE_FILE.exists():
        raise FileNotFoundError(
            "Oxford segment feature file not found:\n"
            f"{SEGMENT_FEATURE_FILE}"
        )

    print("✓ Oxford segment feature file found")

    print()
    print("Expected Oxford candidate file:")
    print(f"    {CANDIDATE_FILE}")

    if CANDIDATE_FILE.exists():

        size = CANDIDATE_FILE.stat().st_size

        if size == 0:

            print(
                "✓ Candidate file exists but is empty."
            )

            print(
                "  This is consistent with the reported "
                "0 Oxford candidates."
            )

        else:

            print(
                f"✓ Candidate file found "
                f"({size:,} bytes)"
            )

    else:

        print(
            "⚠ Oxford candidate file does not exist."
        )


# =============================================================================
# LOAD SEGMENT FEATURES
# =============================================================================

def load_segment_features() -> pd.DataFrame:

    section("LOADING OXFORD SEGMENT FEATURES")

    df = pd.read_csv(
        SEGMENT_FEATURE_FILE
    )

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    print()
    print("Columns:")

    for column in df.columns:
        print(f"    {column}")

    return df


# =============================================================================
# STEP 5A
# FEATURE AVAILABILITY AUDIT
# =============================================================================

def inspect_feature_availability(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section(
        "STEP 5A — OXFORD FEATURE AVAILABILITY"
    )

    print(
        "Question:"
    )

    print(
        "Are Oxford accelerometer/gyroscope features actually "
        "available to the artifact detector?"
    )

    groups = {
        "accelerometer": ACC_FEATURES,
        "jerk": JERK_FEATURES,
        "gyroscope": GYRO_FEATURES,
        "accelerometer_frequency": ACC_FREQ_FEATURES,
        "gyroscope_frequency": GYRO_FREQ_FEATURES,
        "bcg": BCG_FEATURES,
    }

    rows = []

    for group_name, features in groups.items():

        for feature in features:

            if feature not in df.columns:

                rows.append(
                    {
                        "feature_group": group_name,
                        "feature": feature,
                        "exists": False,
                        "finite_count": 0,
                        "nan_count": len(df),
                        "inf_count": 0,
                        "availability_percent": 0.0,
                    }
                )

                continue

            values = numeric_series(
                df,
                feature,
            )

            array = values.to_numpy(
                dtype=float
            )

            finite_mask = np.isfinite(
                array
            )

            finite_count = int(
                finite_mask.sum()
            )

            nan_count = int(
                np.isnan(array).sum()
            )

            inf_count = int(
                np.isinf(array).sum()
            )

            availability = (
                finite_count
                / len(df)
                * 100.0
                if len(df) > 0
                else 0.0
            )

            rows.append(
                {
                    "feature_group": group_name,
                    "feature": feature,
                    "exists": True,
                    "finite_count": finite_count,
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "availability_percent": availability,
                }
            )

    audit = pd.DataFrame(rows)

    print()
    print(
        audit.to_string(
            index=False
        )
    )

    audit_file = (
        OUTPUT_DIR
        / "oxford_feature_availability.csv"
    )

    audit.to_csv(
        audit_file,
        index=False,
    )

    print()
    print("Saved:")
    print(f"    {audit_file}")

    # -------------------------------------------------------------------------
    # Group-level interpretation
    # -------------------------------------------------------------------------

    print()

    for group_name in groups:

        group = audit[
            audit["feature_group"] == group_name
        ]

        if len(group) == 0:
            continue

        availability = float(
            group[
                "availability_percent"
            ].mean()
        )

        print(
            f"{group_name:30s}: "
            f"{availability:7.2f}% feature availability"
        )

    # -------------------------------------------------------------------------
    # Determine broad availability
    # -------------------------------------------------------------------------

    acc_available = (
        audit[
            audit["feature_group"].isin(
                [
                    "accelerometer",
                    "jerk",
                    "accelerometer_frequency",
                ]
            )
        ]["availability_percent"] > 0
    ).any()

    gyro_available = (
        audit[
            audit["feature_group"].isin(
                [
                    "gyroscope",
                    "gyroscope_frequency",
                ]
            )
        ]["availability_percent"] > 0
    ).any()

    bcg_available = (
        audit[
            audit["feature_group"] == "bcg"
        ]["availability_percent"] > 0
    ).any()

    # -------------------------------------------------------------------------
    # Explicit conclusion
    # -------------------------------------------------------------------------

    subsection(
        "5A INTERPRETATION"
    )

    if not acc_available:

        print(
            "⚠ Oxford accelerometer/jerk features are "
            "NOT available."
        )

        print(
            "  They are absent or contain no finite values "
            "in the Oxford feature table."
        )

    else:

        print(
            "✓ Oxford accelerometer/jerk features are available."
        )

    if not gyro_available:

        print(
            "⚠ Oxford gyroscope features are NOT available."
        )

        print(
            "  They are absent or contain no finite values "
            "in the Oxford feature table."
        )

    else:

        print(
            "✓ Oxford gyroscope features are available."
        )

    if bcg_available:

        print(
            "✓ Oxford BCG features are available."
        )

    else:

        print(
            "⚠ Oxford BCG features are not available."
        )

    if not acc_available and not gyro_available:

        print()
        print(
            "IMPORTANT CONCLUSION:"
        )

        print(
            "The current IMU-based artifact candidate "
            "generator cannot meaningfully operate on Oxford."
        )

        print()
        print(
            "The zero-candidate result therefore does NOT mean:"
        )

        print(
            "    'Oxford contains no artifacts.'"
        )

        print()
        print(
            "It means:"
        )

        print(
            "    'The Oxford feature table used by this stage "
            "does not contain usable IMU features.'"
        )

    return audit


# =============================================================================
# STEP 5B
# FEATURE DISTRIBUTION ANALYSIS
# =============================================================================

def feature_distribution_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section(
        "STEP 5B — OXFORD FEATURE DISTRIBUTIONS"
    )

    print(
        "This section examines the numerical distributions "
        "without modifying the detector."
    )

    feature_list = (
        ACC_FEATURES
        + JERK_FEATURES
        + GYRO_FEATURES
        + ACC_FREQ_FEATURES
        + GYRO_FREQ_FEATURES
        + BCG_FEATURES
    )

    rows = []

    for feature in feature_list:

        if feature not in df.columns:

            rows.append(
                {
                    "feature": feature,
                    "n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "median": np.nan,
                    "max": np.nan,
                    "zscore_max": np.nan,
                }
            )

            continue

        values = finite_values(
            df,
            feature,
        )

        if len(values) == 0:

            rows.append(
                {
                    "feature": feature,
                    "n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "median": np.nan,
                    "max": np.nan,
                    "zscore_max": np.nan,
                }
            )

            continue

        mean = float(
            values.mean()
        )

        std = float(
            values.std(
                ddof=0
            )
        )

        if std > 0:

            zscores = (
                values - mean
            ) / std

            zscore_max = float(
                zscores.max()
            )

        else:

            zscore_max = 0.0

        rows.append(
            {
                "feature": feature,
                "n": int(len(values)),
                "mean": mean,
                "std": std,
                "min": float(values.min()),
                "median": float(values.median()),
                "max": float(values.max()),
                "zscore_max": zscore_max,
            }
        )

    distribution = pd.DataFrame(
        rows
    )

    print()
    print(
        distribution.to_string(
            index=False
        )
    )

    distribution_file = (
        OUTPUT_DIR
        / "oxford_feature_distributions.csv"
    )

    distribution.to_csv(
        distribution_file,
        index=False,
    )

    print()
    print("Saved:")
    print(
        f"    {distribution_file}"
    )

    # -------------------------------------------------------------------------
    # Features capable of exceeding current Z threshold
    # -------------------------------------------------------------------------

    subsection(
        "CURRENT Z-THRESHOLD DIAGNOSTIC"
    )

    available_distribution = distribution[
        distribution["n"] > 0
    ].copy()

    if len(available_distribution) > 0:

        threshold_crossers = (
            available_distribution[
                available_distribution[
                    "zscore_max"
                ] >= CURRENT_Z_THRESHOLD
            ]
        )

        print(
            f"Current diagnostic threshold: "
            f"Z > {CURRENT_Z_THRESHOLD}"
        )

        print()

        if len(threshold_crossers) == 0:

            print(
                "No available Oxford feature reaches "
                f"Z >= {CURRENT_Z_THRESHOLD}."
            )

            print(
                "This would make a Z-based candidate rule "
                "unable to identify events from these features."
            )

        else:

            print(
                "Features reaching the current diagnostic "
                "Z threshold:"
            )

            print(
                threshold_crossers[
                    [
                        "feature",
                        "n",
                        "mean",
                        "std",
                        "max",
                        "zscore_max",
                    ]
                ].to_string(
                    index=False
                )
            )

    else:

        print(
            "No finite Oxford feature distributions are "
            "available for analysis."
        )

    # -------------------------------------------------------------------------
    # BCG diagnostic
    # -------------------------------------------------------------------------

    subsection(
        "BCG DISTRIBUTION — DIAGNOSTIC ONLY"
    )

    bcg_distribution = distribution[
        distribution["feature"].isin(
            BCG_FEATURES
        )
    ]

    if len(bcg_distribution) > 0:

        print(
            "Oxford BCG statistics are displayed for "
            "understanding Oxford signal behavior."
        )

        print(
            "They are NOT being converted into artifact labels."
        )

        print()

        print(
            bcg_distribution.to_string(
                index=False
            )
        )

    else:

        print(
            "No BCG features were found."
        )

    # -------------------------------------------------------------------------
    # Plot BCG feature distributions
    # -------------------------------------------------------------------------

    available_bcg = []

    for feature in BCG_FEATURES:

        if feature not in df.columns:
            continue

        values = finite_values(
            df,
            feature,
        )

        if len(values) > 0:

            available_bcg.append(
                feature
            )

    if available_bcg:

        fig, axes = plt.subplots(
            2,
            4,
            figsize=(18, 9),
        )

        axes = axes.flatten()

        for ax, feature in zip(
            axes,
            available_bcg,
        ):

            values = finite_values(
                df,
                feature,
            )

            bins = min(
                12,
                max(
                    3,
                    len(values)
                )
            )

            ax.hist(
                values,
                bins=bins,
            )

            ax.set_title(
                feature
            )

            ax.set_xlabel(
                "Value"
            )

            ax.set_ylabel(
                "Count"
            )

            ax.grid(
                True,
                alpha=0.25,
            )

        for ax in axes[
            len(available_bcg):
        ]:

            ax.axis(
                "off"
            )

        fig.suptitle(
            "Oxford BCG Feature Distributions",
            fontsize=16,
        )

        fig.tight_layout()

        plot_file = (
            OUTPUT_DIR
            / "oxford_bcg_feature_distributions.png"
        )

        fig.savefig(
            plot_file,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

        print()
        print(
            "Saved BCG distribution plot:"
        )

        print(
            f"    {plot_file}"
        )

    return distribution


# =============================================================================
# STEP 5C
# REPRESENTATIVE OXFORD SIGNAL AUDIT
# =============================================================================

def representative_signal_audit(
    df: pd.DataFrame,
) -> None:

    section(
        "STEP 5C — REPRESENTATIVE OXFORD SIGNAL AUDIT"
    )

    print(
        "The representative Oxford plots supplied for this "
        "investigation are being interpreted together with "
        "the Oxford feature table."
    )

    print()
    print(
        "Observed structure from the supplied plots:"
    )

    print(
        "    Accelerometer panel : blank"
    )

    print(
        "    Gyroscope panel     : blank"
    )

    print(
        "    BCG panel           : populated"
    )

    print()
    print(
        "This is consistent with the feature-table diagnosis:"
    )

    print(
        "    IMU-derived features : unavailable / NaN"
    )

    print(
        "    BCG-derived features : finite"
    )

    # -------------------------------------------------------------------------
    # Sensor identity
    # -------------------------------------------------------------------------

    subsection(
        "OXFORD SENSOR IDENTITY"
    )

    if "sensor_id" in df.columns:

        print(
            df[
                "sensor_id"
            ].value_counts(
                dropna=False
            ).to_string()
        )

    else:

        print(
            "sensor_id column not present."
        )

    # -------------------------------------------------------------------------
    # Oxford records
    # -------------------------------------------------------------------------

    subsection(
        "OXFORD RECORD INFORMATION"
    )

    id_columns = [
        "record_id",
        "record_identifier",
        "sample_count",
        "duration_seconds",
        "sampling_rate_hz",
    ]

    available = [
        column
        for column in id_columns
        if column in df.columns
    ]

    if available:

        print(
            df[
                available
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "No standard record-information columns "
            "were found."
        )

    # -------------------------------------------------------------------------
    # BCG finite-value summary
    # -------------------------------------------------------------------------

    subsection(
        "BCG SIGNAL AVAILABILITY"
    )

    for feature in BCG_FEATURES:

        if feature not in df.columns:

            print(
                f"{feature:40s}: column absent"
            )

            continue

        values = finite_values(
            df,
            feature,
        )

        print(
            f"{feature:40s}: "
            f"{len(values):,} finite values"
        )

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------

    subsection(
        "5C INTERPRETATION"
    )

    print(
        "The supplied representative Oxford plots do not "
        "show usable IMU traces in the plotted accelerometer "
        "and gyroscope panels."
    )

    print()
    print(
        "The BCG traces clearly contain signal variation "
        "and isolated disturbances/spikes in some records."
    )

    print()
    print(
        "However, this stage must NOT automatically call "
        "those BCG disturbances artifacts."
    )

    print()
    print(
        "Why?"
    )

    print(
        "Because Step 5 is intended to discover and validate "
        "artifact definitions before building the final detector."
    )

    print()
    print(
        "Therefore:"
    )

    print(
        "    BCG behavior = worth investigating"
    )

    print(
        "    BCG spike = NOT automatically an artifact"
    )

    print(
        "    Oxford 0 IMU candidates = NOT evidence of clean data"
    )


# =============================================================================
# STEP 5D
# CURRENT CANDIDATE LOGIC AUDIT
# =============================================================================

def candidate_logic_audit(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section(
        "STEP 5D — CURRENT CANDIDATE-GENERATION LOGIC AUDIT"
    )

    print(
        f"Current diagnostic Z threshold: "
        f"{CURRENT_Z_THRESHOLD}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This script does NOT change this threshold."
    )

    print(
        "It only determines whether the Oxford feature table "
        "contains the variables required to apply an IMU-based rule."
    )

    # -------------------------------------------------------------------------
    # Determine feature availability
    # -------------------------------------------------------------------------

    required_groups = {
        "acceleration": ACC_FEATURES,
        "jerk": JERK_FEATURES,
        "gyroscope": GYRO_FEATURES,
    }

    rows = []

    for group_name, features in required_groups.items():

        available_features = []

        for feature in features:

            if feature_available(
                df,
                feature,
            ):

                available_features.append(
                    feature
                )

        rows.append(
            {
                "feature_group": group_name,
                "required_features": len(features),
                "available_features": len(
                    available_features
                ),
                "available_feature_names": (
                    ", ".join(
                        available_features
                    )
                    if available_features
                    else ""
                ),
            }
        )

    logic_df = pd.DataFrame(
        rows
    )

    print()
    print(
        logic_df.to_string(
            index=False
        )
    )

    logic_file = (
        OUTPUT_DIR
        / "oxford_candidate_logic_audit.csv"
    )

    logic_df.to_csv(
        logic_file,
        index=False,
    )

    print()
    print(
        "Saved:"
    )

    print(
        f"    {logic_file}"
    )

    # -------------------------------------------------------------------------
    # Explicit rule applicability
    # -------------------------------------------------------------------------

    accel_available = (
        logic_df.loc[
            logic_df["feature_group"]
            == "acceleration",
            "available_features",
        ].iloc[0]
        > 0
    )

    jerk_available = (
        logic_df.loc[
            logic_df["feature_group"]
            == "jerk",
            "available_features",
        ].iloc[0]
        > 0
    )

    gyro_available = (
        logic_df.loc[
            logic_df["feature_group"]
            == "gyroscope",
            "available_features",
        ].iloc[0]
        > 0
    )

    subsection(
        "CURRENT CANDIDATE RULE APPLICABILITY"
    )

    print(
        f"Acceleration available : "
        f"{accel_available}"
    )

    print(
        f"Jerk available         : "
        f"{jerk_available}"
    )

    print(
        f"Gyroscope available    : "
        f"{gyro_available}"
    )

    print()

    if not accel_available:

        print(
            "✗ Acceleration-based candidate detection "
            "cannot operate on Oxford."
        )

    if not jerk_available:

        print(
            "✗ Jerk-based candidate detection "
            "cannot operate on Oxford."
        )

    if not gyro_available:

        print(
            "✗ Gyroscope-based candidate detection "
            "cannot operate on Oxford."
        )

    # -------------------------------------------------------------------------
    # Candidate file status
    # -------------------------------------------------------------------------

    subsection(
        "EXISTING OXFORD CANDIDATE FILE"
    )

    if CANDIDATE_FILE.exists():

        size = CANDIDATE_FILE.stat().st_size

        if size == 0:

            print(
                "Candidate file is empty."
            )

            print(
                "This matches the reported:"
            )

            print(
                "    Candidate events: 0"
            )

        else:

            try:

                candidate_df = pd.read_csv(
                    CANDIDATE_FILE
                )

                print(
                    f"Existing candidate rows: "
                    f"{len(candidate_df):,}"
                )

                print()
                print(
                    "Candidate columns:"
                )

                print(
                    candidate_df.columns.tolist()
                )

            except Exception as exc:

                print(
                    "Candidate file could not be "
                    "read as a normal CSV."
                )

                print(
                    f"Reason: {exc}"
                )

    else:

        print(
            "No candidate file exists."
        )

    # -------------------------------------------------------------------------
    # Final interpretation
    # -------------------------------------------------------------------------

    subsection(
        "5D INTERPRETATION"
    )

    if (
        not accel_available
        and not jerk_available
        and not gyro_available
    ):

        print(
            "The current IMU-based candidate-generation "
            "rule is NOT applicable to Oxford."
        )

        print()
        print(
            "Therefore the zero-candidate result is explained "
            "by missing IMU feature availability rather than "
            "by a demonstrated absence of artifacts."
        )

    else:

        print(
            "At least some IMU-derived features are available."
        )

        print(
            "Further threshold-level analysis is therefore "
            "appropriate."
        )

    return logic_df


# =============================================================================
# GENERATE FINAL REPORT
# =============================================================================

def generate_final_report(
    feature_audit: pd.DataFrame,
    distribution: pd.DataFrame,
    logic_audit: pd.DataFrame,
) -> None:

    section(
        "FINAL STEP 5A–5D CONCLUSION"
    )

    # -------------------------------------------------------------------------
    # Availability summary
    # -------------------------------------------------------------------------

    acc_available = (
        feature_audit[
            feature_audit["feature_group"].isin(
                [
                    "accelerometer",
                    "jerk",
                    "accelerometer_frequency",
                ]
            )
        ]["availability_percent"] > 0
    ).any()

    gyro_available = (
        feature_audit[
            feature_audit["feature_group"].isin(
                [
                    "gyroscope",
                    "gyroscope_frequency",
                ]
            )
        ]["availability_percent"] > 0
    ).any()

    bcg_available = (
        feature_audit[
            feature_audit["feature_group"] == "bcg"
        ]["availability_percent"] > 0
    ).any()

    current_imu_rule_applicable = (
        acc_available
        and gyro_available
    )

    # -------------------------------------------------------------------------
    # Conclusion table
    # -------------------------------------------------------------------------

    conclusion_rows = [
        {
            "check": "Oxford accelerometer available",
            "result": acc_available,
        },
        {
            "check": "Oxford gyroscope available",
            "result": gyro_available,
        },
        {
            "check": "Oxford BCG available",
            "result": bcg_available,
        },
        {
            "check": "Current IMU candidate rule applicable",
            "result": current_imu_rule_applicable,
        },
    ]

    report = pd.DataFrame(
        conclusion_rows
    )

    report_file = (
        OUTPUT_DIR
        / "oxford_5a_5d_conclusion.csv"
    )

    report.to_csv(
        report_file,
        index=False,
    )

    print(
        report.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Scientific interpretation
    # -------------------------------------------------------------------------

    print()
    print(
        "SCIENTIFIC INTERPRETATION:"
    )

    print()

    if bcg_available:

        print(
            "1. Oxford BCG data is present and contains "
            "finite feature values."
        )

    else:

        print(
            "1. Oxford BCG feature availability was not "
            "demonstrated in the feature table."
        )

    if acc_available:

        print(
            "2. Oxford accelerometer/jerk-derived features "
            "are available."
        )

    else:

        print(
            "2. Oxford accelerometer/jerk-derived features "
            "are unavailable in the current feature table."
        )

    if gyro_available:

        print(
            "3. Oxford gyroscope-derived features "
            "are available."
        )

    else:

        print(
            "3. Oxford gyroscope-derived features "
            "are unavailable in the current feature table."
        )

    if not current_imu_rule_applicable:

        print(
            "4. Therefore the existing IMU-based artifact "
            "candidate logic cannot fully evaluate Oxford."
        )

        print(
            "5. Zero Oxford candidates must NOT be interpreted "
            "as evidence of zero artifacts."
        )

    else:

        print(
            "4. The current IMU candidate rule has the basic "
            "feature inputs needed for Oxford."
        )

        print(
            "5. Threshold-level behavior should therefore "
            "be investigated without changing the threshold yet."
        )

    print()
    print(
        "NEXT DECISION:"
    )

    print(
        "Before manual annotation, inspect the Oxford dataset "
        "loader/output architecture to determine whether:"
    )

    print(
        "    A. Oxford intentionally contains BCG only,"
    )

    print(
        "    B. Oxford contains IMU in the raw files but the "
        "loader does not expose it,"
    )

    print(
        "    C. Oxford IMU is stored under another sensor/stream,"
    )

    print(
        "    D. Oxford is intentionally a BCG-only validation "
        "dataset."
    )

    print()
    print(
        "DO NOT change the artifact threshold yet."
    )

    print()
    print(
        "Investigation outputs:"
    )

    print(
        f"    {OUTPUT_DIR}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)

    print(
        "OXFORD ARTIFACT INVESTIGATION — STEPS 5A–5D"
    )

    print("=" * 78)

    print()

    print(
        "This is a diagnostic investigation only."
    )

    print(
        "No artifact removal is performed."
    )

    print(
        "No ML model is trained."
    )

    print(
        "No thresholds are changed."
    )

    print(
        "No timing-layer data is modified."
    )

    print()

    # -------------------------------------------------------------------------
    # Validate files
    # -------------------------------------------------------------------------

    validate_input_files()

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    df = load_segment_features()

    # -------------------------------------------------------------------------
    # STEP 5A
    # -------------------------------------------------------------------------

    feature_audit = inspect_feature_availability(
        df
    )

    # -------------------------------------------------------------------------
    # STEP 5B
    # -------------------------------------------------------------------------

    distribution = feature_distribution_analysis(
        df
    )

    # -------------------------------------------------------------------------
    # STEP 5C
    # -------------------------------------------------------------------------

    representative_signal_audit(
        df
    )

    # -------------------------------------------------------------------------
    # STEP 5D
    # -------------------------------------------------------------------------

    logic_audit = candidate_logic_audit(
        df
    )

    # -------------------------------------------------------------------------
    # FINAL REPORT
    # -------------------------------------------------------------------------

    generate_final_report(
        feature_audit,
        distribution,
        logic_audit,
    )

    print()
    print("=" * 78)

    print(
        "OXFORD 5A–5D INVESTIGATION COMPLETE"
    )

    print("=" * 78)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()