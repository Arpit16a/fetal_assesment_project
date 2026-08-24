"""
Master validation script for the dataset loader layer.

Validates the currently configured datasets:

    - Multimodal Cough Dataset
    - Four-IMU fetal movement dataset
    - Oxford female fetal dataset

The master test validates the complete loader lifecycle:

    discover()
        ↓
    load()
        ↓
    standardize()
        ↓
    StandardizedData

The script intentionally validates the loader layer only.
It does not perform signal processing, artifact detection,
feature extraction, behavioral analysis, or machine learning.

A failed loader validation should block downstream work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dataset import (
    STANDARD_COLUMNS,
    BaseDatasetLoader,
    CoughLoader,
    FourIMULoader,
    MaternalMPU6050Loader,
    OxfordLoader,
    StandardizedData,
)


# =============================================================================
# DATASET PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Asus\OneDrive\Desktop\fetal_assesment_project"
)

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

# Real maternal MPU6050 data is not collected yet (expected in the
# next ~7-8 days). This path is wired in now so that when it arrives,
# validating it is a one-line change (set RUN_MATERNAL_MPU6050 =
# True below) rather than new plumbing. Until the folder exists, this
# dataset is skipped, not failed.
MATERNAL_MPU6050_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "maternal"
    / "mpu6050"
)

RUN_MATERNAL_MPU6050 = MATERNAL_MPU6050_PATH.exists()


# =============================================================================
# VALIDATION CONTRACT
# =============================================================================

COMMON_COLUMNS = [
    "timestamp",
    "sensor_id",
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "mx", "my", "mz",
    "bcg_x", "bcg_y", "bcg_z",
    "label",
    "subject_id",
    "dataset_source",
    "record_id",
]

DATASET_EXPECTED_SOURCES = {
    "COUGH": "COUGH",
    "FOUR_IMU": "FOUR_IMU",
    "OXFORD": "OXFORD",
    "MATERNAL_MPU6050": "MATERNAL_MPU6050",
}

# Minimum unique subject_id values required for a dataset's loader
# output to be considered capable of subject-independent validation.
# COUGH's folder-per-subject structure and Oxford's one-file-per-woman
# structure both give real subject identity, so both are held to a
# real minimum. FOUR_IMU's subject_id is currently a record-level
# proxy (see dataset.py docstring), so its minimum is set low and
# should be raised once true participant identity is confirmed.
# MATERNAL_MPU6050 has no minimum yet — there may be exactly one
# maternal test recording at first, and that's expected, not a bug.
MIN_EXPECTED_SUBJECTS = {
    "COUGH": 2,
    "FOUR_IMU": 2,
    "OXFORD": 2,
    "MATERNAL_MPU6050": None,
}


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def print_header(title: str) -> None:
    """Print a consistent section header."""

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


def validate_loader_interface(
    loader: BaseDatasetLoader,
) -> bool:
    """
    Validate that the object follows the BaseDatasetLoader abstraction.
    """

    required_methods = [
        "discover",
        "load",
        "standardize",
        "run",
    ]

    missing_methods = [
        method
        for method in required_methods
        if not callable(
            getattr(loader, method, None)
        )
    ]

    if missing_methods:

        print(
            "\n❌ Loader interface invalid."
        )

        print(
            f"Missing methods: {missing_methods}"
        )

        return False

    if not isinstance(
        loader,
        BaseDatasetLoader,
    ):

        print(
            "\n❌ Loader does not inherit "
            "from BaseDatasetLoader."
        )

        return False

    print(
        "\n✓ BaseDatasetLoader interface validated"
    )

    return True


def validate_standardized_output(
    standardized: StandardizedData,
    expected_source: str,
) -> bool:
    """
    Validate the StandardizedData object returned by loader.run().
    """

    if not isinstance(
        standardized,
        StandardizedData,
    ):

        print(
            "\n❌ Loader did not return "
            "StandardizedData."
        )

        print(
            f"Returned type: "
            f"{type(standardized).__name__}"
        )

        return False

    data = standardized.data

    print(
        "\nStandardizedData object:"
    )

    print(
        f"  Dataset name: "
        f"{standardized.dataset_name}"
    )

    print(
        f"  Sampling rate: "
        f"{standardized.sampling_rate}"
    )

    print(
        f"  Shape: "
        f"{data.shape}"
    )

    # -------------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------------

    missing_columns = [
        column
        for column in COMMON_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:

        print(
            f"\n❌ Missing common columns: "
            f"{missing_columns}"
        )

        return False

    print(
        "\n✓ Complete common schema present"
    )

    # -------------------------------------------------------------------------
    # Compare against canonical schema
    # -------------------------------------------------------------------------

    missing_canonical = [
        column
        for column in STANDARD_COLUMNS
        if column not in data.columns
    ]

    if missing_canonical:

        print(
            f"\n❌ Missing canonical columns: "
            f"{missing_canonical}"
        )

        return False

    print(
        "✓ Canonical STANDARD_COLUMNS present"
    )

    # -------------------------------------------------------------------------
    # Duplicate columns
    # -------------------------------------------------------------------------

    duplicated_columns = (
        data.columns[
            data.columns.duplicated()
        ].tolist()
    )

    if duplicated_columns:

        print(
            f"\n❌ Duplicate columns: "
            f"{duplicated_columns}"
        )

        return False

    print(
        "✓ No duplicate columns"
    )

    # -------------------------------------------------------------------------
    # Basic dataset integrity
    # -------------------------------------------------------------------------

    if len(data) == 0:

        print(
            "\n❌ Dataset is empty"
        )

        return False

    print(
        "✓ Dataset is non-empty"
    )

    # -------------------------------------------------------------------------
    # Dataset source
    # -------------------------------------------------------------------------

    sources = (
        data["dataset_source"]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        "\nDataset sources:"
    )

    print(sources)

    if expected_source not in sources:

        print(
            f"\n❌ Expected source "
            f"'{expected_source}' not found"
        )

        return False

    print(
        "✓ Dataset source validated"
    )

    # -------------------------------------------------------------------------
    # Subject validation
    # -------------------------------------------------------------------------

    unique_subjects = data["subject_id"].nunique()

    print(
        "\nUnique subjects:",
        unique_subjects,
    )

    if data["subject_id"].isna().any():

        print(
            "❌ Missing subject IDs"
        )

        return False

    print(
        "✓ Subject IDs valid"
    )

    # -------------------------------------------------------------------------
    # Subject diversity — catches the "everything is one fake subject"
    # bug class, which passed silently before because this check
    # only looked for NaN, never for collapsed/constant subject_id.
    # That bug is what made subject-independent validation (Step 6)
    # impossible on FOUR_IMU and OXFORD despite both loaders "passing."
    # -------------------------------------------------------------------------

    min_expected = MIN_EXPECTED_SUBJECTS.get(expected_source)

    if min_expected is not None and unique_subjects < min_expected:

        print(
            f"\n❌ Subject diversity too low: {unique_subjects} unique "
            f"subject(s), expected at least {min_expected}. "
            "Subject-independent validation is not possible on this "
            "output — check whether subject_id is a real identity or "
            "an accidentally-constant/proxy value."
        )

        return False

    print(
        f"✓ Subject diversity check passed "
        f"(>= {min_expected} expected)"
        if min_expected is not None
        else "✓ Subject diversity check skipped (no minimum configured)"
    )

    # -------------------------------------------------------------------------
    # Timestamp validation
    # -------------------------------------------------------------------------

    if data["timestamp"].isna().any():

        print(
            "\n❌ Missing timestamps"
        )

        return False

    print(
        "✓ Timestamp field valid"
    )

    # -------------------------------------------------------------------------
    # Label validation
    # -------------------------------------------------------------------------

    if data["label"].isna().any():

        print(
            "\n❌ Missing labels"
        )

        return False

    print(
        "✓ Labels valid"
    )

    print(
        "\nLabel distribution:"
    )

    print(
        data["label"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    # -------------------------------------------------------------------------
    # Signal availability
    # -------------------------------------------------------------------------

    signal_columns = [
        "ax", "ay", "az",
        "gx", "gy", "gz",
        "mx", "my", "mz",
        "bcg_x", "bcg_y", "bcg_z",
    ]

    print(
        "\nSignal availability:"
    )

    print(
        data[signal_columns]
        .notna()
        .sum()
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    print(
        "\nMetadata:"
    )

    metadata = (
        standardized.metadata
        if standardized.metadata is not None
        else {}
    )

    if metadata:

        for key, value in metadata.items():

            print(
                f"  {key}: {value}"
            )

    else:

        print(
            "  No additional metadata."
        )

    return True


def validate_loader(
    name: str,
    loader: BaseDatasetLoader,
    expected_source: str,
) -> bool:
    """
    Execute and validate one complete dataset loader.
    """

    print_header(
        f"{name} LOADER VALIDATION"
    )

    # -------------------------------------------------------------------------
    # Interface
    # -------------------------------------------------------------------------

    if not validate_loader_interface(
        loader
    ):

        return False

    # -------------------------------------------------------------------------
    # Path
    # -------------------------------------------------------------------------

    print(
        "\nDataset path:"
    )

    print(
        loader.root_path
    )

    if not loader.root_path.exists():

        print(
            "\n❌ Dataset path does not exist."
        )

        return False

    print(
        "✓ Dataset path exists"
    )

    # -------------------------------------------------------------------------
    # Complete lifecycle
    # -------------------------------------------------------------------------

    try:

        standardized = loader.run()

    except Exception as exc:

        print(
            "\n❌ LOADER EXECUTION FAILED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return False

    print(
        "\n✓ Complete loader lifecycle executed"
    )

    # -------------------------------------------------------------------------
    # Standardized output
    # -------------------------------------------------------------------------

    if not validate_standardized_output(
        standardized,
        expected_source,
    ):

        return False

    print(
        f"\n✅ {name} LOADER PASSED"
    )

    return True


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    """Run all configured loader validations."""

    print_header(
        "MASTER DATASET LOADER VALIDATION"
    )

    # -------------------------------------------------------------------------
    # Dataset paths
    # -------------------------------------------------------------------------

    print(
        "\nDataset paths:"
    )

    print(
        "\nCOUGH:"
    )

    print(
        COUGH_PATH
    )

    print(
        "\nFOUR-IMU:"
    )

    print(
        FOUR_IMU_PATH
    )

    print(
        "\nOXFORD:"
    )

    print(
        OXFORD_PATH
    )

    if RUN_MATERNAL_MPU6050:

        print(
            "\nMATERNAL MPU6050:"
        )

        print(
            MATERNAL_MPU6050_PATH
        )

    else:

        print(
            "\nMATERNAL MPU6050: not yet available, skipping "
            f"(expected path: {MATERNAL_MPU6050_PATH})"
        )

    # -------------------------------------------------------------------------
    # Create loaders
    # -------------------------------------------------------------------------

    try:

        cough_loader = CoughLoader(
            COUGH_PATH
        )

        four_imu_loader = FourIMULoader(
            FOUR_IMU_PATH
        )

        oxford_loader = OxfordLoader(
            OXFORD_PATH
        )

        maternal_loader = (
            MaternalMPU6050Loader(MATERNAL_MPU6050_PATH)
            if RUN_MATERNAL_MPU6050
            else None
        )

    except Exception as exc:

        print(
            "\n❌ LOADER INITIALIZATION FAILED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    # -------------------------------------------------------------------------
    # Validate loaders
    # -------------------------------------------------------------------------

    results = {
        "COUGH": validate_loader(
            "COUGH",
            cough_loader,
            DATASET_EXPECTED_SOURCES[
                "COUGH"
            ],
        ),

        "FOUR_IMU": validate_loader(
            "FOUR-IMU",
            four_imu_loader,
            DATASET_EXPECTED_SOURCES[
                "FOUR_IMU"
            ],
        ),

        "OXFORD": validate_loader(
            "OXFORD",
            oxford_loader,
            DATASET_EXPECTED_SOURCES[
                "OXFORD"
            ],
        ),
    }

    if RUN_MATERNAL_MPU6050:

        results["MATERNAL_MPU6050"] = validate_loader(
            "MATERNAL_MPU6050",
            maternal_loader,
            DATASET_EXPECTED_SOURCES[
                "MATERNAL_MPU6050"
            ],
        )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print_header(
        "MASTER VALIDATION SUMMARY"
    )

    for dataset, passed in results.items():

        status = (
            "PASS ✓"
            if passed
            else
            "FAIL ❌"
        )

        print(
            f"{dataset:<15} : {status}"
        )

    # -------------------------------------------------------------------------
    # Global result
    # -------------------------------------------------------------------------

    if all(results.values()):

        print("\n")
        print("=" * 80)
        print(
            "🎯 ALL DATASET LOADERS PASSED"
        )
        print("=" * 80)

        print(
            "\nLoader abstraction is frozen."
        )

        print(
            "All datasets now follow:"
        )

        print(
            "discover() → load() → standardize() → run()"
        )

        print(
            "\nNext layer: 02_signal_processing"
        )

        return 0

    print("\n")
    print("=" * 80)
    print(
        "❌ LOADER VALIDATION FAILED"
    )
    print("=" * 80)

    print(
        "\nDo NOT move to signal processing yet."
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )