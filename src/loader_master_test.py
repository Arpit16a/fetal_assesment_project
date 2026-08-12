from pathlib import Path
import numpy as np
import pandas as pd

from dataset import (
    CoughLoader,
    FourIMULoader,
    OxfordLoader,
)


class ConcreteFourIMULoader(FourIMULoader):
    def standardize(self, data):
        return data


class ConcreteOxfordLoader(OxfordLoader):
    def standardize(self, data):
        return data


# Expose a concrete Oxford loader implementation for test instantiation.
OxfordLoader = ConcreteOxfordLoader


# ============================================================
# DATASET PATHS
# ============================================================

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


# ============================================================
# COMMON CONTRACT
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
# OPTIONAL DATASET-SPECIFIC COLUMNS
# ============================================================

OPTIONAL_COLUMNS = {
    "COUGH": [
        "audio_available",
    ],

    "FOUR_IMU": [
        "record_id",
        "sub_dataset",
    ],

    "OXFORD": [
        "record_id",
        "bcg_x",
        "bcg_y",
        "bcg_z",
    ],
}


# ============================================================
# VALIDATE LOADER
# ============================================================

def validate_loader(
    name,
    loader,
    expected_source,
):

    print("\n")
    print("=" * 80)
    print(f"{name} LOADER VALIDATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    try:

        data = loader.load()

    except Exception as e:

        print("\n❌ LOAD FAILED")
        print(type(e).__name__, ":", e)

        return False


    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    print("\n✓ Loader executed successfully")

    print("\nShape:")
    print(data.shape)

    # --------------------------------------------------------
    # Required common columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in COMMON_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:

        print(
            "\n❌ Missing common columns:",
            missing_columns
        )

        return False

    print("\n✓ Common schema present")


    # --------------------------------------------------------
    # Check unexpected duplicate columns
    # --------------------------------------------------------

    duplicated_columns = (
        data.columns[
            data.columns.duplicated()
        ]
        .tolist()
    )

    if duplicated_columns:

        print(
            "\n❌ Duplicate columns:",
            duplicated_columns
        )

        return False

    print("✓ No duplicate columns")


    # --------------------------------------------------------
    # Check empty dataset
    # --------------------------------------------------------

    if len(data) == 0:

        print("\n❌ Dataset is empty")

        return False

    print("✓ Dataset is non-empty")


    # --------------------------------------------------------
    # Source validation
    # --------------------------------------------------------

    sources = (
        data["dataset_source"]
        .dropna()
        .unique()
        .tolist()
    )

    print("\nDataset sources:")
    print(sources)

    if expected_source not in sources:

        print(
            f"\n❌ Expected source "
            f"'{expected_source}' not found"
        )

        return False

    print("✓ Dataset source validated")


    # --------------------------------------------------------
    # Subject validation
    # --------------------------------------------------------

    print(
        "\nUnique subjects:",
        data["subject_id"].nunique()
    )

    if data["subject_id"].isna().any():

        print(
            "❌ Missing subject IDs"
        )

        return False

    print("✓ Subject IDs valid")


    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------

    if data["timestamp"].isna().any():

        print(
            "\n❌ Missing timestamps"
        )

        return False

    print("✓ Timestamp field valid")


    # --------------------------------------------------------
    # Label validation
    # --------------------------------------------------------

    if data["label"].isna().any():

        print(
            "\n❌ Missing labels"
        )

        return False

    print("✓ Labels valid")

    print("\nLabel distribution:")
    print(
        data["label"]
        .value_counts(dropna=False)
        .sort_index()
    )


    # --------------------------------------------------------
    # Numeric signal validation
    # --------------------------------------------------------

    signal_columns = [
        "ax", "ay", "az",
        "gx", "gy", "gz",
        "mx", "my", "mz",
    ]

    print("\nSignal availability:")

    availability = (
        data[signal_columns]
        .notna()
        .sum()
    )

    print(availability)


    # --------------------------------------------------------
    # Dataset-specific fields
    # --------------------------------------------------------

    print("\nDataset-specific validation:")

    for column in OPTIONAL_COLUMNS.get(
        expected_source,
        []
    ):

        if column in data.columns:

            if column.startswith("bcg_"):

                count = data[column].notna().sum()

                print(
                    f"  ✓ {column}: "
                    f"{count} valid samples"
                )

            else:

                print(
                    f"  ✓ {column}: present"
                )

        else:

            print(
                f"  - {column}: not present"
            )


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    print("\nMetadata:")

    for key, value in loader.metadata.items():

        print(
            f"  {key}: {value}"
        )


    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        f"\n✅ {name} LOADER PASSED"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MASTER DATASET LOADER VALIDATION")
    print("=" * 80)

    print("\nDataset paths:")

    print("COUGH:")
    print(COUGH_PATH)

    print("\nFOUR-IMU:")
    print(FOUR_IMU_PATH)

    print("\nOXFORD:")
    print(OXFORD_PATH)


    # ========================================================
    # CREATE LOADERS
    # ========================================================

    cough_loader = CoughLoader(
        COUGH_PATH
    )

    four_imu_loader = ConcreteFourIMULoader(
        FOUR_IMU_PATH
    )

    oxford_loader = ConcreteOxfordLoader(
        OXFORD_PATH
    )


    # ========================================================
    # VALIDATE
    # ========================================================

    results = {}

    results["COUGH"] = validate_loader(
        "COUGH",
        cough_loader,
        "COUGH",
    )

    results["FOUR_IMU"] = validate_loader(
        "FOUR-IMU",
        four_imu_loader,
        "FOUR_IMU",
    )

    results["OXFORD"] = validate_loader(
        "OXFORD",
        oxford_loader,
        "OXFORD",
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("MASTER VALIDATION SUMMARY")
    print("=" * 80)

    for dataset, passed in results.items():

        status = (
            "PASS ✓"
            if passed
            else "FAIL ❌"
        )

        print(
            f"{dataset:<15} : {status}"
        )


    # ========================================================
    # GLOBAL RESULT
    # ========================================================

    if all(results.values()):

        print("\n")
        print("=" * 80)
        print("🎯 ALL DATASET LOADERS PASSED")
        print("=" * 80)

        print(
            "\nLoader layer is ready."
        )

        print(
            "Next layer: 02_signal_processing"
        )

    else:

        print("\n")
        print("=" * 80)
        print("❌ LOADER VALIDATION FAILED")
        print("=" * 80)

        print(
            "\nDo NOT move to signal processing yet."
        )


if __name__ == "__main__":
    main()