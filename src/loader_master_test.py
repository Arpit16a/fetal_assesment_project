"""
Master validation script for the dataset loader layer.

Validates the three currently configured fetal/multimodal loaders:
    - Multimodal Cough Dataset
    - Four-IMU fetal movement dataset
    - Oxford female fetal dataset

The script intentionally validates the loader outputs rather than performing
signal processing. A failed loader validation should block downstream work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dataset import CoughLoader, FourIMULoader, OxfordLoader


# =============================================================================
# CONCRETE TEST LOADERS
# =============================================================================

class ConcreteFourIMULoader(FourIMULoader):
    """Concrete Four-IMU loader used only for validation instantiation."""

    def standardize(self, data: pd.DataFrame | None = None):
        """Return already-standardized Four-IMU data unchanged."""
        return self.raw_data if data is None else data


class ConcreteOxfordLoader(OxfordLoader):
    """Concrete Oxford loader used only for validation instantiation."""

    def standardize(self, data: pd.DataFrame | None = None):
        """Return already-standardized Oxford data unchanged."""
        return self.raw_data if data is None else data


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


# =============================================================================
# VALIDATION CONTRACT
# =============================================================================

COMMON_COLUMNS = [
    "timestamp",
    "sensor_id",
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "mx", "my", "mz",
    "label",
    "subject_id",
    "dataset_source",
]

OPTIONAL_COLUMNS = {
    "COUGH": ["audio_available"],
    "FOUR_IMU": ["record_id", "sub_dataset"],
    "OXFORD": ["record_id", "bcg_x", "bcg_y", "bcg_z"],
}

SIGNAL_COLUMNS = [
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "mx", "my", "mz",
]


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def print_header(title: str) -> None:
    """Print a consistent section header."""
    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


def validate_optional_fields(
    data: pd.DataFrame,
    loader: Any,
    expected_source: str,
) -> None:
    """Report dataset-specific fields without making optional fields mandatory."""
    print("\nDataset-specific validation:")

    for column in OPTIONAL_COLUMNS.get(expected_source, []):
        if column in data.columns:
            if column.startswith("bcg_"):
                count = data[column].notna().sum()
                print(f"  ✓ {column}: {count:,} valid samples")
            else:
                print(f"  ✓ {column}: present")
            continue

        if column == "audio_available" and column in loader.metadata:
            print(f"  ✓ {column}: {loader.metadata[column]}")
        else:
            print(f"  - {column}: not present")


def validate_loader(
    name: str,
    loader: Any,
    expected_source: str,
) -> bool:
    """Load one dataset and validate the common loader contract."""
    print_header(f"{name} LOADER VALIDATION")

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    try:
        data = loader.load()
    except Exception as exc:
        print("\n❌ LOAD FAILED")
        print(f"{type(exc).__name__}: {exc}")
        return False

    print("\n✓ Loader executed successfully")
    print("\nShape:")
    print(data.shape)

    # -------------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------------
    missing_columns = [
        column for column in COMMON_COLUMNS if column not in data.columns
    ]

    if missing_columns:
        print(f"\n❌ Missing common columns: {missing_columns}")
        return False

    print("\n✓ Common schema present")

    duplicated_columns = data.columns[data.columns.duplicated()].tolist()
    if duplicated_columns:
        print(f"\n❌ Duplicate columns: {duplicated_columns}")
        return False

    print("✓ No duplicate columns")

    # -------------------------------------------------------------------------
    # Basic dataset integrity
    # -------------------------------------------------------------------------
    if len(data) == 0:
        print("\n❌ Dataset is empty")
        return False

    print("✓ Dataset is non-empty")

    sources = (
        data["dataset_source"]
        .dropna()
        .unique()
        .tolist()
    )

    print("\nDataset sources:")
    print(sources)

    if expected_source not in sources:
        print(f"\n❌ Expected source '{expected_source}' not found")
        return False

    print("✓ Dataset source validated")

    # -------------------------------------------------------------------------
    # Subject validation
    # -------------------------------------------------------------------------
    print("\nUnique subjects:", data["subject_id"].nunique())

    if data["subject_id"].isna().any():
        print("❌ Missing subject IDs")
        return False

    print("✓ Subject IDs valid")

    # -------------------------------------------------------------------------
    # Timestamp validation
    # -------------------------------------------------------------------------
    if data["timestamp"].isna().any():
        print("\n❌ Missing timestamps")
        return False

    print("✓ Timestamp field valid")

    # -------------------------------------------------------------------------
    # Label validation
    # -------------------------------------------------------------------------
    if data["label"].isna().any():
        print("\n❌ Missing labels")
        return False

    print("✓ Labels valid")
    print("\nLabel distribution:")
    print(data["label"].value_counts(dropna=False).sort_index())

    # -------------------------------------------------------------------------
    # Signal availability
    # -------------------------------------------------------------------------
    print("\nSignal availability:")
    print(data[SIGNAL_COLUMNS].notna().sum())

    # -------------------------------------------------------------------------
    # Dataset-specific fields
    # -------------------------------------------------------------------------
    validate_optional_fields(data, loader, expected_source)

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    print("\nMetadata:")
    for key, value in loader.metadata.items():
        print(f"  {key}: {value}")

    print(f"\n✅ {name} LOADER PASSED")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    """Run all configured loader validations and return a process status."""
    print_header("MASTER DATASET LOADER VALIDATION")

    print("\nDataset paths:")
    print("COUGH:")
    print(COUGH_PATH)
    print("\nFOUR-IMU:")
    print(FOUR_IMU_PATH)
    print("\nOXFORD:")
    print(OXFORD_PATH)

    # -------------------------------------------------------------------------
    # Create loaders
    # -------------------------------------------------------------------------
    cough_loader = CoughLoader(COUGH_PATH)
    four_imu_loader = ConcreteFourIMULoader(FOUR_IMU_PATH)
    oxford_loader = ConcreteOxfordLoader(OXFORD_PATH)

    # -------------------------------------------------------------------------
    # Validate loaders
    # -------------------------------------------------------------------------
    results = {
        "COUGH": validate_loader(
            "COUGH",
            cough_loader,
            "COUGH",
        ),
        "FOUR_IMU": validate_loader(
            "FOUR-IMU",
            four_imu_loader,
            "FOUR_IMU",
        ),
        "OXFORD": validate_loader(
            "OXFORD",
            oxford_loader,
            "OXFORD",
        ),
    }

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    print_header("MASTER VALIDATION SUMMARY")

    for dataset, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ❌"
        print(f"{dataset:<15} : {status}")

    # -------------------------------------------------------------------------
    # Global result
    # -------------------------------------------------------------------------
    if all(results.values()):
        print("\n")
        print("=" * 80)
        print("🎯 ALL DATASET LOADERS PASSED")
        print("=" * 80)
        print("\nLoader layer is ready.")
        print("Next layer: 02_signal_processing")
        return 0

    print("\n")
    print("=" * 80)
    print("❌ LOADER VALIDATION FAILED")
    print("=" * 80)
    print("\nDo NOT move to signal processing yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())