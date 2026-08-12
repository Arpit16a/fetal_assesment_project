from pathlib import Path
from dataset import OxfordLoader


# ============================================================
# OXFORD DATASET PATH
# ============================================================

OXFORD_PATH = Path(
    r"C:\Users\Asus\OneDrive\Desktop\fetal_assesment_project"
    r"\data\raw\fetal\oxford_female"
)


# ============================================================
# CREATE LOADER
# ============================================================

loader = OxfordLoader(
    root_path=OXFORD_PATH
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("OXFORD LOADER TEST")
print("=" * 70)

print("\nDataset path:")
print(OXFORD_PATH)


# ============================================================
# DISCOVERY TEST
# ============================================================

records = loader.discover()

print(f"\nDiscovered records: {len(records)}")

print("\nFirst 5 records:")

for record in records[:5]:

    print(
        f"{record['record_id']} | "
        f"{record['signal_file'].name} | "
        f"{record['bp_file'].name}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING OXFORD DATASET")
print("=" * 70)

data = loader.load()


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZED OXFORD DATASET")
print("=" * 70)

print("\nShape:")
print(data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 10 rows:")
print(data.head(10).to_string(index=False))


# ============================================================
# DATA TYPES
# ============================================================

print("\nDtypes:")
print(data.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")
print(data.isna().sum())


# ============================================================
# LABEL VALIDATION
# ============================================================

print("\nLabel distribution:")

print(
    data["label"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# RECORD VALIDATION
# ============================================================

print("\nRecord distribution:")
print(
    data["record_id"].nunique()
)


# ============================================================
# BCG VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("BCG VALIDATION")
print("=" * 70)

bcg_columns = [
    "bcg_x",
    "bcg_y",
    "bcg_z"
]

print("\nValid BCG samples:")

print(
    data[bcg_columns]
    .notna()
    .sum()
)


print("\nBCG missing values:")

print(
    data[bcg_columns]
    .isna()
    .sum()
)


# ============================================================
# IMU VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("IMU FIELD VALIDATION")
print("=" * 70)

imu_columns = [
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "mx",
    "my",
    "mz"
]

print(
    data[imu_columns]
    .notna()
    .sum()
)


# ============================================================
# METADATA
# ============================================================

print("\n" + "=" * 70)
print("METADATA")
print("=" * 70)

for key, value in loader.metadata.items():

    print(f"{key}: {value}")


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL OXFORD VALIDATION")
print("=" * 70)


required_columns = [
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

    "bcg_x",
    "bcg_y",
    "bcg_z",

    "label",
    "subject_id",
    "dataset_source",
    "record_id"
]


missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]


if missing_columns:

    print(
        "✗ Missing required columns:",
        missing_columns
    )

else:

    print("✓ Common schema present")


if data.empty:

    print("✗ Dataset is empty")

else:

    print("✓ Dataset is non-empty")


if data["dataset_source"].nunique() == 1:

    print(
        "✓ Dataset source:",
        data["dataset_source"].unique().tolist()
    )

else:

    print("✗ Multiple dataset sources detected")


if data["record_id"].nunique() == len(records):

    print(
        f"✓ All {len(records)} Oxford records represented"
    )

else:

    print(
        "⚠ Record count mismatch:",
        data["record_id"].nunique()
    )


if data[bcg_columns].notna().all().all():

    print("✓ BCG channels contain valid samples")

else:

    print("⚠ Some BCG channels contain missing values")


if data[imu_columns].isna().all().all():

    print("✓ IMU fields correctly empty for Oxford")

else:

    print(
        "⚠ Unexpected IMU values found in Oxford dataset"
    )


print("\n" + "=" * 70)
print("OXFORD LOADER TEST COMPLETE")
print("=" * 70)