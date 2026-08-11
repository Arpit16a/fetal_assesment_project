from pathlib import Path

from dataset import CoughLoader


COUGH_PATH = Path(
    r"C:\Users\Asus\OneDrive\Desktop\fetal_assesment_project"
    r"\data\raw\artifacts\cough_imu"
)


loader = CoughLoader(COUGH_PATH)


# ============================================================
# Discovery
# ============================================================

trials = loader.discover_trials()

print("\n" + "=" * 70)
print("COUGH DATASET DISCOVERY")
print("=" * 70)

print(
    f"Total discovered trials: {len(trials)}"
)

print("\nFirst 5 trials:")

for trial in trials[:5]:

    print(
        trial["subject_id"],
        "|",
        trial["trial"],
        "|",
        trial["accelerometer"]
    )


# ============================================================
# Load
# ============================================================

data = loader.load()


# ============================================================
# Basic information
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZED COUGH DATASET")
print("=" * 70)

print(
    "\nShape:",
    data.shape
)

print(
    "\nColumns:"
)

print(
    data.columns.tolist()
)


# ============================================================
# First rows
# ============================================================

print("\nFirst 5 rows:")

print(
    data.head()
)


# ============================================================
# Data types
# ============================================================

print("\nData types:")

print(
    data.dtypes
)


# ============================================================
# Missing values
# ============================================================

print("\nMissing values:")

print(
    data.isna().sum()
)


# ============================================================
# Modality availability
# ============================================================

print("\n" + "=" * 70)
print("MODALITY VALIDATION")
print("=" * 70)

print(
    "\nAccelerometer:"
)

print(
    data[
        ["ax", "ay", "az"]
    ].notna().sum()
)


print(
    "\nGyroscope:"
)

print(
    data[
        ["gx", "gy", "gz"]
    ].notna().sum()
)


print(
    "\nMagnetometer:"
)

print(
    data[
        ["mx", "my", "mz"]
    ].notna().sum()
)


# ============================================================
# Dataset metadata
# ============================================================

print("\n" + "=" * 70)
print("METADATA")
print("=" * 70)

print(
    loader.metadata
)


# ============================================================
# Labels
# ============================================================

print("\nTrial labels:")

print(
    data["label"].value_counts()
)


# ============================================================
# Subjects
# ============================================================

print("\nSubjects:")

print(
    data["subject_id"].nunique()
)