from pathlib import Path
import pandas as pd


# ============================================================
# PATH
# ============================================================

FETAL_PATH = Path(
    r"C:\Users\Asus\OneDrive\Desktop\fetal_assesment_project\data\raw\fetal\four_imu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FOUR-IMU FETAL DATASET DISCOVERY")
print("=" * 70)

print("\nDataset path:")
print(FETAL_PATH)


# ============================================================
# CHECK PATH
# ============================================================

if not FETAL_PATH.exists():
    raise FileNotFoundError(
        f"\nFetal Four-IMU dataset path does not exist:\n{FETAL_PATH}"
    )


# ============================================================
# DISCOVER CSV FILES
# ============================================================

csv_files = sorted(
    [
        p for p in FETAL_PATH.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".csv"
    ]
)


print("\n" + "=" * 70)
print("CSV FILE DISCOVERY")
print("=" * 70)

print(f"\nTotal CSV files found: {len(csv_files)}")


if not csv_files:
    raise RuntimeError(
        "No CSV files were found in the Four-IMU dataset."
    )


# ============================================================
# SHOW ALL CSV FILES
# ============================================================

print("\nAll CSV files:")

for i, file in enumerate(csv_files, start=1):
    print(f"{i:3d}. {file}")


# ============================================================
# CLASSIFY FILES
# ============================================================

additional_files = [
    p for p in csv_files
    if "additional data" in p.name.lower()
]

recording_files = [
    p for p in csv_files
    if p not in additional_files
]


print("\n" + "=" * 70)
print("FILE CLASSIFICATION")
print("=" * 70)

print(f"\nAdditional metadata CSVs : {len(additional_files)}")
print(f"Recording CSVs           : {len(recording_files)}")


print("\nAdditional metadata files:")

for file in additional_files:
    print(" -", file.name)


print("\nFirst recording files:")

for file in recording_files[:10]:
    print(" -", file)


# ============================================================
# INSPECT FIRST RECORDING FILE
# ============================================================

if not recording_files:
    raise RuntimeError(
        "No recording CSV files were identified."
    )


first_file = recording_files[0]


print("\n" + "=" * 70)
print("FIRST RECORDING FILE")
print("=" * 70)

print("\nFile:")
print(first_file)


try:
    df = pd.read_csv(first_file)

except Exception as e:
    raise RuntimeError(
        f"Could not read recording file:\n{first_file}\n\n{e}"
    )


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nRaw CSV columns:")
print(list(df.columns))


print("\nShape:")
print(df.shape)


print("\nFirst 10 rows:")
print(df.head(10).to_string())


print("\nData types:")
print(df.dtypes)


print("\nMissing values:")
print(df.isna().sum())


# ============================================================
# UNIQUE LABEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("LABEL INSPECTION")
print("=" * 70)


label_columns = [
    col for col in df.columns
    if "label" in str(col).lower()
]


if label_columns:

    for col in label_columns:

        print(f"\nLabel column: {col}")

        print(
            df[col]
            .value_counts(dropna=False)
            .sort_index()
        )

else:

    print("\nNo column containing 'label' was found.")


# ============================================================
# TIME INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TIME COLUMN INSPECTION")
print("=" * 70)


time_columns = [
    col for col in df.columns
    if "time" in str(col).lower()
]


if time_columns:

    for col in time_columns:

        print(f"\nTime column: {col}")

        print("First values:")
        print(df[col].head(10).tolist())

        print("Last values:")
        print(df[col].tail(10).tolist())

else:

    print("\nNo column containing 'time' was found.")


# ============================================================
# COLUMN NAME PATTERN INSPECTION
# ============================================================

print("\n" + "=" * 70)
print("SENSOR COLUMN INSPECTION")
print("=" * 70)


print("\nColumns grouped by likely sensor type:")


accelerometer_columns = [
    col for col in df.columns
    if "acc" in str(col).lower()
    or str(col).lower().startswith("a")
]


gyroscope_columns = [
    col for col in df.columns
    if "gyro" in str(col).lower()
    or str(col).lower().startswith("g")
]


print("\nPossible accelerometer columns:")
print(accelerometer_columns)


print("\nPossible gyroscope columns:")
print(gyroscope_columns)


# ============================================================
# INSPECT ADDITIONAL METADATA FILES
# ============================================================

print("\n" + "=" * 70)
print("ADDITIONAL METADATA FILES")
print("=" * 70)


for file in additional_files:

    print("\n" + "-" * 70)
    print("File:")
    print(file)

    try:

        meta_df = pd.read_csv(file)

        print("\nColumns:")
        print(list(meta_df.columns))

        print("\nShape:")
        print(meta_df.shape)

        print("\nFirst 10 rows:")
        print(meta_df.head(10).to_string())

        print("\nData types:")
        print(meta_df.dtypes)

        print("\nMissing values:")
        print(meta_df.isna().sum())

    except Exception as e:

        print(f"Could not read file: {e}")


# ============================================================
# INSPECT ONE FILE FROM EACH SUB-DATASET
# ============================================================

print("\n" + "=" * 70)
print("SUB-DATASET INSPECTION")
print("=" * 70)


subdatasets = {}

for file in recording_files:

    parts_lower = [
        part.lower()
        for part in file.parts
    ]

    matched = None

    if "sub-dataset one" in parts_lower:
        matched = "Sub-dataset One"

    elif "sub-dataset two" in parts_lower:
        matched = "Sub-dataset Two"

    elif "sub-dataset three" in parts_lower:
        matched = "Sub-dataset Three"

    if matched and matched not in subdatasets:
        subdatasets[matched] = file


for name, file in subdatasets.items():

    print("\n" + "-" * 70)
    print(name)

    print("File:")
    print(file)

    try:

        sub_df = pd.read_csv(file)

        print("\nColumns:")
        print(list(sub_df.columns))

        print("\nShape:")
        print(sub_df.shape)

        print("\nFirst 5 rows:")
        print(sub_df.head().to_string())

        print("\nData types:")
        print(sub_df.dtypes)

        print("\nMissing values:")
        print(sub_df.isna().sum())

        label_cols = [
            col for col in sub_df.columns
            if "label" in str(col).lower()
        ]

        if label_cols:

            print("\nLabel distributions:")

            for col in label_cols:

                print(f"\n{col}:")
                print(
                    sub_df[col]
                    .value_counts(dropna=False)
                    .sort_index()
                )

    except Exception as e:

        print(f"Could not inspect file: {e}")


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("FOUR-IMU DISCOVERY COMPLETE")
print("=" * 70)