from pathlib import Path
import numpy as np
from scipy.io import loadmat


# ============================================================
# PATH
# ============================================================

OXFORD_PATH = Path(
    r"C:\Users\Asus\OneDrive\Desktop\fetal_assesment_project\data\raw\fetal\oxford_female"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("OXFORD FETAL DATASET DISCOVERY")
print("=" * 70)

print("\nDataset path:")
print(OXFORD_PATH)


# ============================================================
# CHECK PATH
# ============================================================

if not OXFORD_PATH.exists():
    raise FileNotFoundError(
        f"\nOxford dataset path does not exist:\n{OXFORD_PATH}"
    )


# ============================================================
# DISCOVER MAT FILES
# ============================================================

mat_files = sorted(
    [
        p for p in OXFORD_PATH.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".mat"
    ]
)


print("\n" + "=" * 70)
print("MAT FILE DISCOVERY")
print("=" * 70)

print(f"\nTotal MAT files found: {len(mat_files)}")


if not mat_files:
    raise RuntimeError(
        "No MATLAB .mat files were found."
    )


# ============================================================
# CLASSIFY SIGNAL / BP FILES
# ============================================================

signal_files = [
    p for p in mat_files
    if p.stem.lower().endswith("_signal")
]


bp_files = [
    p for p in mat_files
    if p.stem.lower().endswith("_bp")
]


print("\nSignal files:")
print(f"Count: {len(signal_files)}")

for file in signal_files[:20]:
    print(" -", file.name)


print("\nBP files:")
print(f"Count: {len(bp_files)}")

for file in bp_files[:20]:
    print(" -", file.name)


# ============================================================
# PAIR SIGNAL AND BP FILES
# ============================================================

print("\n" + "=" * 70)
print("SIGNAL / BP PAIRING")
print("=" * 70)


signal_map = {
    file.stem.lower().replace("_signal", ""): file
    for file in signal_files
}


bp_map = {
    file.stem.lower().replace("_bp", ""): file
    for file in bp_files
}


all_record_ids = sorted(
    set(signal_map.keys()) |
    set(bp_map.keys())
)


print(f"\nUnique record IDs: {len(all_record_ids)}")


paired_count = 0
signal_only_count = 0
bp_only_count = 0


for record_id in all_record_ids:

    signal_file = signal_map.get(record_id)
    bp_file = bp_map.get(record_id)

    if signal_file and bp_file:

        status = "PAIRED"
        paired_count += 1

    elif signal_file:

        status = "SIGNAL ONLY"
        signal_only_count += 1

    else:

        status = "BP ONLY"
        bp_only_count += 1

    print(
        f"{record_id:15s} | {status}"
    )


print("\nPairing summary:")
print("Paired       :", paired_count)
print("Signal only  :", signal_only_count)
print("BP only      :", bp_only_count)


# ============================================================
# INSPECT FIRST SIGNAL FILE
# ============================================================

if not signal_files:
    raise RuntimeError(
        "No *_signal.mat files were found."
    )


first_signal = signal_files[0]


print("\n" + "=" * 70)
print("FIRST SIGNAL FILE")
print("=" * 70)

print("\nFile:")
print(first_signal)


try:

    signal_data = loadmat(
        first_signal,
        squeeze_me=False,
        struct_as_record=False
    )

except Exception as e:

    raise RuntimeError(
        f"Could not load signal file:\n"
        f"{first_signal}\n\n"
        f"{e}"
    )


# ============================================================
# SIGNAL MATLAB VARIABLES
# ============================================================

print("\nMATLAB variables:")

signal_variables = [
    key for key in signal_data.keys()
    if not key.startswith("__")
]


for key in signal_variables:

    value = signal_data[key]

    print(
        f"\nVariable: {key}"
    )

    print(
        "Type:",
        type(value)
    )

    print(
        "Shape:",
        getattr(value, "shape", "N/A")
    )

    print(
        "Dtype:",
        getattr(value, "dtype", "N/A")
    )


# ============================================================
# SIGNAL ARRAY CONTENT
# ============================================================

print("\n" + "=" * 70)
print("SIGNAL ARRAY CONTENT INSPECTION")
print("=" * 70)


for key in signal_variables:

    value = signal_data[key]

    if isinstance(value, np.ndarray):

        print("\n" + "-" * 70)
        print(f"Variable: {key}")

        print("Shape:")
        print(value.shape)

        print("First values:")

        try:

            print(
                value.flatten()[:20]
            )

        except Exception:

            print(
                "Could not flatten variable."
            )


# ============================================================
# INSPECT FIRST BP FILE
# ============================================================

if bp_files:

    first_bp = bp_files[0]

    print("\n" + "=" * 70)
    print("FIRST BP FILE")
    print("=" * 70)

    print("\nFile:")
    print(first_bp)

    try:

        bp_data = loadmat(
            first_bp,
            squeeze_me=False,
            struct_as_record=False
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not load BP file:\n"
            f"{first_bp}\n\n"
            f"{e}"
        )


    # ========================================================
    # BP MATLAB VARIABLES
    # ========================================================

    print("\nMATLAB variables:")

    bp_variables = [
        key for key in bp_data.keys()
        if not key.startswith("__")
    ]


    for key in bp_variables:

        value = bp_data[key]

        print(
            f"\nVariable: {key}"
        )

        print(
            "Type:",
            type(value)
        )

        print(
            "Shape:",
            getattr(value, "shape", "N/A")
        )

        print(
            "Dtype:",
            getattr(value, "dtype", "N/A")
        )


    # ========================================================
    # BP ARRAY CONTENT
    # ========================================================

    print("\n" + "=" * 70)
    print("BP ARRAY CONTENT INSPECTION")
    print("=" * 70)


    for key in bp_variables:

        value = bp_data[key]

        if isinstance(value, np.ndarray):

            print("\n" + "-" * 70)
            print(f"Variable: {key}")

            print("Shape:")
            print(value.shape)

            print("First values:")

            try:

                print(
                    value.flatten()[:50]
                )

            except Exception:

                print(
                    "Could not flatten variable."
                )


# ============================================================
# SAMPLE PAIR
# ============================================================

print("\n" + "=" * 70)
print("FIRST SIGNAL / BP PAIR")
print("=" * 70)


if all_record_ids:

    first_record = all_record_ids[0]

    print("\nRecord ID:")
    print(first_record)

    print("\nSignal:")
    print(
        signal_map.get(first_record)
    )

    print("\nBP:")
    print(
        bp_map.get(first_record)
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("OXFORD DISCOVERY COMPLETE")
print("=" * 70)