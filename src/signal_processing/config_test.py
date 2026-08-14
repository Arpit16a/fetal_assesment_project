from config import (
    COUGH_CONFIG,
    FOUR_IMU_CONFIG,
    OXFORD_CONFIG,
    DATASET_CONFIGS,
    PROCESSING_POLICY,
    get_dataset_config,
)


print("=" * 70)
print("SIGNAL PROCESSING CONFIGURATION TEST")
print("=" * 70)


# ============================================================
# Registry
# ============================================================

print("\nRegistered datasets:")

for name in DATASET_CONFIGS:
    print(f"- {name}")


# ============================================================
# COUGH
# ============================================================

print("\n" + "-" * 70)
print("COUGH")
print("-" * 70)

print(COUGH_CONFIG)


# ============================================================
# FOUR-IMU
# ============================================================

print("\n" + "-" * 70)
print("FOUR-IMU")
print("-" * 70)

print(FOUR_IMU_CONFIG)


# ============================================================
# OXFORD
# ============================================================

print("\n" + "-" * 70)
print("OXFORD")
print("-" * 70)

print(OXFORD_CONFIG)


# ============================================================
# Lookup test
# ============================================================

print("\n" + "-" * 70)
print("LOOKUP TEST")
print("-" * 70)

for name in ["COUGH", "FOUR_IMU", "OXFORD"]:

    config = get_dataset_config(name)

    print(
        f"{name}: "
        f"{config.name} | "
        f"signal={config.signal_type} | "
        f"fs={config.sampling_rate_hz}"
    )


# ============================================================
# Policy
# ============================================================

print("\n" + "-" * 70)
print("PROCESSING POLICY")
print("-" * 70)

print(PROCESSING_POLICY)


print("\n" + "=" * 70)
print("CONFIGURATION TEST PASSED")
print("=" * 70)