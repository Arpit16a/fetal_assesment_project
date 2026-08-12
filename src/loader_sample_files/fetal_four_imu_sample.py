from pathlib import Path
from dataset import FourIMULoader


class ConcreteFourIMULoader(FourIMULoader):
    def standardize(self, df):
        return df

FOUR_IMU_PATH = Path(
    r"C:\Users\Asus\OneDrive\Desktop\fetal_assesment_project\data\raw\fetal\four_imu"
)

loader = ConcreteFourIMULoader(FOUR_IMU_PATH)

print("=" * 70)
print("FOUR-IMU LOADER TEST")
print("=" * 70)

records = loader.discover()

print(f"\nDiscovered recording files: {len(records)}")
print("\nFirst 5 records:")

for r in records[:5]:
    print(
        f"{r['sub_dataset']} | "
        f"{r['record_id']} | "
        f"{r['path']}"
    )

data = loader.load()

print("\n" + "=" * 70)
print("STANDARDIZED FOUR-IMU DATASET")
print("=" * 70)

print("\nShape:", data.shape)
print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 10 rows:")
print(data.head(10).to_string(index=False))

print("\nDtypes:")
print(data.dtypes)

print("\nMissing values:")
print(data.isna().sum())

print("\nIMU distribution:")
print(data["sensor_id"].value_counts())

print("\nSub-dataset distribution:")
print(data["sub_dataset"].value_counts())

print("\nLabel distribution:")
print(data.groupby(["sub_dataset", "label"]).size())

print("\nMetadata:")
print(loader.metadata)
