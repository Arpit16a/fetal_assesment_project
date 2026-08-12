from pathlib import Path
import pandas as pd


COUGH_PATH = Path(
    r"C:\Users\Asus\OneDrive\Desktop\fetal_assesment_project\data\raw\artifacts\cough_imu"
)


mag_file = next(
    COUGH_PATH.rglob("Magnetometer.csv")
)

print("=" * 70)
print("FIRST COUGH MAGNETOMETER FILE")
print("=" * 70)

print("\nFile:")
print(mag_file)

df = pd.read_csv(mag_file)

print("\nRaw CSV columns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10))

print("\nShape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())