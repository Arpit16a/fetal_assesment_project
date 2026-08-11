from pathlib import Path

from dataset import HARLoader


HAR_PATH = Path(
    r"C:\Users\Asus\OneDrive\Desktop\fetal_assesment_project"
    r"\data\raw\artifacts\har"
)

loader = HARLoader(
    root_path=HAR_PATH,
)

standardized = loader.run()

df_har = standardized.data

print(df_har.head())
print()
print(df_har.shape)
print()
print(df_har.columns.tolist())

print(standardized.summary())

# ============================================================
# HAR RAW DATA VERIFICATION
# ============================================================

print("Original columns:")
print(loader.raw_data.columns.tolist())

print("\nFirst 5 raw rows:")
print(loader.raw_data.head())

print("\nData types:")
print(loader.raw_data.dtypes)

print("\nMissing values:")
print(loader.raw_data.isna().sum())

print("\nActivity distribution:")
print(
    loader.raw_data["activity"]
    .value_counts()
    .sort_index()
)