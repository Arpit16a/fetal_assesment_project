from pathlib import Path
root = Path("data/raw/artifacts/cough_imu/Multimodal Cough Dataset")
print(sorted(set(p.parent.name for p in root.rglob("Accelerometer.csv"))))