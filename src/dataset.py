"""
Unified Dataset Standardization Layer.

Provides dataset-specific loaders that convert heterogeneous raw datasets
into a common sensor-oriented representation for downstream processing.

Datasets
--------
1. Oxford female fetal movement dataset
2. Four-IMU fetal movement dataset
3. HAR artifact dataset
4. Multimodal cough + IMU dataset

The loaders preserve the source data and do not invent unavailable modalities.
Missing fields are represented with NaN.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat


# =============================================================================
# 1. STANDARD DATA SCHEMA
# =============================================================================

STANDARD_COLUMNS = [
    "timestamp",
    "sensor_id",
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "mx", "my", "mz",
    "bcg_x", "bcg_y", "bcg_z",
    "label",
    "subject_id",
    "dataset_source",
    "record_id",
]


@dataclass
class StandardizedData:
    """Container for standardized sensor data."""

    data: pd.DataFrame
    dataset_name: str
    sampling_rate: Optional[float] = None

    def validate(self) -> None:
        """Validate that the dataframe contains the complete common schema."""
        missing_columns = [
            column for column in STANDARD_COLUMNS if column not in self.data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing standardized columns: {missing_columns}"
            )

    def summary(self) -> dict:
        """Return basic information about the standardized dataset."""
        self.validate()
        return {
            "dataset_name": self.dataset_name,
            "rows": len(self.data),
            "columns": list(self.data.columns),
            "subjects": self.data["subject_id"].nunique(),
            "labels": self.data["label"].nunique(),
            "sampling_rate": self.sampling_rate,
        }


# =============================================================================
# 2. BASE DATASET LOADER
# =============================================================================

class BaseDatasetLoader(ABC):
    """Abstract base class for dataset-specific loaders."""

    def __init__(self, root_path: str | Path, dataset_name: str) -> None:
        self.root_path = Path(root_path)
        self.dataset_name = dataset_name

        if not self.root_path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist:\n{self.root_path}"
            )

    @abstractmethod
    def discover(self):
        """Discover files belonging to the dataset."""
        raise NotImplementedError

    @abstractmethod
    def load(self):
        """Load raw dataset files."""
        raise NotImplementedError

    @abstractmethod
    def standardize(self) -> StandardizedData:
        """Convert raw data into the common schema."""
        raise NotImplementedError

    def check_path(self) -> None:
        """Print basic dataset path information."""
        print(f"Dataset       : {self.dataset_name}")
        print(f"Dataset path  : {self.root_path}")
        print(f"Path exists   : {self.root_path.exists()}")


# =============================================================================
# 3. STANDARDIZATION HELPERS
# =============================================================================

def create_empty_standard_dataframe(length: int = 0) -> pd.DataFrame:
    """Create a dataframe containing every standard column initialized to NaN."""
    df = pd.DataFrame(index=np.arange(length))
    for column in STANDARD_COLUMNS:
        df[column] = np.nan
    return df


def ensure_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* containing exactly the standard columns."""
    standardized = df.copy()
    for column in STANDARD_COLUMNS:
        if column not in standardized.columns:
            standardized[column] = np.nan
    return standardized[STANDARD_COLUMNS]


def validate_standardized_data(df: pd.DataFrame) -> None:
    """Validate a dataframe against the unified schema."""
    missing = [column for column in STANDARD_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Standardization failed. Missing columns: {missing}"
        )
    print("Standardized schema validation: PASSED")


# =============================================================================
# 4. HAR DATASET LOADER
# =============================================================================

class HARLoader(BaseDatasetLoader):
    """Loader for the IMU-based Human Activity Recognition dataset."""

    def __init__(
        self,
        root_path: str | Path,
        sampling_rate: Optional[float] = None,
    ) -> None:
        super().__init__(root_path=root_path, dataset_name="HAR")
        self.sampling_rate = sampling_rate
        self.csv_file: Optional[Path] = None
        self.raw_data: Optional[pd.DataFrame] = None

    def discover(self):
        """Find CSV files inside the HAR dataset directory."""
        csv_files = list(self.root_path.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV file found inside:\n{self.root_path}"
            )

        if len(csv_files) > 1:
            print(
                f"Warning: found {len(csv_files)} CSV files. "
                "Using the first one."
            )

        self.csv_file = csv_files[0]
        print("HAR Dataset Discovery")
        print("---------------------")
        print(f"CSV file: {self.csv_file.name}")
        return self.csv_file

    def load(self) -> pd.DataFrame:
        """Load the HAR CSV file into a pandas dataframe."""
        if self.csv_file is None:
            self.discover()

        self.raw_data = pd.read_csv(self.csv_file)
        print("\nHAR Dataset Loaded")
        print("-----------------")
        print(f"Rows    : {len(self.raw_data):,}")
        print(f"Columns : {len(self.raw_data.columns)}")
        return self.raw_data

    @staticmethod
    def _find_column(columns, candidates, required: bool = True):
        """Find a column using case-insensitive exact/partial matching."""
        normalized = {
            str(column).strip().lower(): column for column in columns
        }

        for candidate in candidates:
            key = candidate.lower()
            if key in normalized:
                return normalized[key]

        for column in columns:
            column_lower = str(column).strip().lower()
            for candidate in candidates:
                if candidate.lower() in column_lower:
                    return column

        if required:
            raise ValueError(
                "Could not find required column.\n"
                f"Possible names: {candidates}\n"
                f"Available columns:\n{list(columns)}"
            )
        return None

    def standardize(self) -> StandardizedData:
        """Convert the HAR dataset into the unified schema."""
        if self.raw_data is None:
            self.load()

        df = self.raw_data.copy()

        ax_col = self._find_column(
            df.columns,
            ["ax", "acc_x", "accel_x", "accelerometer_x", "acceleration_x", "x_acceleration"],
        )
        ay_col = self._find_column(
            df.columns,
            ["ay", "acc_y", "accel_y", "accelerometer_y", "acceleration_y", "y_acceleration"],
        )
        az_col = self._find_column(
            df.columns,
            ["az", "acc_z", "accel_z", "accelerometer_z", "acceleration_z", "z_acceleration"],
        )

        gx_col = self._find_column(
            df.columns, ["gx", "gyro_x", "gyroscope_x", "gyr_x"], required=False
        )
        gy_col = self._find_column(
            df.columns, ["gy", "gyro_y", "gyroscope_y", "gyr_y"], required=False
        )
        gz_col = self._find_column(
            df.columns, ["gz", "gyro_z", "gyroscope_z", "gyr_z"], required=False
        )

        label_col = self._find_column(
            df.columns, ["activity", "activity_label", "label", "class", "action"]
        )
        timestamp_col = self._find_column(
            df.columns, ["timestamp", "time", "time_stamp"], required=False
        )
        subject_col = self._find_column(
            df.columns,
            ["subject", "subject_id", "participant", "participant_id", "user", "user_id"],
            required=False,
        )

        standardized = create_empty_standard_dataframe(length=len(df))

        standardized["timestamp"] = (
            df[timestamp_col].values if timestamp_col is not None else np.arange(len(df))
        )

        standardized["ax"] = pd.to_numeric(df[ax_col], errors="coerce")
        standardized["ay"] = pd.to_numeric(df[ay_col], errors="coerce")
        standardized["az"] = pd.to_numeric(df[az_col], errors="coerce")

        if gx_col is not None:
            standardized["gx"] = pd.to_numeric(df[gx_col], errors="coerce")
        if gy_col is not None:
            standardized["gy"] = pd.to_numeric(df[gy_col], errors="coerce")
        if gz_col is not None:
            standardized["gz"] = pd.to_numeric(df[gz_col], errors="coerce")

        standardized["label"] = df[label_col].astype(str).values
        standardized["subject_id"] = (
            df[subject_col].astype(str).values if subject_col is not None else "HAR"
        )
        standardized["sensor_id"] = "IMU"
        standardized["dataset_source"] = "HAR"

        standardized = ensure_standard_columns(standardized)
        validate_standardized_data(standardized)

        return StandardizedData(
            data=standardized,
            dataset_name="HAR",
            sampling_rate=self.sampling_rate,
        )

    def run(self) -> StandardizedData:
        """Run discovery, loading, and standardization."""
        self.discover()
        self.load()
        return self.standardize()


# =============================================================================
# 5. MULTIMODAL COUGH DATASET LOADER
# =============================================================================

class CoughLoader:
    """Loader for the Multimodal Cough Dataset."""

    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path)

        dataset_folder = self.root_path / "Multimodal Cough Dataset"
        self.dataset_root = (
            dataset_folder if dataset_folder.exists() else self.root_path
        )

        self.raw_data: Optional[pd.DataFrame] = None
        self.metadata: dict = {}
        self.trial_records: list[dict] = []

    @staticmethod
    def _find_file(directory: Path, filename: str) -> Optional[Path]:
        """Find a file by case-insensitive filename."""
        filename = filename.lower()
        for file in directory.iterdir():
            if file.is_file() and file.name.lower() == filename:
                return file
        return None

    def discover_trials(self) -> list[dict]:
        """Discover subject/trial directories containing accelerometer data."""
        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Cough dataset path does not exist:\n{self.dataset_root}"
            )

        trials = []
        for accel_file in self.dataset_root.rglob("*"):
            if not accel_file.is_file() or accel_file.name.lower() != "accelerometer.csv":
                continue

            trial_dir = accel_file.parent
            subject_dir = trial_dir.parent
            trials.append({
                "subject_id": subject_dir.name,
                "trial": trial_dir.name,
                "trial_path": trial_dir,
                "accelerometer": accel_file,
                "gyroscope": self._find_file(trial_dir, "gyroscope.csv"),
                "magnetometer": self._find_file(trial_dir, "magnetometer.csv"),
                "audio_files": [
                    path
                    for path in trial_dir.iterdir()
                    if path.is_file() and path.suffix.lower() == ".wav"
                ],
            })

        self.trial_records = trials
        return trials

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names without changing their values."""
        result = df.copy()
        result.columns = [str(col).strip().lower() for col in result.columns]
        return result

    def load(self) -> pd.DataFrame:
        """Load and align accelerometer, gyroscope, and magnetometer data."""
        trials = self.discover_trials()
        print(f"Discovered trials: {len(trials)}")

        if not trials:
            raise RuntimeError(
                "No valid cough IMU recordings were discovered.\n"
                f"Dataset root checked:\n{self.dataset_root}"
            )

        records = []

        for trial in trials:
            accel_file = trial["accelerometer"]
            gyro_file = trial["gyroscope"]
            mag_file = trial["magnetometer"]

            try:
                accel = self._normalize_columns(pd.read_csv(accel_file))
                accel_required = ["elapsed (s)", "x-axis (g)", "y-axis (g)", "z-axis (g)"]
                missing = [col for col in accel_required if col not in accel.columns]

                if missing:
                    print(
                        f"Skipping {accel_file.name}: missing accelerometer columns {missing}"
                    )
                    continue

                accel = accel[accel_required].copy()
                accel.rename(
                    columns={
                        "elapsed (s)": "timestamp",
                        "x-axis (g)": "ax",
                        "y-axis (g)": "ay",
                        "z-axis (g)": "az",
                    },
                    inplace=True,
                )
                accel["timestamp"] = pd.to_numeric(accel["timestamp"], errors="coerce")

                if gyro_file is not None:
                    gyro = self._normalize_columns(pd.read_csv(gyro_file))
                    gyro_required = [
                        "elapsed (s)",
                        "x-axis (deg/s)",
                        "y-axis (deg/s)",
                        "z-axis (deg/s)",
                    ]

                    if all(col in gyro.columns for col in gyro_required):
                        gyro = gyro[gyro_required].copy()
                        gyro.rename(
                            columns={
                                "elapsed (s)": "timestamp",
                                "x-axis (deg/s)": "gx",
                                "y-axis (deg/s)": "gy",
                                "z-axis (deg/s)": "gz",
                            },
                            inplace=True,
                        )
                        gyro["timestamp"] = pd.to_numeric(
                            gyro["timestamp"], errors="coerce"
                        )
                        accel = pd.merge_asof(
                            accel.sort_values("timestamp"),
                            gyro.sort_values("timestamp"),
                            on="timestamp",
                            direction="nearest",
                        )
                    else:
                        print(
                            f"Warning: Gyroscope format not recognized for {gyro_file.name}"
                        )
                        accel[["gx", "gy", "gz"]] = np.nan
                else:
                    accel[["gx", "gy", "gz"]] = np.nan

                if mag_file is not None:
                    mag = self._normalize_columns(pd.read_csv(mag_file))
                    mag_required = [
                        "elapsed (s)",
                        "x-axis (t)",
                        "y-axis (t)",
                        "z-axis (t)",
                    ]

                    if all(col in mag.columns for col in mag_required):
                        mag = mag[mag_required].copy()
                        mag.rename(
                            columns={
                                "elapsed (s)": "timestamp",
                                "x-axis (t)": "mx",
                                "y-axis (t)": "my",
                                "z-axis (t)": "mz",
                            },
                            inplace=True,
                        )
                        mag["timestamp"] = pd.to_numeric(
                            mag["timestamp"], errors="coerce"
                        )
                        accel = pd.merge_asof(
                            accel.sort_values("timestamp"),
                            mag.sort_values("timestamp"),
                            on="timestamp",
                            direction="nearest",
                        )
                    else:
                        print(
                            f"Warning: Magnetometer format not recognized for {mag_file.name}"
                        )
                        accel[["mx", "my", "mz"]] = np.nan
                else:
                    accel[["mx", "my", "mz"]] = np.nan

                accel["sensor_id"] = "IMU"
                accel["label"] = trial["trial"]
                accel["subject_id"] = trial["subject_id"]
                accel["dataset_source"] = "COUGH"
                records.append(accel)

            except Exception as exc:
                print(
                    f"Skipping trial {trial['subject_id']} / {trial['trial']}: {exc}"
                )

        if not records:
            raise RuntimeError(
                "Trials were discovered, but no valid multimodal IMU recordings could be loaded."
            )

        self.raw_data = pd.concat(records, ignore_index=True)

        required_schema = [
            "timestamp",
            "sensor_id",
            "ax", "ay", "az",
            "gx", "gy", "gz",
            "mx", "my", "mz",
            "label",
            "subject_id",
            "dataset_source",
        ]

        for column in required_schema:
            if column not in self.raw_data.columns:
                self.raw_data[column] = np.nan

        self.raw_data = self.raw_data[required_schema]

        self.metadata = {
            "dataset_name": "Multimodal Cough Dataset",
            "rows": len(self.raw_data),
            "subjects": self.raw_data["subject_id"].nunique(),
            "trials": len(trials),
            "labels": self.raw_data["label"].nunique(),
            "columns": list(self.raw_data.columns),
            "accelerometer_available": self.raw_data[["ax", "ay", "az"]].notna().any().any(),
            "gyroscope_available": self.raw_data[["gx", "gy", "gz"]].notna().any().any(),
            "magnetometer_available": self.raw_data[["mx", "my", "mz"]].notna().any().any(),
            "audio_available": any(
                len(trial.get("audio_files", [])) > 0 for trial in trials
            ),
        }

        return self.raw_data


# =============================================================================
# 6. FOUR-IMU FETAL DATASET LOADER
# =============================================================================

class FourIMULoader(BaseDatasetLoader):
    """Loader for the Fetal Movement Dataset Recorded Using Four IMUs."""

    STANDARD_COLUMNS = [
        "timestamp",
        "sensor_id",
        "ax", "ay", "az",
        "gx", "gy", "gz",
        "mx", "my", "mz",
        "label",
        "subject_id",
        "dataset_source",
        "record_id",
        "sub_dataset",
    ]

    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path)
        self.dataset_root = self.root_path
        self.raw_data: Optional[pd.DataFrame] = None
        self.metadata: dict = {}
        self.record_files: list[dict] = []

    def discover(self) -> list[dict]:
        """Discover recording CSVs while excluding metadata CSVs."""
        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Four-IMU dataset path does not exist:\n{self.dataset_root}"
            )

        records = []
        for subdir in sorted(self.dataset_root.glob("Sub-dataset *")):
            if not subdir.is_dir():
                continue
            for csv_file in sorted(subdir.glob("record_*.csv")):
                if csv_file.is_file():
                    records.append({
                        "sub_dataset": subdir.name,
                        "record_id": csv_file.stem,
                        "path": csv_file,
                    })

        self.record_files = records
        return records

    @staticmethod
    def _required_columns() -> list[str]:
        columns = ["time", "label"]
        for imu in range(1, 5):
            columns.extend([
                f"ax{imu}", f"ay{imu}", f"az{imu}",
                f"gx{imu}", f"gy{imu}", f"gz{imu}",
            ])
        return columns

    def _validate_record(self, df: pd.DataFrame, path: Path) -> None:
        """Strictly validate a Four-IMU recording before conversion."""
        required = self._required_columns()
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(
                f"{path.name}: missing required columns: {missing}"
            )

        if df[required].isna().any().any():
            raise ValueError(
                f"{path.name}: required columns contain missing values"
            )

        if not pd.api.types.is_numeric_dtype(df["time"]):
            raise ValueError(f"{path.name}: time column is not numeric")

        for column in required:
            if column == "label":
                continue
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(
                    f"{path.name}: column '{column}' is not numeric"
                )

    def load(self) -> pd.DataFrame:
        """Load and convert all Four-IMU recordings to long format."""
        records = self.discover()
        print(f"Discovered Four-IMU recordings: {len(records)}")

        if not records:
            raise RuntimeError(
                "No Four-IMU recording CSV files were discovered."
            )

        standardized_records = []
        skipped = 0

        for record in records:
            path = record["path"]
            try:
                df = pd.read_csv(path)
                df.columns = [str(column).strip().lower() for column in df.columns]
                self._validate_record(df, path)

                time = df["time"].to_numpy()
                chunks = []

                for imu in range(1, 5):
                    chunks.append(pd.DataFrame({
                        "timestamp": time,
                        "sensor_id": f"IMU{imu}",
                        "ax": df[f"ax{imu}"].to_numpy(),
                        "ay": df[f"ay{imu}"].to_numpy(),
                        "az": df[f"az{imu}"].to_numpy(),
                        "gx": df[f"gx{imu}"].to_numpy(),
                        "gy": df[f"gy{imu}"].to_numpy(),
                        "gz": df[f"gz{imu}"].to_numpy(),
                        "mx": np.nan,
                        "my": np.nan,
                        "mz": np.nan,
                        "label": df["label"].to_numpy(),
                        "subject_id": "single_subject",
                        "dataset_source": "FOUR_IMU",
                        "record_id": record["record_id"],
                        "sub_dataset": record["sub_dataset"],
                    }))

                standardized_records.append(
                    pd.concat(chunks, ignore_index=True)
                )

            except Exception as exc:
                skipped += 1
                print(f"Skipping {path.name}: {exc}")

        if not standardized_records:
            raise RuntimeError(
                "Four-IMU files were discovered, but none passed schema/data validation."
            )

        self.raw_data = pd.concat(standardized_records, ignore_index=True)
        self.raw_data = self.raw_data[self.STANDARD_COLUMNS]

        if set(self.raw_data.columns) != set(self.STANDARD_COLUMNS):
            raise RuntimeError("Four-IMU standardized schema validation failed.")

        sensor_columns = ["ax", "ay", "az", "gx", "gy", "gz"]
        if self.raw_data[sensor_columns].isna().any().any():
            raise RuntimeError(
                "Four-IMU accelerometer/gyroscope data contains unexpected missing values."
            )

        self.metadata = {
            "dataset_name": "Fetal Movement Dataset Recorded Using Four IMUs",
            "rows": len(self.raw_data),
            "raw_recordings_discovered": len(records),
            "recordings_loaded": self.raw_data["record_id"].nunique(),
            "recordings_skipped": skipped,
            "subjects": self.raw_data["subject_id"].nunique(),
            "sub_datasets": self.raw_data["sub_dataset"].nunique(),
            "labels": self.raw_data["label"].nunique(),
            "imu_count": self.raw_data["sensor_id"].nunique(),
            "sampling_rate_hz": 30.0,
            "accelerometer_available": True,
            "gyroscope_available": True,
            "magnetometer_available": False,
            "columns": list(self.raw_data.columns),
        }

        return self.raw_data


# =============================================================================
# 7. OXFORD FEMALE FETAL DATASET LOADER
# =============================================================================

class OxfordLoader(BaseDatasetLoader):
    """Loader for the Oxford female fetal dataset."""

    def __init__(self, root_path: str | Path) -> None:
        super().__init__(root_path, "Oxford Female Fetal Dataset")
        self.raw_data: Optional[pd.DataFrame] = None
        self.metadata: dict = {}
        self.signal_variable = "BCG_PREPROC_3AXIS"
        self.label_variable = "BP_MOUV_FILES"
        self.records: list[dict] = []

    def discover(self) -> list[dict]:
        """Discover and pair Oxford signal and BP MATLAB files."""
        signal_files = sorted(self.root_path.rglob("*_signal.mat"))
        bp_files = sorted(self.root_path.rglob("*_bp.mat"))

        signal_map = {
            file.name.replace("_signal.mat", ""): file for file in signal_files
        }
        bp_map = {
            file.name.replace("_bp.mat", ""): file for file in bp_files
        }

        record_ids = sorted(set(signal_map) | set(bp_map))
        self.records = [
            {
                "record_id": record_id,
                "signal_file": signal_map.get(record_id),
                "bp_file": bp_map.get(record_id),
            }
            for record_id in record_ids
        ]
        return self.records

    def load(self) -> pd.DataFrame:
        """Load, align, and standardize Oxford BCG records."""
        records = self.discover()
        print(f"Discovered Oxford records: {len(records)}")

        if not records:
            raise RuntimeError(
                "No Oxford signal/BP records were discovered.\n"
                f"Dataset root checked:\n{self.root_path}"
            )

        loaded_records = []
        records_loaded = 0
        records_skipped = 0
        total_signal_samples = 0
        total_label_samples = 0
        total_aligned_samples = 0
        truncated_records = []

        for record in records:
            record_id = record["record_id"]
            signal_file = record["signal_file"]
            bp_file = record["bp_file"]

            if signal_file is None:
                print(f"Skipping {record_id}: signal file missing")
                records_skipped += 1
                continue
            if bp_file is None:
                print(f"Skipping {record_id}: BP file missing")
                records_skipped += 1
                continue

            try:
                signal_mat = loadmat(signal_file)
                if self.signal_variable not in signal_mat:
                    print(
                        f"Skipping {record_id}: signal variable "
                        f"'{self.signal_variable}' not found"
                    )
                    records_skipped += 1
                    continue

                signal = np.asarray(signal_mat[self.signal_variable])
                if signal.ndim != 2:
                    print(
                        f"Skipping {record_id}: unexpected signal dimensions {signal.shape}"
                    )
                    records_skipped += 1
                    continue
                if signal.shape[1] != 3:
                    print(
                        f"Skipping {record_id}: expected 3 BCG axes, got shape {signal.shape}"
                    )
                    records_skipped += 1
                    continue

                bp_mat = loadmat(bp_file)
                if self.label_variable not in bp_mat:
                    print(
                        f"Skipping {record_id}: label variable "
                        f"'{self.label_variable}' not found"
                    )
                    records_skipped += 1
                    continue

                labels = np.asarray(bp_mat[self.label_variable]).reshape(-1)
                signal_length = len(signal)
                label_length = len(labels)
                total_signal_samples += signal_length
                total_label_samples += label_length

                aligned_length = min(signal_length, label_length)
                if aligned_length == 0:
                    print(f"Skipping {record_id}: empty signal or label array")
                    records_skipped += 1
                    continue

                if signal_length != label_length:
                    print(
                        f"Warning: {record_id} length mismatch "
                        f"(signal={signal_length}, labels={label_length}). "
                        f"Truncating both to {aligned_length} samples."
                    )
                    truncated_records.append({
                        "record_id": record_id,
                        "signal_length": signal_length,
                        "label_length": label_length,
                        "aligned_length": aligned_length,
                    })

                signal = signal[:aligned_length]
                labels = labels[:aligned_length]
                total_aligned_samples += aligned_length

                loaded_records.append(pd.DataFrame({
                    "timestamp": np.arange(aligned_length, dtype=np.int64),
                    "sensor_id": "BCG_3AXIS",
                    "ax": np.nan,
                    "ay": np.nan,
                    "az": np.nan,
                    "gx": np.nan,
                    "gy": np.nan,
                    "gz": np.nan,
                    "mx": np.nan,
                    "my": np.nan,
                    "mz": np.nan,
                    "bcg_x": signal[:, 0],
                    "bcg_y": signal[:, 1],
                    "bcg_z": signal[:, 2],
                    "label": labels,
                    "subject_id": "oxford_female",
                    "dataset_source": "OXFORD",
                    "record_id": record_id,
                }))
                records_loaded += 1

            except Exception as exc:
                print(f"Skipping Oxford record {record_id}: {exc}")
                records_skipped += 1

        if not loaded_records:
            raise RuntimeError(
                "Oxford records were discovered, but no valid records could be loaded."
            )

        required_schema = [
            "timestamp",
            "sensor_id",
            "ax", "ay", "az",
            "gx", "gy", "gz",
            "mx", "my", "mz",
            "bcg_x", "bcg_y", "bcg_z",
            "label",
            "subject_id",
            "dataset_source",
            "record_id",
        ]

        self.raw_data = pd.concat(loaded_records, ignore_index=True)
        for column in required_schema:
            if column not in self.raw_data.columns:
                self.raw_data[column] = np.nan
        self.raw_data = self.raw_data[required_schema]

        self.metadata = {
            "dataset_name": "Oxford Female Fetal Dataset",
            "rows": len(self.raw_data),
            "records_discovered": len(records),
            "records_loaded": records_loaded,
            "records_skipped": records_skipped,
            "records_truncated": len(truncated_records),
            "subjects": self.raw_data["subject_id"].nunique(),
            "labels": self.raw_data["label"].nunique(),
            "signal_variable": self.signal_variable,
            "label_variable": self.label_variable,
            "signal_type": "BCG",
            "signal_axes": 3,
            "sampling_rate_hz": None,
            "timestamp_unit": "sample_index",
            "original_signal_samples": total_signal_samples,
            "original_label_samples": total_label_samples,
            "aligned_samples": total_aligned_samples,
            "accelerometer_available": False,
            "gyroscope_available": False,
            "magnetometer_available": False,
            "bcg_available": True,
            "columns": list(self.raw_data.columns),
            "truncated_records": truncated_records,
        }

        return self.raw_data