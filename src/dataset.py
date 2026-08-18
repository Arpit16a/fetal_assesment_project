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

Architecture
------------
Every dataset loader follows the same lifecycle:

    discover()
        ↓
    load()
        ↓
    standardize()
        ↓
    StandardizedData

The public `run()` method executes the complete lifecycle.

The loaders preserve the source data and do not invent unavailable
modalities. Missing sensor modalities are represented with NaN.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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


# =============================================================================
# 2. STANDARDIZED DATA CONTAINER
# =============================================================================

@dataclass
class StandardizedData:
    """Container for standardized sensor data."""

    data: pd.DataFrame
    dataset_name: str
    sampling_rate: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None

    def validate(self) -> None:
        """Validate that the dataframe contains the complete common schema."""

        missing_columns = [
            column
            for column in STANDARD_COLUMNS
            if column not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing standardized columns: {missing_columns}"
            )

        if len(self.data) == 0:
            raise ValueError("Standardized dataset is empty.")

        if self.data["timestamp"].isna().any():
            raise ValueError("Standardized dataset contains missing timestamps.")

        if self.data["label"].isna().any():
            raise ValueError("Standardized dataset contains missing labels.")

        if self.data["subject_id"].isna().any():
            raise ValueError(
                "Standardized dataset contains missing subject IDs."
            )

        if self.data["dataset_source"].isna().any():
            raise ValueError(
                "Standardized dataset contains missing dataset sources."
            )

    def summary(self) -> dict[str, Any]:
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
# 3. BASE DATASET LOADER
# =============================================================================

class BaseDatasetLoader(ABC):
    """
    Abstract base class for all dataset loaders.

    Every concrete loader must implement:

        discover()
        load()
        standardize()

    `run()` provides the common orchestration layer.
    """

    def __init__(
        self,
        root_path: str | Path,
        dataset_name: str,
        sampling_rate: Optional[float] = None,
    ) -> None:

        self.root_path = Path(root_path)
        self.dataset_name = dataset_name
        self.sampling_rate = sampling_rate

        if not self.root_path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist:\n{self.root_path}"
            )

        self.metadata: dict[str, Any] = {}
        self.raw_data: Any = None
        self.standardized_data: Optional[StandardizedData] = None

    @abstractmethod
    def discover(self) -> Any:
        """Discover files belonging to the dataset."""
        raise NotImplementedError

    @abstractmethod
    def load(self) -> Any:
        """Load discovered raw dataset files into memory."""
        raise NotImplementedError

    @abstractmethod
    def standardize(self) -> StandardizedData:
        """Convert loaded raw data into the common schema."""
        raise NotImplementedError

    def run(self) -> StandardizedData:
        """
        Execute the complete dataset loading lifecycle.

        Lifecycle
        ---------
        1. Discover raw files
        2. Load raw data
        3. Standardize raw data
        4. Validate standardized output
        """

        self.discover()
        self.load()

        standardized = self.standardize()

        if not isinstance(standardized, StandardizedData):
            raise TypeError(
                f"{self.__class__.__name__}.standardize() must return "
                "a StandardizedData object."
            )

        standardized.validate()

        self.standardized_data = standardized
        return standardized

    def check_path(self) -> None:
        """Print basic dataset path information."""

        print(f"Dataset       : {self.dataset_name}")
        print(f"Dataset path  : {self.root_path}")
        print(f"Path exists   : {self.root_path.exists()}")


# =============================================================================
# 4. STANDARDIZATION HELPERS
# =============================================================================

def create_empty_standard_dataframe(length: int = 0) -> pd.DataFrame:
    """
    Create a dataframe containing every standard column initialized to NaN.
    """

    df = pd.DataFrame(index=np.arange(length))

    for column in STANDARD_COLUMNS:
        df[column] = np.nan

    return df


def ensure_standard_columns(
    df: pd.DataFrame,
    keep_extra_columns: bool = False,
) -> pd.DataFrame:
    """
    Ensure that all standard columns exist.

    Parameters
    ----------
    df:
        Input dataframe.

    keep_extra_columns:
        If False, return exactly STANDARD_COLUMNS.
        If True, preserve dataset-specific columns after STANDARD_COLUMNS.
    """

    standardized = df.copy()

    for column in STANDARD_COLUMNS:
        if column not in standardized.columns:
            standardized[column] = np.nan

    if keep_extra_columns:
        extra_columns = [
            column
            for column in standardized.columns
            if column not in STANDARD_COLUMNS
        ]

        return standardized[
            STANDARD_COLUMNS + extra_columns
        ]

    return standardized[STANDARD_COLUMNS]


def validate_standardized_data(df: pd.DataFrame) -> None:
    """Validate a dataframe against the unified schema."""

    missing = [
        column
        for column in STANDARD_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Standardization failed. Missing columns: {missing}"
        )

    if len(df) == 0:
        raise ValueError("Standardized dataframe is empty.")

    print("Standardized schema validation: PASSED")


# =============================================================================
# 5. HAR DATASET LOADER
# =============================================================================

class HARLoader(BaseDatasetLoader):
    """Loader for the IMU-based Human Activity Recognition dataset."""

    def __init__(
        self,
        root_path: str | Path,
        sampling_rate: Optional[float] = None,
    ) -> None:

        super().__init__(
            root_path=root_path,
            dataset_name="HAR",
            sampling_rate=sampling_rate,
        )

        self.csv_file: Optional[Path] = None

    def discover(self) -> Path:
        """Find CSV files inside the HAR dataset directory."""

        csv_files = sorted(self.root_path.rglob("*.csv"))

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
        print(f"CSV file: {self.csv_file}")

        return self.csv_file

    def load(self) -> pd.DataFrame:
        """Load the raw HAR CSV file."""

        if self.csv_file is None:
            self.discover()

        self.raw_data = pd.read_csv(self.csv_file)

        print("\nHAR Dataset Loaded")
        print("-----------------")
        print(f"Rows    : {len(self.raw_data):,}")
        print(f"Columns : {len(self.raw_data.columns)}")

        return self.raw_data

    @staticmethod
    def _find_column(
        columns,
        candidates,
        required: bool = True,
    ):
        """Find a column using case-insensitive exact/partial matching."""

        normalized = {
            str(column).strip().lower(): column
            for column in columns
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
        """Convert the raw HAR dataset into the unified schema."""

        if self.raw_data is None:
            self.load()

        df = self.raw_data.copy()

        ax_col = self._find_column(
            df.columns,
            [
                "ax",
                "acc_x",
                "accel_x",
                "accelerometer_x",
                "acceleration_x",
                "x_acceleration",
            ],
        )

        ay_col = self._find_column(
            df.columns,
            [
                "ay",
                "acc_y",
                "accel_y",
                "accelerometer_y",
                "acceleration_y",
                "y_acceleration",
            ],
        )

        az_col = self._find_column(
            df.columns,
            [
                "az",
                "acc_z",
                "accel_z",
                "accelerometer_z",
                "acceleration_z",
                "z_acceleration",
            ],
        )

        gx_col = self._find_column(
            df.columns,
            ["gx", "gyro_x", "gyroscope_x", "gyr_x"],
            required=False,
        )

        gy_col = self._find_column(
            df.columns,
            ["gy", "gyro_y", "gyroscope_y", "gyr_y"],
            required=False,
        )

        gz_col = self._find_column(
            df.columns,
            ["gz", "gyro_z", "gyroscope_z", "gyr_z"],
            required=False,
        )

        label_col = self._find_column(
            df.columns,
            [
                "activity",
                "activity_label",
                "label",
                "class",
                "action",
            ],
        )

        timestamp_col = self._find_column(
            df.columns,
            ["timestamp", "time", "time_stamp"],
            required=False,
        )

        subject_col = self._find_column(
            df.columns,
            [
                "subject",
                "subject_id",
                "participant",
                "participant_id",
                "user",
                "user_id",
            ],
            required=False,
        )

        standardized = create_empty_standard_dataframe(
            length=len(df)
        )

        if timestamp_col is not None:
            standardized["timestamp"] = pd.to_numeric(
                df[timestamp_col],
                errors="coerce",
            )
        else:
            standardized["timestamp"] = np.arange(len(df))

        standardized["ax"] = pd.to_numeric(
            df[ax_col],
            errors="coerce",
        )

        standardized["ay"] = pd.to_numeric(
            df[ay_col],
            errors="coerce",
        )

        standardized["az"] = pd.to_numeric(
            df[az_col],
            errors="coerce",
        )

        if gx_col is not None:
            standardized["gx"] = pd.to_numeric(
                df[gx_col],
                errors="coerce",
            )

        if gy_col is not None:
            standardized["gy"] = pd.to_numeric(
                df[gy_col],
                errors="coerce",
            )

        if gz_col is not None:
            standardized["gz"] = pd.to_numeric(
                df[gz_col],
                errors="coerce",
            )

        standardized["label"] = df[label_col].astype(str).values

        if subject_col is not None:
            standardized["subject_id"] = (
                df[subject_col].astype(str).values
            )
        else:
            standardized["subject_id"] = "HAR"

        standardized["sensor_id"] = "IMU"
        standardized["dataset_source"] = "HAR"
        standardized["record_id"] = "HAR"

        standardized = ensure_standard_columns(standardized)
        validate_standardized_data(standardized)

        return StandardizedData(
            data=standardized,
            dataset_name=self.dataset_name,
            sampling_rate=self.sampling_rate,
            metadata=self.metadata,
        )


# =============================================================================
# 6. MULTIMODAL COUGH DATASET LOADER
# =============================================================================

class CoughLoader(BaseDatasetLoader):
    """Loader for the Multimodal Cough Dataset."""

    def __init__(
        self,
        root_path: str | Path,
    ) -> None:

        super().__init__(
            root_path=root_path,
            dataset_name="Multimodal Cough Dataset",
        )

        dataset_folder = self.root_path / "Multimodal Cough Dataset"

        self.dataset_root = (
            dataset_folder
            if dataset_folder.exists()
            else self.root_path
        )

        self.trial_records: list[dict] = []

    @staticmethod
    def _find_file(
        directory: Path,
        filename: str,
    ) -> Optional[Path]:
        """Find a file by case-insensitive filename."""

        filename = filename.lower()

        for file in directory.iterdir():
            if file.is_file() and file.name.lower() == filename:
                return file

        return None

    def discover(self) -> list[dict]:
        """Discover subject/trial directories containing accelerometer data."""

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Cough dataset path does not exist:\n"
                f"{self.dataset_root}"
            )

        trials = []

        for accel_file in self.dataset_root.rglob("*"):

            if (
                not accel_file.is_file()
                or accel_file.name.lower() != "accelerometer.csv"
            ):
                continue

            trial_dir = accel_file.parent
            subject_dir = trial_dir.parent

            trials.append(
                {
                    "subject_id": subject_dir.name,
                    "trial": trial_dir.name,
                    "record_id": trial_dir.name,
                    "trial_path": trial_dir,
                    "accelerometer": accel_file,
                    "gyroscope": self._find_file(
                        trial_dir,
                        "gyroscope.csv",
                    ),
                    "magnetometer": self._find_file(
                        trial_dir,
                        "magnetometer.csv",
                    ),
                    "audio_files": [
                        path
                        for path in trial_dir.iterdir()
                        if (
                            path.is_file()
                            and path.suffix.lower() == ".wav"
                        )
                    ],
                }
            )

        self.trial_records = trials

        print(f"Discovered cough trials: {len(trials)}")

        return trials

    @staticmethod
    def _normalize_columns(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize column names without changing their values."""

        result = df.copy()

        result.columns = [
            str(col).strip().lower()
            for col in result.columns
        ]

        return result

    def load(self) -> list[dict]:
        """
        Load raw accelerometer, gyroscope, and magnetometer files.

        The raw records remain separated by trial. Alignment and conversion
        into the unified schema are performed by `standardize()`.
        """

        trials = self.trial_records

        if not trials:
            trials = self.discover()

        if not trials:
            raise RuntimeError(
                "No valid cough IMU recordings were discovered.\n"
                f"Dataset root checked:\n{self.dataset_root}"
            )

        raw_trials = []

        for trial in trials:

            try:
                accel = self._normalize_columns(
                    pd.read_csv(trial["accelerometer"])
                )

                gyro = None

                if trial["gyroscope"] is not None:
                    gyro = self._normalize_columns(
                        pd.read_csv(trial["gyroscope"])
                    )

                mag = None

                if trial["magnetometer"] is not None:
                    mag = self._normalize_columns(
                        pd.read_csv(trial["magnetometer"])
                    )

                raw_trials.append(
                    {
                        "metadata": trial,
                        "accelerometer": accel,
                        "gyroscope": gyro,
                        "magnetometer": mag,
                    }
                )

            except Exception as exc:
                print(
                    f"Skipping trial "
                    f"{trial['subject_id']} / "
                    f"{trial['trial']}: {exc}"
                )

        if not raw_trials:
            raise RuntimeError(
                "Trials were discovered, but no raw cough recordings "
                "could be loaded."
            )

        self.raw_data = raw_trials

        return self.raw_data

    def standardize(self) -> StandardizedData:
        """Align multimodal IMU streams and convert them to common schema."""

        if self.raw_data is None:
            self.load()

        records = []

        for raw_trial in self.raw_data:

            trial = raw_trial["metadata"]
            accel = raw_trial["accelerometer"]
            gyro = raw_trial["gyroscope"]
            mag = raw_trial["magnetometer"]

            accel_required = [
                "elapsed (s)",
                "x-axis (g)",
                "y-axis (g)",
                "z-axis (g)",
            ]

            missing = [
                col
                for col in accel_required
                if col not in accel.columns
            ]

            if missing:
                print(
                    f"Skipping {trial['record_id']}: "
                    f"missing accelerometer columns {missing}"
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

            accel["timestamp"] = pd.to_numeric(
                accel["timestamp"],
                errors="coerce",
            )

            accel = accel.dropna(
                subset=["timestamp"]
            ).sort_values("timestamp")

            # -----------------------------------------------------------------
            # Gyroscope
            # -----------------------------------------------------------------

            if gyro is not None:

                gyro_required = [
                    "elapsed (s)",
                    "x-axis (deg/s)",
                    "y-axis (deg/s)",
                    "z-axis (deg/s)",
                ]

                if all(
                    col in gyro.columns
                    for col in gyro_required
                ):

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
                        gyro["timestamp"],
                        errors="coerce",
                    )

                    gyro = gyro.dropna(
                        subset=["timestamp"]
                    ).sort_values("timestamp")

                    accel = pd.merge_asof(
                        accel,
                        gyro,
                        on="timestamp",
                        direction="nearest",
                    )

                else:

                    print(
                        f"Warning: Gyroscope format not recognized "
                        f"for {trial['record_id']}"
                    )

                    accel[["gx", "gy", "gz"]] = np.nan

            else:

                accel[["gx", "gy", "gz"]] = np.nan

            # -----------------------------------------------------------------
            # Magnetometer
            # -----------------------------------------------------------------

            if mag is not None:

                mag_required = [
                    "elapsed (s)",
                    "x-axis (t)",
                    "y-axis (t)",
                    "z-axis (t)",
                ]

                if all(
                    col in mag.columns
                    for col in mag_required
                ):

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
                        mag["timestamp"],
                        errors="coerce",
                    )

                    mag = mag.dropna(
                        subset=["timestamp"]
                    ).sort_values("timestamp")

                    accel = pd.merge_asof(
                        accel.sort_values("timestamp"),
                        mag,
                        on="timestamp",
                        direction="nearest",
                    )

                else:

                    print(
                        f"Warning: Magnetometer format not recognized "
                        f"for {trial['record_id']}"
                    )

                    accel[["mx", "my", "mz"]] = np.nan

            else:

                accel[["mx", "my", "mz"]] = np.nan

            # -----------------------------------------------------------------
            # Dataset metadata
            # -----------------------------------------------------------------

            accel["bcg_x"] = np.nan
            accel["bcg_y"] = np.nan
            accel["bcg_z"] = np.nan

            accel["sensor_id"] = "IMU"
            accel["label"] = trial["trial"]
            accel["subject_id"] = trial["subject_id"]
            accel["dataset_source"] = "COUGH"
            accel["record_id"] = trial["record_id"]

            records.append(accel)

        if not records:
            raise RuntimeError(
                "Raw cough recordings were loaded, but no valid "
                "recordings could be standardized."
            )

        standardized = pd.concat(
            records,
            ignore_index=True,
        )

        standardized = ensure_standard_columns(
            standardized
        )

        validate_standardized_data(standardized)

        self.metadata = {
            "dataset_name": self.dataset_name,
            "rows": len(standardized),
            "subjects": standardized["subject_id"].nunique(),
            "trials": len(self.trial_records),
            "labels": standardized["label"].nunique(),
            "columns": list(standardized.columns),
            "accelerometer_available": standardized[
                ["ax", "ay", "az"]
            ].notna().any().any(),
            "gyroscope_available": standardized[
                ["gx", "gy", "gz"]
            ].notna().any().any(),
            "magnetometer_available": standardized[
                ["mx", "my", "mz"]
            ].notna().any().any(),
            "audio_available": any(
                len(trial.get("audio_files", [])) > 0
                for trial in self.trial_records
            ),
        }

        return StandardizedData(
            data=standardized,
            dataset_name=self.dataset_name,
            sampling_rate=self.sampling_rate,
            metadata=self.metadata,
        )


# =============================================================================
# 7. FOUR-IMU FETAL DATASET LOADER
# =============================================================================

class FourIMULoader(BaseDatasetLoader):
    """Loader for the Fetal Movement Dataset Recorded Using Four IMUs."""

    def __init__(
        self,
        root_path: str | Path,
    ) -> None:

        super().__init__(
            root_path=root_path,
            dataset_name=(
                "Fetal Movement Dataset Recorded Using Four IMUs"
            ),
            sampling_rate=30.0,
        )

        self.dataset_root = self.root_path
        self.record_files: list[dict] = []

    def discover(self) -> list[dict]:
        """Discover Four-IMU recording CSVs."""

        records = []

        for subdir in sorted(
            self.dataset_root.glob("Sub-dataset *")
        ):

            if not subdir.is_dir():
                continue

            for csv_file in sorted(
                subdir.glob("record_*.csv")
            ):

                if csv_file.is_file():

                    records.append(
                        {
                            "sub_dataset": subdir.name,
                            "record_id": csv_file.stem,
                            "path": csv_file,
                        }
                    )

        self.record_files = records

        print(
            f"Discovered Four-IMU recordings: "
            f"{len(records)}"
        )

        return records

    @staticmethod
    def _required_columns() -> list[str]:
        """Return required Four-IMU raw columns."""

        columns = ["time", "label"]

        for imu in range(1, 5):

            columns.extend(
                [
                    f"ax{imu}",
                    f"ay{imu}",
                    f"az{imu}",
                    f"gx{imu}",
                    f"gy{imu}",
                    f"gz{imu}",
                ]
            )

        return columns

    def _validate_record(
        self,
        df: pd.DataFrame,
        path: Path,
    ) -> None:
        """Strictly validate a Four-IMU raw recording."""

        required = self._required_columns()

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{path.name}: missing required columns: {missing}"
            )

        if df[required].isna().any().any():
            raise ValueError(
                f"{path.name}: required columns contain missing values"
            )

        if not pd.api.types.is_numeric_dtype(df["time"]):
            raise ValueError(
                f"{path.name}: time column is not numeric"
            )

        for column in required:

            if column == "label":
                continue

            if not pd.api.types.is_numeric_dtype(
                df[column]
            ):
                raise ValueError(
                    f"{path.name}: column '{column}' "
                    "is not numeric"
                )

    def load(self) -> list[dict]:
        """
        Load raw Four-IMU recordings.

        No conversion to the unified schema is performed here.
        """

        records = self.record_files

        if not records:
            records = self.discover()

        if not records:
            raise RuntimeError(
                "No Four-IMU recording CSV files were discovered."
            )

        raw_records = []
        skipped = 0

        for record in records:

            path = record["path"]

            try:

                df = pd.read_csv(path)

                df.columns = [
                    str(column).strip().lower()
                    for column in df.columns
                ]

                self._validate_record(
                    df,
                    path,
                )

                raw_records.append(
                    {
                        "metadata": record,
                        "data": df,
                    }
                )

            except Exception as exc:

                skipped += 1

                print(
                    f"Skipping {path.name}: {exc}"
                )

        if not raw_records:
            raise RuntimeError(
                "Four-IMU files were discovered, but none "
                "passed raw data validation."
            )

        self.raw_data = raw_records

        self.metadata = {
            "dataset_name": self.dataset_name,
            "raw_recordings_discovered": len(records),
            "raw_recordings_loaded": len(raw_records),
            "raw_recordings_skipped": skipped,
            "sampling_rate_hz": 30.0,
        }

        return self.raw_data

    def standardize(self) -> StandardizedData:
        """
        Convert all Four-IMU recordings from wide raw format
        into long sensor-oriented format.
        """

        if self.raw_data is None:
            self.load()

        standardized_records = []

        for raw_record in self.raw_data:

            record = raw_record["metadata"]
            df = raw_record["data"]

            time = df["time"].to_numpy()

            chunks = []

            for imu in range(1, 5):

                chunks.append(
                    pd.DataFrame(
                        {
                            "timestamp": time,
                            "sensor_id": f"IMU{imu}",

                            "ax": df[
                                f"ax{imu}"
                            ].to_numpy(),

                            "ay": df[
                                f"ay{imu}"
                            ].to_numpy(),

                            "az": df[
                                f"az{imu}"
                            ].to_numpy(),

                            "gx": df[
                                f"gx{imu}"
                            ].to_numpy(),

                            "gy": df[
                                f"gy{imu}"
                            ].to_numpy(),

                            "gz": df[
                                f"gz{imu}"
                            ].to_numpy(),

                            "mx": np.nan,
                            "my": np.nan,
                            "mz": np.nan,

                            "bcg_x": np.nan,
                            "bcg_y": np.nan,
                            "bcg_z": np.nan,

                            "label": df[
                                "label"
                            ].to_numpy(),

                            "subject_id": "single_subject",

                            "dataset_source": "FOUR_IMU",

                            "record_id": record[
                                "record_id"
                            ],
                        }
                    )
                )

            standardized_records.append(
                pd.concat(
                    chunks,
                    ignore_index=True,
                )
            )

        standardized = pd.concat(
            standardized_records,
            ignore_index=True,
        )

        standardized = ensure_standard_columns(
            standardized
        )

        validate_standardized_data(
            standardized
        )

        sensor_columns = [
            "ax", "ay", "az",
            "gx", "gy", "gz",
        ]

        if standardized[
            sensor_columns
        ].isna().any().any():

            raise RuntimeError(
                "Four-IMU accelerometer/gyroscope data "
                "contains unexpected missing values."
            )

        self.metadata.update(
            {
                "rows": len(standardized),
                "recordings_loaded": standardized[
                    "record_id"
                ].nunique(),

                "subjects": standardized[
                    "subject_id"
                ].nunique(),

                "sub_datasets": len(
                    set(
                        record["metadata"][
                            "sub_dataset"
                        ]
                        for record in self.raw_data
                    )
                ),

                "labels": standardized[
                    "label"
                ].nunique(),

                "imu_count": standardized[
                    "sensor_id"
                ].nunique(),

                "accelerometer_available": True,
                "gyroscope_available": True,
                "magnetometer_available": False,
                "bcg_available": False,

                "columns": list(
                    standardized.columns
                ),
            }
        )

        return StandardizedData(
            data=standardized,
            dataset_name=self.dataset_name,
            sampling_rate=self.sampling_rate,
            metadata=self.metadata,
        )


# =============================================================================
# 8. OXFORD FEMALE FETAL DATASET LOADER
# =============================================================================

class OxfordLoader(BaseDatasetLoader):
    """Loader for the Oxford female fetal dataset."""

    def __init__(
        self,
        root_path: str | Path,
    ) -> None:

        super().__init__(
            root_path=root_path,
            dataset_name="Oxford Female Fetal Dataset",
        )

        self.signal_variable = "BCG_PREPROC_3AXIS"
        self.label_variable = "BP_MOUV_FILES"

        self.records: list[dict] = []

    def discover(self) -> list[dict]:
        """Discover and pair Oxford signal and BP MATLAB files."""

        signal_files = sorted(
            self.root_path.rglob("*_signal.mat")
        )

        bp_files = sorted(
            self.root_path.rglob("*_bp.mat")
        )

        signal_map = {
            file.name.replace(
                "_signal.mat",
                "",
            ): file
            for file in signal_files
        }

        bp_map = {
            file.name.replace(
                "_bp.mat",
                "",
            ): file
            for file in bp_files
        }

        record_ids = sorted(
            set(signal_map) | set(bp_map)
        )

        self.records = [
            {
                "record_id": record_id,
                "signal_file": signal_map.get(
                    record_id
                ),
                "bp_file": bp_map.get(
                    record_id
                ),
            }
            for record_id in record_ids
        ]

        print(
            f"Discovered Oxford records: "
            f"{len(self.records)}"
        )

        return self.records

    def load(self) -> list[dict]:
        """
        Load raw Oxford MATLAB arrays.

        Alignment and conversion into the common schema happen
        in `standardize()`.
        """

        records = self.records

        if not records:
            records = self.discover()

        if not records:
            raise RuntimeError(
                "No Oxford signal/BP records were discovered.\n"
                f"Dataset root checked:\n{self.root_path}"
            )

        raw_records = []
        skipped = 0

        for record in records:

            record_id = record["record_id"]

            signal_file = record["signal_file"]
            bp_file = record["bp_file"]

            if signal_file is None:
                print(
                    f"Skipping {record_id}: "
                    "signal file missing"
                )
                skipped += 1
                continue

            if bp_file is None:
                print(
                    f"Skipping {record_id}: "
                    "BP file missing"
                )
                skipped += 1
                continue

            try:

                signal_mat = loadmat(
                    signal_file
                )

                if self.signal_variable not in signal_mat:
                    print(
                        f"Skipping {record_id}: signal variable "
                        f"'{self.signal_variable}' not found"
                    )
                    skipped += 1
                    continue

                signal = np.asarray(
                    signal_mat[
                        self.signal_variable
                    ]
                )

                bp_mat = loadmat(
                    bp_file
                )

                if self.label_variable not in bp_mat:
                    print(
                        f"Skipping {record_id}: label variable "
                        f"'{self.label_variable}' not found"
                    )
                    skipped += 1
                    continue

                labels = np.asarray(
                    bp_mat[
                        self.label_variable
                    ]
                ).reshape(-1)

                raw_records.append(
                    {
                        "record_id": record_id,
                        "signal": signal,
                        "labels": labels,
                    }
                )

            except Exception as exc:

                print(
                    f"Skipping Oxford record "
                    f"{record_id}: {exc}"
                )

                skipped += 1

        if not raw_records:
            raise RuntimeError(
                "Oxford records were discovered, but no "
                "valid raw records could be loaded."
            )

        self.raw_data = raw_records

        self.metadata = {
            "dataset_name": self.dataset_name,
            "records_discovered": len(records),
            "raw_records_loaded": len(raw_records),
            "raw_records_skipped": skipped,
            "signal_variable": self.signal_variable,
            "label_variable": self.label_variable,
            "signal_type": "BCG",
            "signal_axes": 3,
            "sampling_rate_hz": None,
        }

        return self.raw_data

    def standardize(self) -> StandardizedData:
        """
        Align Oxford BCG signals and movement labels and convert
        them into the common schema.
        """

        if self.raw_data is None:
            self.load()

        loaded_records = []

        records_loaded = 0
        records_skipped = 0

        total_signal_samples = 0
        total_label_samples = 0
        total_aligned_samples = 0

        truncated_records = []

        for record in self.raw_data:

            record_id = record["record_id"]
            signal = np.asarray(
                record["signal"]
            )
            labels = np.asarray(
                record["labels"]
            ).reshape(-1)

            if signal.ndim != 2:

                print(
                    f"Skipping {record_id}: "
                    f"unexpected signal dimensions "
                    f"{signal.shape}"
                )

                records_skipped += 1
                continue

            if signal.shape[1] != 3:

                print(
                    f"Skipping {record_id}: expected "
                    f"3 BCG axes, got shape "
                    f"{signal.shape}"
                )

                records_skipped += 1
                continue

            signal_length = len(signal)
            label_length = len(labels)

            total_signal_samples += signal_length
            total_label_samples += label_length

            aligned_length = min(
                signal_length,
                label_length,
            )

            if aligned_length == 0:

                print(
                    f"Skipping {record_id}: "
                    "empty signal or label array"
                )

                records_skipped += 1
                continue

            if signal_length != label_length:

                print(
                    f"Warning: {record_id} length mismatch "
                    f"(signal={signal_length}, "
                    f"labels={label_length}). "
                    f"Truncating both to "
                    f"{aligned_length} samples."
                )

                truncated_records.append(
                    {
                        "record_id": record_id,
                        "signal_length": signal_length,
                        "label_length": label_length,
                        "aligned_length": aligned_length,
                    }
                )

            signal = signal[
                :aligned_length
            ]

            labels = labels[
                :aligned_length
            ]

            total_aligned_samples += (
                aligned_length
            )

            loaded_records.append(
                pd.DataFrame(
                    {
                        "timestamp": np.arange(
                            aligned_length,
                            dtype=np.int64,
                        ),

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
                    }
                )
            )

            records_loaded += 1

        if not loaded_records:
            raise RuntimeError(
                "Oxford records were loaded, but no valid "
                "records could be standardized."
            )

        standardized = pd.concat(
            loaded_records,
            ignore_index=True,
        )

        standardized = ensure_standard_columns(
            standardized
        )

        validate_standardized_data(
            standardized
        )

        self.metadata.update(
            {
                "rows": len(standardized),
                "records_loaded": records_loaded,
                "records_skipped": records_skipped,
                "records_truncated": len(
                    truncated_records
                ),
                "subjects": standardized[
                    "subject_id"
                ].nunique(),
                "labels": standardized[
                    "label"
                ].nunique(),
                "timestamp_unit": "sample_index",
                "original_signal_samples":
                    total_signal_samples,
                "original_label_samples":
                    total_label_samples,
                "aligned_samples":
                    total_aligned_samples,
                "accelerometer_available": False,
                "gyroscope_available": False,
                "magnetometer_available": False,
                "bcg_available": True,
                "columns": list(
                    standardized.columns
                ),
                "truncated_records":
                    truncated_records,
            }
        )

        return StandardizedData(
            data=standardized,
            dataset_name=self.dataset_name,
            sampling_rate=self.sampling_rate,
            metadata=self.metadata,
        )