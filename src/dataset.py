"""
Unified Dataset Standardization Layer

Purpose
-------
Provide a common interface for loading heterogeneous datasets into
a standardized representation suitable for the downstream pipeline.

Current datasets
----------------
1. Oxford fetal movement dataset
2. Four-IMU fetal movement dataset
3. HAR artifact dataset
4. Multimodal cough + IMU dataset

The loaders should eventually expose the same conceptual fields:

    timestamp
    sensor_id
    ax, ay, az
    gx, gy, gz
    label
    subject_id
    dataset_source

Not every dataset contains every field. Missing modalities are represented
with None / NaN rather than inventing data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. STANDARD DATA SCHEMA
# ============================================================

STANDARD_COLUMNS = [
    "timestamp",
    "sensor_id",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "label",
    "subject_id",
    "dataset_source",
]


@dataclass
class StandardizedData:
    """
    Container for standardized sensor data.

    The actual signal data is stored as a pandas DataFrame using
    STANDARD_COLUMNS.
    """

    data: pd.DataFrame

    dataset_name: str
    sampling_rate: Optional[float] = None

    def validate(self) -> None:
        """
        Validate that the standardized dataframe follows the
        common schema.
        """

        missing_columns = [
            column
            for column in STANDARD_COLUMNS
            if column not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing standardized columns: {missing_columns}"
            )

    def summary(self) -> dict:
        """
        Return basic information about the standardized dataset.
        """

        self.validate()

        return {
            "dataset_name": self.dataset_name,
            "rows": len(self.data),
            "columns": list(self.data.columns),
            "subjects": self.data["subject_id"].nunique(),
            "labels": self.data["label"].nunique(),
            "sampling_rate": self.sampling_rate,
        }


# ============================================================
# 2. BASE DATASET LOADER
# ============================================================

class BaseDatasetLoader(ABC):
    """
    Abstract base class for every dataset loader.

    Every dataset-specific loader must implement:

        discover()
        load()
        standardize()
    """

    def __init__(
        self,
        root_path: str | Path,
        dataset_name: str,
    ):
        self.root_path = Path(root_path)
        self.dataset_name = dataset_name

        if not self.root_path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist:\n{self.root_path}"
            )

    @abstractmethod
    def discover(self):
        """
        Discover files belonging to the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self):
        """
        Load raw dataset files.
        """
        raise NotImplementedError

    @abstractmethod
    def standardize(self) -> StandardizedData:
        """
        Convert raw data into the common schema.
        """
        raise NotImplementedError

    def check_path(self) -> None:
        """
        Basic dataset path check.
        """

        print(f"Dataset       : {self.dataset_name}")
        print(f"Dataset path  : {self.root_path}")
        print(f"Path exists   : {self.root_path.exists()}")


# ============================================================
# 3. STANDARD COLUMN CREATION HELPER
# ============================================================

def create_empty_standard_dataframe(
    length: int = 0,
) -> pd.DataFrame:
    """
    Create an empty dataframe following the common schema.

    This is useful when a dataset does not contain all modalities.
    """

    df = pd.DataFrame(index=np.arange(length))

    for column in STANDARD_COLUMNS:
        df[column] = np.nan

    return df


# ============================================================
# 4. COLUMN NORMALIZATION HELPER
# ============================================================

def ensure_standard_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure that all standard columns exist.

    Existing columns are preserved.
    Missing columns are added as NaN.

    The original dataframe is not modified.
    """

    standardized = df.copy()

    for column in STANDARD_COLUMNS:
        if column not in standardized.columns:
            standardized[column] = np.nan

    return standardized[STANDARD_COLUMNS]


# ============================================================
# 5. BASIC STANDARDIZATION VALIDATION
# ============================================================

def validate_standardized_data(
    df: pd.DataFrame,
) -> None:
    """
    Validate a dataframe against the unified schema.
    """

    missing = [
        column
        for column in STANDARD_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Standardization failed. Missing columns: {missing}"
        )

    print("Standardized schema validation: PASSED")

#================================================================================================================================================================================================================================================
#================================================================================================================================================================================================================================================



# ============================================================
# 6. HAR DATASET LOADER
# ============================================================

class HARLoader(BaseDatasetLoader):
    """
    Loader for the IMU-based Human Activity Recognition dataset.

    Expected dataset structure:

        har/
        └── IMU-based Human Activity Recignition Dataset.csv

    The loader converts the original HAR columns into the
    project's unified sensor-data schema.
    """

    def __init__(
        self,
        root_path: str | Path,
        sampling_rate: Optional[float] = None,
    ):
        super().__init__(
            root_path=root_path,
            dataset_name="HAR",
        )

        self.sampling_rate = sampling_rate
        self.csv_file: Optional[Path] = None
        self.raw_data: Optional[pd.DataFrame] = None

    # --------------------------------------------------------
    # File Discovery
    # --------------------------------------------------------

    def discover(self):
        """
        Find CSV files inside the HAR dataset directory.
        """

        csv_files = list(self.root_path.rglob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV file found inside:\n{self.root_path}"
            )

        if len(csv_files) > 1:
            print(
                f"Warning: found {len(csv_files)} CSV files. "
                f"Using the first one."
            )

        self.csv_file = csv_files[0]

        print("HAR Dataset Discovery")
        print("---------------------")
        print(f"CSV file: {self.csv_file.name}")

        return self.csv_file

    # --------------------------------------------------------
    # Raw Data Loading
    # --------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Load the HAR CSV file into a pandas DataFrame.
        """

        if self.csv_file is None:
            self.discover()

        self.raw_data = pd.read_csv(self.csv_file)

        print("\nHAR Dataset Loaded")
        print("-----------------")
        print(f"Rows    : {len(self.raw_data):,}")
        print(f"Columns : {len(self.raw_data.columns)}")

        return self.raw_data

    # --------------------------------------------------------
    # Column Detection
    # --------------------------------------------------------

    @staticmethod
    def _find_column(
        columns,
        candidates,
        required=True,
    ):
        """
        Find a column using case-insensitive matching.

        Parameters
        ----------
        columns:
            Available dataframe columns.

        candidates:
            Possible names for the desired field.

        required:
            If True, raise an error when no match is found.
        """

        normalized = {
            str(column).strip().lower(): column
            for column in columns
        }

        # Exact matching first
        for candidate in candidates:
            key = candidate.lower()

            if key in normalized:
                return normalized[key]

        # Partial matching second
        for column in columns:

            column_lower = str(column).strip().lower()

            for candidate in candidates:

                if candidate.lower() in column_lower:
                    return column

        if required:
            raise ValueError(
                f"Could not find required column.\n"
                f"Possible names: {candidates}\n"
                f"Available columns:\n{list(columns)}"
            )

        return None

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    def standardize(self) -> StandardizedData:
        """
        Convert the HAR dataset into the unified schema.
        """

        if self.raw_data is None:
            self.load()

        df = self.raw_data.copy()

        # ----------------------------------------------------
        # Identify acceleration columns
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Identify gyroscope columns if available
        # ----------------------------------------------------

        gx_col = self._find_column(
            df.columns,
            [
                "gx",
                "gyro_x",
                "gyroscope_x",
                "gyr_x",
            ],
            required=False,
        )

        gy_col = self._find_column(
            df.columns,
            [
                "gy",
                "gyro_y",
                "gyroscope_y",
                "gyr_y",
            ],
            required=False,
        )

        gz_col = self._find_column(
            df.columns,
            [
                "gz",
                "gyro_z",
                "gyroscope_z",
                "gyr_z",
            ],
            required=False,
        )

        # ----------------------------------------------------
        # Identify label/activity column
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Identify timestamp if available
        # ----------------------------------------------------

        timestamp_col = self._find_column(
            df.columns,
            [
                "timestamp",
                "time",
                "time_stamp",
            ],
            required=False,
        )

        # ----------------------------------------------------
        # Create standardized dataframe
        # ----------------------------------------------------

        standardized = create_empty_standard_dataframe(
            length=len(df)
        )

        # Timestamp
        if timestamp_col is not None:
            standardized["timestamp"] = df[timestamp_col].values
        else:
            standardized["timestamp"] = np.arange(len(df))

        # Accelerometer
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

        # Gyroscope
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

        # Label
        standardized["label"] = df[label_col].astype(str).values

        # HAR is not necessarily guaranteed to contain a
        # subject column, so use a dataset-level identifier
        # unless one exists.
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

        if subject_col is not None:
            standardized["subject_id"] = (
                df[subject_col].astype(str).values
            )
        else:
            standardized["subject_id"] = "HAR"

        
        # The HAR dataset contains accelerometer and gyroscope
        # measurements from one IMU sensor.
        standardized["sensor_id"] = "IMU"

        # Dataset source
        standardized["dataset_source"] = "HAR"

        # ----------------------------------------------------
        # Final validation
        # ----------------------------------------------------

        standardized = ensure_standard_columns(
            standardized
        )

        validate_standardized_data(
            standardized
        )

        return StandardizedData(
            data=standardized,
            dataset_name="HAR",
            sampling_rate=self.sampling_rate,
        )

    # --------------------------------------------------------
    # Complete Pipeline
    # --------------------------------------------------------

    def run(self) -> StandardizedData:
        """
        Run the complete HAR loading pipeline.
        """

        self.discover()
        self.load()

        return self.standardize()    
    


#====================================================================================================================================================================================
#====================================================================================================================================================================================




# ============================================================
# CoughLoader
# ============================================================

class CoughLoader:
    """
    Loader for the Multimodal Cough Dataset.

    Expected structure:

    cough_imu/
        Multimodal Cough Dataset/
            005/
                Trial_1_No_Talking/
                    Accelerometer.csv
                    Gyroscope.csv
                    Magnetometer.csv
                Trial_2_Talking/
                    Accelerometer.csv
                    Gyroscope.csv
                    Magnetometer.csv
                Trial_3_Nonverbal/
                    Accelerometer.csv
                    Gyroscope.csv
                    Magnetometer.csv
            006/
            ...

    Output schema:

        timestamp
        sensor_id

        ax
        ay
        az

        gx
        gy
        gz

        mx
        my
        mz

        label
        subject_id
        dataset_source
    """

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(self, root_path):

        self.root_path = Path(root_path)

        # Handle both:
        #
        # cough_imu/
        #
        # and
        #
        # cough_imu/Multimodal Cough Dataset/

        dataset_folder = (
            self.root_path / "Multimodal Cough Dataset"
        )

        if dataset_folder.exists():
            self.dataset_root = dataset_folder
        else:
            self.dataset_root = self.root_path

        self.raw_data = None
        self.metadata = {}

        self.trial_records = []

    # ============================================================
    # File discovery helper
    # ============================================================

    @staticmethod
    def _find_file(directory, filename):

        filename = filename.lower()

        for file in directory.iterdir():

            if (
                file.is_file()
                and file.name.lower() == filename
            ):
                return file

        return None

    # ============================================================
    # Dataset discovery
    # ============================================================

    def discover_trials(self):
        """
        Discover all subject/trial directories containing
        accelerometer recordings.
        """

        trials = []

        if not self.dataset_root.exists():

            raise FileNotFoundError(
                f"Cough dataset path does not exist:\n"
                f"{self.dataset_root}"
            )

        # --------------------------------------------------------
        # Search recursively for Accelerometer.csv
        # --------------------------------------------------------

        for accel_file in self.dataset_root.rglob("*"):

            if not accel_file.is_file():
                continue

            if accel_file.name.lower() != "accelerometer.csv":
                continue

            trial_dir = accel_file.parent

            subject_dir = trial_dir.parent

            trials.append({

                "subject_id":
                    subject_dir.name,

                "trial":
                    trial_dir.name,

                "trial_path":
                    trial_dir,

                "accelerometer":
                    accel_file,

                "gyroscope":
                    self._find_file(
                        trial_dir,
                        "gyroscope.csv"
                    ),

                "magnetometer":
                    self._find_file(
                        trial_dir,
                        "magnetometer.csv"
                    ),

                "audio_files": [
                    p
                    for p in trial_dir.iterdir()
                    if (
                        p.is_file()
                        and p.suffix.lower() == ".wav"
                    )
                ]
            })

        self.trial_records = trials

        return trials

    # ============================================================
    # Load dataset
    # ============================================================

    def load(self):

        trials = self.discover_trials()

        print(
            f"Discovered trials: {len(trials)}"
        )

        if len(trials) == 0:

            raise RuntimeError(
                "No valid cough IMU recordings were discovered.\n"
                f"Dataset root checked:\n"
                f"{self.dataset_root}"
            )

        records = []

        # ========================================================
        # Process every trial
        # ========================================================

        for trial in trials:

            accel_file = trial["accelerometer"]

            gyro_file = trial["gyroscope"]

            mag_file = trial["magnetometer"]

            try:

                # =================================================
                # ACCELEROMETER
                # =================================================

                accel = pd.read_csv(
                    accel_file
                )

                accel.columns = [
                    str(col).strip().lower()
                    for col in accel.columns
                ]

                accel_required = [
                    "elapsed (s)",
                    "x-axis (g)",
                    "y-axis (g)",
                    "z-axis (g)"
                ]

                missing = [
                    col
                    for col in accel_required
                    if col not in accel.columns
                ]

                if missing:

                    print(
                        f"Skipping {accel_file.name}: "
                        f"missing accelerometer columns "
                        f"{missing}"
                    )

                    continue

                accel = accel[
                    [
                        "elapsed (s)",
                        "x-axis (g)",
                        "y-axis (g)",
                        "z-axis (g)"
                    ]
                ].copy()

                accel.rename(
                    columns={
                        "elapsed (s)": "timestamp",
                        "x-axis (g)": "ax",
                        "y-axis (g)": "ay",
                        "z-axis (g)": "az"
                    },
                    inplace=True
                )

                # Make sure timestamp is numeric

                accel["timestamp"] = pd.to_numeric(
                    accel["timestamp"],
                    errors="coerce"
                )

                # =================================================
                # GYROSCOPE
                # =================================================

                if gyro_file is not None:

                    gyro = pd.read_csv(
                        gyro_file
                    )

                    gyro.columns = [
                        str(col).strip().lower()
                        for col in gyro.columns
                    ]

                    gyro_required = [
                        "elapsed (s)",
                        "x-axis (deg/s)",
                        "y-axis (deg/s)",
                        "z-axis (deg/s)"
                    ]

                    if all(
                        col in gyro.columns
                        for col in gyro_required
                    ):

                        gyro = gyro[
                            [
                                "elapsed (s)",
                                "x-axis (deg/s)",
                                "y-axis (deg/s)",
                                "z-axis (deg/s)"
                            ]
                        ].copy()

                        gyro.rename(
                            columns={
                                "elapsed (s)": "timestamp",
                                "x-axis (deg/s)": "gx",
                                "y-axis (deg/s)": "gy",
                                "z-axis (deg/s)": "gz"
                            },
                            inplace=True
                        )

                        gyro["timestamp"] = pd.to_numeric(
                            gyro["timestamp"],
                            errors="coerce"
                        )

                        # -----------------------------------------
                        # Align gyro to accelerometer timestamps
                        # -----------------------------------------

                        accel = pd.merge_asof(

                            accel.sort_values(
                                "timestamp"
                            ),

                            gyro.sort_values(
                                "timestamp"
                            ),

                            on="timestamp",

                            direction="nearest"
                        )

                    else:

                        print(
                            f"Warning: Gyroscope format "
                            f"not recognized for "
                            f"{gyro_file.name}"
                        )

                        accel["gx"] = np.nan
                        accel["gy"] = np.nan
                        accel["gz"] = np.nan

                else:

                    accel["gx"] = np.nan
                    accel["gy"] = np.nan
                    accel["gz"] = np.nan

                # =================================================
                # MAGNETOMETER
                # =================================================

                if mag_file is not None:

                    mag = pd.read_csv(
                        mag_file
                    )

                    mag.columns = [
                        str(col).strip().lower()
                        for col in mag.columns
                    ]

                    # ---------------------------------------------
                    # IMPORTANT:
                    #
                    # Actual dataset uses Tesla:
                    #
                    # x-axis (T)
                    # y-axis (T)
                    # z-axis (T)
                    # ---------------------------------------------

                    mag_required = [
                        "elapsed (s)",
                        "x-axis (t)",
                        "y-axis (t)",
                        "z-axis (t)"
                    ]

                    if all(
                        col in mag.columns
                        for col in mag_required
                    ):

                        mag = mag[
                            [
                                "elapsed (s)",
                                "x-axis (t)",
                                "y-axis (t)",
                                "z-axis (t)"
                            ]
                        ].copy()

                        mag.rename(
                            columns={
                                "elapsed (s)": "timestamp",
                                "x-axis (t)": "mx",
                                "y-axis (t)": "my",
                                "z-axis (t)": "mz"
                            },
                            inplace=True
                        )

                        mag["timestamp"] = pd.to_numeric(
                            mag["timestamp"],
                            errors="coerce"
                        )

                        # -----------------------------------------
                        # Align magnetometer to accelerometer
                        # timestamps
                        # -----------------------------------------

                        accel = pd.merge_asof(

                            accel.sort_values(
                                "timestamp"
                            ),

                            mag.sort_values(
                                "timestamp"
                            ),

                            on="timestamp",

                            direction="nearest"
                        )

                    else:

                        print(
                            f"Warning: Magnetometer format "
                            f"not recognized for "
                            f"{mag_file.name}"
                        )

                        accel["mx"] = np.nan
                        accel["my"] = np.nan
                        accel["mz"] = np.nan

                else:

                    accel["mx"] = np.nan
                    accel["my"] = np.nan
                    accel["mz"] = np.nan

                # =================================================
                # METADATA
                # =================================================

                accel["sensor_id"] = "IMU"

                accel["label"] = trial["trial"]

                accel["subject_id"] = (
                    trial["subject_id"]
                )

                accel["dataset_source"] = "COUGH"

                records.append(accel)

            except Exception as e:

                print(
                    f"Skipping trial "
                    f"{trial['subject_id']} / "
                    f"{trial['trial']}: {e}"
                )

        # ========================================================
        # Check successful loading
        # ========================================================

        if not records:

            raise RuntimeError(
                "Trials were discovered, but no valid "
                "multimodal IMU recordings could be loaded."
            )

        # ========================================================
        # Combine trials
        # ========================================================

        self.raw_data = pd.concat(
            records,
            ignore_index=True
        )

        # ========================================================
        # Standardized schema
        # ========================================================

        required_schema = [

            "timestamp",

            "sensor_id",

            "ax",
            "ay",
            "az",

            "gx",
            "gy",
            "gz",

            "mx",
            "my",
            "mz",

            "label",

            "subject_id",

            "dataset_source"
        ]

        # Add missing columns if necessary

        for column in required_schema:

            if column not in self.raw_data.columns:

                self.raw_data[column] = np.nan

        # Keep only common schema

        self.raw_data = self.raw_data[
            required_schema
        ]

        # ========================================================
        # Metadata
        # ========================================================

        self.metadata = {

            "dataset_name":
                "Multimodal Cough Dataset",

            "rows":
                len(self.raw_data),

            "subjects":
                self.raw_data[
                    "subject_id"
                ].nunique(),

            "trials":
                len(trials),

            "labels":
                self.raw_data[
                    "label"
                ].nunique(),

            "columns":
                list(self.raw_data.columns),

            "accelerometer_available":
                self.raw_data[
                    ["ax", "ay", "az"]
                ].notna().any().any(),

            "gyroscope_available":
                self.raw_data[
                    ["gx", "gy", "gz"]
                ].notna().any().any(),

            "magnetometer_available":
                self.raw_data[
                    ["mx", "my", "mz"]
                ].notna().any().any(),

            "audio_available":
                any(
                    len(
                        trial.get(
                            "audio_files",
                            []
                        )
                    ) > 0
                    for trial in trials
                )
        }

        return self.raw_data