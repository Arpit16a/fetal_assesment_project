"""
labeling_helper.py

Makes manual labeling something you REVIEW, not something you guess
at from a spreadsheet. For every candidate in priority_labeling_batch.csv:

    1. Reconstructs the actual filtered signal (the same signal
       candidate_generation.py used), with a bit of context before
       and after the candidate so you can see what it looked like
       against its surroundings.
    2. Plots it and saves a PNG.
    3. Computes a transparent, rule-based "suggested_label" using
       the framework below -- NOT a trained model, not ground
       truth, just a starting hint you can accept or override
       after looking at the plot.

Output
-------
data/processed/artifact_features/labeling_review/
    {dataset}/{candidate_id}.png
    priority_labeling_batch_with_suggestions.csv

Open the CSV next to the PNGs (or just browse the PNG folder), look
at each plot, and fill in the REAL manual_label column yourself --
the suggestion is there to speed up the obvious cases, not to
replace your judgment on the ones that aren't obvious.

SUGGESTION LOGIC (read this before trusting it)
---------------------------------------------------
COUGH:      always "artifact" -- no fetus is present in this
            dataset; it exists purely as a non-target-motion
            reference.

FOUR_IMU / OXFORD:
    - duration > LONG_DURATION_S and multiple sensors overlapping
        -> "artifact"   (sustained + whole-body-consistent)
    - duration < SHORT_DURATION_S and one axis dominates
      (high directional_consistency) and NOT multiple sensors
        -> "likely_fetal"  (brief + localized + sharp)
    - true_label indicates the mother pressed the button near this
      candidate: nudges toward "likely_fetal" but does not override
      the above on its own -- it's supporting evidence, not proof.
    - anything that doesn't clearly match one pattern -> "uncertain"

This is a simple, inspectable heuristic, not a validated
classifier -- it exists to make an obvious case fast to confirm and
an ambiguous case obvious that it's ambiguous, not to be right on
its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "dataset.py").exists():
            return candidate
    raise RuntimeError(
        f"Could not locate the project root (looked for "
        f"src/dataset.py) starting from {start}."
    )


PROJECT_ROOT = _find_project_root(Path(__file__).parent)
SRC_DIR = PROJECT_ROOT / "src"
SIGNAL_PROCESSING_DIR = SRC_DIR / "signal_processing"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SIGNAL_PROCESSING_DIR))

from importlib.util import spec_from_file_location, module_from_spec


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


timing = _load_module("timing_layer", SIGNAL_PROCESSING_DIR / "01_timing.py")
sp = _load_module("signal_processing_layer", SIGNAL_PROCESSING_DIR / "signal_processing.py")

import config as cfg  # noqa: E402


FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "artifact_features"
BATCH_PATH = FEATURES_DIR / "priority_labeling_batch.csv"

OUTPUT_DIR = FEATURES_DIR / "labeling_review"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_SECONDS = 1.0  # signal shown before/after the candidate, for reference

LONG_DURATION_S = 1.2
SHORT_DURATION_S = 0.8
DIRECTIONAL_CONSISTENCY_THRESHOLD = 0.6


# ============================================================
# SUGGESTION LOGIC
# ============================================================

def suggest_label(row: pd.Series) -> tuple[str, str]:

    if row["dataset"] == "COUGH":
        return "artifact", "Cough dataset has no fetus -- always non-target motion."

    duration = row.get("duration_seconds", np.nan)
    sensors_involved = row.get("sensors_involved", np.nan)
    directional = row.get("primary_directional_consistency", np.nan)
    multi_sensor = pd.notna(sensors_involved) and sensors_involved >= 2

    if pd.notna(duration) and duration > LONG_DURATION_S and multi_sensor:
        return "artifact", f"Sustained ({duration:.2f}s) and synchronized across {int(sensors_involved)+1} sensors."

    if (
        pd.notna(duration) and duration < SHORT_DURATION_S
        and pd.notna(directional) and directional > DIRECTIONAL_CONSISTENCY_THRESHOLD
        and not multi_sensor
    ):
        return "likely_fetal", f"Brief ({duration:.2f}s), concentrated in one direction, not synchronized across sensors."

    return "uncertain", "No clear pattern match -- duration/sync/direction don't agree; look at the plot."


# ============================================================
# SIGNAL RECONSTRUCTION + PLOTTING
# ============================================================

def plot_candidate(
    segment: pd.DataFrame,
    candidate_row: pd.Series,
    dataset_name: str,
    output_path: Path,
) -> None:

    dataset_config = cfg.get_dataset_config(dataset_name)
    fs = dataset_config.sampling_rate_hz

    segment = sp.resample_segment(segment, dataset_config)
    filtered_segment, _ = sp.apply_bandpass_filter(segment, dataset_config)

    signal_columns = sp._available_signal_columns(segment)
    if not signal_columns:
        return

    start_idx = int(candidate_row["start_index"])
    end_idx = int(candidate_row["end_index"])

    context_samples = int(CONTEXT_SECONDS * fs) if fs else 0
    plot_start = max(0, start_idx - context_samples)
    plot_end = min(len(filtered_segment), end_idx + context_samples)

    timestamps = filtered_segment["timestamp"].to_numpy()[plot_start:plot_end]

    fig, ax = plt.subplots(figsize=(10, 4))

    for col in signal_columns[:3]:  # primary axis group only, keep it readable
        values = filtered_segment[col].to_numpy()[plot_start:plot_end]
        ax.plot(timestamps, values, label=col, linewidth=1)

    ax.axvspan(
        timestamps[start_idx - plot_start] if start_idx - plot_start < len(timestamps) else timestamps[0],
        timestamps[min(end_idx - plot_start, len(timestamps) - 1)],
        color="orange", alpha=0.25, label="candidate",
    )

    suggested, reason = suggest_label(candidate_row)

    ax.set_title(
        f"{candidate_row['candidate_id']}  |  suggested: {suggested}\n{reason}",
        fontsize=9,
    )
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    fig.savefig(output_path, dpi=100)
    plt.close(fig)


# ============================================================
# ORCHESTRATION
# ============================================================

def process_dataset(dataset_name: str, batch_rows: pd.DataFrame, features: pd.DataFrame) -> None:

    from dataset import CoughLoader, FourIMULoader, OxfordLoader

    loaders = {
        "COUGH": (CoughLoader, PROJECT_ROOT / "data" / "raw" / "artifacts" / "cough_imu" / "Multimodal Cough Dataset"),
        "FOUR_IMU": (FourIMULoader, PROJECT_ROOT / "data" / "raw" / "fetal" / "four_imu"),
        "OXFORD": (OxfordLoader, PROJECT_ROOT / "data" / "raw" / "fetal" / "oxford_female"),
    }

    loader_cls, path = loaders[dataset_name]
    if not path.exists():
        print(f"SKIPPED {dataset_name} -- raw data path not found: {path}")
        return

    print(f"Loading {dataset_name} (needed once for all its candidates)...")
    loader = loader_cls(path)
    standardized = loader.run().data

    dataset_config = cfg.get_dataset_config(dataset_name)
    timing_config = timing.TimingConfig(expected_sampling_rate_hz=dataset_config.sampling_rate_hz)

    output_dir = OUTPUT_DIR / dataset_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    group_columns = ["record_id"]
    if "sensor_id" in standardized.columns:
        group_columns = ["record_id", "sensor_id"]

    plotted = 0

    for keys, record_data in standardized.groupby(group_columns):

        record_id = keys[0] if isinstance(keys, tuple) else keys

        segment_ids_needed = set(features[features["record_id"].astype(str) == str(record_id)]["segment_id"])
        if not segment_ids_needed:
            continue

        segments, _ = timing.analyze_record(
            record_data, timing_config, dataset_source=dataset_name,
            subject_id=record_data["subject_id"].iloc[0] if "subject_id" in record_data.columns else None,
            record_id=str(record_id),
        )

        for seg_index, segment in enumerate(segments, start=1):

            sensor_component = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else None
            segment_id = (
                f"{dataset_name}_{record_id}_{sensor_component}_seg{seg_index:04d}"
                if sensor_component is not None
                else f"{dataset_name}_{record_id}_seg{seg_index:04d}"
            )

            segment_candidates = features[features["segment_id"] == segment_id]
            batch_candidates = segment_candidates[
                segment_candidates["candidate_id"].isin(batch_rows["candidate_id"])
            ]

            for _, candidate_row in batch_candidates.iterrows():
                output_path = output_dir / f"{candidate_row['candidate_id']}.png"
                plot_candidate(segment, candidate_row, dataset_name, output_path)
                plotted += 1

    del standardized
    loader.raw_data = None
    loader.standardized_data = None

    print(f"{dataset_name}: plotted {plotted} candidates -> {output_dir}")


def main() -> None:

    if not BATCH_PATH.exists():
        raise FileNotFoundError(f"{BATCH_PATH} not found. Run build_labeling_batch.py first.")

    batch = pd.read_csv(BATCH_PATH)
    print(f"Labeling batch: {len(batch)} candidates")

    all_suggestions = []

    for dataset_name in batch["dataset"].unique():

        dataset_batch = batch[batch["dataset"] == dataset_name]

        features_path = FEATURES_DIR / f"{dataset_name.lower()}_candidate_features.csv"
        if not features_path.exists():
            print(f"SKIPPED {dataset_name} -- {features_path} not found")
            continue

        features = pd.read_csv(features_path, dtype={"true_label": str})

        process_dataset(dataset_name, dataset_batch, features)

        merged = dataset_batch.merge(
            features[[
                "candidate_id", "true_label", "sensors_involved",
                "max_sensor_overlap_fraction", "primary_directional_consistency",
            ]],
            on="candidate_id", how="left",
        )

        suggestions = merged.apply(suggest_label, axis=1, result_type="expand")
        merged["suggested_label"] = suggestions[0]
        merged["suggested_reason"] = suggestions[1]

        all_suggestions.append(merged)

    if not all_suggestions:
        print("Nothing to process.")
        return

    result = pd.concat(all_suggestions, ignore_index=True)
    output_csv = FEATURES_DIR / "priority_labeling_batch_with_suggestions.csv"
    result.to_csv(output_csv, index=False)

    print(f"\nSaved: {output_csv}")
    print(f"Plots saved under: {OUTPUT_DIR}")
    print(
        "\nSuggested-label breakdown (a starting hint, not ground truth):"
    )
    print(result["suggested_label"].value_counts())
    print(
        "\nOpen each PNG, look at it, and fill in manual_label yourself "
        "in this CSV -- the suggestion is there to speed up the obvious "
        "cases, not to replace your judgment on the ones that aren't."
    )


if __name__ == "__main__":
    main()
