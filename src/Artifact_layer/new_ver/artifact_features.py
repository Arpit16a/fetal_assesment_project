"""
07_artifact_features.py

PHASE 7 — ARTIFACT FEATURES
=============================

Converts each Phase 6 candidate event (a start/end index range) into
a feature vector — the actual input a baseline model can consume.
Nothing in this module decides artifact vs. non-artifact; it only
describes what a candidate event looks like, numerically.

Feature groups, per the project plan:
    - Time-domain:   duration, RMS, peak amplitude, peak-to-peak,
                      variance, std, zero-crossing rate
    - Energy:        signal energy, mean energy per sample
    - Frequency:     dominant frequency, spectral centroid,
                      band power (in the dataset's own configured
                      band), spectral entropy
    - Multi-axis:    per-axis energy contribution, directional
                      consistency (how dominated by one axis the
                      event is, vs. spread evenly across axes)
    - Multi-sensor:  (Four-IMU only) how many OTHER sensors on the
                      same recording show a temporally-overlapping
                      candidate at the same time, and by how much —
                      real motion (maternal repositioning, belt
                      shift) tends to show up on multiple abdominal
                      sensors at once; a single-sensor spike is more
                      consistent with local sensor noise.

INTEGRATION
-------------
Reuses signal_processing.py's resample_segment()/apply_bandpass_filter()
and 01_timing.py's segmentation — the exact same filtered signal
06_candidate_generation.py used to find these candidates in the
first place, sliced by the same start_index/end_index it already
computed. No re-detection, no second definition of "filtered."
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from importlib.util import spec_from_file_location, module_from_spec


def _find_project_root(start: Path) -> Path:
    """
    Walk up from `start` until finding the directory that contains
    src/dataset.py, instead of hardcoding a fixed number of .parent
    hops. A fixed hop-count broke the moment this script moved from
    src/signal_processing/ to src/Artifact_layer/new_ver/ -- it
    silently pointed at the wrong directory. dataset.py is the most
    stable anchor in this project (the frozen loader contract), so
    this stays correct even if these artifact-layer scripts move
    again.
    """
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "dataset.py").exists():
            return candidate
    raise RuntimeError(
        f"Could not locate the project root (looked for "
        f"src/dataset.py) starting from {start}. If the project "
        f"layout changed, dataset.py should still exist somewhere "
        f"above this script -- check it wasn't moved or renamed."
    )


PROJECT_ROOT = _find_project_root(Path(__file__).parent)
SRC_DIR = PROJECT_ROOT / "src"
SIGNAL_PROCESSING_DIR = SRC_DIR / "signal_processing"

# So `from dataset import ...` and `import config` work regardless
# of how this script is invoked (python -m, direct run, an IDE's
# run button) -- not dependent on the current working directory
# matching a specific -m invocation, unlike relying on cwd alone.
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SIGNAL_PROCESSING_DIR))


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


timing = _load_module("timing_layer", SIGNAL_PROCESSING_DIR / "01_timing.py")
sp = _load_module("signal_processing_layer", SIGNAL_PROCESSING_DIR / "signal_processing.py")

import config as cfg  # noqa: E402


_PRIMARY_AXIS_GROUPS = {
    "IMU": ["ax", "ay", "az"],
    "BCG": ["bcg_x", "bcg_y", "bcg_z"],
}

_SECONDARY_AXIS_GROUP = ["gx", "gy", "gz"]


# =============================================================================
# TIME-DOMAIN + ENERGY FEATURES  (single axis)
# =============================================================================

def _time_domain_features(values: np.ndarray, prefix: str) -> dict:

    if len(values) == 0 or np.all(np.isnan(values)):
        return {
            f"{prefix}_rms": np.nan,
            f"{prefix}_peak": np.nan,
            f"{prefix}_peak_to_peak": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_variance": np.nan,
            f"{prefix}_zero_crossing_rate": np.nan,
            f"{prefix}_energy": np.nan,
            f"{prefix}_mean_energy": np.nan,
        }

    valid = values[~np.isnan(values)]

    rms = float(np.sqrt(np.mean(valid ** 2)))
    peak = float(np.max(np.abs(valid)))
    peak_to_peak = float(valid.max() - valid.min())
    std = float(np.std(valid))
    variance = float(np.var(valid))

    # Zero-crossing rate: fraction of consecutive-sample sign changes.
    # Computed on the mean-removed signal so a constant offset (e.g.
    # gravity leakage) doesn't suppress crossings that would
    # otherwise be there.
    centered = valid - np.mean(valid)
    signs = np.sign(centered)
    signs[signs == 0] = 1  # treat an exact-zero sample as a continuation
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    zero_crossing_rate = float(crossings / max(len(valid) - 1, 1))

    energy = float(np.sum(valid ** 2))
    mean_energy = float(energy / len(valid))

    return {
        f"{prefix}_rms": rms,
        f"{prefix}_peak": peak,
        f"{prefix}_peak_to_peak": peak_to_peak,
        f"{prefix}_std": std,
        f"{prefix}_variance": variance,
        f"{prefix}_zero_crossing_rate": zero_crossing_rate,
        f"{prefix}_energy": energy,
        f"{prefix}_mean_energy": mean_energy,
    }


# =============================================================================
# FREQUENCY-DOMAIN FEATURES  (single axis)
# =============================================================================

def _frequency_domain_features(
    values: np.ndarray,
    fs: float,
    band_hz: tuple[float, float] | None,
    prefix: str,
) -> dict:

    valid = values[~np.isnan(values)]

    # Welch needs a minimum number of samples to produce a
    # meaningful PSD estimate; short candidate events (near the
    # 0.3s minimum duration) may not have enough samples at low
    # sampling rates. Return NaN rather than a PSD estimate built
    # from too few points to mean anything.
    min_samples = 16

    if len(valid) < min_samples:
        return {
            f"{prefix}_dominant_frequency_hz": np.nan,
            f"{prefix}_spectral_centroid_hz": np.nan,
            f"{prefix}_band_power": np.nan,
            f"{prefix}_spectral_entropy": np.nan,
        }

    nperseg = min(len(valid), max(min_samples, len(valid) // 2))

    freqs, psd = welch(valid, fs=fs, nperseg=nperseg)

    if psd.sum() <= 0 or not np.isfinite(psd.sum()):
        return {
            f"{prefix}_dominant_frequency_hz": np.nan,
            f"{prefix}_spectral_centroid_hz": np.nan,
            f"{prefix}_band_power": np.nan,
            f"{prefix}_spectral_entropy": np.nan,
        }

    dominant_frequency = float(freqs[np.argmax(psd)])
    spectral_centroid = float(np.sum(freqs * psd) / np.sum(psd))

    if band_hz is not None:
        low, high = band_hz
        in_band = (freqs >= low) & (freqs <= high)
        band_power = float(psd[in_band].sum())
    else:
        band_power = float(psd.sum())

    # Spectral entropy: Shannon entropy of the normalized PSD,
    # normalized to [0, 1] by log2(N). Low entropy = energy
    # concentrated in a narrow band (tonal/rhythmic); high entropy =
    # energy spread broadly (noise-like).
    psd_norm = psd / psd.sum()
    psd_norm = psd_norm[psd_norm > 0]  # avoid log(0)
    entropy = float(-np.sum(psd_norm * np.log2(psd_norm)))
    max_entropy = float(np.log2(len(psd))) if len(psd) > 1 else 1.0
    spectral_entropy = entropy / max_entropy if max_entropy > 0 else np.nan

    return {
        f"{prefix}_dominant_frequency_hz": dominant_frequency,
        f"{prefix}_spectral_centroid_hz": spectral_centroid,
        f"{prefix}_band_power": band_power,
        f"{prefix}_spectral_entropy": spectral_entropy,
    }


# =============================================================================
# MULTI-AXIS FEATURES
# =============================================================================

def _multi_axis_features(
    axis_arrays: dict[str, np.ndarray],
    axis_names: list[str],
    prefix: str,
) -> dict:
    """
    Per-axis energy contribution and directional consistency —
    is this event dominated by one axis, or spread across all of
    them?
    """

    energies = {}
    for axis in axis_names:
        values = axis_arrays.get(axis)
        if values is None:
            energies[axis] = 0.0
            continue
        valid = values[~np.isnan(values)]
        energies[axis] = float(np.sum(valid ** 2)) if len(valid) else 0.0

    total_energy = sum(energies.values())

    contributions = {
        f"{prefix}_{axis}_contribution": (
            energies[axis] / total_energy if total_energy > 0 else np.nan
        )
        for axis in axis_names
    }

    # Directional consistency: max single-axis share of total energy.
    # 1.0 = entirely one axis (e.g. a clean push/kick in one
    # direction); close to 1/3 (for 3 axes) = evenly spread (more
    # consistent with an omnidirectional disturbance like shifting).
    directional_consistency = (
        max(energies.values()) / total_energy if total_energy > 0 else np.nan
    )

    return {
        **contributions,
        f"{prefix}_directional_consistency": directional_consistency,
    }


# =============================================================================
# MULTI-SENSOR FEATURES  (Four-IMU only)
# =============================================================================

def add_multi_sensor_features(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    For datasets with multiple sensors on one recording, compute,
    for every candidate: how many OTHER sensors show a temporally
    overlapping candidate on the same record, and what fraction of
    this candidate's duration that overlap covers.

    Real non-fetal motion (maternal repositioning, belt shift) tends
    to move every abdominal sensor at once; something showing up on
    only one of four sensors is more consistent with local sensor
    noise than whole-body motion. This does NOT decide which is
    which — it's a feature for the Phase 8 model to weigh, not a
    rule applied here.
    """

    if "sensor_id" not in candidates_df.columns or candidates_df.empty:
        candidates_df["sensors_involved"] = np.nan
        candidates_df["max_sensor_overlap_fraction"] = np.nan
        return candidates_df

    n_sensors_col = candidates_df.groupby("record_id")["sensor_id"].transform("nunique")
    is_multi_sensor_record = n_sensors_col > 1

    sensors_involved = np.zeros(len(candidates_df), dtype=int)
    max_overlap_fraction = np.zeros(len(candidates_df), dtype=float)

    for record_id, record_candidates in candidates_df[is_multi_sensor_record].groupby("record_id"):

        starts = record_candidates["start_timestamp"].to_numpy()
        ends = record_candidates["end_timestamp"].to_numpy()
        sensors = record_candidates["sensor_id"].to_numpy()
        idx = record_candidates.index.to_numpy()

        for i in range(len(record_candidates)):

            this_start, this_end = starts[i], ends[i]
            this_sensor = sensors[i]
            this_duration = max(this_end - this_start, 1e-9)

            involved = set()
            best_overlap = 0.0

            for j in range(len(record_candidates)):
                if sensors[j] == this_sensor:
                    continue

                overlap = min(this_end, ends[j]) - max(this_start, starts[j])
                if overlap > 0:
                    involved.add(sensors[j])
                    best_overlap = max(best_overlap, overlap / this_duration)

            sensors_involved[candidates_df.index.get_loc(idx[i])] = len(involved)
            max_overlap_fraction[candidates_df.index.get_loc(idx[i])] = best_overlap

    candidates_df = candidates_df.copy()
    candidates_df["sensors_involved"] = np.where(is_multi_sensor_record, sensors_involved, np.nan)
    candidates_df["max_sensor_overlap_fraction"] = np.where(
        is_multi_sensor_record, max_overlap_fraction, np.nan
    )

    return candidates_df


# =============================================================================
# ORCHESTRATION — ONE SEGMENT'S CANDIDATES
# =============================================================================

def extract_features_for_segment(
    segment: pd.DataFrame,
    segment_candidates: pd.DataFrame,
    dataset_name: str,
) -> list[dict]:
    """
    Reconstruct the same filtered signal 06_candidate_generation.py
    used, then extract features for every candidate belonging to
    this segment by slicing [start_index:end_index].
    """

    dataset_config = cfg.get_dataset_config(dataset_name)

    segment = sp.resample_segment(segment, dataset_config)
    filtered_segment, filter_skip_reason = sp.apply_bandpass_filter(segment, dataset_config)

    signal_columns = sp._available_signal_columns(segment)
    filtered_arrays = {
        col: filtered_segment[col].to_numpy(dtype=float) for col in signal_columns
    }

    fs = dataset_config.sampling_rate_hz
    primary_axes = [a for a in _PRIMARY_AXIS_GROUPS.get(dataset_config.signal_type, []) if a in filtered_arrays]
    secondary_axes = [a for a in _SECONDARY_AXIS_GROUP if a in filtered_arrays]

    true_label = None
    if "label" in segment.columns:
        mode = segment["label"].mode()
        true_label = str(mode.iloc[0]) if not mode.empty else None

    rows = []

    for _, candidate in segment_candidates.iterrows():

        start = int(candidate["start_index"])
        end = int(candidate["end_index"])

        row = candidate.to_dict()
        row["true_label"] = true_label

        # --- Primary axis group: full time+freq+energy features ---
        if primary_axes:
            magnitude = np.sqrt(
                sum(filtered_arrays[a][start:end] ** 2 for a in primary_axes)
            )
            row.update(_time_domain_features(magnitude, "primary"))
            if fs:
                row.update(
                    _frequency_domain_features(
                        magnitude, fs, dataset_config.bandpass_hz, "primary"
                    )
                )
            row.update(
                _multi_axis_features(
                    {a: filtered_arrays[a][start:end] for a in primary_axes},
                    primary_axes,
                    "primary",
                )
            )

        # --- Secondary axis group (gyro, if present): time-domain only ---
        if secondary_axes:
            gyro_magnitude = np.sqrt(
                sum(filtered_arrays[a][start:end] ** 2 for a in secondary_axes)
            )
            row.update(_time_domain_features(gyro_magnitude, "secondary"))

        row["filtered"] = filter_skip_reason is None
        row["filter_skip_reason"] = filter_skip_reason

        rows.append(row)

    return rows


# =============================================================================
# ORCHESTRATION — ONE DATASET
# =============================================================================

def process_dataset(
    standardized_data: pd.DataFrame,
    candidates_df: pd.DataFrame,
    dataset_name: str,
    timing_config: "timing.TimingConfig",
) -> pd.DataFrame:

    if candidates_df.empty:
        return candidates_df

    all_rows = []

    group_columns = ["record_id"]
    if "sensor_id" in standardized_data.columns:
        group_columns = ["record_id", "sensor_id"]

    for keys, record_data in standardized_data.groupby(group_columns):

        # BUG FIX: segment_id previously used record_id only, so
        # every sensor in a multi-sensor recording (e.g. Four-IMU's
        # 4 IMUs) got IDENTICAL segment_id strings. Downstream code
        # that looks candidates/windows up BY segment_id (Phase 7's
        # feature extraction, in particular) then matched candidates
        # from all four sensors at once and silently computed
        # features against the wrong sensor's signal whenever index
        # ranges happened to overlap. Caught via a synthetic test
        # with a deliberately un-synchronized 4th sensor — 8 real
        # candidates produced 32 feature rows before this fix.
        if isinstance(keys, tuple) and len(keys) > 1:
            record_id, sensor_component = keys[0], "_".join(str(k) for k in keys[1:])
        else:
            record_id, sensor_component = keys, None

        segments, _ = timing.analyze_record(
            record_data,
            timing_config,
            dataset_source=dataset_name,
            subject_id=(
                record_data["subject_id"].iloc[0]
                if "subject_id" in record_data.columns
                else None
            ),
            record_id=str(record_id),
        )

        for seg_index, segment in enumerate(segments, start=1):

            segment_id = (
                f"{dataset_name}_{record_id}_{sensor_component}_seg{seg_index:04d}"
                if sensor_component is not None
                else f"{dataset_name}_{record_id}_seg{seg_index:04d}"
            )

            segment_candidates = candidates_df[candidates_df["segment_id"] == segment_id]
            if segment_candidates.empty:
                continue

            rows = extract_features_for_segment(segment, segment_candidates, dataset_name)
            all_rows.extend(rows)

    features_df = pd.DataFrame(all_rows)
    features_df = add_multi_sensor_features(features_df)

    return features_df


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    from dataset import CoughLoader, FourIMULoader, OxfordLoader

    project_root = PROJECT_ROOT
    candidates_dir = project_root / "data" / "processed" / "artifact_candidates"
    output_dir = project_root / "data" / "processed" / "artifact_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        "COUGH": (
            CoughLoader,
            project_root / "data" / "raw" / "artifacts" / "cough_imu" / "Multimodal Cough Dataset",
        ),
        "FOUR_IMU": (
            FourIMULoader,
            project_root / "data" / "raw" / "fetal" / "four_imu",
        ),
        "OXFORD": (
            OxfordLoader,
            project_root / "data" / "raw" / "fetal" / "oxford_female",
        ),
    }

    all_features = []
    summary_rows = []

    for dataset_name, (loader_cls, path) in loaders.items():

        print("=" * 78)
        print(dataset_name)
        print("=" * 78)

        candidates_path = candidates_dir / f"{dataset_name.lower()}_candidate_events.csv"
        if not candidates_path.exists():
            print(f"SKIPPED — {candidates_path} not found. Run 06_candidate_generation.py first.")
            summary_rows.append({"dataset": dataset_name, "status": "NO_CANDIDATES"})
            continue

        candidates_df = pd.read_csv(candidates_path)
        if candidates_df.empty:
            print(f"SKIPPED — {candidates_path} has zero candidates.")
            summary_rows.append({"dataset": dataset_name, "status": "ZERO_CANDIDATES"})
            continue

        if not path.exists():
            print(f"SKIPPED — raw data path does not exist: {path}")
            summary_rows.append({"dataset": dataset_name, "status": "SKIPPED"})
            continue

        try:
            loader = loader_cls(path)
            standardized = loader.run().data
        except Exception as exc:
            print(f"LOAD FAILED: {exc}")
            summary_rows.append({"dataset": dataset_name, "status": f"LOAD_FAILED: {exc}"})
            continue

        dataset_config = cfg.get_dataset_config(dataset_name)
        timing_config = timing.TimingConfig(expected_sampling_rate_hz=dataset_config.sampling_rate_hz)

        features_df = process_dataset(standardized, candidates_df, dataset_name, timing_config)

        del standardized
        loader.raw_data = None
        loader.standardized_data = None

        features_df["dataset"] = dataset_name
        all_features.append(features_df)

        output_path = output_dir / f"{dataset_name.lower()}_candidate_features.csv"
        features_df.to_csv(output_path, index=False)

        print(f"Feature rows: {len(features_df)}")
        print(f"Saved: {output_path}")

        summary_rows.append(
            {"dataset": dataset_name, "status": "OK", "candidates_with_features": len(features_df)}
        )

    # Combined manual-labeling template across all datasets, keyed on
    # candidate_id (the true labeling unit now, not the whole segment).
    if all_features:

        combined = pd.concat(all_features, ignore_index=True)

        template = combined[
            ["dataset", "record_id", "subject_id", "segment_id", "candidate_id",
             "start_timestamp", "end_timestamp", "duration_seconds",
             "peak_activity_score", "sensor_id"]
        ].copy()

        template["manual_label"] = ""
        template["confidence"] = ""
        template["reviewer_notes"] = ""
        template["allowed_label_values"] = "artifact / likely_fetal / uncertain"

        template_path = output_dir / "manual_label_template.csv"
        template.to_csv(template_path, index=False)
        print(f"\nCombined manual-labeling template saved: {template_path} ({len(template)} candidates)")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "feature_extraction_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 78)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
