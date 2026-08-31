"""
06_candidate_generation.py

PHASE 6 — ARTIFACT / NON-TARGET MOTION: CANDIDATE GENERATION
================================================================

This replaces 02_artifact_inspection.py's candidate logic, which
flagged an entire continuous SEGMENT (sometimes minutes long) as one
"candidate" based on whole-segment aggregate z-scores. That doesn't
answer the actual question this layer exists to answer — WHEN inside
a segment did something happen — and it doesn't consume the
per-window signal quality flags 02_signal_processing.py now produces.

This module implements the deterministic algorithm from the project
plan, and nothing else:

    filtered signal
        -> activity score (vector magnitude)
        -> adaptive threshold (median + k*MAD, PER RECORD)
        -> contiguous active regions
        -> merge close regions
        -> minimum duration filter
        -> candidate event

CRITICAL DISTINCTION (read before changing thresholds)
----------------------------------------------------------
A candidate event means "elevated activity here — worth
investigating." It does NOT mean "artifact," and it does NOT mean
"not fetal movement." Signal quality, non-target motion, and target
movement are three different questions:

    - Signal quality:   already answered by 02_signal_processing.py
                        (GOOD / QUESTIONABLE / INVALID per window)
    - This module:      "is there elevated activity here?" (candidate)
    - Artifact baseline (Phase 8): "is this candidate likely
                        non-fetal motion?" — requires labels this
                        module does not produce.

INTEGRATION WITH THE SIGNAL PROCESSING LAYER
-----------------------------------------------
- Filtering is NOT reimplemented here. This module imports and
  reuses signal_processing.py's resample_segment() and
  apply_bandpass_filter() directly, so there is exactly one
  definition of "the filtered signal" in the whole project.
- INVALID-quality windows (from *_windows.csv, produced by
  02_signal_processing.py) are excluded from BOTH the adaptive
  threshold's baseline estimate and from candidate generation
  itself. A sensor fault must not (a) get reported as a candidate
  event, or (b) skew the "normal activity" baseline that everything
  else gets compared against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


# =============================================================================
# ACTIVITY SCORE
# =============================================================================

# Which axis group defines "activity" depends on what the dataset
# actually measures — never invented, matches signal_type in
# config.py. Datasets with only one populated group have no choice
# to make here; this exists so a future dataset with both isn't
# silently guessed at.
_PRIMARY_AXIS_GROUPS = {
    "IMU": ["ax", "ay", "az"],
    "BCG": ["bcg_x", "bcg_y", "bcg_z"],
}


def compute_activity_score(
    filtered_arrays: dict[str, np.ndarray],
    dataset_config: "cfg.DatasetConfig",
) -> np.ndarray:
    """
    Vector magnitude of the primary signal group, computed on the
    ALREADY BAND-PASS-FILTERED signal (so gravity/DC offset and
    high-frequency noise are already suppressed — magnitude of the
    filtered signal directly reflects dynamic movement energy,
    without needing a separate gravity-removal step here).
    """

    axes = _PRIMARY_AXIS_GROUPS.get(dataset_config.signal_type)

    if axes is None or not all(a in filtered_arrays for a in axes):
        raise ValueError(
            f"No primary axis group available for signal_type="
            f"{dataset_config.signal_type!r}. Available columns: "
            f"{list(filtered_arrays.keys())}"
        )

    squared_sum = sum(filtered_arrays[a] ** 2 for a in axes)
    return np.sqrt(squared_sum)


def compute_adaptive_threshold(
    activity_score: np.ndarray,
    valid_mask: np.ndarray,
    candidate_config: "cfg.CandidateGenerationConfig",
) -> tuple[float, bool]:
    """
    median + k * MAD, computed ONLY over samples not excluded by
    valid_mask (i.e. not inside an INVALID-quality window). MAD is
    scaled by 1.4826 to be comparable to a standard deviation under
    a roughly-symmetric distribution, while staying robust to the
    exact outliers (real movements) we're trying to detect — using
    std() directly would let those same outliers inflate the
    threshold that's supposed to catch them.

    Returns (threshold, baseline_is_reliable). baseline_is_reliable
    is False when too few valid samples exist to trust the estimate
    (see min_valid_samples_for_baseline) — the caller should skip
    candidate generation for this record rather than threshold
    against a baseline built from almost nothing.
    """

    valid_scores = activity_score[valid_mask]

    if len(valid_scores) < candidate_config.min_valid_samples_for_baseline:
        return np.inf, False

    median = np.median(valid_scores)
    mad = np.median(np.abs(valid_scores - median))
    scaled_mad = 1.4826 * mad

    if scaled_mad == 0:
        # Degenerate case: near-constant signal. Fall back to a
        # tiny epsilon rather than a zero-width threshold that
        # would flag literally every nonzero sample as a candidate.
        scaled_mad = 1e-9

    threshold = median + candidate_config.threshold_mad_multiplier * scaled_mad

    return float(threshold), True


# =============================================================================
# CONTIGUOUS REGIONS
# =============================================================================

def find_contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Start/end index pairs (end exclusive) of every contiguous run of
    True in a boolean array. Vectorized — see the same pattern used
    for the quality-check flatline detector in signal_processing.py.
    """

    if not mask.any():
        return []

    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)

    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    return list(zip(starts.tolist(), ends.tolist()))


def merge_close_regions(
    regions: list[tuple[int, int]],
    gap_samples: int,
) -> list[tuple[int, int]]:
    """Merge regions separated by a gap smaller than gap_samples."""

    if not regions:
        return []

    merged = [regions[0]]

    for start, end in regions[1:]:
        last_start, last_end = merged[-1]

        if start - last_end <= gap_samples:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def filter_min_duration(
    regions: list[tuple[int, int]],
    min_samples: int,
) -> list[tuple[int, int]]:
    return [(s, e) for s, e in regions if (e - s) >= min_samples]


# =============================================================================
# QUALITY MASK — EXCLUDE INVALID WINDOWS FROM CANDIDATE GENERATION
# =============================================================================

def build_valid_mask(
    n_samples: int,
    timestamps: np.ndarray,
    windows_df: pd.DataFrame | None,
    segment_id: str,
) -> np.ndarray:
    """
    True for samples NOT covered by an INVALID-quality window for
    this segment. If no window-quality data is available for this
    segment (e.g. 02_signal_processing.py hasn't been run for this
    dataset yet), everything is treated as valid — candidate
    generation still works, it just isn't quality-screened yet,
    which is different from silently pretending it was.
    """

    valid = np.ones(n_samples, dtype=bool)

    if windows_df is None or windows_df.empty:
        return valid

    segment_windows = windows_df[windows_df["segment_id"] == segment_id]
    invalid_windows = segment_windows[segment_windows["quality"] == "INVALID"]

    for _, window in invalid_windows.iterrows():
        in_window = (
            (timestamps >= window["start_timestamp"])
            & (timestamps <= window["end_timestamp"])
        )
        valid[in_window] = False

    return valid


# =============================================================================
# ORCHESTRATION — ONE SEGMENT
# =============================================================================

def generate_candidates_for_segment(
    segment: pd.DataFrame,
    dataset_name: str,
    segment_id: str,
    windows_df: pd.DataFrame | None,
) -> list[dict]:
    """
    Run the full Phase 6 algorithm on one continuous, timing-valid
    segment and return one row per resulting candidate event.
    """

    dataset_config = cfg.get_dataset_config(dataset_name)
    candidate_config = cfg.CANDIDATE_GENERATION_CONFIG

    segment = sp.resample_segment(segment, dataset_config)

    if segment.empty or "timestamp" not in segment.columns:
        return []

    filtered_segment, filter_skip_reason = sp.apply_bandpass_filter(
        segment, dataset_config
    )

    signal_columns = sp._available_signal_columns(segment)
    filtered_arrays = {
        col: filtered_segment[col].to_numpy(dtype=float) for col in signal_columns
    }

    try:
        activity_score = compute_activity_score(filtered_arrays, dataset_config)
    except ValueError:
        return []

    timestamps = segment["timestamp"].to_numpy(dtype=float)

    valid_mask = build_valid_mask(len(segment), timestamps, windows_df, segment_id)
    # A sample with any NaN in the primary axis group can't produce
    # a meaningful activity score either — exclude it the same way.
    valid_mask &= ~np.isnan(activity_score)

    threshold, baseline_reliable = compute_adaptive_threshold(
        activity_score, valid_mask, candidate_config
    )

    if not baseline_reliable:
        return []

    fs = dataset_config.sampling_rate_hz
    if fs is None:
        return []

    active_mask = valid_mask & (activity_score > threshold)

    regions = find_contiguous_regions(active_mask)

    merge_gap_samples = max(int(round(candidate_config.merge_gap_seconds * fs)), 1)
    regions = merge_close_regions(regions, merge_gap_samples)

    min_duration_samples = max(
        int(round(candidate_config.min_event_duration_seconds * fs)), 1
    )
    regions = filter_min_duration(regions, min_duration_samples)

    subject_id = segment["subject_id"].iloc[0] if "subject_id" in segment.columns else None
    record_id = segment["record_id"].iloc[0] if "record_id" in segment.columns else None
    sensor_id = segment["sensor_id"].iloc[0] if "sensor_id" in segment.columns else None

    candidates = []

    for index, (start, end) in enumerate(regions, start=1):

        candidates.append(
            {
                "dataset": dataset_name,
                "segment_id": segment_id,
                "candidate_id": f"{segment_id}_c{index:04d}",
                "start_timestamp": float(timestamps[start]),
                "end_timestamp": float(timestamps[end - 1]),
                "duration_seconds": float(timestamps[end - 1] - timestamps[start]),
                "sample_count": end - start,
                "start_index": start,
                "end_index": end,
                "peak_activity_score": float(activity_score[start:end].max()),
                "threshold_used": threshold,
                "filtered": filter_skip_reason is None,
                "filter_skip_reason": filter_skip_reason,
                "subject_id": subject_id,
                "record_id": record_id,
                "sensor_id": sensor_id,
            }
        )

    return candidates


# =============================================================================
# ORCHESTRATION — ONE DATASET
# =============================================================================

def process_dataset(
    standardized_data: pd.DataFrame,
    dataset_name: str,
    timing_config: "timing.TimingConfig",
    windows_df: pd.DataFrame | None,
) -> pd.DataFrame:

    all_candidates = []

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

            candidates = generate_candidates_for_segment(
                segment, dataset_name, segment_id, windows_df
            )
            all_candidates.extend(candidates)

    return pd.DataFrame(all_candidates)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    from dataset import CoughLoader, FourIMULoader, OxfordLoader

    project_root = PROJECT_ROOT
    sp_output_dir = project_root / "data" / "processed" / "signal_processing"
    output_dir = project_root / "data" / "processed" / "artifact_candidates"
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

    summary_rows = []

    for dataset_name, (loader_cls, path) in loaders.items():

        print("=" * 78)
        print(dataset_name)
        print("=" * 78)

        if not path.exists():
            print(f"SKIPPED — path does not exist: {path}")
            summary_rows.append({"dataset": dataset_name, "status": "SKIPPED"})
            continue

        try:
            loader = loader_cls(path)
            standardized = loader.run().data
        except Exception as exc:
            print(f"LOAD FAILED: {exc}")
            summary_rows.append({"dataset": dataset_name, "status": f"LOAD_FAILED: {exc}"})
            continue

        windows_path = sp_output_dir / f"{dataset_name.lower()}_windows.csv"
        if windows_path.exists():
            windows_df = pd.read_csv(windows_path)
            print(f"Loaded quality windows: {len(windows_df)} rows from {windows_path.name}")
        else:
            windows_df = None
            print(
                f"No quality windows found at {windows_path} — run "
                "signal_processing.py first for quality-screened candidates. "
                "Continuing WITHOUT quality screening for now."
            )

        dataset_config = cfg.get_dataset_config(dataset_name)
        timing_config = timing.TimingConfig(
            expected_sampling_rate_hz=dataset_config.sampling_rate_hz,
        )

        candidates_df = process_dataset(standardized, dataset_name, timing_config, windows_df)

        # Explicit release — same lesson learned in loader_master_test.py:
        # don't let the largest dataset's raw+standardized data sit in
        # memory while later datasets are processed.
        del standardized
        loader.raw_data = None
        loader.standardized_data = None

        output_path = output_dir / f"{dataset_name.lower()}_candidate_events.csv"
        candidates_df.to_csv(output_path, index=False)

        print(f"Candidates generated: {len(candidates_df)}")
        print(f"Saved: {output_path}")

        summary_rows.append(
            {
                "dataset": dataset_name,
                "status": "OK",
                "candidates": len(candidates_df),
                "median_duration_s": (
                    float(candidates_df["duration_seconds"].median())
                    if len(candidates_df)
                    else None
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "candidate_generation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("CANDIDATE GENERATION SUMMARY")
    print("=" * 78)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
