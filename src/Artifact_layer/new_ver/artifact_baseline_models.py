"""
04_artifact_baseline_models.py

ARTIFACT BASELINE MODELS + SUBJECT-INDEPENDENT VALIDATION
============================================================

Purpose
-------
Step 5:  Train and compare simple baseline artifact classifiers
         (Random Forest, XGBoost, SVM) on the tabular segment-level
         features produced by the timing + inspection layers.

Step 6:  Every comparison below is subject-independent by
         construction (GroupKFold on subject_id, plus a final
         held-out group split). There is no step in this script
         where a subject appears in both train and test.

This script answers ONE question:

    "Can we establish a reproducible baseline for identifying
     artifact / non-target motion before we trust downstream
     movement events?"

It does NOT:
    - perform final artifact removal
    - get integrated into the pipeline (that's 05_..._integration.py)
    - claim these numbers are final; they are a baseline to beat

Input
-----
data/processed/artifact_labels/validated_dataset.csv   (from Step 4)

Outputs
-------
data/processed/artifact_baseline/
    model_comparison.csv
    best_model.joblib
    best_model_features.json
    feature_importance_xgboost.png
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

def _find_project_root(start: Path) -> Path:
    """
    Walk up from `start` until finding the directory that contains
    src/dataset.py, instead of hardcoding a fixed number of .parent
    hops -- a fixed hop-count broke once already when this script
    moved from src/signal_processing/ to src/Artifact_layer/new_ver/.
    dataset.py is this project's most stable anchor point.
    """
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "dataset.py").exists():
            return candidate
    raise RuntimeError(
        f"Could not locate the project root (looked for "
        f"src/dataset.py) starting from {start}."
    )


PROJECT_ROOT = _find_project_root(Path(__file__).parent)

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "artifact_labels" / "validated_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "artifact_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

# Matches 07_artifact_features.py's actual output columns — the old
# list here (acc_mean/gyro_rms/bcg_spectral_energy/etc.) was written
# for 02_artifact_inspection.py's whole-SEGMENT features and no
# longer exists now that Phase 6/7 produce per-CANDIDATE-EVENT
# features instead. "primary" = the dataset's main signal group
# (accelerometer for IMU datasets, BCG for Oxford); "secondary" =
# gyroscope, where available. sensors_involved /
# max_sensor_overlap_fraction are NaN for single-sensor datasets by
# design (see add_multi_sensor_features in 07_artifact_features.py)
# and are left in the list — the imputer below handles that the same
# way it handles Oxford's missing gyro features.
FEATURE_COLUMNS = [
    "duration_seconds", "sample_count", "peak_activity_score",

    "primary_rms", "primary_peak", "primary_peak_to_peak",
    "primary_std", "primary_variance", "primary_zero_crossing_rate",
    "primary_energy", "primary_mean_energy",
    "primary_dominant_frequency_hz", "primary_spectral_centroid_hz",
    "primary_band_power", "primary_spectral_entropy",
    "primary_directional_consistency",

    "secondary_rms", "secondary_peak", "secondary_peak_to_peak",
    "secondary_std", "secondary_variance", "secondary_zero_crossing_rate",
    "secondary_energy", "secondary_mean_energy",

    "sensors_involved", "max_sensor_overlap_fraction",
]


# ============================================================
# DATA PREP
# ============================================================

# Features that labeling_helper.py's suggest_label() heuristic uses
# directly to decide artifact vs. likely_fetal. If a row's label
# came from that heuristic (label_source == "heuristic_suggestion")
# rather than a human, these features are not independent evidence
# for the model to learn from -- they're the exact rule that
# produced the label. Training on them in that case doesn't measure
# "can this model detect artifacts," it measures "can this model
# reconstruct the heuristic" -- a materially different, much weaker
# claim that will look deceptively perfect (100% accuracy, zero
# variance) without ever being told apart from real signal.
LABEL_GENERATING_FEATURES = {
    "duration_seconds", "sensors_involved", "primary_directional_consistency",
}

FEATURE_COLUMNS = [
    "duration_seconds", "sample_count", "peak_activity_score",

    "primary_rms", "primary_peak", "primary_peak_to_peak",
    "primary_std", "primary_variance", "primary_zero_crossing_rate",
    "primary_energy", "primary_mean_energy",
    "primary_dominant_frequency_hz", "primary_spectral_centroid_hz",
    "primary_band_power", "primary_spectral_entropy",
    "primary_directional_consistency",

    "secondary_rms", "secondary_peak", "secondary_peak_to_peak",
    "secondary_std", "secondary_variance", "secondary_zero_crossing_rate",
    "secondary_energy", "secondary_mean_energy",

    "sensors_involved", "max_sensor_overlap_fraction",
]


# ============================================================
# DATA PREP
# ============================================================

def load_training_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run label_validation.py first."
        )

    # true_label mixes numeric labels (Four-IMU/Oxford) and strings
    # (Cough's talking-condition folder names) across rows, which
    # pandas can't infer a single dtype for — this is expected
    # given the project's own data, not a data-quality problem, so
    # silence the warning rather than let it look like an error.
    df = pd.read_csv(INPUT_PATH, dtype={"true_label": str})

    labeled = df[df["is_artifact"].notna()].copy()

    # FAIL CLEARLY, NOT WITH A GroupKFold(n_splits=0) STACK TRACE.
    # Zero labeled rows here almost always means
    # manual_label_template.csv hasn't been hand-labeled yet, not a
    # code bug — say that plainly rather than let sklearn's error
    # message be the only clue.
    min_required = max(N_SPLITS, 2)

    if len(labeled) == 0:
        raise RuntimeError(
            "No labeled candidates found (0 rows with is_artifact "
            "set). This means manual_label_template.csv (produced by "
            "artifact_features.py) hasn't been hand-labeled yet -- "
            "every row's manual_label is still blank. Label a sample "
            "of candidates first (see build_labeling_batch.py for a "
            "manageable, stratified sample instead of all of them), "
            "then re-run label_validation.py before this script."
        )

    if labeled["subject_id"].nunique() < min_required:
        raise RuntimeError(
            f"Only {labeled['subject_id'].nunique()} unique labeled "
            f"subject(s) found, but subject-independent "
            f"{min_required}-fold validation needs at least "
            f"{min_required}. Label candidates from more distinct "
            f"subjects, not just more candidates from the same few."
        )

    if labeled["subject_id"].isna().any():
        raise ValueError(
            "Some labeled rows have missing subject_id — cannot "
            "guarantee subject-independent splitting."
        )

    # CIRCULARITY GUARD — this is the fix for the "100% accuracy"
    # result: detect when labels came from the heuristic and
    # automatically drop the features that heuristic used to decide
    # them, rather than trust everyone running this script to
    # remember that caveat by hand.
    if "label_source" in labeled.columns:
        heuristic_fraction = (labeled["label_source"] == "heuristic_suggestion").mean()
    else:
        heuristic_fraction = 0.0

    feature_columns = FEATURE_COLUMNS

    if heuristic_fraction > 0:
        excluded = [c for c in LABEL_GENERATING_FEATURES if c in FEATURE_COLUMNS]
        feature_columns = [c for c in FEATURE_COLUMNS if c not in LABEL_GENERATING_FEATURES]

        print(
            f"\nCIRCULARITY GUARD: {heuristic_fraction:.0%} of labeled rows "
            f"came from the suggest_label() heuristic, not human review. "
            f"Excluding {excluded} from training/evaluation — those are "
            f"the exact features that heuristic used to assign the "
            f"label, so leaving them in would let the model trivially "
            f"reconstruct the rule instead of learning anything about "
            f"the signal. The resulting metrics below are the honest "
            f"(much harder, likely much lower) number: can OTHER signal "
            f"characteristics predict what the heuristic decided.\n"
            f"This is still not a validated artifact detector — it's a "
            f"check on whether the heuristic's decisions correlate with "
            f"independent signal properties at all. Real human-reviewed "
            f"labels are what turns this into an actual result."
        )

    available_features = [c for c in feature_columns if c in labeled.columns]
    missing = set(feature_columns) - set(available_features)
    if missing:
        print(f"NOTE: feature columns not present, skipping: {missing}")

    X = labeled[available_features]
    y = labeled["is_artifact"].astype(int)
    groups = labeled["subject_id"].astype(str)

    print(f"Training rows: {len(X)}  |  Unique subjects: {groups.nunique()}")
    print(f"Class balance:\n{y.value_counts()}")

    return X, y, groups


# ============================================================
# MODELS
# ============================================================

def build_models() -> dict:

    return {
        "RandomForest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "XGBoost": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "SVM": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
    }


# ============================================================
# SUBJECT-INDEPENDENT CROSS-VALIDATION  (STEP 6)
# ============================================================

def evaluate_subject_independent(model, X, y, groups, n_splits=N_SPLITS) -> dict:
    """
    GroupKFold guarantees no subject appears in both the train
    and validation fold in any split.
    """

    n_splits = min(n_splits, groups.nunique())
    splitter = GroupKFold(n_splits=n_splits)

    fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}

    for train_idx, val_idx in splitter.split(X, y, groups):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]

        fold_metrics["accuracy"].append(accuracy_score(y_val, pred))
        fold_metrics["precision"].append(precision_score(y_val, pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_val, pred, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_val, pred, zero_division=0))

        if len(np.unique(y_val)) > 1:
            fold_metrics["roc_auc"].append(roc_auc_score(y_val, proba))

    summary = {
        metric: (float(np.mean(values)), float(np.std(values)))
        for metric, values in fold_metrics.items()
        if values
    }

    return summary


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 78)
    print("STEP 5 + 6 — BASELINE ARTIFACT MODELS, SUBJECT-INDEPENDENT VALIDATION")
    print("=" * 78)

    X, y, groups = load_training_data()

    models = build_models()

    results_rows = []

    for name, pipeline in models.items():

        print(f"\n--- {name} : {N_SPLITS}-fold subject-independent CV ---")

        summary = evaluate_subject_independent(pipeline, X, y, groups)

        row = {"model": name}
        for metric, (mean, std) in summary.items():
            print(f"  {metric:10s}: {mean:.3f} +/- {std:.3f}")
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std

        results_rows.append(row)

    results_df = pd.DataFrame(results_rows).sort_values(
        "f1_mean", ascending=False
    )

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    print("\n" + "=" * 78)
    print("MODEL COMPARISON (sorted by mean F1)")
    print("=" * 78)
    print(results_df.to_string(index=False))

    # ------------------------------------------------------------
    # FINAL HELD-OUT FIT — best model, on a group-disjoint split
    # ------------------------------------------------------------

    best_model_name = results_df.iloc[0]["model"]
    print(f"\nBest model by mean F1: {best_model_name}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    best_pipeline = build_models()[best_model_name]
    best_pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])

    held_out_pred = best_pipeline.predict(X.iloc[test_idx])
    held_out_proba = best_pipeline.predict_proba(X.iloc[test_idx])[:, 1]
    y_test = y.iloc[test_idx]

    print("\nHeld-out (disjoint-subject) evaluation of the best model:")
    print(f"  accuracy : {accuracy_score(y_test, held_out_pred):.3f}")
    print(f"  precision: {precision_score(y_test, held_out_pred, zero_division=0):.3f}")
    print(f"  recall   : {recall_score(y_test, held_out_pred, zero_division=0):.3f}")
    print(f"  f1       : {f1_score(y_test, held_out_pred, zero_division=0):.3f}")
    if len(np.unique(y_test)) > 1:
        print(f"  roc_auc  : {roc_auc_score(y_test, held_out_proba):.3f}")

    # ------------------------------------------------------------
    # SAVE BEST MODEL (refit on ALL labeled data for deployment)
    # ------------------------------------------------------------

    final_pipeline = build_models()[best_model_name]
    final_pipeline.fit(X, y)

    model_path = OUTPUT_DIR / "best_model.joblib"
    joblib.dump(final_pipeline, model_path)

    features_path = OUTPUT_DIR / "best_model_features.json"
    features_path.write_text(json.dumps(list(X.columns), indent=2))

    print(f"\nSaved best model ({best_model_name}): {model_path}")
    print(f"Saved feature list: {features_path}")

    # ------------------------------------------------------------
    # FEATURE IMPORTANCE (XGBoost only, if it's in the comparison)
    # ------------------------------------------------------------

    if "XGBoost" in models:

        xgb_pipeline = build_models()["XGBoost"]
        xgb_pipeline.fit(X, y)

        importances = xgb_pipeline.named_steps["model"].feature_importances_
        order = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 8))
        plt.barh(
            [X.columns[i] for i in order][:15][::-1],
            importances[order][:15][::-1],
        )
        plt.title("XGBoost Feature Importance (top 15)")
        plt.tight_layout()

        plot_path = OUTPUT_DIR / "feature_importance_xgboost.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"Saved feature importance plot: {plot_path}")

    print("\n" + "=" * 78)
    print("STEP 5 + 6 COMPLETE")
    print("=" * 78)
    print(
        "\nThis is a BASELINE, not a final detector. Next: "
        "05_artifact_detector_integration.py (Step 7)."
    )


if __name__ == "__main__":
    main()