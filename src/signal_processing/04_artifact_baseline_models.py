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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

FEATURE_COLUMNS = [
    "duration_seconds", "sample_count",
    "acc_mean", "acc_std", "acc_rms", "acc_peak_to_peak",
    "jerk_rms", "jerk_peak_to_peak",
    "acc_dominant_frequency_hz", "acc_spectral_centroid_hz",
    "acc_spectral_bandwidth_hz", "acc_spectral_energy",
    "gyro_mean", "gyro_std", "gyro_rms", "gyro_peak_to_peak",
    "gyro_dominant_frequency_hz", "gyro_spectral_centroid_hz",
    "gyro_spectral_bandwidth_hz", "gyro_spectral_energy",
    "bcg_mean", "bcg_std", "bcg_rms", "bcg_peak_to_peak",
    "bcg_dominant_frequency_hz", "bcg_spectral_centroid_hz",
    "bcg_spectral_bandwidth_hz", "bcg_spectral_energy",
]


# ============================================================
# DATA PREP
# ============================================================

def load_training_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run 03_label_validation.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    labeled = df[df["is_artifact"].notna()].copy()

    if labeled["subject_id"].isna().any():
        raise ValueError(
            "Some labeled rows have missing subject_id — cannot "
            "guarantee subject-independent splitting."
        )

    available_features = [c for c in FEATURE_COLUMNS if c in labeled.columns]
    missing = set(FEATURE_COLUMNS) - set(available_features)
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