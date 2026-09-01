"""
04_artifact_baseline_models.py

ARTIFACT BASELINE MODELS + SUBJECT-INDEPENDENT VALIDATION
===========================================================

Scientific purpose
------------------
This script evaluates whether candidate-event features can distinguish
artifact/non-target motion from usable events.

IMPORTANT LABEL POLICY
----------------------
There are two fundamentally different situations:

1. HUMAN REVIEWED LABELS
   These are the only labels used for a real artifact-detector result.
   Rows with label_source == "manual" are eligible for training/evaluation.

2. HEURISTIC LABELS
   These are NOT ground truth. They are useful for auditing whether the
   heuristic is internally correlated with other signal features, but the
   resulting metrics must NOT be called artifact-detector performance.

The previous implementation trained on heuristic labels after removing
five directly used features. That is still scientifically unsafe because
the remaining waveform features are correlated transformations of the same
candidate event, and candidate generation itself selected events using
activity information. A model can therefore predict the heuristic without
learning the real artifact concept.

This version therefore:
    - uses MANUAL labels only for actual model training;
    - runs a clearly named HEURISTIC AUDIT when labels are heuristic-only;
    - uses a truly untouched subject-disjoint test set for final evaluation;
    - uses StratifiedGroupKFold when possible;
    - reports class balance by dataset and subject;
    - reports feature-label correlations for leakage diagnostics;
    - runs a small permutation/null audit for heuristic-only data;
    - never saves a heuristic-trained model as a deployable artifact detector.

Input
-----
data/processed/artifact_labels/validated_dataset.csv

Outputs
-------
data/processed/artifact_baseline/
    model_comparison.csv
    held_out_evaluation.csv
    dataset_label_balance.csv
    subject_label_balance.csv
    feature_diagnostics.csv
    best_model.joblib                 # only when manual labels are used
    best_model_features.json          # only when manual labels are used
    feature_importance_xgboost.png    # only when manual labels are used
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

def _find_project_root(start: Path) -> Path:
    """Find the repository root using src/dataset.py as the anchor."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "dataset.py").exists():
            return candidate
    raise RuntimeError(
        f"Could not locate project root starting from {start}. "
        "Expected src/dataset.py."
    )


PROJECT_ROOT = _find_project_root(Path(__file__).parent)

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artifact_labels"
    / "validated_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "artifact_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
REQUESTED_FOLDS = 5
TEST_SIZE = 0.20
PERMUTATION_RUNS = 20

# Candidate-event features produced by artifact_features.py.
FEATURE_COLUMNS = [
    "duration_seconds",
    "sample_count",
    "peak_activity_score",

    "primary_rms",
    "primary_peak",
    "primary_peak_to_peak",
    "primary_std",
    "primary_variance",
    "primary_zero_crossing_rate",
    "primary_energy",
    "primary_mean_energy",
    "primary_dominant_frequency_hz",
    "primary_spectral_centroid_hz",
    "primary_band_power",
    "primary_spectral_entropy",
    "primary_directional_consistency",

    "secondary_rms",
    "secondary_peak",
    "secondary_peak_to_peak",
    "secondary_std",
    "secondary_variance",
    "secondary_zero_crossing_rate",
    "secondary_energy",
    "secondary_mean_energy",

    "sensors_involved",
    "max_sensor_overlap_fraction",
]

# These were directly used by labeling_helper.py's heuristic.
DIRECT_LABEL_FEATURES = {
    "duration_seconds",
    "sample_count",
    "sensors_involved",
    "max_sensor_overlap_fraction",
    "primary_directional_consistency",
}

# peak_activity_score deserves special treatment:
# candidate_generation.py uses activity score to decide which regions
# become candidates. It is therefore not an independent observation of
# all signal events; it is part of the candidate-selection mechanism.
CANDIDATE_SELECTION_FEATURES = {
    "peak_activity_score",
}

# The remaining amplitude/energy features are legitimate signal features
# for a human-labeled model. They are NOT automatically "leakage".
# With heuristic labels, however, they can still be correlated with the
# rule because the rule operates on the same candidate waveform.
STRICT_HEURISTIC_AUDIT_FEATURES = [
    "primary_zero_crossing_rate",
    "primary_dominant_frequency_hz",
    "primary_spectral_centroid_hz",
    "primary_spectral_entropy",
    "secondary_zero_crossing_rate",
    "secondary_dominant_frequency_hz",  # optional / may be absent
    "secondary_spectral_centroid_hz",   # optional / may be absent
    "secondary_spectral_entropy",       # optional / may be absent
]

# For the actual human-label model, all measured event features are allowed
# except the direct label-generating variables. This is a modeling choice,
# not a claim that the features are independent of one another.
MANUAL_MODEL_EXCLUDED = DIRECT_LABEL_FEATURES


# ============================================================
# DATA LOADING + LABEL PROVENANCE
# ============================================================

def _normalise_label_source(df: pd.DataFrame) -> pd.Series:
    if "label_source" not in df.columns:
        return pd.Series("unknown", index=df.index)

    return (
        df["label_source"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def load_labeled_dataframe() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run label_validation.py first."
        )

    df = pd.read_csv(INPUT_PATH, dtype={"true_label": str})

    if "is_artifact" not in df.columns:
        raise ValueError(
            "validated_dataset.csv does not contain 'is_artifact'. "
            "Run label_validation.py again."
        )

    labeled = df[df["is_artifact"].notna()].copy()

    if labeled.empty:
        raise RuntimeError(
            "No labeled candidates found. "
            "Manual labels are required before a real artifact detector "
            "can be trained."
        )

    if "subject_id" not in labeled.columns:
        raise ValueError("Missing subject_id; subject-independent evaluation is impossible.")

    if labeled["subject_id"].isna().any():
        raise ValueError(
            "Some labeled rows have missing subject_id. "
            "Fix subject assignment before training."
        )

    labeled["label_source"] = _normalise_label_source(labeled)
    labeled["is_artifact"] = pd.to_numeric(
        labeled["is_artifact"], errors="coerce"
    )

    labeled = labeled[labeled["is_artifact"].isin([0, 1])].copy()
    labeled["is_artifact"] = labeled["is_artifact"].astype(int)
    labeled["subject_id"] = labeled["subject_id"].astype(str)

    if labeled.empty:
        raise RuntimeError("No valid binary artifact labels remain after cleaning.")

    return labeled


# ============================================================
# DATASET / SUBJECT DIAGNOSTICS
# ============================================================

def save_label_balance_diagnostics(df: pd.DataFrame) -> None:
    if "dataset" in df.columns:
        dataset_balance = (
            pd.crosstab(df["dataset"], df["is_artifact"])
            .rename(columns={0: "non_artifact", 1: "artifact"})
            .reset_index()
        )
        dataset_balance.to_csv(
            OUTPUT_DIR / "dataset_label_balance.csv", index=False
        )

        print("\nLABEL BALANCE BY DATASET")
        print(dataset_balance.to_string(index=False))

    subject_balance = (
        df.groupby("subject_id")["is_artifact"]
        .agg(
            n_candidates="size",
            n_artifact="sum",
            artifact_fraction="mean",
            n_classes="nunique",
        )
        .reset_index()
    )
    subject_balance.to_csv(
        OUTPUT_DIR / "subject_label_balance.csv", index=False
    )

    print("\nSUBJECT LABEL DIAGNOSTICS")
    print(
        f"Subjects: {len(subject_balance)} | "
        f"subjects containing both classes: "
        f"{int((subject_balance['n_classes'] == 2).sum())}"
    )


# ============================================================
# FEATURE DIAGNOSTICS
# ============================================================

def save_feature_diagnostics(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    rows = []

    for feature in feature_columns:
        if feature not in df.columns:
            continue

        numeric = pd.to_numeric(df[feature], errors="coerce")
        valid = numeric.notna()

        if valid.sum() < 5 or df.loc[valid, "is_artifact"].nunique() < 2:
            corr = np.nan
        else:
            corr = numeric[valid].corr(
                df.loc[valid, "is_artifact"].astype(float)
            )

        rows.append(
            {
                "feature": feature,
                "missing_fraction": float(1.0 - valid.mean()),
                "unique_values": int(numeric.nunique(dropna=True)),
                "label_point_biserial_like_corr": (
                    float(corr) if pd.notna(corr) else np.nan
                ),
            }
        )

    diagnostics = pd.DataFrame(rows)
    if not diagnostics.empty:
        diagnostics["abs_corr"] = diagnostics[
            "label_point_biserial_like_corr"
        ].abs()
        diagnostics = diagnostics.sort_values("abs_corr", ascending=False)

    diagnostics.to_csv(
        OUTPUT_DIR / "feature_diagnostics.csv", index=False
    )


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_manual_features(df: pd.DataFrame) -> list[str]:
    features = [
        c
        for c in FEATURE_COLUMNS
        if c in df.columns and c not in MANUAL_MODEL_EXCLUDED
    ]

    if not features:
        raise RuntimeError("No usable model features remain.")

    return features


def get_heuristic_audit_features(df: pd.DataFrame) -> list[str]:
    """
    Deliberately conservative feature set for the heuristic audit.

    It excludes:
      - direct label-generating features;
      - peak_activity_score, because it participates in candidate selection;
      - amplitude/energy features, because those can be tightly coupled to
        the activity-driven candidate-selection mechanism.

    This does NOT make heuristic labels into ground truth. It only asks
    whether the heuristic labels remain predictable from relatively
    independent spectral/shape descriptors.
    """
    features = [
        c
        for c in STRICT_HEURISTIC_AUDIT_FEATURES
        if c in df.columns
        and c not in DIRECT_LABEL_FEATURES
        and c not in CANDIDATE_SELECTION_FEATURES
    ]

    if not features:
        raise RuntimeError(
            "No strict heuristic-audit features are available. "
            "This is a diagnostic condition, not a model failure."
        )

    return features


# ============================================================
# MODELS
# ============================================================

def build_models() -> dict:
    return {
        "RandomForest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=3,
                        min_child_weight=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=2.0,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# ============================================================
# SPLITTING
# ============================================================

def choose_n_splits(y: pd.Series, groups: pd.Series) -> int:
    class_group_counts = (
        pd.DataFrame({"y": y.to_numpy(), "group": groups.to_numpy()})
        .groupby("y")["group"]
        .nunique()
    )

    if len(class_group_counts) < 2:
        raise RuntimeError(
            "Both artifact classes must be present in the labeled data."
        )

    max_possible = int(class_group_counts.min())
    n_splits = min(REQUESTED_FOLDS, max_possible)

    if n_splits < 2:
        raise RuntimeError(
            "Subject-independent CV requires at least two distinct subjects "
            "in EACH class. Current class/group structure is insufficient."
        )

    return n_splits


def make_stratified_group_splitter(n_splits: int):
    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def make_untouched_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create the final test set BEFORE model comparison.

    The test subjects are never used for selecting the best model.
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise RuntimeError(
            "The untouched group split produced a single-class train/test "
            "partition. Increase the labeled subject count or adjust "
            "the split strategy."
        )

    return train_idx, test_idx


# ============================================================
# METRICS
# ============================================================

def metric_dict(
    y_true: pd.Series,
    pred: np.ndarray,
    proba: np.ndarray,
) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, pred, zero_division=0)
        ),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    else:
        out["roc_auc"] = np.nan

    return out


# ============================================================
# CROSS-VALIDATION
# ============================================================

def evaluate_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
) -> tuple[dict, list[dict]]:
    splitter = make_stratified_group_splitter(n_splits)

    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]

        metrics = metric_dict(y_val, pred, proba)
        metrics["fold"] = fold
        metrics["n_train"] = len(train_idx)
        metrics["n_val"] = len(val_idx)
        metrics["train_subjects"] = groups.iloc[train_idx].nunique()
        metrics["val_subjects"] = groups.iloc[val_idx].nunique()

        fold_rows.append(metrics)

    fold_df = pd.DataFrame(fold_rows)

    summary = {}
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        values = pd.to_numeric(fold_df[metric], errors="coerce").dropna()
        if not values.empty:
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=0))

    return summary, fold_rows


# ============================================================
# HEURISTIC-ONLY AUDIT
# ============================================================

def run_heuristic_audit(df: pd.DataFrame) -> None:
    """
    Never presents heuristic-label performance as artifact-detector
    performance.

    The audit is intentionally conservative. If the score is still very
    high, that is evidence that the heuristic and the signal representation
    are strongly coupled — not evidence that an artifact detector works.
    """
    print("\n" + "=" * 78)
    print("HEURISTIC-LABEL AUDIT — NOT AN ARTIFACT DETECTOR")
    print("=" * 78)

    features = get_heuristic_audit_features(df)
    X = df[features].copy()
    y = df["is_artifact"].copy()
    groups = df["subject_id"].copy()

    n_splits = choose_n_splits(y, groups)

    print(f"Audit features: {features}")
    print(f"Rows: {len(df)} | Subjects: {groups.nunique()} | Folds: {n_splits}")

    model = build_models()["RandomForest"]

    summary, _ = evaluate_cv(
        model, X, y, groups, n_splits
    )

    print("\nObserved heuristic-audit performance:")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        key = f"{metric}_mean"
        if key in summary:
            print(
                f"  {metric:10s}: "
                f"{summary[key]:.3f} +/- "
                f"{summary[f'{metric}_std']:.3f}"
            )

    print(
        "\nInterpretation: these numbers measure how predictable the "
        "heuristic labels are from a restricted signal representation. "
        "They are NOT evidence of artifact-detection accuracy."
    )

    # Null/permutation audit: labels are shuffled while subject groups
    # remain fixed. A real predictive relationship should disappear.
    rng = np.random.default_rng(RANDOM_STATE)
    null_auc = []

    print(f"\nRunning {PERMUTATION_RUNS} label-permutation null tests...")

    for i in range(PERMUTATION_RUNS):
        y_perm = pd.Series(
            rng.permutation(y.to_numpy()),
            index=y.index,
        )

        # A permutation can occasionally produce a fold without both
        # classes; skip such a permutation rather than invent an AUC.
        try:
            perm_summary, _ = evaluate_cv(
                build_models()["RandomForest"],
                X,
                y_perm,
                groups,
                n_splits,
            )
        except Exception:
            continue

        if "roc_auc_mean" in perm_summary:
            null_auc.append(perm_summary["roc_auc_mean"])

    if null_auc:
        print(
            f"Permutation ROC-AUC: "
            f"{np.mean(null_auc):.3f} +/- {np.std(null_auc):.3f}"
        )
        print(
            f"Observed ROC-AUC minus null mean: "
            f"{summary.get('roc_auc_mean', np.nan) - np.mean(null_auc):.3f}"
        )

    print(
        "\nNEXT SCIENTIFIC ACTION: obtain human-reviewed artifact labels. "
        "Do not freeze or deploy a detector from this heuristic audit."
    )


# ============================================================
# HUMAN-LABEL BASELINE
# ============================================================

def run_human_label_baseline(df: pd.DataFrame) -> None:
    """
    Actual artifact-model evaluation using human-reviewed labels only.
    """
    manual = df[
        df["label_source"].isin(
            ["manual", "human", "human_review", "manual_review"]
        )
    ].copy()

    if manual.empty:
        raise RuntimeError(
            "No human-reviewed labels were found. "
            "A real artifact detector cannot be trained yet."
        )

    if manual["is_artifact"].nunique() < 2:
        raise RuntimeError(
            "Human-reviewed labels contain only one class. "
            "Both artifact and non-artifact labels are required."
        )

    if manual["subject_id"].nunique() < 5:
        raise RuntimeError(
            "At least five distinct subjects with human-reviewed labels "
            "are required for the planned subject-independent baseline."
        )

    features = get_manual_features(manual)

    X = manual[features].copy()
    y = manual["is_artifact"].copy()
    groups = manual["subject_id"].copy()

    n_splits = choose_n_splits(y, groups)

    print("\n" + "=" * 78)
    print("HUMAN-LABEL ARTIFACT BASELINE")
    print("=" * 78)
    print(
        f"Human-reviewed rows: {len(manual)} | "
        f"Subjects: {groups.nunique()} | "
        f"Features: {len(features)} | "
        f"CV folds: {n_splits}"
    )
    print(f"Class balance:\n{y.value_counts()}")

    # ------------------------------------------------------------
    # Create untouched test subjects BEFORE model comparison.
    # ------------------------------------------------------------
    train_idx, test_idx = make_untouched_test_split(X, y, groups)

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    g_train = groups.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print(f"\nUntouched test subjects: {groups.iloc[test_idx].nunique()}")
    print(f"Training subjects: {g_train.nunique()}")
    print(f"Test subjects: {groups.iloc[test_idx].nunique()}")

    # ------------------------------------------------------------
    # Model comparison occurs ONLY on the training subjects.
    # ------------------------------------------------------------
    results_rows = []

    for name, pipeline in build_models().items():
        print(
            f"\n--- {name}: "
            f"{n_splits}-fold StratifiedGroupKFold on training subjects ---"
        )

        summary, _ = evaluate_cv(
            pipeline,
            X_train,
            y_train,
            g_train,
            n_splits,
        )

        row = {"model": name}
        row.update(summary)

        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key in summary:
                print(
                    f"  {metric:10s}: "
                    f"{summary[mean_key]:.3f} +/- "
                    f"{summary[std_key]:.3f}"
                )

        results_rows.append(row)

    results_df = pd.DataFrame(results_rows).sort_values(
        "f1_mean",
        ascending=False,
    )

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 78)
    print("MODEL COMPARISON — TRAINING SUBJECTS ONLY")
    print("=" * 78)
    print(results_df.to_string(index=False))

    best_name = str(results_df.iloc[0]["model"])
    print(f"\nSelected model: {best_name}")

    # ------------------------------------------------------------
    # One final evaluation on subjects never used for model selection.
    # ------------------------------------------------------------
    best_model = build_models()[best_name]
    best_model.fit(X_train, y_train)

    test_pred = best_model.predict(X_test)
    test_proba = best_model.predict_proba(X_test)[:, 1]

    held_out = metric_dict(y_test, test_pred, test_proba)

    print("\nFINAL HELD-OUT SUBJECT-DISJOINT TEST")
    print("------------------------------------")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{metric:10s}: {held_out[metric]:.3f}")

    cm = confusion_matrix(y_test, test_pred)
    print("\nConfusion matrix [rows=true, columns=predicted]:")
    print(cm)

    held_out_row = pd.DataFrame(
        [
            {
                "model": best_name,
                "test_subjects": groups.iloc[test_idx].nunique(),
                "test_rows": len(test_idx),
                **held_out,
            }
        ]
    )
    held_out_row.to_csv(
        OUTPUT_DIR / "held_out_evaluation.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Refit ONLY after the evaluation is finished.
    # The held-out result above remains valid because it was produced
    # before this refit.
    # ------------------------------------------------------------
    final_model = build_models()[best_name]
    final_model.fit(X, y)

    joblib.dump(
        final_model,
        OUTPUT_DIR / "best_model.joblib",
    )

    (OUTPUT_DIR / "best_model_features.json").write_text(
        json.dumps(features, indent=2)
    )

    # XGBoost feature importance, fitted only for interpretation.
    xgb = build_models()["XGBoost"]
    xgb.fit(X, y)

    importances = xgb.named_steps["model"].feature_importances_
    order = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(10, 8))
    plt.barh(
        [X.columns[i] for i in order][::-1],
        importances[order][::-1],
    )
    plt.title("XGBoost Feature Importance — Human Labels")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "feature_importance_xgboost.png",
        dpi=150,
    )
    plt.close()

    print(
        f"\nSaved deployable baseline model: "
        f"{OUTPUT_DIR / 'best_model.joblib'}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 78)
    print("STEP 5 + 6 — ARTIFACT BASELINE + SUBJECT-INDEPENDENT VALIDATION")
    print("=" * 78)

    df = load_labeled_dataframe()

    print(f"\nTotal labeled rows: {len(df)}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    print(f"\nLabel source counts:\n{df['label_source'].value_counts()}")

    save_label_balance_diagnostics(df)
    save_feature_diagnostics(df, FEATURE_COLUMNS)

    heuristic_rows = df[
        df["label_source"] == "heuristic_suggestion"
    ]
    manual_rows = df[
        df["label_source"].isin(
            ["manual", "human", "human_review", "manual_review"]
        )
    ]

    # ------------------------------------------------------------
    # Scientific gate:
    # If the current labeled set is entirely heuristic, do NOT train
    # or save a detector. Run only the audit.
    # ------------------------------------------------------------
    if not manual_rows.empty:
        print(
            f"\nHuman-reviewed labels detected: {len(manual_rows)} rows."
        )
        run_human_label_baseline(df)

    if not heuristic_rows.empty:
        if manual_rows.empty:
            run_heuristic_audit(df)
        else:
            print(
                f"\nNOTE: {len(heuristic_rows)} heuristic-labeled rows are "
                "present but excluded from the real detector training. "
                "Only human-reviewed rows are used for the deployable model."
            )

    if manual_rows.empty:
        print("\n" + "=" * 78)
        print("STEP 5 + 6 NOT FROZEN — HUMAN LABELS STILL REQUIRED")
        print("=" * 78)
        print(
            "\nNo deployable artifact model was trained or saved. "
            "The heuristic audit is diagnostic only."
        )
    else:
        print("\n" + "=" * 78)
        print("STEP 5 + 6 COMPLETE — HUMAN-LABEL BASELINE")
        print("=" * 78)
        print(
            "\nThe held-out test result is the only number that should be "
            "reported as baseline artifact-detector performance."
        )


if __name__ == "__main__":
    main()
