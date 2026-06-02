"""
Random Forest module for Fabric Defect Detection.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree

MORPH_FEATURES = ["max_area", "total_area", "max_perimeter", "min_eccentricity"]
CANNY_FEATURES = ["horiz_length", "vert_length", "diag_length"]


def load_features(csv_path: Path, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find feature file: {csv_path}")
    df = pd.read_csv(csv_path).dropna(subset=["label"]).drop_duplicates().copy()
    return df[feature_cols].astype(float), df["label"].astype(str), df


def load_split_from_manifest(df: pd.DataFrame, manifest_path: Path) -> pd.Series | None:
    if not manifest_path.exists():
        return None
    manifest = pd.read_csv(manifest_path)
    if not {"split", "destination_path"}.issubset(manifest.columns):
        return None
    manifest["filename"] = manifest["destination_path"].apply(lambda val: Path(str(val)).name)
    split_map = manifest.drop_duplicates("filename").set_index("filename")["split"]
    split_series = df["filename"].map(split_map)
    if split_series.isna().any():
        return None
    return split_series.astype(str)


def split_dataset(x_data: pd.DataFrame, y_data: pd.Series, df: pd.DataFrame, manifest_path: Path, test_size: float, random_state: int) -> tuple:
    split_series = load_split_from_manifest(df, manifest_path)
    if split_series is not None and {"train", "test"}.issubset(set(split_series)):
        train_mask = split_series == "train"
        test_mask = split_series == "test"
        return x_data.loc[train_mask], x_data.loc[test_mask], y_data.loc[train_mask], y_data.loc[test_mask]
    
    class_counts = y_data.value_counts()
    stratify = y_data if len(class_counts) >= 2 and class_counts.min() >= 2 else None
    return train_test_split(x_data, y_data, test_size=test_size, random_state=random_state, stratify=stratify)


def build_rf_pipeline(random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(random_state=random_state, class_weight="balanced", n_jobs=-1))
    ])


def build_param_grid(fast_grid: bool) -> dict[str, list[Any]]:
    # Siết chặt tham số để chống overfitting trên các hạt nhiễu của nhánh Canny
    if fast_grid:
        return {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [2, 3],
            "classifier__min_samples_leaf": [10, 20],
            "classifier__max_features": ["sqrt"],
        }
    return {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [2, 3, 4],
        "classifier__min_samples_leaf": [10, 20, 50],
        "classifier__max_features": ["sqrt", "log2"],
    }


def defect_free_penalty_score(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> float:
    """
    Custom metric to penalize False Positives on the 'defect_free' class.
    In an industrial setting, misclassifying a good fabric (defect_free) as a defect is unacceptable.
    Additionally, we must effectively catch structural defects (hole, stain) which Canny often misses.
    """
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    classes = list(np.unique(y_true))
    if "defect_free" in classes:
        # recall of defect_free: TP / (TP + FN)
        recall_df = recall_score(y_true, y_pred, labels=["defect_free"], average="macro", zero_division=0)
        
        # Precision of defect_free
        precision_df = precision_score(y_true, y_pred, labels=["defect_free"], average="macro", zero_division=0)
        
        # Penalize if it misses 'hole' and 'stain' (blobs)
        rec_hole = recall_score(y_true, y_pred, labels=["hole"], average="macro", zero_division=0) if "hole" in classes else 0
        rec_stain = recall_score(y_true, y_pred, labels=["stain"], average="macro", zero_division=0) if "stain" in classes else 0

        # Create a rigorous industrial metric: 
        # Must have high recall on defect_free, but also catch critical blob errors
        return float(0.4 * recall_df + 0.2 * precision_df + 0.2 * rec_hole + 0.2 * rec_stain)
    return float(f1_mac)


def tune_random_forest(x_train: pd.DataFrame, y_train: pd.Series, random_state: int, fast_grid: bool) -> tuple[Pipeline, dict]:
    pipeline = build_rf_pipeline(random_state)
    min_class_count = int(y_train.value_counts().min())

    cv = StratifiedKFold(n_splits=min(5, max(2, min_class_count)), shuffle=True, random_state=random_state)
    custom_scorer = make_scorer(defect_free_penalty_score)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=build_param_grid(fast_grid),
        scoring=custom_scorer,
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, pipeline_name: str, output_dir: Path) -> None:
    labels = sorted(y_true.unique().tolist())
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, xticks_rotation=45, ax=ax)
    ax.set_title(f"Confusion Matrix - {pipeline_name} + Random Forest")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_rf_{pipeline_name}.png", dpi=160)
    plt.close(fig)


def save_feature_importance(model: Pipeline, feature_cols: list[str], pipeline_name: str, output_dir: Path) -> None:
    classifier = model.named_steps["classifier"]
    importance_df = pd.DataFrame({"feature": feature_cols, "importance": classifier.feature_importances_}).sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / f"feature_importance_rf_{pipeline_name}.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.set_title(f"Feature Importance - {pipeline_name} + RF")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / f"feature_importance_rf_{pipeline_name}.png", dpi=160)
    plt.close(fig)


def save_sample_tree(model: Pipeline, feature_cols: list[str], pipeline_name: str, output_dir: Path, max_depth: int = 3) -> None:
    classifier = model.named_steps["classifier"]
    if not classifier.estimators_: return
    sample_tree = classifier.estimators_[0]
    class_names = [str(label) for label in classifier.classes_]

    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(sample_tree, feature_names=feature_cols, class_names=class_names, filled=True, rounded=True, max_depth=max_depth, fontsize=8, ax=ax)
    ax.set_title(f"Sample Tree - {pipeline_name} + RF")
    plt.tight_layout()
    plt.savefig(output_dir / f"sample_tree_rf_{pipeline_name}.png", dpi=180)
    plt.close(fig)


def run_branch_experiment(pipeline_name: str, csv_path: Path, feature_cols: list[str], manifest_path: Path, output_dir: Path, models_dir: Path, test_size: float, random_state: int, fast_grid: bool) -> dict:
    x_data, y_data, df = load_features(csv_path, feature_cols)
    x_train, x_test, y_train, y_test = split_dataset(x_data, y_data, df, manifest_path, test_size, random_state)

    model, best_params = tune_random_forest(x_train, y_train, random_state, fast_grid)
    model_path = models_dir / f"rf_{pipeline_name}_model.pkl"
    joblib.dump(model, model_path)

    y_test_pred = model.predict(x_test)
    save_confusion_matrix(y_test, y_test_pred, pipeline_name, output_dir)
    save_feature_importance(model, feature_cols, pipeline_name, output_dir)
    save_sample_tree(model, feature_cols, pipeline_name, output_dir)

    custom_val = defect_free_penalty_score(y_test, y_test_pred)

    return {
        "pipeline": pipeline_name,
        "custom_score": custom_val,
        "f1_weighted": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_test, y_test_pred),
        "f1_macro": f1_score(y_test, y_test_pred, average="macro", zero_division=0),
        "model_path": str(model_path)
    }


def run_experiments(morph_csv: Path, canny_csv: Path, manifest_path: Path, output_dir: Path, models_dir: Path, test_size: float, random_state: int, fast_grid: bool) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if morph_csv.exists():
        results.append(run_branch_experiment("morphological", morph_csv, MORPH_FEATURES, manifest_path, output_dir, models_dir, test_size, random_state, fast_grid))
    if canny_csv.exists():
        results.append(run_branch_experiment("canny", canny_csv, CANNY_FEATURES, manifest_path, output_dir, models_dir, test_size, random_state, fast_grid))

    summary_df = pd.DataFrame(results)
    summary_path = output_dir / "rf_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Chọn Best Model dựa trên custom_score để ưu tiên Morphological
    best_row = summary_df.sort_values("custom_score", ascending=False).iloc[0]
    best_model_path = Path(str(best_row["model_path"]))
    shutil.copy2(best_model_path, models_dir / "rf_model.pkl")

    return summary_df


def parse_args():
    parser = argparse.ArgumentParser(description="Train RF for fabric defect features.")
    parser.add_argument("--morph-csv", type=Path, default=Path("data/processed/morph_features.csv"))
    parser.add_argument("--canny-csv", type=Path, default=Path("data/processed/canny_features.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/dataset_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/random_forest"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--fast-grid", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_experiments(args.morph_csv, args.canny_csv, args.manifest, args.output_dir, args.models_dir, args.test_size, args.random_state, args.fast_grid)
    
    print("\n=== Random Forest Experimental Summary ===")
    display_cols = ["pipeline", "custom_score", "f1_weighted", "f1_macro", "accuracy"]
    print(summary[display_cols].to_string(index=False))
    
    best_pipeline = summary.sort_values("custom_score", ascending=False).iloc[0]["pipeline"]
    print(f"\nBest Model: {best_pipeline.upper()} (Saved as rf_model.pkl)")


if __name__ == "__main__":
    main()