"""
Support Vector Machine (SVM) module for Fabric Defect Detection.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

MORPH_FEATURES = ["max_area", "total_area", "max_perimeter", "min_eccentricity"]
DIRECTIONAL_FEATURES = ["horiz_length", "vert_length", "diag_length"]


def load_features(csv_path: Path, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = pd.read_csv(csv_path).dropna(subset=["label"]).drop_duplicates().copy()
    return df[feature_cols].astype(float), df["label"].astype(str), df


def split_dataset(x_data: pd.DataFrame, y_data: pd.Series, df: pd.DataFrame, manifest_path: Path, test_size: float, random_state: int) -> tuple:
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        manifest["filename"] = manifest["destination_path"].apply(lambda val: Path(str(val)).name)
        split_map = manifest.drop_duplicates("filename").set_index("filename")["split"]
        split_series = df["filename"].map(split_map).astype(str)
        if {"train", "test"}.issubset(set(split_series)):
            train_mask = split_series == "train"
            test_mask = split_series == "test"
            return x_data.loc[train_mask], x_data.loc[test_mask], y_data.loc[train_mask], y_data.loc[test_mask]
    return train_test_split(x_data, y_data, test_size=test_size, random_state=random_state, stratify=y_data)


def build_svm_pipeline(random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(random_state=random_state, class_weight="balanced", probability=True))
    ])


def tune_svm(x_train: pd.DataFrame, y_train: pd.Series, random_state: int, fast_grid: bool) -> tuple[Pipeline, dict]:
    pipeline = build_svm_pipeline(random_state)
    param_grid = {
        "svm__C": [1, 10] if fast_grid else [0.1, 1, 10, 50, 100],
        "svm__kernel": ["rbf", "linear"],
        "svm__gamma": ["scale"] if fast_grid else ["scale", "auto", 0.01, 0.1]
    }
    min_class_count = int(y_train.value_counts().min())
    cv = StratifiedKFold(n_splits=min(5, max(2, min_class_count)), shuffle=True, random_state=random_state)
    
    # Sửa scoring thành f1_macro để đồng bộ chuẩn công nghiệp với Random Forest
    grid_search = GridSearchCV(pipeline, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=3)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def plot_decision_boundary(model: Pipeline, X: pd.DataFrame, y: pd.Series, pipeline_name: str, output_dir: Path):
    imputer = model.named_steps['imputer']
    scaler = model.named_steps['scaler']
    X_processed = scaler.transform(imputer.transform(X))
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)
    
    labels = sorted(y.unique())
    label_map = {label: idx for idx, label in enumerate(labels)}
    y_num = y.map(label_map).values
    
    best_svm = model.named_steps['svm']
    svm_2d = SVC(C=best_svm.C, kernel=best_svm.kernel, gamma=best_svm.gamma, class_weight='balanced')
    svm_2d.fit(X_pca, y_num)
    
    h = .02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    contour = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_num, cmap=plt.cm.coolwarm, edgecolors='k', s=40)
    
    handles, _ = scatter.legend_elements()
    ax.legend(handles, labels, title="Classes")
    ax.set_title(f"SVM Hyperplane Boundary (PCA 2D)\nBranch: {pipeline_name.upper()}")
    plt.tight_layout()
    plt.savefig(output_dir / f"hyperplane_svm_{pipeline_name}.png", dpi=160)
    plt.close(fig)


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, pipeline_name: str, output_dir: Path):
    labels = sorted(y_true.unique().tolist())
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, xticks_rotation=45, ax=ax)
    ax.set_title(f"Confusion Matrix - {pipeline_name} + SVM")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_svm_{pipeline_name}.png", dpi=160)
    plt.close(fig)


def run_branch_experiment(pipeline_name: str, csv_path: Path, feature_cols: list[str], manifest_path: Path, output_dir: Path, models_dir: Path, fast_grid: bool) -> dict:
    x_data, y_data, df = load_features(csv_path, feature_cols)
    x_train, x_test, y_train, y_test = split_dataset(x_data, y_data, df, manifest_path, 0.2, 42)

    model, best_params = tune_svm(x_train, y_train, 42, fast_grid)
    model_path = models_dir / f"svm_{pipeline_name}_model.pkl"
    joblib.dump(model, model_path)
    
    y_test_pred = model.predict(x_test)
    save_confusion_matrix(y_test, y_test_pred, pipeline_name, output_dir)
    
    if len(np.unique(y_train)) > 1: 
        plot_decision_boundary(model, x_test, y_test, pipeline_name, output_dir)

    return {
        "pipeline": pipeline_name,
        "f1_macro": f1_score(y_test, y_test_pred, average="macro", zero_division=0), # BỔ SUNG ĐO ĐẠC F1-MACRO
        "f1_weighted": f1_score(y_test, y_test_pred, average="weighted", zero_division=0), 
        "accuracy": accuracy_score(y_test, y_test_pred),
        "model_path": str(model_path)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-grid", action="store_true")
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/dataset_manifest.csv"))
    args = parser.parse_args()

    output_dir, models_dir = Path("reports/experiments/svm"), Path("models")
    output_dir.mkdir(parents=True, exist_ok=True); models_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if (morph_csv := Path("data/processed/morph_features.csv")).exists():
        results.append(run_branch_experiment("morphological", morph_csv, MORPH_FEATURES, args.manifest, output_dir, models_dir, args.fast_grid))
    if (directional_csv := Path("data/processed/directional_features.csv")).exists():
        results.append(run_branch_experiment("directional", directional_csv, DIRECTIONAL_FEATURES, args.manifest, output_dir, models_dir, args.fast_grid))

    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(output_dir / "svm_metrics_summary.csv", index=False)
        
        # Chọn mô hình tốt nhất dựa trên F1-Macro
        best_row = summary_df.sort_values("f1_macro", ascending=False).iloc[0]
        shutil.copy2(best_row["model_path"], models_dir / "svm_model.pkl")
        
        print("\n=== SVM Experimental Summary ===")
        # In ra Terminal đầy đủ các cột đo đạc
        display_cols = ["pipeline", "f1_macro", "f1_weighted", "accuracy"]
        print(summary_df[display_cols].to_string(index=False))
        
        print(f"\nBest Model: {best_row['pipeline'].upper()} (Saved as svm_model.pkl)")


if __name__ == "__main__":
    main()