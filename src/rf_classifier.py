"""
Random Forest module for Fabric Defect Detection.

Responsibility:
- Train Random Forest models only.
- Evaluate Random Forest on both traditional image-processing branches:
  1. Morphological features
  2. Canny directional-line features
- Export confusion matrix, Precision, Recall, F1-score.
- Export feature-importance chart and a sample decision-tree visualization.

Default inputs:
    data/processed/morph_features.csv
    data/processed/canny_features.csv
    data/raw/dataset_manifest.csv  # optional, used to preserve the existing train/test split

Default outputs:
    models/rf_morphological_model.pkl
    models/rf_canny_model.pkl
    models/rf_model.pkl  # copy of the best branch model by macro F1
    reports/experiments/random_forest/*
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree


MORPH_FEATURES = [
    "max_area",
    "total_area",
    "max_perimeter",
    "min_eccentricity",
]

CANNY_FEATURES = [
    "horiz_length",
    "vert_length",
    "diag_length",
]


def load_features(csv_path: Path, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load one feature CSV file.

    Args:
        csv_path: Path to the feature CSV file.
        feature_cols: List of numeric feature column names.

    Returns:
        X feature matrix, y label vector and original dataframe.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find feature file: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["filename", "label", *feature_cols]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing columns in {csv_path}: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["label"]).drop_duplicates().copy()

    x_data = df[feature_cols].astype(float)
    y_data = df["label"].astype(str)

    return x_data, y_data, df


def load_split_from_manifest(
    df: pd.DataFrame,
    manifest_path: Path,
) -> pd.Series | None:
    """
    Load train/test split from dataset_manifest.csv if available.

    Args:
        df: Feature dataframe containing filename column.
        manifest_path: Path to dataset_manifest.csv.

    Returns:
        Series with split values aligned with df, or None if unavailable.
    """
    if not manifest_path.exists():
        return None

    manifest = pd.read_csv(manifest_path)
    required_cols = {"split", "destination_path"}

    if not required_cols.issubset(manifest.columns):
        return None

    manifest = manifest.copy()
    manifest["filename"] = manifest["destination_path"].apply(
        lambda value: Path(str(value)).name
    )

    split_map = manifest.drop_duplicates("filename").set_index("filename")["split"]
    split_series = df["filename"].map(split_map)

    if split_series.isna().any():
        missing_count = int(split_series.isna().sum())
        print(
            f"Warning: {missing_count} rows were not found in dataset_manifest.csv. "
            "Fallback to stratified random split."
        )
        return None

    return split_series.astype(str)


def split_dataset(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    df: pd.DataFrame,
    manifest_path: Path,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data using dataset_manifest.csv if possible, otherwise use stratified split.

    Args:
        x_data: Feature matrix.
        y_data: Label vector.
        df: Original feature dataframe.
        manifest_path: Path to dataset_manifest.csv.
        test_size: Test ratio used for fallback split.
        random_state: Random seed.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    split_series = load_split_from_manifest(df, manifest_path)

    if split_series is not None and {"train", "test"}.issubset(set(split_series)):
        train_mask = split_series == "train"
        test_mask = split_series == "test"

        return (
            x_data.loc[train_mask],
            x_data.loc[test_mask],
            y_data.loc[train_mask],
            y_data.loc[test_mask],
        )

    class_counts = y_data.value_counts()
    stratify = y_data if len(class_counts) >= 2 and class_counts.min() >= 2 else None

    return train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def build_rf_pipeline(random_state: int) -> Pipeline:
    """
    Build Random Forest pipeline.

    Args:
        random_state: Random seed.

    Returns:
        Scikit-learn pipeline.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_param_grid(fast_grid: bool) -> dict[str, list[Any]]:
    """
    Build hyperparameter grid for Random Forest tuning.

    Args:
        fast_grid: Whether to use a smaller grid for quick testing.

    Returns:
        Parameter grid for GridSearchCV.
    """
    if fast_grid:
        return {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10],
            "classifier__min_samples_leaf": [1, 2],
            "classifier__max_features": ["sqrt"],
        }

    return {
        "classifier__n_estimators": [50, 100, 200, 300],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
    }


def tune_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    fast_grid: bool,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Tune Random Forest hyperparameters using macro F1-score.

    Args:
        x_train: Training feature matrix.
        y_train: Training labels.
        random_state: Random seed.
        fast_grid: Whether to use a small grid.

    Returns:
        Best model and best parameter dictionary.
    """
    pipeline = build_rf_pipeline(random_state)
    min_class_count = int(y_train.value_counts().min())

    if min_class_count < 2:
        fallback_params = {
            "classifier__n_estimators": 100,
            "classifier__max_depth": 10,
            "classifier__min_samples_leaf": 2,
            "classifier__max_features": "sqrt",
        }
        pipeline.set_params(**fallback_params)
        pipeline.fit(x_train, y_train)
        return pipeline, fallback_params

    cv = StratifiedKFold(
        n_splits=min(5, min_class_count),
        shuffle=True,
        random_state=random_state,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=build_param_grid(fast_grid),
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    grid_search.fit(x_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    pipeline_name: str,
    output_dir: Path,
) -> None:
    """
    Save confusion matrix as CSV and PNG.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        pipeline_name: Name of the image-processing branch.
        output_dir: Output directory.
    """
    labels = sorted(y_true.unique().tolist())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_df.to_csv(
        output_dir / f"confusion_matrix_rf_{pipeline_name}.csv",
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        xticks_rotation=45,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {pipeline_name} + Random Forest")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_rf_{pipeline_name}.png", dpi=160)
    plt.close(fig)


def save_feature_importance(
    model: Pipeline,
    feature_cols: list[str],
    pipeline_name: str,
    output_dir: Path,
) -> None:
    """
    Save Random Forest feature importance as CSV and PNG.

    Args:
        model: Trained Random Forest pipeline.
        feature_cols: Feature column names.
        pipeline_name: Name of the image-processing branch.
        output_dir: Output directory.
    """
    classifier = model.named_steps["classifier"]

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(
        output_dir / f"feature_importance_rf_{pipeline_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.set_title(f"Feature Importance - {pipeline_name} + Random Forest")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / f"feature_importance_rf_{pipeline_name}.png", dpi=160)
    plt.close(fig)


def save_sample_tree(
    model: Pipeline,
    feature_cols: list[str],
    pipeline_name: str,
    output_dir: Path,
    max_depth: int = 3,
) -> None:
    """
    Save one sample decision tree from the Random Forest.

    Args:
        model: Trained Random Forest pipeline.
        feature_cols: Feature column names.
        pipeline_name: Name of the image-processing branch.
        output_dir: Output directory.
        max_depth: Maximum visualization depth.
    """
    classifier = model.named_steps["classifier"]

    if not classifier.estimators_:
        return

    sample_tree = classifier.estimators_[0]
    class_names = [str(label) for label in classifier.classes_]

    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        sample_tree,
        feature_names=feature_cols,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(f"Sample Decision Tree - {pipeline_name} + Random Forest")
    plt.tight_layout()
    plt.savefig(output_dir / f"sample_tree_rf_{pipeline_name}.png", dpi=180)
    plt.close(fig)


def evaluate_random_forest(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    pipeline_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Evaluate Random Forest and save detailed reports.

    Args:
        model: Trained model.
        x_train: Training features.
        y_train: Training labels.
        x_test: Testing features.
        y_test: Testing labels.
        pipeline_name: Name of the image-processing branch.
        output_dir: Output directory.

    Returns:
        Summary metrics.
    """
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    report = classification_report(
        y_test,
        y_test_pred,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        output_dir / f"classification_report_rf_{pipeline_name}.csv",
        encoding="utf-8-sig",
    )

    save_confusion_matrix(y_test, y_test_pred, pipeline_name, output_dir)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    return {
        "pipeline": pipeline_name,
        "model": "Random Forest",
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "accuracy_gap": train_accuracy - test_accuracy,
        "precision_macro": precision_score(
            y_test,
            y_test_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_test,
            y_test_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_test,
            y_test_pred,
            average="macro",
            zero_division=0,
        ),
        "precision_weighted": precision_score(
            y_test,
            y_test_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": recall_score(
            y_test,
            y_test_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_test,
            y_test_pred,
            average="weighted",
            zero_division=0,
        ),
    }


def run_branch_experiment(
    pipeline_name: str,
    csv_path: Path,
    feature_cols: list[str],
    manifest_path: Path,
    output_dir: Path,
    models_dir: Path,
    test_size: float,
    random_state: int,
    fast_grid: bool,
) -> dict[str, Any]:
    """
    Train and evaluate Random Forest for one image-processing branch.

    Args:
        pipeline_name: Branch name, e.g. morphological or canny.
        csv_path: Feature CSV path.
        feature_cols: Feature columns.
        manifest_path: Optional manifest path.
        output_dir: Report output directory.
        models_dir: Model output directory.
        test_size: Fallback test ratio.
        random_state: Random seed.
        fast_grid: Whether to use a small grid search.

    Returns:
        Summary metrics.
    """
    x_data, y_data, df = load_features(csv_path, feature_cols)

    x_train, x_test, y_train, y_test = split_dataset(
        x_data=x_data,
        y_data=y_data,
        df=df,
        manifest_path=manifest_path,
        test_size=test_size,
        random_state=random_state,
    )

    model, best_params = tune_random_forest(
        x_train=x_train,
        y_train=y_train,
        random_state=random_state,
        fast_grid=fast_grid,
    )

    model_path = models_dir / f"rf_{pipeline_name}_model.pkl"
    joblib.dump(model, model_path)

    with (output_dir / f"best_params_rf_{pipeline_name}.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(best_params, file, ensure_ascii=False, indent=4, default=str)

    metrics = evaluate_random_forest(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        pipeline_name=pipeline_name,
        output_dir=output_dir,
    )

    save_feature_importance(model, feature_cols, pipeline_name, output_dir)
    save_sample_tree(model, feature_cols, pipeline_name, output_dir)

    metrics["model_path"] = str(model_path)
    metrics["best_params"] = json.dumps(best_params, ensure_ascii=False)

    return metrics


def run_experiments(
    morph_csv: Path,
    canny_csv: Path,
    manifest_path: Path,
    output_dir: Path,
    models_dir: Path,
    test_size: float,
    random_state: int,
    fast_grid: bool,
) -> pd.DataFrame:
    """
    Run Random Forest experiments for both feature branches.

    Args:
        morph_csv: Morphological feature CSV.
        canny_csv: Canny feature CSV.
        manifest_path: Dataset manifest path.
        output_dir: Report output directory.
        models_dir: Model output directory.
        test_size: Fallback test ratio.
        random_state: Random seed.
        fast_grid: Whether to use a small grid search.

    Returns:
        Summary dataframe.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run_branch_experiment(
            pipeline_name="morphological",
            csv_path=morph_csv,
            feature_cols=MORPH_FEATURES,
            manifest_path=manifest_path,
            output_dir=output_dir,
            models_dir=models_dir,
            test_size=test_size,
            random_state=random_state,
            fast_grid=fast_grid,
        ),
        run_branch_experiment(
            pipeline_name="canny",
            csv_path=canny_csv,
            feature_cols=CANNY_FEATURES,
            manifest_path=manifest_path,
            output_dir=output_dir,
            models_dir=models_dir,
            test_size=test_size,
            random_state=random_state,
            fast_grid=fast_grid,
        ),
    ]

    summary_df = pd.DataFrame(results)
    summary_path = output_dir / "rf_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    best_row = summary_df.sort_values("f1_macro", ascending=False).iloc[0]
    best_model_path = Path(str(best_row["model_path"]))
    shutil.copy2(best_model_path, models_dir / "rf_model.pkl")

    with (output_dir / "rf_best_model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_pipeline": best_row["pipeline"],
                "selection_metric": "f1_macro",
                "best_model_source": str(best_model_path),
                "exported_as": str(models_dir / "rf_model.pkl"),
            },
            file,
            ensure_ascii=False,
            indent=4,
        )

    return summary_df


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate Random Forest for fabric defect features."
    )

    parser.add_argument(
        "--morph-csv",
        type=Path,
        default=Path("data/processed/morph_features.csv"),
        help="Path to morphological feature CSV.",
    )
    parser.add_argument(
        "--canny-csv",
        type=Path,
        default=Path("data/processed/canny_features.csv"),
        help="Path to Canny feature CSV.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/dataset_manifest.csv"),
        help="Optional dataset manifest to preserve train/test split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/experiments/random_forest"),
        help="Output report directory.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Output model directory.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fallback test size if manifest is unavailable.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--fast-grid",
        action="store_true",
        help="Use a smaller hyperparameter grid for faster testing.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    summary = run_experiments(
        morph_csv=args.morph_csv,
        canny_csv=args.canny_csv,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        fast_grid=args.fast_grid,
    )

    print("\n=== Random Forest Experimental Summary ===")
    print(summary)
