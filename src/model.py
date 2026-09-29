"""Simple leakage-aware logistic regression models for win probability."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Settings


MODEL_FEATURES = {
    "rating_only": {
        "numeric": ["rating_diff"],
        "categorical": [],
    },
    "baseline": {
        "numeric": ["rating_diff"],
        "categorical": ["speed", "user_color"],
    },
    "behavioral": {
        "numeric": [
            "hour_sin",
            "hour_cos",
            "rating_diff",
            "break_after_previous",
            "loss_streak_before",
            "win_streak_before",
            "session_game_number",
        ],
        "categorical": ["day_of_week", "speed", "user_color", "tilt_proxy"],
    },
}


def _safe_auc(y_true: pd.Series, proba: np.ndarray) -> float | None:
    """Compute ROC-AUC when both classes are present."""
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, proba))


def _build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    settings: Settings,
) -> Pipeline:
    """Build one preprocessing + logistic regression pipeline."""
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipe, categorical_features))

    preprocessor = ColumnTransformer(transformers)
    model = LogisticRegression(
        C=settings.model.C,
        max_iter=settings.model.max_iter,
        class_weight=settings.model.class_weight,
        random_state=settings.model.random_state,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _prepare_data(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.Series]:
    """Keep decisive games and return the modeling target."""
    data = df.loc[df["is_decisive"].astype(bool)].copy()
    data = data.sort_values("created_at").reset_index(drop=True)
    if len(data) < 30:
        raise ValueError(f"Need at least 30 decisive games for modeling, got {len(data)}")
    y = data[settings.model.target].astype(int)
    return data, y


def _split_data(data: pd.DataFrame, y: pd.Series, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int]:
    """Split observations chronologically without shuffling."""
    split_idx = max(1, int(len(data) * (1 - test_size)))
    return (
        data.iloc[:split_idx],
        data.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
        split_idx,
    )


def _fit_one_model(
    data: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
    settings: Settings,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit one model and return metrics plus coefficients."""
    X_train_df, X_test_df, y_train, y_test, split_idx = _split_data(
        data, y, settings.model.test_size
    )
    feature_cols = numeric_features + categorical_features
    pipeline = _build_pipeline(numeric_features, categorical_features, settings)
    pipeline.fit(X_train_df[feature_cols], y_train)

    proba = pipeline.predict_proba(X_test_df[feature_cols])[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "model": model_name,
        "n_total": len(data),
        "n_train": len(X_train_df),
        "n_test": len(X_test_df),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "roc_auc": _safe_auc(y_test, proba),
        "log_loss": float(log_loss(y_test, proba, labels=[0, 1])),
        "split": "chronological",
        "test_start": str(data.loc[split_idx, "created_at"]),
        "features": feature_cols,
    }

    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
        }
    ).sort_values("coefficient", ascending=False)
    return metrics, coef_df


def fit_model_suite(
    df: pd.DataFrame,
    settings: Settings,
    logger: logging.Logger,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Fit rating-only, baseline, and behavioral models on one time split.

    Returns:
        Tuple of (metrics payload, comparison DataFrame, behavioral coefficients).
    """
    data, y = _prepare_data(df, settings)
    models: dict[str, Any] = {}
    coefficients = pd.DataFrame()

    for model_name, feature_set in MODEL_FEATURES.items():
        metrics, model_coefficients = _fit_one_model(
            data,
            y,
            model_name,
            feature_set["numeric"],
            feature_set["categorical"],
            settings,
        )
        models[model_name] = metrics
        if model_name == "behavioral":
            coefficients = model_coefficients
        logger.info(
            "%s model: ROC-AUC=%.3f, balanced_accuracy=%.3f, log_loss=%.3f",
            model_name,
            metrics["roc_auc"] or float("nan"),
            metrics["balanced_accuracy"],
            metrics["log_loss"],
        )

    comparison = pd.DataFrame(
        [
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "roc_auc": metrics["roc_auc"],
                "log_loss": metrics["log_loss"],
            }
            for name, metrics in models.items()
        ]
    )

    payload = {
        "total_games": int(len(df)),
        "decisive_games": int(len(data)),
        "excluded_non_decisive": int(len(df) - len(data)),
        "models": models,
        "behavioral_model_interpretation": (
            "Numeric coefficients are standardized. Odds ratios describe a one-standard-deviation "
            "change for numeric predictors and a category relative to its reference category for one-hot features."
        ),
    }
    return payload, comparison, coefficients


def save_model_results(
    metrics: dict[str, Any],
    comparison: pd.DataFrame,
    coefficients: pd.DataFrame,
    settings: Settings,
    *,
    metrics_path: Path | None = None,
    comparison_path: Path | None = None,
    coefficients_path: Path | None = None,
) -> None:
    """Save model metrics, comparison table, and behavioral coefficients."""
    metrics_path = settings.resolve_path(metrics_path or settings.paths.model_metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison_path = settings.resolve_path(comparison_path or settings.paths.model_comparison_csv)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)

    coef_path = settings.resolve_path(coefficients_path or settings.paths.model_coefficients_csv)
    coef_path.parent.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(coef_path, index=False)



def generate_walk_forward_splits(
    data: pd.DataFrame,
    initial_train_size: int,
    test_window_size: int,
    min_training_size: int,
) -> list[dict[str, int | str]]:
    """Create expanding-window chronological train/test splits.

    Train windows always start at the first observation and expand after each
    test window. The split boundary is moved forward when multiple games share
    the same timestamp so the train/test timestamp condition remains strict.
    """
    if min_training_size < 2:
        raise ValueError("min_training_size must be at least 2")
    if initial_train_size < min_training_size:
        raise ValueError("initial_train_size must be >= min_training_size")
    if test_window_size <= 0:
        raise ValueError("test_window_size must be > 0")

    ordered = data.sort_values("created_at").reset_index(drop=True)
    n = len(ordered)
    if n < initial_train_size + test_window_size:
        raise ValueError(
            "Not enough observations for one full walk-forward fold: "
            f"need at least {initial_train_size + test_window_size}, got {n}"
        )

    splits: list[dict[str, int | str]] = []
    train_end = initial_train_size
    fold = 1
    while train_end + test_window_size <= n:
        while (
            train_end < n
            and train_end > 0
            and ordered.loc[train_end - 1, "created_at"] >= ordered.loc[train_end, "created_at"]
        ):
            train_end += 1

        if train_end + test_window_size > n:
            break
        if train_end < min_training_size:
            raise ValueError("Walk-forward split produced a training window below min_training_size")

        test_end = train_end + test_window_size
        train_timestamp_max = ordered.loc[train_end - 1, "created_at"]
        test_timestamp_min = ordered.loc[train_end, "created_at"]
        if train_timestamp_max >= test_timestamp_min:
            raise ValueError("Temporal leakage detected at walk-forward split boundary")

        splits.append(
            {
                "fold": fold,
                "train_start_idx": 0,
                "train_end_idx": train_end,
                "test_start_idx": train_end,
                "test_end_idx": test_end,
                "train_start": str(ordered.loc[0, "created_at"]),
                "train_end": str(train_timestamp_max),
                "test_start": str(test_timestamp_min),
                "test_end": str(ordered.loc[test_end - 1, "created_at"]),
            }
        )
        train_end = test_end
        fold += 1

    if not splits:
        raise ValueError("No valid walk-forward folds could be generated")
    return splits


def _score_split(
    data: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    feature_set: dict[str, list[str]],
    train_start_idx: int,
    train_end_idx: int,
    test_start_idx: int,
    test_end_idx: int,
    settings: Settings,
) -> dict[str, Any]:
    """Fit preprocessing/model on one training window and score one test window."""
    numeric_features = feature_set["numeric"]
    categorical_features = feature_set["categorical"]
    feature_cols = numeric_features + categorical_features
    train = data.iloc[train_start_idx:train_end_idx]
    test = data.iloc[test_start_idx:test_end_idx]
    y_train = y.iloc[train_start_idx:train_end_idx]
    y_test = y.iloc[test_start_idx:test_end_idx]

    if train.empty or test.empty:
        raise ValueError("Walk-forward train and test windows must be non-empty")
    if y_train.nunique() < 2:
        raise ValueError("Training window must contain both target classes")

    pipeline = _build_pipeline(numeric_features, categorical_features, settings)
    pipeline.fit(train[feature_cols], y_train)
    proba = pipeline.predict_proba(test[feature_cols])[:, 1]

    return {
        "model": model_name,
        "n_train": len(train),
        "n_test": len(test),
        "roc_auc": _safe_auc(y_test, proba),
        "log_loss": float(log_loss(y_test, proba, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_test, proba)),
    }


def evaluate_chronological_baseline(
    df: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Evaluate the existing chronological 80/20 split with added Brier score."""
    data, y = _prepare_data(df, settings)
    _, _, _, _, split_idx = _split_data(data, y, settings.model.test_size)
    rows: list[dict[str, Any]] = []
    for model_name, feature_set in MODEL_FEATURES.items():
        scored = _score_split(
            data,
            y,
            model_name,
            feature_set,
            0,
            split_idx,
            split_idx,
            len(data),
            settings,
        )
        scored["n_total"] = len(data)
        scored["split"] = "chronological_80_20"
        rows.append(scored)
    return pd.DataFrame(rows)


def walk_forward_validate(
    df: pd.DataFrame,
    settings: Settings,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run expanding-window validation and return aggregate, fold, and baseline metrics."""
    data, y = _prepare_data(df, settings)
    split_defs = generate_walk_forward_splits(
        data,
        settings.walk_forward.initial_train_size,
        settings.walk_forward.test_window_size,
        settings.walk_forward.min_training_size,
    )

    fold_rows: list[dict[str, Any]] = []
    for split in split_defs:
        train_end = int(split["train_end_idx"])
        test_start = int(split["test_start_idx"])
        test_end = int(split["test_end_idx"])
        if set(range(0, train_end)).intersection(range(test_start, test_end)):
            raise ValueError("Temporal split indices overlap")

        train_max = data.loc[train_end - 1, "created_at"]
        test_min = data.loc[test_start, "created_at"]
        if train_max >= test_min:
            raise ValueError("Temporal leakage detected in walk-forward fold")

        for model_name, feature_set in MODEL_FEATURES.items():
            scored = _score_split(
                data,
                y,
                model_name,
                feature_set,
                0,
                train_end,
                test_start,
                test_end,
                settings,
            )
            fold_rows.append(
                {
                    **split,
                    **scored,
                    "train_n": scored["n_train"],
                    "test_n": scored["n_test"],
                }
            )

    folds = pd.DataFrame(fold_rows)
    baseline = evaluate_chronological_baseline(df, settings)
    aggregate_rows: list[dict[str, Any]] = []
    for model_name in MODEL_FEATURES:
        model_folds = folds.loc[folds["model"] == model_name]
        auc_values = model_folds["roc_auc"].dropna().astype(float)
        baseline_row = baseline.loc[baseline["model"] == model_name].iloc[0]
        aggregate_rows.append(
            {
                "model": model_name,
                "n_folds": int(len(model_folds)),
                "valid_auc_folds": int(model_folds["roc_auc"].notna().sum()),
                "mean_roc_auc": float(auc_values.mean()),
                "std_roc_auc": float(auc_values.std(ddof=1)) if len(auc_values) > 1 else np.nan,
                "mean_log_loss": float(model_folds["log_loss"].mean()),
                "std_log_loss": float(model_folds["log_loss"].std(ddof=1)) if len(model_folds) > 1 else np.nan,
                "mean_brier_score": float(model_folds["brier_score"].mean()),
                "std_brier_score": float(model_folds["brier_score"].std(ddof=1)) if len(model_folds) > 1 else np.nan,
                "chronological_80_20_roc_auc": baseline_row["roc_auc"],
                "chronological_80_20_log_loss": baseline_row["log_loss"],
                "chronological_80_20_brier_score": baseline_row["brier_score"],
                "chronological_80_20_n_test": int(baseline_row["n_test"]),
            }
        )
        logger.info(
            "%s walk-forward: mean ROC-AUC=%.3f, log-loss=%.3f, Brier=%.3f (%d folds)",
            model_name,
            aggregate_rows[-1]["mean_roc_auc"],
            aggregate_rows[-1]["mean_log_loss"],
            aggregate_rows[-1]["mean_brier_score"],
            aggregate_rows[-1]["n_folds"],
        )

    aggregate = pd.DataFrame(aggregate_rows)
    return aggregate, folds, baseline


def save_walk_forward_results(
    aggregate: pd.DataFrame,
    folds: pd.DataFrame,
    baseline: pd.DataFrame,
    settings: Settings,
    *,
    aggregate_path: Path | None = None,
    folds_path: Path | None = None,
    markdown_path: Path | None = None,
) -> None:
    """Save walk-forward aggregate/fold metrics and a compact markdown summary."""
    aggregate_path = settings.resolve_path(aggregate_path or settings.paths.walk_forward_metrics_csv)
    folds_path = settings.resolve_path(folds_path or settings.paths.walk_forward_fold_metrics_csv)
    markdown_path = settings.resolve_path(markdown_path or settings.paths.walk_forward_metrics_md)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(aggregate_path, index=False)
    folds.to_csv(folds_path, index=False)

    aggregate_view = aggregate[
        [
            "model",
            "mean_roc_auc",
            "std_roc_auc",
            "mean_log_loss",
            "std_log_loss",
            "mean_brier_score",
            "std_brier_score",
            "n_folds",
            "valid_auc_folds",
        ]
    ]
    baseline_view = baseline[
        ["model", "roc_auc", "log_loss", "brier_score", "n_test"]
    ].rename(
        columns={
            "roc_auc": "80/20 ROC-AUC",
            "log_loss": "80/20 log-loss",
            "brier_score": "80/20 Brier",
            "n_test": "80/20 test n",
        }
    )

    folds_view = folds[
        [
            "fold",
            "model",
            "n_train",
            "n_test",
            "train_end",
            "test_start",
            "roc_auc",
            "log_loss",
            "brier_score",
        ]
    ].copy()
    folds_view["roc_auc"] = folds_view["roc_auc"].fillna("NA")

    md = [
        "# Walk-Forward Temporal Validation",
        "",
        "The existing chronological 80/20 split remains the predictive baseline. "
        "This analysis adds expanding-window validation without changing the three model specifications.",
        "",
        "## Method",
        "",
        f"Initial training size: **{settings.walk_forward.initial_train_size}** decisive games.",
        f"Test window: **{settings.walk_forward.test_window_size}** games per fold.",
        f"Minimum training size: **{settings.walk_forward.min_training_size}** games.",
        "No random shuffle is used. Preprocessing and logistic regression are fit separately inside each fold using training data only.",
        "Every fold satisfies `max(train timestamp) < min(test timestamp)` and has disjoint train/test indices.",
        "",
        "## Aggregate results",
        "",
        aggregate_view.to_markdown(index=False),
        "",
        "## 80/20 comparison",
        "",
        baseline_view.to_markdown(index=False),
        "",
        "## Fold-level results",
        "",
        folds_view.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "Walk-forward validation describes out-of-sample performance across several future time windows instead of relying on one holdout period. "
        "Variation between folds is evidence about temporal stability, not a reason to select a more favorable period.",
        "A missing ROC-AUC is kept as `NA` when a test fold contains only one class; it is not replaced with a guessed value.",
    ]
    markdown_path.write_text("\n".join(md) + "\n", encoding="utf-8")
