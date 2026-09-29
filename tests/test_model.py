from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_settings
from src.model import (
    _build_pipeline,
    evaluate_chronological_baseline,
    fit_model_suite,
    generate_walk_forward_splits,
    walk_forward_validate,
)


def _make_model_data(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    created_at = pd.date_range("2026-01-01", periods=n, freq="3h", tz="UTC")
    rating_diff = rng.normal(0, 150, n)
    score = rating_diff / 300 + rng.normal(0, 0.8, n)
    is_win = (score > 0).astype(int)
    return pd.DataFrame(
        {
            "created_at": created_at,
            "is_decisive": True,
            "is_win": is_win,
            "rating_diff": rating_diff,
            "hour_sin": np.sin(2 * np.pi * created_at.hour / 24),
            "hour_cos": np.cos(2 * np.pi * created_at.hour / 24),
            "break_after_previous": rng.uniform(1, 120, n),
            "loss_streak_before": rng.integers(0, 4, n),
            "win_streak_before": rng.integers(0, 4, n),
            "session_game_number": rng.integers(1, 8, n),
            "day_of_week": created_at.dayofweek,
            "speed": rng.choice(["blitz", "rapid", "bullet"], n),
            "user_color": rng.choice(["white", "black"], n),
            "tilt_proxy": rng.choice([True, False], n),
        }
    )


def test_model_suite_has_three_models() -> None:
    settings = load_settings(Path("config/config.yaml"))
    metrics, comparison, coefficients = fit_model_suite(_make_model_data(), settings, logger=_logger())

    assert set(metrics["models"]) == {"rating_only", "baseline", "behavioral"}
    assert len(comparison) == 3
    assert not coefficients.empty
    assert metrics["decisive_games"] == 80




def test_walk_forward_splits_are_temporal_and_expanding() -> None:
    settings = load_settings(Path("config/config.yaml")).model_copy(
        update={
            "walk_forward": load_settings(Path("config/config.yaml")).walk_forward.model_copy(
                update={"initial_train_size": 40, "test_window_size": 10, "min_training_size": 30}
            )
        }
    )
    data = _make_model_data(80)
    splits = generate_walk_forward_splits(
        data,
        settings.walk_forward.initial_train_size,
        settings.walk_forward.test_window_size,
        settings.walk_forward.min_training_size,
    )

    assert len(splits) == 4
    train_sizes = [int(split["train_end_idx"]) for split in splits]
    test_starts = [int(split["test_start_idx"]) for split in splits]
    assert train_sizes == [40, 50, 60, 70]
    assert test_starts == train_sizes
    for split in splits:
        assert int(split["train_end_idx"]) <= int(split["test_start_idx"])
        assert int(split["train_end_idx"]) < int(split["test_end_idx"])
        assert split["train_end"] < split["test_start"]


def test_walk_forward_rejects_invalid_window_configuration() -> None:
    data = _make_model_data(50)
    with pytest.raises(ValueError, match="initial_train_size must be >= min_training_size"):
        generate_walk_forward_splits(data, 20, 10, 25)
    with pytest.raises(ValueError, match="Not enough observations"):
        generate_walk_forward_splits(data, 45, 10, 30)


def test_walk_forward_preprocessing_is_fit_on_each_training_window(monkeypatch) -> None:
    settings = load_settings(Path("config/config.yaml")).model_copy(
        update={
            "walk_forward": load_settings(Path("config/config.yaml")).walk_forward.model_copy(
                update={"initial_train_size": 40, "test_window_size": 10, "min_training_size": 30}
            )
        }
    )
    data = _make_model_data(80)
    fit_sizes: list[int] = []

    class DummyPipeline:
        def fit(self, X, y):
            fit_sizes.append(len(X))
            return self

        def predict_proba(self, X):
            return np.tile(np.array([[0.4, 0.6]]), (len(X), 1))

    monkeypatch.setattr("src.model._build_pipeline", lambda *args, **kwargs: DummyPipeline())
    _, folds, baseline = walk_forward_validate(data, settings, _logger())

    assert fit_sizes.count(40) == 3
    assert fit_sizes.count(50) == 3
    assert fit_sizes.count(60) == 3
    assert fit_sizes.count(70) == 3
    assert fit_sizes.count(64) == 3
    assert baseline["n_test"].tolist() == [16, 16, 16]
    assert folds["n_train"].tolist() == [40] * 3 + [50] * 3 + [60] * 3 + [70] * 3


def test_walk_forward_preserves_baseline_split_metrics() -> None:
    settings = load_settings(Path("config/config.yaml"))
    data = _make_model_data(80)
    current, _, _ = fit_model_suite(data, settings, _logger())
    baseline = evaluate_chronological_baseline(data, settings)

    for model_name, metrics in current["models"].items():
        row = baseline.loc[baseline["model"] == model_name].iloc[0]
        assert metrics["roc_auc"] == row["roc_auc"]
        assert metrics["log_loss"] == row["log_loss"]




def test_walk_forward_keeps_single_class_auc_as_missing() -> None:
    settings = load_settings(Path("config/config.yaml")).model_copy(
        update={
            "walk_forward": load_settings(Path("config/config.yaml")).walk_forward.model_copy(
                update={"initial_train_size": 30, "test_window_size": 10, "min_training_size": 20}
            )
        }
    )
    data = _make_model_data(50)
    data.loc[30:, "is_win"] = 1
    _, folds, _ = walk_forward_validate(data, settings, _logger())

    assert folds["roc_auc"].isna().all()


def _logger():
    class Logger:
        def info(self, *args, **kwargs):
            return None

    return Logger()
