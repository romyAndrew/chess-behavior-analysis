from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("chess.pgn")

from src.config import load_settings
from src.features import build_features_from_csv
from src.model import fit_model_suite, save_model_results, walk_forward_validate, save_walk_forward_results
from src.parse import parse_pgn_file
from src.stats import (
    analyze_tilt_proxy,
    analyze_tilt_sensitivity,
    analyze_tilt_sensitivity_session_bootstrap,
    compare_tilt_bootstrap_methods,
    render_summary,
    save_stats,
    save_tilt_bootstrap_comparison,
    save_tilt_sensitivity,
)
from src.viz import generate_all


class _Logger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


def test_pipeline_stages_create_consistent_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    base = load_settings(project_root / "config/config.yaml")
    paths = base.paths.model_copy(
        update={
            "raw_pgn": project_root / "tests/fixtures/sample_pipeline.pgn",
            "games_csv": tmp_path / "games.csv",
            "features_csv": tmp_path / "games_features.csv",
            "stats_json": tmp_path / "results/tilt_test.json",
            "model_metrics_json": tmp_path / "results/model_metrics.json",
            "model_coefficients_csv": tmp_path / "results/model_coefficients.csv",
            "model_comparison_csv": tmp_path / "results/model_comparison.csv",
            "analysis_summary_md": tmp_path / "results/analysis_summary.md",
            "tilt_sensitivity_csv": tmp_path / "results/tilt_sensitivity.csv",
            "tilt_sensitivity_md": tmp_path / "results/tilt_sensitivity.md",
            "tilt_bootstrap_comparison_csv": tmp_path / "results/tilt_bootstrap_comparison.csv",
            "tilt_bootstrap_comparison_md": tmp_path / "results/tilt_bootstrap_comparison.md",
            "tilt_sensitivity_session_bootstrap_csv": tmp_path / "results/tilt_sensitivity_session_bootstrap.csv",
            "walk_forward_metrics_csv": tmp_path / "results/walk_forward_metrics.csv",
            "walk_forward_fold_metrics_csv": tmp_path / "results/walk_forward_fold_metrics.csv",
            "walk_forward_metrics_md": tmp_path / "results/walk_forward_metrics.md",
            "log_file": tmp_path / "pipeline.log",
            "figures_dir": tmp_path / "figures",
        }
    )
    settings = base.model_copy(
        update={
            "paths": paths,
            "repo_root": project_root,
            "walk_forward": base.walk_forward.model_copy(
                update={"initial_train_size": 30, "test_window_size": 10, "min_training_size": 20}
            ),
        }
    )
    logger = _Logger()

    parsed = parse_pgn_file(
        project_root / "tests/fixtures/sample_pipeline.pgn", settings, logger
    )
    assert len(parsed) == 60

    features = build_features_from_csv(paths.games_csv, settings, logger)
    assert len(features) == 60
    assert settings.project.timezone == "UTC"
    assert "rating_diff" in features.columns
    assert "tilt_proxy" in features.columns
    assert features["session_id"].notna().all()
    assert all(group["created_at"].is_monotonic_increasing for _, group in features.groupby("session_id"))

    stats_result = analyze_tilt_proxy(features, settings, logger)
    save_stats(stats_result, settings.resolve_path(paths.stats_json))
    settings.resolve_path(paths.analysis_summary_md).write_text(
        render_summary(stats_result), encoding="utf-8"
    )

    sensitivity = analyze_tilt_sensitivity(features, settings, logger)
    save_tilt_sensitivity(
        sensitivity,
        settings.resolve_path(paths.tilt_sensitivity_csv),
        settings.resolve_path(paths.tilt_sensitivity_md),
    )
    bootstrap_comparison = compare_tilt_bootstrap_methods(features, stats_result, settings)
    save_tilt_bootstrap_comparison(
        bootstrap_comparison,
        settings.resolve_path(paths.tilt_bootstrap_comparison_csv),
        settings.resolve_path(paths.tilt_bootstrap_comparison_md),
    )
    session_sensitivity = analyze_tilt_sensitivity_session_bootstrap(features, settings, logger)
    session_sensitivity.to_csv(
        settings.resolve_path(paths.tilt_sensitivity_session_bootstrap_csv),
        index=False,
    )

    metrics, comparison, coefficients = fit_model_suite(features, settings, logger)
    save_model_results(metrics, comparison, coefficients, settings)
    walk_forward_aggregate, walk_forward_folds, baseline = walk_forward_validate(features, settings, logger)
    save_walk_forward_results(walk_forward_aggregate, walk_forward_folds, baseline, settings)

    generate_all(
        features,
        stats_result,
        settings.resolve_path(paths.figures_dir),
        settings.project.timezone,
        sensitivity=sensitivity,
        bootstrap_comparison=bootstrap_comparison,
        walk_forward_folds=walk_forward_folds,
    )

    expected_results = {
        settings.resolve_path(paths.stats_json),
        settings.resolve_path(paths.analysis_summary_md),
        settings.resolve_path(paths.model_metrics_json),
        settings.resolve_path(paths.model_coefficients_csv),
        settings.resolve_path(paths.model_comparison_csv),
        settings.resolve_path(paths.tilt_sensitivity_csv),
        settings.resolve_path(paths.tilt_sensitivity_md),
        settings.resolve_path(paths.tilt_bootstrap_comparison_csv),
        settings.resolve_path(paths.tilt_bootstrap_comparison_md),
        settings.resolve_path(paths.tilt_sensitivity_session_bootstrap_csv),
        settings.resolve_path(paths.walk_forward_metrics_csv),
        settings.resolve_path(paths.walk_forward_fold_metrics_csv),
        settings.resolve_path(paths.walk_forward_metrics_md),
    }
    assert all(path.exists() for path in expected_results)

    expected_figures = {
        "win_rate_by_hour.png",
        "break_by_tilt_proxy.png",
        "win_rate_heatmap.png",
        "tilt_loss_rate_ci.png",
        "correlation_heatmap.png",
        "tilt_sensitivity.png",
        "tilt_bootstrap_comparison.png",
        "walk_forward_roc_auc.png",
    }
    actual_figures = {path.name for path in settings.resolve_path(paths.figures_dir).glob("*.png")}
    assert actual_figures == expected_figures

    comparison_loaded = pd.read_csv(settings.resolve_path(paths.model_comparison_csv))
    assert set(comparison_loaded["model"]) == {"rating_only", "baseline", "behavioral"}

    sensitivity_loaded = pd.read_csv(settings.resolve_path(paths.tilt_sensitivity_csv))
    assert len(sensitivity_loaded) == 6
    assert sensitivity_loaded.loc[sensitivity_loaded["is_baseline"], "threshold_description"].tolist() == [
        "break <= 5 min, streak >= 2"
    ]

    bootstrap_loaded = pd.read_csv(settings.resolve_path(paths.tilt_bootstrap_comparison_csv))
    assert bootstrap_loaded["method"].tolist() == ["row_level_bootstrap", "session_bootstrap"]
    assert bootstrap_loaded["observed_difference_pp"].nunique() == 1

    session_sensitivity_loaded = pd.read_csv(
        settings.resolve_path(paths.tilt_sensitivity_session_bootstrap_csv)
    )
    assert len(session_sensitivity_loaded) == 6

    walk_forward_loaded = pd.read_csv(settings.resolve_path(paths.walk_forward_metrics_csv))
    assert set(walk_forward_loaded["model"]) == {"rating_only", "baseline", "behavioral"}
    folds_loaded = pd.read_csv(settings.resolve_path(paths.walk_forward_fold_metrics_csv))
    assert len(folds_loaded) == 9
    assert folds_loaded["train_end_idx"].groupby(folds_loaded["model"]).nunique().eq(3).all()
