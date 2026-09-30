from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_settings
from src.features import add_behavioral_features
from src.model import fit_model_suite, save_model_results, walk_forward_validate, save_walk_forward_results
from src.sampling import analyze_multi_player
from src.viz import plot_pooled_tilt_effect, plot_walk_forward_roc_auc


class _Logger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


def _raw_fixture(n: int = 120) -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    rows = []
    for i in range(n):
        # Repeated loss clusters create a few tilt-proxy observations after short breaks.
        result = "loss" if i % 7 in {0, 1, 2} else ("draw" if i % 17 == 0 else "win")
        rows.append(
            {
                "game_id": f"g{i}",
                "created_at": base + pd.Timedelta(minutes=3 * i + (60 if i % 30 == 0 and i else 0)),
                "player_id": "p1",
                "username": "p1",
                "user_result": result,
                "is_win": int(result == "win"),
                "is_loss": int(result == "loss"),
                "is_draw": int(result == "draw"),
                "is_decisive": result in {"win", "loss"},
                "result_points": {"win": 1.0, "draw": 0.5, "loss": 0.0}[result],
                "rating_diff": float(((i * 37) % 240) - 120),
                "speed": "bullet",
                "user_color": "black" if i % 2 else "white",
            }
        )
    return pd.DataFrame(rows)


def test_final_analysis_stages_create_retained_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    base = load_settings(project_root / "config/config.yaml")
    paths = base.paths.model_copy(update={
        "multi_features_csv": tmp_path / "games_features.csv",
        "multi_player_pooled_csv": tmp_path / "results/multi_player_pooled_summary.csv",
        "multi_player_data_quality_csv": tmp_path / "results/data_quality.csv",
        "multi_model_metrics_json": tmp_path / "results/multi_player_model_metrics.json",
        "multi_model_coefficients_csv": tmp_path / "results/multi_player_model_coefficients.csv",
        "multi_model_comparison_csv": tmp_path / "results/multi_player_model_comparison.csv",
        "multi_walk_forward_metrics_csv": tmp_path / "results/multi_player_walk_forward_metrics.csv",
        "multi_walk_forward_fold_metrics_csv": tmp_path / "results/multi_player_walk_forward_fold_metrics.csv",
        "multi_walk_forward_metrics_md": tmp_path / "results/multi_player_walk_forward_metrics.md",
        "player_summary_csv": tmp_path / "results/player_summary.csv",
        "figures_dir": tmp_path / "figures",
        "log_file": tmp_path / "pipeline.log",
    })
    settings = base.model_copy(update={
        "paths": paths,
        "repo_root": project_root,
        "walk_forward": base.walk_forward.model_copy(update={
            "initial_train_size": 50,
            "test_window_size": 10,
            "min_training_size": 30,
        }),
        "analysis": base.analysis.model_copy(update={
            "min_group_size": 2,
            "bootstrap_iterations": 100,
        }),
    })
    settings.ensure_directories()

    raw = _raw_fixture()
    features = add_behavioral_features(raw, settings)
    features.to_csv(paths.multi_features_csv, index=False)

    player_summary, _player_effects, pooled, quality = analyze_multi_player(features, settings, _Logger())
    assert int(pooled.loc[0, "games"]) == len(features)
    pooled.to_csv(paths.multi_player_pooled_csv, index=False)
    player_summary.to_csv(paths.player_summary_csv, index=False)
    quality.to_csv(paths.multi_player_data_quality_csv, index=False)

    metrics, comparison, coefficients = fit_model_suite(features, settings, _Logger())
    save_model_results(
        metrics,
        comparison,
        coefficients,
        settings,
        metrics_path=paths.multi_model_metrics_json,
        comparison_path=paths.multi_model_comparison_csv,
        coefficients_path=paths.multi_model_coefficients_csv,
    )
    aggregate, folds, baseline = walk_forward_validate(features, settings, _Logger())
    save_walk_forward_results(
        aggregate,
        folds,
        baseline,
        settings,
        aggregate_path=paths.multi_walk_forward_metrics_csv,
        folds_path=paths.multi_walk_forward_fold_metrics_csv,
        markdown_path=paths.multi_walk_forward_metrics_md,
    )

    plot_pooled_tilt_effect(pooled, paths.figures_dir / "pooled_tilt_effect.png")
    plot_walk_forward_roc_auc(folds, paths.figures_dir / "walk_forward_roc_auc.png")

    assert paths.multi_player_pooled_csv.exists()
    assert paths.multi_model_metrics_json.exists()
    assert paths.multi_walk_forward_metrics_csv.exists()
    assert {path.name for path in paths.figures_dir.glob("*.png")} == {
        "pooled_tilt_effect.png",
        "walk_forward_roc_auc.png",
    }


def test_multi_pipeline_declares_final_stage_order() -> None:
    from src.pipeline import FINAL_MULTI_STAGES

    assert FINAL_MULTI_STAGES == (
        "sample",
        "collect_multi",
        "parse_multi",
        "features_multi",
        "analyze_multi",
        "heterogeneity",
        "model_multi",
        "v13",
    )
