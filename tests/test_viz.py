from pathlib import Path

import pandas as pd

from src.config import load_settings
from src.stats import analyze_tilt_proxy
from src.viz import generate_all


def test_generate_all_writes_documented_figure_names(tmp_path: Path) -> None:
    settings = load_settings(Path("config/config.yaml"))
    rows = []
    base = pd.Timestamp("2026-01-01T12:00:00Z")

    for idx in range(40):
        tilt = idx < 20
        is_loss = idx % 3 == 0 if tilt else True
        rows.append(
            {
                "game_id": f"g{idx}",
                "created_at": base + pd.Timedelta(hours=idx),
                "username": "bat1skaf",
                "user_result": "loss" if is_loss else "win",
                "is_win": int(not is_loss),
                "is_loss": int(is_loss),
                "is_draw": 0,
                "is_decisive": True,
                "result_points": float(not is_loss),
                "hour": idx % 24,
                "speed": "blitz" if idx % 2 else "rapid",
                "tilt_proxy": tilt,
                "break_after_previous": float(idx + 1),
                "rating_diff": float(idx - 20),
                "loss_streak_before": idx % 4,
                "win_streak_before": (idx + 1) % 4,
                "session_game_number": idx % 8 + 1,
                "avg_move_time": float(idx + 2),
                "time_pressure": float(idx % 5) / 5.0,
            }
        )

    df = pd.DataFrame(rows)
    stats_result = analyze_tilt_proxy(df, settings, _logger())
    sensitivity = pd.DataFrame(
        [
            {
                "threshold_description": "break <= 2 min, streak >= 2",
                "is_baseline": False,
                "difference_pp": 3.0,
                "difference_ci_low_pp": -4.0,
                "difference_ci_high_pp": 10.0,
            },
            {
                "threshold_description": "break <= 5 min, streak >= 2",
                "is_baseline": True,
                "difference_pp": 8.8,
                "difference_ci_low_pp": -5.2,
                "difference_ci_high_pp": 22.9,
            },
        ]
    )
    walk_forward_folds = pd.DataFrame(
        [
            {"fold": 1, "model": "rating_only", "roc_auc": 0.51},
            {"fold": 1, "model": "baseline", "roc_auc": 0.52},
            {"fold": 1, "model": "behavioral", "roc_auc": 0.50},
            {"fold": 2, "model": "rating_only", "roc_auc": 0.49},
            {"fold": 2, "model": "baseline", "roc_auc": 0.53},
            {"fold": 2, "model": "behavioral", "roc_auc": None},
        ]
    )
    bootstrap_comparison = pd.DataFrame(
        [
            {
                "method": "row_level_bootstrap",
                "observed_difference_pp": 8.8,
                "ci_low": -5.2,
                "ci_high": 22.9,
            },
            {
                "method": "session_bootstrap",
                "observed_difference_pp": 8.8,
                "ci_low": -2.0,
                "ci_high": 25.0,
            },
        ]
    )
    generate_all(
        df,
        stats_result,
        tmp_path,
        settings.project.timezone,
        sensitivity=sensitivity,
        bootstrap_comparison=bootstrap_comparison,
        walk_forward_folds=walk_forward_folds,
    )

    expected = {
        "win_rate_by_hour.png",
        "break_by_tilt_proxy.png",
        "win_rate_heatmap.png",
        "tilt_loss_rate_ci.png",
        "correlation_heatmap.png",
        "tilt_sensitivity.png",
        "tilt_bootstrap_comparison.png",
        "walk_forward_roc_auc.png",
    }
    assert {path.name for path in tmp_path.glob("*.png")} == expected


def _logger():
    class Logger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

    return Logger()
