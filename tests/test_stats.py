import pandas as pd

from src.config import load_settings
from src.stats import (
    analyze_tilt_proxy,
    analyze_tilt_sensitivity,
    compare_tilt_bootstrap_methods,
    _session_bootstrap_difference,
    build_tilt_sensitivity_configs,
    render_tilt_sensitivity_summary,
)


def test_tilt_stats_returns_effect_and_ci() -> None:
    settings = load_settings("config/config.yaml")
    rows = []
    for i in range(30):
        rows.append(
            {
                "is_decisive": True,
                "tilt_proxy": i < 10,
                "is_loss": i < 7 if i < 10 else i % 4 == 0,
                "mean_eval_loss": 0.8 if i < 10 else 0.4,
                "eval_coverage": 0.8,
            }
        )
    result = analyze_tilt_proxy(pd.DataFrame(rows), settings, logging_getter())
    assert "odds_ratio" in result
    assert len(result["odds_ratio_ci"]) == 2
    assert 0 <= result["loss_rate_tilt_proxy"] <= 1


def logging_getter():
    class Logger:
        def info(self, *args, **kwargs):
            pass
        def warning(self, *args, **kwargs):
            pass
    return Logger()


def test_sensitivity_configurations_include_baseline_and_requested_alternatives() -> None:
    settings = load_settings("config/config.yaml")
    configs = build_tilt_sensitivity_configs(settings)

    assert len(configs) == 6
    assert [
        (item["break_threshold_minutes"], item["min_loss_streak"])
        for item in configs
    ] == [(2.0, 2), (5.0, 2), (10.0, 2), (15.0, 2), (5.0, 3), (5.0, 4)]
    assert sum(bool(item["is_baseline"]) for item in configs) == 1
    assert configs[1]["is_baseline"] is True


def test_tilt_sensitivity_uses_each_configuration_independently() -> None:
    settings = load_settings("config/config.yaml")
    rows = []
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    breaks = [1, 3, 8, 20]
    streaks = [2, 2, 3, 4]

    for idx in range(4):
        rows.append(
            {
                "game_id": f"g{idx}",
                "created_at": base + pd.Timedelta(minutes=idx * 30),
                "username": "bat1skaf",
                "user_result": "loss" if idx % 2 == 0 else "win",
                "is_win": int(idx % 2 == 1),
                "is_loss": int(idx % 2 == 0),
                "is_draw": 0,
                "is_decisive": True,
                "result_points": float(idx % 2 == 1),
                "previous_loss": True,
                "break_after_previous": float(breaks[idx]),
                "loss_streak_before": streaks[idx],
            }
        )

    result = analyze_tilt_sensitivity(pd.DataFrame(rows), settings, logging_getter())

    assert len(result) == 6
    assert result["configuration"].tolist() == [
        "break_2m_streak_2",
        "break_5m_streak_2",
        "break_10m_streak_2",
        "break_15m_streak_2",
        "break_5m_streak_3",
        "break_5m_streak_4",
    ]
    assert result.loc[0, "n_tilt_observations"] != result.loc[2, "n_tilt_observations"]


def test_sensitivity_marks_stricter_streak_groups_as_smaller_than_baseline() -> None:
    settings = load_settings("config/config.yaml")
    rows = []

    # 50 baseline tilt observations: 26 at streak 2, 12 at streak 3, 12 at streak 4.
    # 50 controls keep the comparison group non-empty. The existing configured
    # min_group_size is 10, so the stricter groups are flagged because they are
    # substantially smaller than the unchanged baseline, not because of a new
    # arbitrary sample-size cutoff.
    for idx in range(100):
        is_tilt = idx < 50
        if is_tilt:
            streak = 2 if idx < 26 else 3 if idx < 38 else 4
            break_minutes = 1.0
        else:
            streak = 2
            break_minutes = 20.0

        is_loss = idx % 2 == 0
        rows.append(
            {
                "game_id": f"g{idx}",
                "created_at": pd.Timestamp("2026-01-01T12:00:00Z") + pd.Timedelta(minutes=idx),
                "username": "bat1skaf",
                "user_result": "loss" if is_loss else "win",
                "is_win": int(not is_loss),
                "is_loss": int(is_loss),
                "is_draw": 0,
                "is_decisive": True,
                "result_points": float(not is_loss),
                "previous_loss": True,
                "break_after_previous": break_minutes,
                "loss_streak_before": streak,
            }
        )

    result = analyze_tilt_sensitivity(pd.DataFrame(rows), settings, logging_getter())

    baseline = result.loc[result["is_baseline"]].iloc[0]
    streak_3 = result.loc[result["configuration"] == "break_5m_streak_3"].iloc[0]
    streak_4 = result.loc[result["configuration"] == "break_5m_streak_4"].iloc[0]

    assert int(baseline["n_tilt_observations"]) == 50
    assert int(streak_3["n_tilt_observations"]) == 24
    assert int(streak_4["n_tilt_observations"]) == 12
    assert bool(streak_3["small_sample"]) is True
    assert bool(streak_4["small_sample"]) is True
    assert "Smaller tilt group than baseline (24 vs 50 observations)" in streak_3["stability_note"]
    assert "Smaller tilt group than baseline (12 vs 50 observations)" in streak_4["stability_note"]

    summary = render_tilt_sensitivity_summary(result)
    assert "break <= 5 min, streak >= 3 (24 vs 50 baseline observations)" in summary
    assert "break <= 5 min, streak >= 4 (12 vs 50 baseline observations)" in summary
    assert "break <= 5 min, streak >= 2 (baseline)" in summary
    assert "| break <= 2 min, streak >= 2 | 50 | 50 |" in summary


def _session_fixture() -> pd.DataFrame:
    rows = []
    # Session 1: tilt losses and control wins.
    for idx in range(4):
        rows.append(
            {
                "game_id": f"s1_{idx}",
                "session_id": 1,
                "is_decisive": True,
                "tilt_proxy": idx < 2,
                "is_loss": idx < 2,
            }
        )
    # Session 2: tilt wins and control losses.
    for idx in range(4):
        rows.append(
            {
                "game_id": f"s2_{idx}",
                "session_id": 2,
                "is_decisive": True,
                "tilt_proxy": idx < 2,
                "is_loss": idx >= 2,
            }
        )
    return pd.DataFrame(rows)


def test_session_bootstrap_is_reproducible_and_keeps_sessions_together() -> None:
    df = _session_fixture()

    first = _session_bootstrap_difference(df, iterations=200, confidence=0.95, seed=42)
    second = _session_bootstrap_difference(df, iterations=200, confidence=0.95, seed=42)

    assert first == second
    assert first[2] == 2
    # With two complete sessions, a whole-session resample can only yield
    # session-specific differences of -1, 0, or +1. Mixed row resampling would
    # produce intermediate values as well.
    assert -1.0 <= first[0] <= 1.0
    assert -1.0 <= first[1] <= 1.0


def test_bootstrap_comparison_keeps_point_estimate_and_config_values() -> None:
    settings = load_settings("config/config.yaml")
    settings = settings.model_copy(
        update={
            "analysis": settings.analysis.model_copy(update={"min_group_size": 2})
        }
    )
    df = _session_fixture()
    result = analyze_tilt_proxy(df, settings, logging_getter())

    comparison = compare_tilt_bootstrap_methods(df, result, settings)

    assert comparison["method"].tolist() == ["row_level_bootstrap", "session_bootstrap"]
    assert comparison["observed_difference_pp"].nunique() == 1
    assert comparison.loc[0, "observed_difference_pp"] == comparison.loc[1, "observed_difference_pp"]
    assert comparison.loc[0, "ci_low"] == 100.0 * result["risk_difference_ci"][0]
    assert comparison.loc[0, "ci_high"] == 100.0 * result["risk_difference_ci"][1]
    assert comparison["bootstrap_iterations"].tolist() == [
        settings.analysis.bootstrap_iterations,
        settings.analysis.bootstrap_iterations,
    ]
    assert comparison["random_seed"].tolist() == [settings.analysis.bootstrap_seed] * 2
    assert comparison["n_sessions"].tolist() == [2, 2]
