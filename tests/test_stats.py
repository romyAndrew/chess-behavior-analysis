import pandas as pd

from src.config import load_settings
from src.stats import _session_bootstrap_difference, analyze_tilt_proxy


def _logger():
    class Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    return Logger()


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
    result = analyze_tilt_proxy(pd.DataFrame(rows), settings, _logger())
    assert "odds_ratio" in result
    assert len(result["odds_ratio_ci"]) == 2
    assert 0 <= result["loss_rate_tilt_proxy"] <= 1


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
    assert -1.0 <= first[0] <= 1.0
    assert -1.0 <= first[1] <= 1.0
