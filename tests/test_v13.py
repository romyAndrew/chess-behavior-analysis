from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_settings
from src.v13 import (
    elo_win_probability,
    expanding_player_calibration,
    fit_opponent_adjusted_models,
    recompute_synthetic_tilt_proxy,
    summarize_null_distribution,
    validate_real_baseline,
)


def _fixture() -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    for player in ["p1", "p2"]:
        previous = []
        for i in range(20):
            if i in {0, 1, 2, 8, 9, 10, 15, 16}:
                result = "loss"
            elif i % 7 == 0:
                result = "draw"
            else:
                result = "win"
            previous_result = previous[-1] if previous else None
            prior_streak = 0
            for prior in reversed(previous):
                if prior == "loss":
                    prior_streak += 1
                else:
                    break
            rows.append(
                {
                    "game_id": f"{player}_{i}",
                    "created_at": base + pd.Timedelta(minutes=10 * i),
                    "player_id": player,
                    "user_result": result,
                    "is_decisive": result in {"win", "loss"},
                    "is_loss": int(result == "loss"),
                    "rating_diff": 0.0 if i % 2 else 100.0,
                    "user_color": "white" if i % 2 else "black",
                    "white_player": player if i % 2 == 0 else f"opp{i % 3}",
                    "black_player": f"opp{i % 3}" if i % 2 == 0 else player,
                    "opponent_rating": 1500.0 + 20 * (i % 3),
                    "break_after_previous": 3.0 if i % 5 != 0 else 20.0,
                    "previous_result": previous_result,
                    "tilt_proxy": bool(
                        previous_result == "loss"
                        and (3.0 if i % 5 != 0 else 20.0) <= 5
                        and prior_streak >= 2
                    ),
                }
            )
            previous.append(result)
    return pd.DataFrame(rows)


def _model_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    players = ["p1", "p2", "p3", "p4"]
    for player_idx, player in enumerate(players):
        prev_opponent = None
        for i in range(80):
            rating_diff = float(rng.normal(0, 120))
            color_black = int(i % 2 == 0)
            tilt = int(i % 7 in {0, 1})
            same_opponent = int(i > 0 and i % 11 == 0)
            p_loss = 1 / (1 + np.exp(0.008 * rating_diff))
            p_loss = np.clip(0.25 + 0.12 * tilt + 0.05 * color_black + (p_loss - 0.5), 0.02, 0.98)
            loss = int(rng.random() < p_loss)
            opponent = prev_opponent if same_opponent and prev_opponent is not None else f"opp_{(i + player_idx) % 9}"
            rows.append(
                {
                    "game_id": f"{player}_{i}",
                    "created_at": base + pd.Timedelta(minutes=10 * (i + 100 * player_idx)),
                    "player_id": player,
                    "user_result": "loss" if loss else "win",
                    "is_decisive": True,
                    "is_loss": loss,
                    "rating_diff": rating_diff,
                    "user_color": "black" if color_black else "white",
                    "white_player": player if not color_black else opponent,
                    "black_player": opponent if not color_black else player,
                    "opponent_rating": 1500 - rating_diff,
                    "break_after_previous": 3.0 if tilt else 20.0,
                    "previous_result": "loss" if tilt else "win",
                    "tilt_proxy": bool(tilt),
                    "opponent_name": opponent,
                    "previous_opponent_name": prev_opponent,
                    "same_opponent": same_opponent,
                    "loss": loss,
                    "tilt_proxy_int": tilt,
                    "color_black": color_black,
                }
            )
            prev_opponent = opponent
    return pd.DataFrame(rows)


def _logger():
    class Logger:
        def info(self, *args, **kwargs):
            return None

    return Logger()


def test_elo_probability_matches_standard_formula() -> None:
    values = elo_win_probability(np.array([-400.0, 0.0, 400.0]))
    assert np.allclose(values, [0.0909090909, 0.5, 0.9090909091], atol=1e-9)


def test_real_baseline_reproduces_fixture_from_same_columns() -> None:
    settings = load_settings("config/config.yaml")
    result = validate_real_baseline(_fixture(), settings)
    assert result["players"] == 2
    assert result["total_games"] == 40
    assert result["decisive_games"] + result["draws"] == 40
    assert result["tilt_observations"] >= 0


def test_synthetic_tilt_proxy_is_recomputed_from_new_outcomes() -> None:
    settings = load_settings("config/config.yaml")
    df = _fixture()
    synthetic_loss = df["is_loss"].to_numpy(dtype=bool)
    rebuilt = recompute_synthetic_tilt_proxy(
        df,
        synthetic_loss,
        settings.features.short_break_minutes,
        settings.features.min_loss_streak,
    )
    expected = df["tilt_proxy"].to_numpy(dtype=bool)
    assert np.array_equal(rebuilt, expected)


def test_player_calibration_uses_only_prior_games() -> None:
    df = _fixture()
    p = np.full(len(df), 0.5)
    calibrated = expanding_player_calibration(df, p)
    assert calibrated[0] == 0.5
    # Changing a later outcome must not change earlier probabilities.
    changed = df.copy()
    changed.loc[changed.index[-1], "is_loss"] = 1 - changed.loc[changed.index[-1], "is_loss"]
    changed_calibrated = expanding_player_calibration(changed, p)
    assert np.allclose(calibrated[:-1], changed_calibrated[:-1])


def test_opponent_adjusted_models_return_clustered_tilt_rows() -> None:
    data = _model_fixture()
    summary, coefficients = fit_opponent_adjusted_models(data, _logger())
    assert set(summary["model"]) == {
        "A_unadjusted",
        "B_opponent_player_adjusted",
        "C_same_opponent_adjusted",
    }
    assert summary["term"].eq("tilt_proxy").all()
    assert summary["player_clusters"].eq(4).all()
    assert not coefficients.empty


def test_null_summary_uses_simulation_interval_and_empirical_tail() -> None:
    settings = load_settings("config/config.yaml")
    distribution = pd.DataFrame(
        {
            "scenario": ["raw_elo"] * 4,
            "simulation_id": [1, 2, 3, 4],
            "tilt_n": [10, 11, 12, 13],
            "control_n": [20, 19, 18, 17],
            "difference_pp": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = summarize_null_distribution(distribution, 3.0, settings, "raw")
    row = result.iloc[0]
    assert row["simulation_interval_low_pp"] == 1.075
    assert row["simulation_interval_high_pp"] == 3.925
    assert row["empirical_upper_tail_proportion"] == 0.5
    assert "confidence" not in "simulation_interval_low_pp"
