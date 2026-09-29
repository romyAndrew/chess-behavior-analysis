from pathlib import Path

import pandas as pd

from src.config import load_settings
from src.features import add_behavioral_features, build_tilt_proxy_mask


def test_streaks_sessions_and_tilt_proxy() -> None:
    settings = load_settings(Path("config/config.yaml"))
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    rows = []
    results = ["loss", "loss", "loss", "win", "loss"]
    gaps = [0, 2, 2, 45, 2]
    current = base
    for idx, (result, gap) in enumerate(zip(results, gaps, strict=True)):
        if idx > 0:
            current = current + pd.Timedelta(minutes=gap)
        rows.append(
            {
                "game_id": f"g{idx}",
                "created_at": current,
                "username": "bat1skaf",
                "user_result": result,
                "is_win": int(result == "win"),
                "is_loss": int(result == "loss"),
                "is_draw": int(result == "draw"),
                "is_decisive": result in {"win", "loss"},
                "result_points": {"win": 1.0, "loss": 0.0, "draw": 0.5}[result],
            }
        )
    df = pd.DataFrame(rows)
    result = add_behavioral_features(df, settings)

    assert result.loc[2, "loss_streak_before"] == 2
    assert bool(result.loc[2, "tilt_proxy"]) is True
    assert bool(result.loc[3, "tilt_proxy"]) is False
    assert result.loc[3, "session_id"] != result.loc[2, "session_id"]


def test_first_game_has_no_previous_break_or_tilt() -> None:
    settings = load_settings(Path("config/config.yaml"))
    df = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "created_at": pd.Timestamp("2026-01-01T12:00:00Z"),
                "username": "bat1skaf",
                "user_result": "win",
                "is_win": 1,
                "is_loss": 0,
                "is_draw": 0,
                "is_decisive": True,
                "result_points": 1.0,
            }
        ]
    )
    result = add_behavioral_features(df, settings)
    assert pd.isna(result.loc[0, "break_after_previous"])
    assert bool(result.loc[0, "tilt_proxy"]) is False


def test_calendar_features_use_configured_timezone() -> None:
    settings = load_settings(Path("config/config.yaml"))
    local_settings = settings.model_copy(
        update={
            "project": settings.project.model_copy(update={"timezone": "Europe/Warsaw"})
        }
    )
    df = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "created_at": pd.Timestamp("2026-01-01T23:30:00Z"),
                "username": "bat1skaf",
                "user_result": "win",
                "is_win": 1,
                "is_loss": 0,
                "is_draw": 0,
                "is_decisive": True,
                "result_points": 1.0,
            }
        ]
    )

    result = add_behavioral_features(df, local_settings)

    assert result.loc[0, "hour"] == 0
    assert result.loc[0, "day_of_week"] == 4
    assert result.loc[0, "date"] == "2026-01-02"


def test_tilt_proxy_mask_respects_break_and_streak_thresholds() -> None:
    df = pd.DataFrame(
        {
            "previous_loss": [True, True, True, False],
            "break_after_previous": [1.0, 6.0, 1.0, 1.0],
            "loss_streak_before": [2, 2, 3, 4],
        }
    )

    baseline = build_tilt_proxy_mask(df, break_threshold_minutes=5, min_loss_streak=2)
    assert baseline.tolist() == [True, False, True, False]

    short_break = build_tilt_proxy_mask(df, break_threshold_minutes=2, min_loss_streak=2)
    assert short_break.tolist() == [True, False, True, False]

    longer_streak = build_tilt_proxy_mask(df, break_threshold_minutes=5, min_loss_streak=3)
    assert longer_streak.tolist() == [False, False, True, False]


def test_multi_player_sequential_features_are_isolated() -> None:
    settings = load_settings(Path("config/config.yaml"))
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    df = pd.DataFrame(
        [
            {"game_id": "a1", "player_id": "p1", "username": "p1", "created_at": base, "user_result": "loss"},
            {"game_id": "b1", "player_id": "p2", "username": "p2", "created_at": base + pd.Timedelta(minutes=1), "user_result": "win"},
            {"game_id": "a2", "player_id": "p1", "username": "p1", "created_at": base + pd.Timedelta(minutes=2), "user_result": "loss"},
            {"game_id": "b2", "player_id": "p2", "username": "p2", "created_at": base + pd.Timedelta(minutes=3), "user_result": "loss"},
            {"game_id": "a3", "player_id": "p1", "username": "p1", "created_at": base + pd.Timedelta(minutes=4), "user_result": "win"},
        ]
    )
    df["is_win"] = (df["user_result"] == "win").astype(int)
    df["is_loss"] = (df["user_result"] == "loss").astype(int)
    df["is_draw"] = 0
    df["is_decisive"] = True
    df["result_points"] = df["user_result"].map({"win": 1.0, "loss": 0.0})

    result = add_behavioral_features(df, settings)
    p2_first = result.loc[result["player_id"] == "p2"].iloc[0]
    p1_second = result.loc[result["player_id"] == "p1"].iloc[1]
    p1_third = result.loc[result["player_id"] == "p1"].iloc[2]

    assert pd.isna(p2_first["previous_result"])
    assert pd.isna(p2_first["break_after_previous"])
    assert p2_first["loss_streak_before"] == 0
    assert p2_first["session_game_number"] == 1

    assert p1_second["previous_result"] == "loss"
    assert p1_second["break_after_previous"] == 2.0
    assert p1_second["loss_streak_before"] == 1
    assert p1_third["previous_result"] == "loss"
    assert p1_third["loss_streak_before"] == 2
    assert bool(p1_third["tilt_proxy"]) is True
    assert result.groupby("player_id")["session_id"].nunique().sum() >= 2
