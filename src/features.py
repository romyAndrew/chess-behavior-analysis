"""Behavioral feature engineering."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings


def _streaks(results: pd.Series, target: str) -> pd.Series:
    """Return current streak length for the requested result label."""
    streak = 0
    values: list[int] = []
    for result in results:
        streak = streak + 1 if result == target else 0
        values.append(streak)
    return pd.Series(values, index=results.index, dtype="int64")


def _player_column(df: pd.DataFrame) -> str:
    """Return the sequential-group key without changing single-player output semantics."""
    return "player_id" if "player_id" in df.columns else "username"


def _add_time_features(out: pd.DataFrame, timezone_name: str) -> None:
    """Add calendar and cyclic time features in the configured project timezone."""
    local_time = out["created_at"].dt.tz_convert(timezone_name)
    out["hour"] = local_time.dt.hour.astype(int)
    out["day_of_week"] = local_time.dt.dayofweek.astype(int)
    out["date"] = local_time.dt.date.astype(str)
    hour_angle = 2 * np.pi * out["hour"] / 24.0
    out["hour_sin"] = np.sin(hour_angle)
    out["hour_cos"] = np.cos(hour_angle)


def _add_result_features(out: pd.DataFrame) -> None:
    """Add current and previous result indicators within each player."""
    player_col = _player_column(out)
    out["is_win"] = (out["user_result"] == "win").astype(int)
    out["is_loss"] = (out["user_result"] == "loss").astype(int)
    out["is_draw"] = (out["user_result"] == "draw").astype(int)
    out["is_decisive"] = out["user_result"].isin(["win", "loss"])
    out["result_points"] = out["user_result"].map({"win": 1.0, "draw": 0.5, "loss": 0.0})

    out["previous_result"] = out.groupby(player_col, sort=False)["user_result"].shift(1)
    out["previous_loss"] = out["previous_result"].eq("loss")
    out["previous_win"] = out["previous_result"].eq("win")


def _add_session_features(out: pd.DataFrame, settings: Settings) -> None:
    """Add inter-game breaks and session information separately for each player."""
    player_col = _player_column(out)
    previous_time = out.groupby(player_col, sort=False)["created_at"].shift(1)
    out["break_after_previous"] = (out["created_at"] - previous_time).dt.total_seconds().div(60)
    first_rows = ~out[player_col].duplicated()
    out.loc[first_rows, "break_after_previous"] = np.nan

    grouped_results = out.groupby(player_col, sort=False)["user_result"]
    out["loss_streak"] = grouped_results.transform(lambda values: _streaks(values, "loss"))
    out["win_streak"] = grouped_results.transform(lambda values: _streaks(values, "win"))
    out["loss_streak_before"] = out.groupby(player_col, sort=False)["loss_streak"].shift(1).fillna(0).astype(int)
    out["win_streak_before"] = out.groupby(player_col, sort=False)["win_streak"].shift(1).fillna(0).astype(int)

    session_break = out["break_after_previous"].isna() | (
        out["break_after_previous"] > settings.features.session_gap_minutes
    )
    local_session = session_break.groupby(out[player_col], sort=False).cumsum().astype(int)
    out["session_game_number"] = (
        out.groupby([player_col, local_session], sort=False).cumcount() + 1
    )
    out["session_length"] = (
        out.groupby([player_col, local_session], sort=False)["game_id"].transform("count")
    )

    if player_col == "player_id":
        session_key = out[player_col].astype(str) + "::" + local_session.astype(str)
        out["session_id"] = pd.factorize(session_key, sort=True)[0].astype(int) + 1
    else:
        out["session_id"] = local_session


def build_tilt_proxy_mask(
    df: pd.DataFrame,
    break_threshold_minutes: float,
    min_loss_streak: int,
) -> pd.Series:
    """Build a tilt-proxy mask from the existing behavioral features.

    The rule remains: previous game was a loss, the break was no longer than
    the configured threshold, and the previous loss streak reached the
    configured minimum.
    """
    return (
        df["previous_loss"].astype(bool)
        & df["break_after_previous"].le(break_threshold_minutes)
        & df["loss_streak_before"].ge(min_loss_streak)
    )


def _add_tilt_proxy(out: pd.DataFrame, settings: Settings) -> None:
    """Build the baseline behavioral proxy from the project configuration."""
    out["tilt_proxy"] = build_tilt_proxy_mask(
        out,
        settings.features.short_break_minutes,
        settings.features.min_loss_streak,
    )
    out["tilt_proxy_reason"] = np.select(
        [out["tilt_proxy"]],
        ["previous_loss + short_break + required_loss_streak"],
        default="not_triggered",
    )


def add_behavioral_features(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Create temporal, session, streak, and tilt-proxy features."""
    if df.empty:
        return df.copy()

    player_col = _player_column(df)
    out = df.copy()
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True)
    duplicate_subset = ["game_id"] if player_col == "username" else ["player_id", "game_id"]
    out = (
        out.sort_values([player_col, "created_at"])
        .drop_duplicates(duplicate_subset, keep="first")
        .reset_index(drop=True)
    )

    _add_time_features(out, settings.project.timezone)
    _add_result_features(out)
    _add_session_features(out, settings)
    _add_tilt_proxy(out, settings)

    # Current-game outcome is the target for the pre-game proxy.
    # Post-move features such as time_pressure are intentionally excluded from
    # the predictive model to avoid target leakage.
    return out


def build_features_from_csv(path: Path, settings: Settings, logger: logging.Logger) -> pd.DataFrame:
    """Load parsed games, add features, and persist the result."""
    df = pd.read_csv(path, parse_dates=["created_at"])
    result = add_behavioral_features(df, settings)
    logger.info("Feature engineering produced %d rows", len(result))
    result.to_csv(settings.resolve_path(settings.paths.features_csv), index=False)
    return result
