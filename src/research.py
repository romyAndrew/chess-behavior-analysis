"""Research-population and observation-window helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from .config import Settings


def research_cutoff(settings: Settings) -> pd.Timestamp:
    """Return the fixed research cutoff as a UTC timestamp."""
    return pd.Timestamp(datetime.fromisoformat(settings.research.cutoff_datetime.replace("Z", "+00:00")))


def filter_research_games(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Keep only analyzable games in the configured research population."""
    if df.empty:
        return df.copy()

    out = df.copy()
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True)
    time_control = out["time_control"].fillna("").astype(str).str.strip()
    mask = time_control.eq(settings.research.time_control)
    mask &= out["created_at"].le(research_cutoff(settings))
    return out.loc[mask].sort_values("created_at").drop_duplicates("game_id", keep="first").reset_index(drop=True)


def select_latest_decisive_window(
    df: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Select the latest N decisive games and retain every analyzable game in the window."""
    filtered = filter_research_games(df, settings)
    if filtered.empty:
        return filtered, None

    decisive = filtered.loc[filtered["user_result"].isin(["win", "loss"])].copy()
    minimum = int(settings.sampling.min_decisive_games)
    available = int(len(decisive))
    if available < minimum:
        return filtered.iloc[0:0].copy(), {
            "available_decisive_games_before_cutoff": available,
            "selected_decisive_games": 0,
        }

    selected_decisive = decisive.tail(minimum).copy()
    window_start = pd.Timestamp(selected_decisive["created_at"].min())
    window_end = research_cutoff(settings)
    window = filtered.loc[filtered["created_at"].between(window_start, window_end, inclusive="both")].copy()
    window = window.sort_values("created_at").reset_index(drop=True)

    decisive_in_window = int(window["user_result"].isin(["win", "loss"]).sum())
    draws_in_window = int(window["user_result"].eq("draw").sum())
    summary = {
        "available_decisive_games_before_cutoff": available,
        "selected_decisive_games": minimum,
        "analysis_window_start": window_start.isoformat(),
        "analysis_window_end": window_end.isoformat(),
        "total_games_in_analysis_window": int(len(window)),
        "decisive_games_in_analysis_window": decisive_in_window,
        "draws_in_analysis_window": draws_in_window,
        "window_duration_days": float((window_end - window_start).total_seconds() / 86400.0),
    }
    return window, summary
