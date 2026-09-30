"""Player-level heterogeneity analysis for the fixed multi-player research sample."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .stats import _session_bootstrap_difference, analyze_tilt_proxy


def analyze_player_heterogeneity(
    df: pd.DataFrame,
    settings: Settings,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Estimate the baseline tilt association separately for each player.

    The observed point estimate, odds ratio and p-value reuse the project's
    existing statistical logic. Difference confidence intervals use the
    existing session-aware bootstrap so games within sessions remain grouped.
    No player is removed because of a small tilt group.
    """
    required = {
        "player_id",
        "session_id",
        "is_decisive",
        "tilt_proxy",
        "is_loss",
        "created_at",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Player heterogeneity analysis requires columns: {sorted(missing)}"
        )

    working = df.copy()
    working["created_at"] = pd.to_datetime(working["created_at"], utc=True)
    working = working.sort_values(["player_id", "created_at"]).reset_index(drop=True)

    selection_path = settings.resolve_path(settings.paths.sample_players_csv)
    if not selection_path.exists():
        raise FileNotFoundError(
            f"Selected player file not found: {selection_path}. Run the sampling stage first."
        )
    selection = pd.read_csv(selection_path)
    selection = selection.loc[selection["selected"].astype(bool), ["player_id", "selection_order"]]
    selection["player_id"] = selection["player_id"].astype(str)

    rows: list[dict[str, Any]] = []
    cutoff = pd.Timestamp(settings.research.cutoff_datetime)

    for player_id, group in working.groupby("player_id", sort=False):
        player_id = str(player_id)
        decisive = group.loc[
            group["is_decisive"].astype(bool) & group["tilt_proxy"].notna()
        ].copy()
        tilt = decisive.loc[decisive["tilt_proxy"].astype(bool), "is_loss"].astype(int).to_numpy()
        control = decisive.loc[~decisive["tilt_proxy"].astype(bool), "is_loss"].astype(int).to_numpy()

        tilt_n = int(len(tilt))
        control_n = int(len(control))
        decisive_n = tilt_n + control_n
        total_games = int(len(group))
        sessions = int(group["session_id"].nunique())
        selection_row = selection.loc[selection["player_id"].eq(player_id)]
        selection_order = (
            int(selection_row.iloc[0]["selection_order"])
            if not selection_row.empty
            else np.nan
        )

        difference_pp = (
            float((tilt.mean() - control.mean()) * 100.0)
            if tilt_n and control_n
            else np.nan
        )

        result: dict[str, Any] | None = None
        session_ci_low_pp = np.nan
        session_ci_high_pp = np.nan
        n_bootstrap_sessions = np.nan
        p_value = np.nan
        odds_ratio = np.nan
        odds_ratio_ci_low = np.nan
        odds_ratio_ci_high = np.nan
        statistical_test = "not estimable"
        inference_status = "not_estimable_empty_comparison_group"
        uncertainty_method = "not_estimable"
        uncertainty_note = ""

        if tilt_n and control_n:
            # Reuse the existing test logic, but do not introduce a new player-level
            # exclusion cutoff. The existing method is the same z-test/Fisher + OR CI.
            result = analyze_tilt_proxy(
                group,
                settings,
                logger,
                min_group_size_override=1,
            )
            p_value = float(result["p_value"])
            odds_ratio = float(result["odds_ratio"])
            odds_ratio_ci_low, odds_ratio_ci_high = map(float, result["odds_ratio_ci"])
            statistical_test = str(result["test"])
            inference_status = "estimable"

            try:
                session_low, session_high, n_bootstrap_sessions = _session_bootstrap_difference(
                    group,
                    settings.analysis.bootstrap_iterations,
                    settings.analysis.confidence_level,
                    settings.analysis.bootstrap_seed,
                )
                session_ci_low_pp = 100.0 * session_low
                session_ci_high_pp = 100.0 * session_high
                uncertainty_method = "session_bootstrap"
            except ValueError as exc:
                uncertainty_note = str(exc)
                inference_status = "point_estimate_and_test_estimable_session_ci_not_estimable"
        elif not tilt_n:
            uncertainty_note = "No tilt-proxy observations for this player; comparison statistics are not estimable."
        else:
            uncertainty_note = "No control observations for this player; comparison statistics are not estimable."

        small_tilt_group = tilt_n < settings.analysis.min_group_size

        rows.append(
            {
                "player_id": player_id,
                "selection_order": selection_order,
                "total_games": total_games,
                "decisive_games": decisive_n,
                "sessions": sessions,
                "tilt_n": tilt_n,
                "control_n": control_n,
                "tilt_loss_rate": float(tilt.mean()) if tilt_n else np.nan,
                "control_loss_rate": float(control.mean()) if control_n else np.nan,
                "difference_pp": difference_pp,
                "difference_ci_low": session_ci_low_pp,
                "difference_ci_high": session_ci_high_pp,
                "odds_ratio": odds_ratio,
                "odds_ratio_ci_low": odds_ratio_ci_low,
                "odds_ratio_ci_high": odds_ratio_ci_high,
                "p_value": p_value,
                "statistical_test": statistical_test,
                "uncertainty_method": uncertainty_method,
                "bootstrap_sessions": n_bootstrap_sessions,
                "small_tilt_group": bool(small_tilt_group),
                "min_group_size_reference": int(settings.analysis.min_group_size),
                "inference_status": inference_status,
                "uncertainty_note": uncertainty_note,
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("Player heterogeneity analysis produced no player records")

    return result_df.sort_values("selection_order").reset_index(drop=True)


def summarize_player_heterogeneity(results: pd.DataFrame) -> dict[str, Any]:
    """Return descriptive pooled-versus-player heterogeneity summaries."""
    effects = results["difference_pp"].dropna().astype(float)
    positive = int((effects > 0).sum())
    negative = int((effects < 0).sum())
    zero = int((effects == 0).sum())
    very_small = results.loc[results["small_tilt_group"].astype(bool), "player_id"].astype(str).tolist()

    return {
        "players": int(len(results)),
        "estimable_effects": int(len(effects)),
        "positive_estimates": positive,
        "negative_estimates": negative,
        "zero_estimates": zero,
        "minimum_difference_pp": float(effects.min()) if not effects.empty else np.nan,
        "maximum_difference_pp": float(effects.max()) if not effects.empty else np.nan,
        "median_difference_pp": float(effects.median()) if not effects.empty else np.nan,
        "mean_difference_pp": float(effects.mean()) if not effects.empty else np.nan,
        "players_below_existing_min_group_size": very_small,
    }


def _render_heterogeneity_section(
    results: pd.DataFrame,
    pooled: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    """Render the player-level heterogeneity section for the multi-player analysis report."""
    pooled_row = pooled.iloc[0]
    small_groups = summary["players_below_existing_min_group_size"]
    lines = [
        "## Player-level heterogeneity",
        "",
        "### Method",
        "",
        "The baseline tilt association is estimated separately for each selected player using the predefined baseline tilt proxy: previous loss, break <= 5 minutes, and previous loss streak >= 2. The observed risk difference, odds ratio and p-value reuse the project's existing statistical logic. Difference confidence intervals use the existing session-aware bootstrap with the configured bootstrap iterations and seed.",
        "",
        "No player is removed because of a small tilt group. Groups below the project's existing `min_group_size` are retained and explicitly flagged as having greater statistical uncertainty.",
        "",
        "### Results",
        "",
        f"Pooled descriptive association: **{float(pooled_row['difference_pp']):.2f} percentage points**.",
        f"Player-level point estimates range from **{summary['minimum_difference_pp']:.2f}** to **{summary['maximum_difference_pp']:.2f} pp**; mean = **{summary['mean_difference_pp']:.2f} pp**, median = **{summary['median_difference_pp']:.2f} pp**.",
        f"Positive estimates: **{summary['positive_estimates']}**; negative estimates: **{summary['negative_estimates']}**; zero estimates: **{summary['zero_estimates']}**.",
        "",
        "The pooled association is not treated as a typical player-specific association. The player-specific estimates describe variation across the sampled players and retain their individual uncertainty.",
        "",
        "### Small tilt groups",
        "",
    ]
    if small_groups:
        lines.append(
            "Players below the existing configured `min_group_size` reference: "
            + ", ".join(small_groups)
            + ". They remain in the analysis and are not filtered out."
        )
    else:
        lines.append("No player has fewer tilt observations than the existing configured `min_group_size` reference.")
    lines.extend(
        [
            "",
            "### Interpretation",
            "",
            "The estimated association varies across players to the extent shown by the observed point-estimate range and confidence intervals. This is descriptive evidence about heterogeneity in the sampled players, not evidence that any individual player is psychologically more or less prone to tilt.",
            "",
            "The sample represents randomly selected active bullet players from the specified Lichess bullet leaderboard sampling frame who met the predefined data-availability criteria. It should not be interpreted as an average effect for all Lichess users.",
        ]
    )
    return "\n".join(lines) + "\n"


def save_player_heterogeneity(
    results: pd.DataFrame,
    pooled: pd.DataFrame,
    settings: Settings,
) -> dict[str, Any]:
    """Persist the player-level heterogeneity analysis artifacts."""
    result_path = settings.resolve_path(settings.paths.player_heterogeneity_csv)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(result_path, index=False)

    summary = summarize_player_heterogeneity(results)
    summary_path = settings.resolve_path(settings.paths.player_heterogeneity_summary_json)
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    report_path = settings.resolve_path(settings.paths.multi_player_analysis_md)
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    marker = "\n## Player-level heterogeneity\n"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    report_path.write_text(
        existing + _render_heterogeneity_section(results, pooled, summary),
        encoding="utf-8",
    )
    return summary

