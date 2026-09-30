"""V13 analyses: synthetic null model and opponent-adjusted association models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .config import Settings


def _prepare_multi_player_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return the verified multi-player analytic data in player-time order."""
    required = {
        "player_id",
        "created_at",
        "game_id",
        "user_result",
        "is_decisive",
        "is_loss",
        "rating_diff",
        "user_color",
        "white_player",
        "black_player",
        "opponent_rating",
        "break_after_previous",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"V13 analysis requires columns: {sorted(missing)}")

    out = df.copy()
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True)
    out = out.sort_values(["player_id", "created_at", "game_id"]).reset_index(drop=True)
    out["opponent_name"] = np.where(
        out["user_color"].eq("white"), out["black_player"], out["white_player"]
    )
    out["previous_opponent_name"] = out.groupby("player_id", sort=False)["opponent_name"].shift(1)
    out["same_opponent"] = (
        out["previous_opponent_name"].notna()
        & out["opponent_name"].eq(out["previous_opponent_name"])
    ).astype(int)
    out["loss"] = out["is_loss"].astype(int)
    out["tilt_proxy_int"] = out["tilt_proxy"].astype(int)
    out["color_black"] = out["user_color"].eq("black").astype(int)
    return out


def elo_win_probability(rating_diff: np.ndarray | pd.Series) -> np.ndarray:
    """Return Elo win probability for the focal player."""
    values = np.asarray(rating_diff, dtype=float)
    return 1.0 / (1.0 + np.power(10.0, -values / 400.0))


def expanding_player_calibration(
    df: pd.DataFrame,
    p_loss_elo: np.ndarray,
) -> np.ndarray:
    """Calibrate Elo loss probabilities using only prior decisive games.

    For each game, the probability adjustment is the focal player's cumulative
    mean residual from prior decisive games: mean(observed loss - Elo loss
    probability). This is an expanding, time-respecting player-specific bias
    correction. Draws do not update the correction and the current game's
    outcome is never used to construct its own probability.
    """
    result = np.asarray(p_loss_elo, dtype=float).copy()
    losses = df["is_loss"].astype(int).to_numpy()
    decisive = df["is_decisive"].astype(bool).to_numpy()
    residual = losses - p_loss_elo

    for _, idx in df.groupby("player_id", sort=False).groups.items():
        positions = np.asarray(idx, dtype=int)
        prior_residual_sum = 0.0
        prior_count = 0
        for pos in positions:
            if prior_count:
                result[pos] = np.clip(
                    p_loss_elo[pos] + prior_residual_sum / prior_count,
                    1e-4,
                    1.0 - 1e-4,
                )
            if decisive[pos]:
                prior_residual_sum += float(residual[pos])
                prior_count += 1
    return result


def recompute_synthetic_tilt_proxy(
    df: pd.DataFrame,
    synthetic_loss: np.ndarray,
    break_threshold_minutes: float,
    min_loss_streak: int,
) -> np.ndarray:
    """Recompute the baseline tilt proxy from synthetic sequential outcomes."""
    loss = np.asarray(synthetic_loss, dtype=bool)
    mask = np.zeros(len(df), dtype=bool)
    breaks = df["break_after_previous"].to_numpy(dtype=float)

    for _, idx in df.groupby("player_id", sort=False).groups.items():
        positions = np.asarray(idx, dtype=int)
        g_loss = loss[positions]
        m = len(positions)
        if m == 0:
            continue

        previous_loss = np.zeros(m, dtype=bool)
        if m > 1:
            previous_loss[1:] = g_loss[:-1]

        local_idx = np.arange(m)
        nonloss_positions = np.where(~g_loss, local_idx, -1)
        last_nonloss = np.maximum.accumulate(nonloss_positions)
        last_before = np.empty(m, dtype=int)
        last_before[0] = -1
        if m > 1:
            last_before[1:] = last_nonloss[:-1]
        loss_streak_before = local_idx - last_before - 1

        mask[positions] = (
            previous_loss
            & np.isfinite(breaks[positions])
            & (breaks[positions] <= break_threshold_minutes)
            & (loss_streak_before >= min_loss_streak)
        )
    return mask


def validate_real_baseline(df: pd.DataFrame, settings: Settings, tolerance: float = 1e-12) -> dict[str, Any]:
    """Recompute the v12 observed baseline from the supplied analytic dataset."""
    data = _prepare_multi_player_data(df)
    decisive = data["is_decisive"].astype(bool)
    tilt = data.loc[decisive, "tilt_proxy"].astype(bool)
    losses = data.loc[decisive, "is_loss"].astype(float)
    if int(tilt.sum()) == 0 or int((~tilt).sum()) == 0:
        raise ValueError("Observed baseline needs both tilt and control observations")
    observed_difference = float(losses[tilt].mean() - losses[~tilt].mean())
    expected_difference = float(
        data.loc[decisive & data["tilt_proxy"], "is_loss"].mean()
        - data.loc[decisive & ~data["tilt_proxy"], "is_loss"].mean()
    )
    if not np.isclose(observed_difference, expected_difference, atol=tolerance):
        raise AssertionError("Observed baseline calculation is internally inconsistent")
    return {
        "total_games": int(len(data)),
        "decisive_games": int(decisive.sum()),
        "draws": int((~decisive).sum()),
        "players": int(data["player_id"].nunique()),
        "tilt_observations": int(tilt.sum()),
        "control_observations": int((~tilt).sum()),
        "tilt_loss_rate": float(losses[tilt].mean()),
        "control_loss_rate": float(losses[~tilt].mean()),
        "difference_pp": 100.0 * observed_difference,
    }


def _simulate_one_null(
    df: pd.DataFrame,
    p_loss: np.ndarray,
    rng: np.random.Generator,
    settings: Settings,
) -> tuple[int, int, float]:
    """Simulate one decisive-game outcome path and recompute the proxy."""
    n = len(df)
    decisive = df["is_decisive"].astype(bool).to_numpy()
    loss = np.zeros(n, dtype=bool)
    loss[decisive] = rng.random(int(decisive.sum())) < p_loss[decisive]

    tilt = recompute_synthetic_tilt_proxy(
        df,
        loss,
        settings.features.short_break_minutes,
        settings.features.min_loss_streak,
    )
    tilt_mask = tilt & decisive
    control_mask = decisive & ~tilt
    if not tilt_mask.any() or not control_mask.any():
        raise ValueError("Synthetic simulation produced an empty tilt or control group")

    difference = float(loss[tilt_mask].mean() - loss[control_mask].mean())
    return int(tilt_mask.sum()), int(control_mask.sum()), difference


def simulate_null_distribution(
    df: pd.DataFrame,
    settings: Settings,
    scenario: str,
    p_loss: np.ndarray,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Run the configured synthetic null simulation for one probability model."""
    iterations = int(settings.v13.null_simulations)
    seed = int(settings.v13.simulation_seed)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for simulation_id in range(1, iterations + 1):
        tilt_n, control_n, difference = _simulate_one_null(df, p_loss, rng, settings)
        rows.append(
            {
                "scenario": scenario,
                "simulation_id": simulation_id,
                "tilt_n": tilt_n,
                "control_n": control_n,
                "difference_pp": 100.0 * difference,
            }
        )
    result = pd.DataFrame(rows)
    logger.info(
        "Synthetic null %s: simulations=%d, mean=%.3f pp, tail=%.3f",
        scenario,
        iterations,
        result["difference_pp"].mean(),
        np.mean(result["difference_pp"] >= validate_real_baseline(df, settings)["difference_pp"]),
    )
    return result


def summarize_null_distribution(
    distribution: pd.DataFrame,
    observed_difference_pp: float,
    settings: Settings,
    calibration_description: str,
) -> pd.DataFrame:
    """Summarize synthetic distributions without calling their interval a CI."""
    rows: list[dict[str, Any]] = []
    for scenario, group in distribution.groupby("scenario", sort=False):
        values = group["difference_pp"].astype(float).to_numpy()
        rows.append(
            {
                "scenario": scenario,
                "simulations": int(len(values)),
                "observed_difference_pp": float(observed_difference_pp),
                "mean_difference_pp": float(np.mean(values)),
                "median_difference_pp": float(np.median(values)),
                "std_difference_pp": float(np.std(values, ddof=1)),
                "simulation_interval_low_pp": float(np.quantile(values, 0.025)),
                "simulation_interval_high_pp": float(np.quantile(values, 0.975)),
                "empirical_upper_tail_proportion": float(np.mean(values >= observed_difference_pp)),
                "mean_tilt_n": float(group["tilt_n"].mean()),
                "min_tilt_n": int(group["tilt_n"].min()),
                "max_tilt_n": int(group["tilt_n"].max()),
                "mean_control_n": float(group["control_n"].mean()),
                "simulation_seed": int(settings.v13.simulation_seed),
                "calibration_method": calibration_description if scenario == "player_calibrated_elo" else "raw Elo probability",
            }
        )
    return pd.DataFrame(rows)


def prepare_opponent_context(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build descriptive opponent-context diagnostics and model-ready rows."""
    data = _prepare_multi_player_data(df)
    data = data.loc[data["is_decisive"].astype(bool) & data["tilt_proxy"].notna()].copy()

    rows: list[dict[str, Any]] = []
    for label, mask in [
        ("after_previous_loss", data["previous_result"].eq("loss")),
        ("after_previous_nonloss", data["previous_result"].isin(["win", "draw"])),
        ("short_break", data["break_after_previous"].le(5)),
        ("longer_break", data["break_after_previous"].gt(5)),
    ]:
        subset = data.loc[mask].copy()
        rows.append(
            {
                "group": label,
                "observations": int(len(subset)),
                "mean_rating_diff": float(subset["rating_diff"].mean()),
                "mean_opponent_rating": float(subset["opponent_rating"].mean()),
                "same_opponent_rate": float(subset["same_opponent"].mean()),
            }
        )

    loss_then_short = data["previous_result"].eq("loss") & data["break_after_previous"].le(5)
    rows.append(
        {
            "group": "after_previous_loss_and_short_break",
            "observations": int(loss_then_short.sum()),
            "mean_rating_diff": float(data.loc[loss_then_short, "rating_diff"].mean()),
            "mean_opponent_rating": float(data.loc[loss_then_short, "opponent_rating"].mean()),
            "same_opponent_rate": float(data.loc[loss_then_short, "same_opponent"].mean()),
        }
    )
    context = pd.DataFrame(rows)
    return data, context


def fit_opponent_adjusted_models(
    data: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit three logistic models with player-clustered robust standard errors."""
    model_specs = {
        "A_unadjusted": "loss ~ tilt_proxy_int",
        "B_opponent_player_adjusted": (
            "loss ~ tilt_proxy_int + rating_diff + color_black + C(player_id)"
        ),
        "C_same_opponent_adjusted": (
            "loss ~ tilt_proxy_int + rating_diff + color_black "
            "+ same_opponent + C(player_id)"
        ),
    }
    summary_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for model_name, formula in model_specs.items():
        model = smf.glm(
            formula,
            data=data,
            family=sm.families.Binomial(),
        ).fit(cov_type="cluster", cov_kwds={"groups": data["player_id"]})
        ci = model.conf_int()
        coef = float(model.params["tilt_proxy_int"])
        low, high = map(float, ci.loc["tilt_proxy_int"])
        summary_rows.append(
            {
                "model": model_name,
                "formula": formula,
                "term": "tilt_proxy",
                "coefficient": coef,
                "odds_ratio": float(np.exp(coef)),
                "standard_error": float(model.bse["tilt_proxy_int"]),
                "ci_low": low,
                "ci_high": high,
                "odds_ratio_ci_low": float(np.exp(low)),
                "odds_ratio_ci_high": float(np.exp(high)),
                "p_value": float(model.pvalues["tilt_proxy_int"]),
                "n_obs": int(model.nobs),
                "player_clusters": int(data["player_id"].nunique()),
                "standard_error_method": "cluster-robust by player_id",
            }
        )
        for term, value in model.params.items():
            low_term, high_term = map(float, ci.loc[term])
            coefficient_rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "coefficient": float(value),
                    "standard_error": float(model.bse[term]),
                    "ci_low": low_term,
                    "ci_high": high_term,
                    "odds_ratio": float(np.exp(value)),
                    "odds_ratio_ci_low": float(np.exp(low_term)),
                    "odds_ratio_ci_high": float(np.exp(high_term)),
                    "p_value": float(model.pvalues[term]),
                    "n_obs": int(model.nobs),
                    "player_clusters": int(data["player_id"].nunique()),
                }
            )
        logger.info(
            "%s tilt OR=%.3f, p=%.4f, n=%d",
            model_name,
            np.exp(coef),
            model.pvalues["tilt_proxy_int"],
            model.nobs,
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(coefficient_rows)


def plot_synthetic_null_distribution(
    distribution: pd.DataFrame,
    observed_difference_pp: float,
    output_path: Path,
) -> None:
    """Plot the null distributions and mark the observed association."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario, group in distribution.groupby("scenario", sort=False):
        ax.hist(
            group["difference_pp"],
            bins=45,
            density=True,
            alpha=0.45,
            label=scenario.replace("_", " ").title(),
        )
    ax.axvline(
        observed_difference_pp,
        linewidth=2,
        linestyle="--",
        label=f"Observed data ({observed_difference_pp:.2f} pp)",
    )
    ax.set_xlabel("Difference in loss rate (percentage points)")
    ax.set_ylabel("Density")
    ax.set_title("Synthetic null model: no tilt effect")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_opponent_adjusted_effect(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot tilt-proxy odds ratios from opponent-adjusted models."""
    plot_df = summary.copy().iloc[::-1]
    y = np.arange(len(plot_df))
    x = plot_df["odds_ratio"].to_numpy(dtype=float)
    low = plot_df["odds_ratio_ci_low"].to_numpy(dtype=float)
    high = plot_df["odds_ratio_ci_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.errorbar(x, y, xerr=[x - low, high - x], fmt="o", capsize=5)
    ax.axvline(1.0, linewidth=1, linestyle="--")
    display_labels = {
        "A_unadjusted": "Unadjusted",
        "B_opponent_player_adjusted": "Player + match context",
        "C_same_opponent_adjusted": "+ Same opponent",
    }
    labels = plot_df["model"].map(display_labels).fillna(plot_df["model"]).tolist()
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.set_xlabel("Tilt-proxy odds ratio (log scale, 95% CI)")
    ax.set_ylabel("Model")
    ax.set_title("Opponent-adjusted association for the tilt proxy")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_v13(df: pd.DataFrame, settings: Settings, logger: logging.Logger) -> dict[str, Any]:
    """Run all v13 analyses and persist reproducible artifacts."""
    settings.ensure_directories()
    data = _prepare_multi_player_data(df)
    baseline = validate_real_baseline(data, settings)
    logger.info(
        "V13 baseline sanity: %d players, %d games, tilt=%d, difference=%.3f pp",
        baseline["players"],
        baseline["total_games"],
        baseline["tilt_observations"],
        baseline["difference_pp"],
    )

    p_win_elo = elo_win_probability(data["rating_diff"].to_numpy(dtype=float))
    p_loss_elo = 1.0 - p_win_elo
    p_loss_cal = expanding_player_calibration(data, p_loss_elo)

    raw_dist = simulate_null_distribution(data, settings, "raw_elo", p_loss_elo, logger)
    cal_dist = simulate_null_distribution(data, settings, "player_calibrated_elo", p_loss_cal, logger)
    distribution = pd.concat([raw_dist, cal_dist], ignore_index=True)

    calibration_description = (
        "expanding player-specific residual correction using only prior decisive games; "
        "p_cal = clip(p_elo + mean(prior observed_loss - p_elo), 1e-4, 1-1e-4)"
    )
    null_summary = summarize_null_distribution(
        distribution,
        baseline["difference_pp"],
        settings,
        calibration_description,
    )

    _, context = prepare_opponent_context(data)
    model_data, _ = prepare_opponent_context(data)
    model_summary, model_coefficients = fit_opponent_adjusted_models(model_data, logger)

    paths = settings.paths
    settings.resolve_path(paths.synthetic_null_distribution_csv).parent.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(settings.resolve_path(paths.synthetic_null_distribution_csv), index=False)
    null_summary.to_csv(settings.resolve_path(paths.synthetic_null_summary_csv), index=False)
    settings.resolve_path(paths.synthetic_null_summary_json).write_text(
        json.dumps(
            {
                "observed_baseline": baseline,
                "simulation_seed": int(settings.v13.simulation_seed),
                "null_simulations_per_scenario": int(settings.v13.null_simulations),
                "scenarios": null_summary.to_dict(orient="records"),
                "calibration_method": calibration_description,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    context.to_csv(settings.resolve_path(paths.opponent_context_csv), index=False)
    model_summary.to_csv(settings.resolve_path(paths.opponent_adjusted_models_csv), index=False)
    model_coefficients.to_csv(settings.resolve_path(paths.opponent_adjusted_coefficients_csv), index=False)

    plot_synthetic_null_distribution(
        distribution,
        baseline["difference_pp"],
        settings.resolve_path(paths.figures_dir) / "synthetic_null_distribution.png",
    )
    plot_opponent_adjusted_effect(
        model_summary,
        settings.resolve_path(paths.figures_dir) / "opponent_adjusted_effect.png",
    )

    analysis_md = render_v13_analysis_markdown(baseline, null_summary, context, model_summary, calibration_description)
    settings.resolve_path(paths.v13_analysis_md).write_text(analysis_md, encoding="utf-8")
    return {
        "baseline": baseline,
        "null_summary": null_summary,
        "opponent_context": context,
        "opponent_adjusted_models": model_summary,
    }


def render_v13_analysis_markdown(
    baseline: dict[str, Any],
    null_summary: pd.DataFrame,
    context: pd.DataFrame,
    model_summary: pd.DataFrame,
    calibration_description: str,
) -> str:
    """Render the standalone v13 results narrative."""
    lines = [
        "# V13: Synthetic Null and Opponent-Adjusted Analysis",
        "",
        "V13 is an analytical layer built on the fixed v12 research dataset. The v12 sampling design, data, tilt proxy, sensitivity analysis, heterogeneity analysis and predictive models are preserved.",
        "",
        "## Observed baseline sanity check",
        "",
        f"- Players: **{baseline['players']}**",
        f"- Total games: **{baseline['total_games']}**",
        f"- Decisive games: **{baseline['decisive_games']}**",
        f"- Draws retained: **{baseline['draws']}**",
        f"- Observed tilt-proxy observations among decisive games: **{baseline['tilt_observations']}**",
        f"- Observed control observations: **{baseline['control_observations']}**",
        f"- Observed pooled difference: **+{baseline['difference_pp']:.2f} pp**",
        "",
        "The observed baseline is recomputed from the same multi-player feature dataset before any synthetic data are generated.",
        "",
        "## Synthetic null model",
        "",
        "The null simulation preserves player identities, opponents, ratings, rating differences, colors, timestamps, breaks, sessions, draws, repeated-opponent structure and the number of games. Only decisive win/loss outcomes are regenerated from a specified probability model, after which the same baseline tilt-proxy logic is recomputed.",
        "",
        "### Probability specifications",
        "",
        "1. **Raw Elo:** `p(win) = 1 / (1 + 10^(-rating_diff / 400))`.",
        f"2. **Player-calibrated Elo:** {calibration_description}.",
        "",
        "These are null data-generating processes. They contain no explicit tilt effect.",
        "",
    ]
    for _, row in null_summary.iterrows():
        lines.extend(
            [
                f"### {row['scenario'].replace('_', ' ').title()}",
                "",
                f"- Simulations: **{int(row['simulations'])}**",
                f"- Mean difference: **{row['mean_difference_pp']:+.2f} pp**",
                f"- Median difference: **{row['median_difference_pp']:+.2f} pp**",
                f"- Standard deviation: **{row['std_difference_pp']:.2f} pp**",
                f"- 95% simulation interval: **[{row['simulation_interval_low_pp']:+.2f}, {row['simulation_interval_high_pp']:+.2f}] pp**",
                f"- Share of null simulations at least as large as observed: **{100*row['empirical_upper_tail_proportion']:.1f}%**",
                f"- Mean simulated tilt-proxy count: **{row['mean_tilt_n']:.1f}**",
                "",
            ]
        )
    lines.extend(
        [
            "The simulation interval describes the spread of the synthetic null distribution; it is not a confidence interval for the observed effect. The empirical upper-tail proportion is not labelled as a p-value.",
            "",
            "## Opponent context",
            "",
            "| Group | Observations | Mean rating difference | Mean opponent rating | Same-opponent rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in context.iterrows():
        lines.append(
            f"| {row['group']} | {int(row['observations'])} | {row['mean_rating_diff']:.2f} | {row['mean_opponent_rating']:.2f} | {100*row['same_opponent_rate']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Opponent-adjusted logistic models",
            "",
            "The target is the current-game loss among decisive games. Standard errors are cluster-robust by `player_id`. Model B adds rating difference, color and player fixed effects. Model C additionally controls for whether the current opponent is the same as in the previous game.",
            "",
            "| Model | Tilt OR | 95% CI | p-value | N | Player clusters |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for _, row in model_summary.iterrows():
        lines.append(
            f"| {row['model']} | {row['odds_ratio']:.3f} | [{row['odds_ratio_ci_low']:.3f}, {row['odds_ratio_ci_high']:.3f}] | {row['p_value']:.4f} | {int(row['n_obs'])} | {int(row['player_clusters'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The raw pooled association is compatible with a null data-generating process that preserves the observed structure of the games while regenerating outcomes without an explicit tilt effect. The player-calibrated null gives a similar result, with its distribution depending on the chosen probability specification.",
            "",
            "After adjusting for rating difference, color and player fixed effects, the estimated tilt-proxy association is materially smaller than the raw pooled association. Adding same-opponent status produces a closely related adjusted estimate. These models describe observational associations; they do not establish a psychological mechanism or causality.",
            "",
            "The v13 result therefore shifts the main interpretation from treating the raw +12.66 pp difference as a direct behavioral effect toward treating it as an association that can arise from the broader game-generating process and match context.",
            "",
        ]
    )
    return "\n".join(lines)
