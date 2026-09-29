"""Statistical analysis of the tilt proxy."""

from __future__ import annotations

import json
import logging
from math import exp, log, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from .config import Settings
from .research import research_cutoff
from .features import build_tilt_proxy_mask


def _odds_ratio_ci(table: np.ndarray, confidence: float) -> tuple[float, float, float]:
    """Return odds ratio and Wald CI with Haldane correction."""
    corrected = table.astype(float) + 0.5
    a, b = corrected[0]
    c, d = corrected[1]
    odds_ratio = (a * d) / (b * c)
    se = sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = 1.959963984540054
    if confidence != 0.95:
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + confidence / 2))
    low = exp(log(odds_ratio) - z * se)
    high = exp(log(odds_ratio) + z * se)
    return float(odds_ratio), float(low), float(high)


def _row_level_bootstrap_difference(
    group_a: np.ndarray,
    group_b: np.ndarray,
    iterations: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap a CI by resampling individual rows independently."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(iterations, dtype=float)
    for i in range(iterations):
        sample_a = rng.choice(group_a, size=len(group_a), replace=True)
        sample_b = rng.choice(group_b, size=len(group_b), replace=True)
        diffs[i] = float(np.mean(sample_a) - np.mean(sample_b))
    alpha = 1 - confidence
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def _session_bootstrap_difference(
    df: pd.DataFrame,
    iterations: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, int]:
    """Bootstrap a loss-rate difference by resampling whole sessions.

    The statistic is defined on decisive games with a non-null ``tilt_proxy``.
    Every selected session contributes all of its relevant games together. For
    this proportion difference, aggregating session-level sufficient statistics
    is equivalent to concatenating the full sampled sessions and recomputing
    the two group loss rates.
    """
    required = {"session_id", "is_decisive", "tilt_proxy", "is_loss"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Session bootstrap requires columns: {sorted(missing)}")

    data = df.loc[df["is_decisive"].astype(bool) & df["tilt_proxy"].notna()].copy()
    if data["session_id"].isna().any():
        raise ValueError("Session bootstrap requires every analytic game to have a session_id")

    session_rows: list[tuple[int, int, int, int, int]] = []
    for session_id, group in data.groupby("session_id", sort=False):
        tilt_rows = group.loc[group["tilt_proxy"].astype(bool), "is_loss"]
        control_rows = group.loc[~group["tilt_proxy"].astype(bool), "is_loss"]
        session_rows.append(
            (
                int(session_id),
                int(len(tilt_rows)),
                int(tilt_rows.sum()),
                int(len(control_rows)),
                int(control_rows.sum()),
            )
        )

    n_sessions = len(session_rows)
    if n_sessions == 0:
        raise ValueError("Session bootstrap requires at least one analytic session")

    summary = np.asarray([row[1:] for row in session_rows], dtype=np.int64)
    rng = np.random.default_rng(seed)
    diffs = np.empty(iterations, dtype=float)
    valid = 0
    attempts = 0
    max_attempts = max(iterations * 1000, 10_000)

    while valid < iterations:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                "Could not generate enough valid session bootstrap replicates with both comparison groups."
            )

        sampled = rng.integers(0, n_sessions, size=n_sessions)
        sampled_summary = summary[sampled].sum(axis=0)
        tilt_n, tilt_losses, control_n, control_losses = sampled_summary.tolist()
        if tilt_n == 0 or control_n == 0:
            continue
        diffs[valid] = (tilt_losses / tilt_n) - (control_losses / control_n)
        valid += 1

    alpha = 1 - confidence
    return (
        float(np.quantile(diffs, alpha / 2)),
        float(np.quantile(diffs, 1 - alpha / 2)),
        n_sessions,
    )


def _analytic_tilt_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return the decisive games used by the primary tilt analysis."""
    data = df.loc[df["is_decisive"].astype(bool)].copy()
    return data.loc[data["tilt_proxy"].notna()]


def compare_tilt_bootstrap_methods(
    df: pd.DataFrame,
    result: dict[str, Any],
    settings: Settings,
) -> pd.DataFrame:
    """Compare the existing row bootstrap with the new session bootstrap."""
    data = _analytic_tilt_data(df)
    if "session_id" not in data.columns:
        raise ValueError("Bootstrap comparison requires the existing session_id feature")
    if data["session_id"].isna().any():
        raise ValueError("Bootstrap comparison requires every analytic game to have a session_id")

    session_low, session_high, n_sessions = _session_bootstrap_difference(
        df,
        settings.analysis.bootstrap_iterations,
        settings.analysis.confidence_level,
        settings.analysis.bootstrap_seed,
    )
    observed_difference_pp = 100.0 * float(result["risk_difference"])
    rows = [
        {
            "method": "row_level_bootstrap",
            "observed_difference_pp": observed_difference_pp,
            "ci_low": 100.0 * float(result["risk_difference_ci"][0]),
            "ci_high": 100.0 * float(result["risk_difference_ci"][1]),
            "n_observations": int(len(data)),
            "n_sessions": int(n_sessions),
            "bootstrap_iterations": int(settings.analysis.bootstrap_iterations),
            "random_seed": int(settings.analysis.bootstrap_seed),
        },
        {
            "method": "session_bootstrap",
            "observed_difference_pp": observed_difference_pp,
            "ci_low": 100.0 * session_low,
            "ci_high": 100.0 * session_high,
            "n_observations": int(len(data)),
            "n_sessions": int(n_sessions),
            "bootstrap_iterations": int(settings.analysis.bootstrap_iterations),
            "random_seed": int(settings.analysis.bootstrap_seed),
        },
    ]
    comparison = pd.DataFrame(rows)

    result["bootstrap_comparison"] = {
        "observed_difference_pp": observed_difference_pp,
        "row_level_bootstrap_ci_pp": [rows[0]["ci_low"], rows[0]["ci_high"]],
        "session_bootstrap_ci_pp": [rows[1]["ci_low"], rows[1]["ci_high"]],
        "n_observations": int(len(data)),
        "n_sessions": int(n_sessions),
        "bootstrap_iterations": int(settings.analysis.bootstrap_iterations),
        "random_seed": int(settings.analysis.bootstrap_seed),
        "row_level_method": "row_level_bootstrap",
        "session_method": "session_bootstrap",
    }
    return comparison


def analyze_tilt_sensitivity_session_bootstrap(
    df: pd.DataFrame,
    settings: Settings,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Compute session-bootstrap CIs for the existing six tilt thresholds."""
    rows: list[dict[str, Any]] = []
    sensitivity_settings = settings.model_copy(
        update={"analysis": settings.analysis.model_copy(update={"min_group_size": 1})}
    )

    for config in build_tilt_sensitivity_configs(settings):
        data = df.copy()
        data["tilt_proxy"] = build_tilt_proxy_mask(
            data,
            config["break_threshold_minutes"],
            config["min_loss_streak"],
        )
        decisive = _analytic_tilt_data(data)
        n_tilt = int(decisive["tilt_proxy"].sum())
        n_control = int((~decisive["tilt_proxy"]).sum())

        row: dict[str, Any] = {
            **config,
            "n_observations": int(len(decisive)),
            "n_tilt_observations": n_tilt,
            "n_control_observations": n_control,
            "session_bootstrap_ci_low_pp": np.nan,
            "session_bootstrap_ci_high_pp": np.nan,
            "n_sessions": np.nan,
            "bootstrap_iterations": int(settings.analysis.bootstrap_iterations),
            "random_seed": int(settings.analysis.bootstrap_seed),
        }

        if n_tilt == 0 or n_control == 0:
            rows.append(row)
            continue

        result = analyze_tilt_proxy(data, sensitivity_settings, logger)
        session_low, session_high, n_sessions = _session_bootstrap_difference(
            data,
            settings.analysis.bootstrap_iterations,
            settings.analysis.confidence_level,
            settings.analysis.bootstrap_seed,
        )
        row.update(
            {
                "observed_difference_pp": 100.0 * result["risk_difference"],
                "session_bootstrap_ci_low_pp": 100.0 * session_low,
                "session_bootstrap_ci_high_pp": 100.0 * session_high,
                "n_sessions": int(n_sessions),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def save_tilt_bootstrap_comparison(
    comparison: pd.DataFrame,
    csv_path: Path,
    markdown_path: Path,
) -> None:
    """Persist the row-level and session-level bootstrap comparison."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(csv_path, index=False)

    row_level = comparison.loc[comparison["method"] == "row_level_bootstrap"].iloc[0]
    session_level = comparison.loc[comparison["method"] == "session_bootstrap"].iloc[0]
    ci_change_low = float(session_level["ci_low"] - row_level["ci_low"])
    ci_change_high = float(session_level["ci_high"] - row_level["ci_high"])
    markdown = (
        "# Tilt Bootstrap Inference Comparison\n\n"
        "The primary tilt estimate is unchanged. The comparison only changes the resampling unit used to quantify uncertainty.\n\n"
        "## Methods\n\n"
        "- **row_level_bootstrap** resamples individual decisive games independently and preserves the existing v7 inference procedure.\n"
        "- **session_bootstrap** resamples complete existing sessions with replacement. All relevant games from a selected session stay together.\n\n"
        "The session-aware method is used to account for dependence among games played close together in the same session. It does not alter the observed point estimate.\n\n"
        "## Results\n\n"
        "| Method | Observed difference (pp) | 95% CI (pp) | Observations | Sessions | Iterations | Seed |\n"
        "|---|---:|---|---:|---:|---:|---:|\n"
        f"| row-level bootstrap | {row_level['observed_difference_pp']:.2f} | [{row_level['ci_low']:.2f}, {row_level['ci_high']:.2f}] | {int(row_level['n_observations'])} | {int(row_level['n_sessions'])} | {int(row_level['bootstrap_iterations'])} | {int(row_level['random_seed'])} |\n"
        f"| session bootstrap | {session_level['observed_difference_pp']:.2f} | [{session_level['ci_low']:.2f}, {session_level['ci_high']:.2f}] | {int(session_level['n_observations'])} | {int(session_level['n_sessions'])} | {int(session_level['bootstrap_iterations'])} | {int(session_level['random_seed'])} |\n\n"
        f"The session-bootstrap CI lower bound differs from the row-level CI by {ci_change_low:.2f} pp, and the upper bound differs by {ci_change_high:.2f} pp."
        " A wider session-level interval indicates less independent information after accounting for within-session dependence; a similar interval indicates that the dependence adjustment has limited impact for this dataset.\n\n"
        "## Interpretation\n\n"
        "The two methods estimate the same observed association. Only the uncertainty calculation changes. Neither method establishes causation or measures psychological tilt directly.\n"
    )
    markdown_path.write_text(markdown, encoding="utf-8")


def analyze_tilt_proxy(
    df: pd.DataFrame,
    settings: Settings,
    logger: logging.Logger,
    min_group_size_override: int | None = None,
) -> dict[str, Any]:
    """Test whether the tilt proxy is associated with current-game losses.

    H0: The loss proportion is the same under the tilt proxy and non-proxy groups.
    H1: The loss proportion differs between the groups.
    """
    data = df.loc[df["is_decisive"].astype(bool)].copy()
    data = data.loc[data["tilt_proxy"].notna()]
    groups = data["tilt_proxy"].astype(bool)
    tilt = data.loc[groups, "is_loss"].astype(int).to_numpy()
    control = data.loc[~groups, "is_loss"].astype(int).to_numpy()

    min_group_size = (
        settings.analysis.min_group_size
        if min_group_size_override is None
        else int(min_group_size_override)
    )
    if min_group_size < 1:
        raise ValueError("min_group_size_override must be at least 1")
    if len(tilt) < min_group_size or len(control) < min_group_size:
        raise ValueError(
            f"Not enough decisive games for the tilt test: tilt={len(tilt)}, control={len(control)}"
        )

    table = np.array(
        [
            [int(tilt.sum()), int(len(tilt) - tilt.sum())],
            [int(control.sum()), int(len(control) - control.sum())],
        ],
        dtype=int,
    )
    odds_ratio, or_low, or_high = _odds_ratio_ci(table, settings.analysis.confidence_level)

    expected = chi2_contingency(table, correction=False)[3]
    use_fisher = bool((expected < 5).any())
    if use_fisher:
        _, p_value = fisher_exact(table)
        test_name = "Fisher exact test"
    else:
        _, p_value = proportions_ztest(
            count=np.array([tilt.sum(), control.sum()]),
            nobs=np.array([len(tilt), len(control)]),
        )
        test_name = "two-proportion z-test"

    tilt_rate = float(np.mean(tilt))
    control_rate = float(np.mean(control))
    rate_low_tilt, rate_high_tilt = proportion_confint(
        int(tilt.sum()), len(tilt), alpha=1 - settings.analysis.confidence_level, method="wilson"
    )
    rate_low_control, rate_high_control = proportion_confint(
        int(control.sum()), len(control), alpha=1 - settings.analysis.confidence_level, method="wilson"
    )
    risk_difference = tilt_rate - control_rate
    rd_low, rd_high = _row_level_bootstrap_difference(
        tilt,
        control,
        settings.analysis.bootstrap_iterations,
        settings.analysis.confidence_level,
        settings.analysis.bootstrap_seed,
    )

    result: dict[str, Any] = {
        "hypotheses": {
            "H0": "Current-game loss proportion is equal for tilt-proxy and control groups.",
            "H1": "Current-game loss proportion differs between tilt-proxy and control groups.",
        },
        "test": test_name,
        "confidence_level": settings.analysis.confidence_level,
        "n_tilt_proxy": len(tilt),
        "n_control": len(control),
        "loss_rate_tilt_proxy": tilt_rate,
        "loss_rate_control": control_rate,
        "loss_rate_tilt_proxy_ci": [float(rate_low_tilt), float(rate_high_tilt)],
        "loss_rate_control_ci": [float(rate_low_control), float(rate_high_control)],
        "risk_difference": risk_difference,
        "risk_difference_ci": [rd_low, rd_high],
        "odds_ratio": odds_ratio,
        "odds_ratio_ci": [or_low, or_high],
        "p_value": float(p_value),
        "alpha": 1 - settings.analysis.confidence_level,
        "decision": (
            "reject H0" if float(p_value) < 1 - settings.analysis.confidence_level else "do not reject H0"
        ),
        "effect_interpretation": (
            "Tilt-proxy group has higher observed loss odds"
            if odds_ratio > 1
            else "Tilt-proxy group has lower observed loss odds"
        ),
        "caution": "Association only; the proxy is not a direct measure of psychological tilt and does not establish causation.",
    }

    quality_col = "mean_eval_loss"
    quality_min_coverage = settings.features.min_eval_coverage_for_quality
    if quality_col in data.columns and "eval_coverage" in data.columns:
        quality = data.loc[
            data["eval_coverage"].fillna(0).ge(quality_min_coverage),
            ["tilt_proxy", quality_col],
        ].dropna()
        q_tilt = quality.loc[quality["tilt_proxy"], quality_col].to_numpy()
        q_control = quality.loc[~quality["tilt_proxy"], quality_col].to_numpy()
        if len(q_tilt) >= settings.analysis.min_group_size and len(q_control) >= settings.analysis.min_group_size:
            stat, q_p = mannwhitneyu(q_tilt, q_control, alternative="two-sided")
            rank_biserial = 1 - (2 * float(stat) / (len(q_tilt) * len(q_control)))
            q_low, q_high = _row_level_bootstrap_difference(
                q_tilt,
                q_control,
                settings.analysis.bootstrap_iterations,
                settings.analysis.confidence_level,
                settings.analysis.bootstrap_seed,
            )
            result["quality_test"] = {
                "metric": quality_col,
                "test": "Mann-Whitney U",
                "n_tilt_proxy": len(q_tilt),
                "n_control": len(q_control),
                "median_tilt_proxy": float(np.median(q_tilt)),
                "median_control": float(np.median(q_control)),
                "rank_biserial_effect": float(rank_biserial),
                "median_difference_bootstrap_ci": [q_low, q_high],
                "p_value": float(q_p),
                "note": "Lower mean_eval_loss indicates smaller observed engine-evaluation deterioration per available evaluated move.",
            }

    return result


# The baseline row is generated from the current project configuration so the
# sensitivity layer cannot silently drift away from the baseline definition.
def build_tilt_sensitivity_configs(settings: Settings) -> list[dict[str, Any]]:
    """Return the fixed sensitivity configurations, including the current baseline."""
    baseline_break = settings.features.short_break_minutes
    baseline_streak = settings.features.min_loss_streak
    return [
        {
            "configuration": "break_2m_streak_2",
            "threshold_description": "break <= 2 min, streak >= 2",
            "break_threshold_minutes": 2.0,
            "min_loss_streak": 2,
            "is_baseline": False,
        },
        {
            "configuration": "break_5m_streak_2",
            "threshold_description": f"break <= {baseline_break:g} min, streak >= {baseline_streak}",
            "break_threshold_minutes": float(baseline_break),
            "min_loss_streak": int(baseline_streak),
            "is_baseline": True,
        },
        {
            "configuration": "break_10m_streak_2",
            "threshold_description": "break <= 10 min, streak >= 2",
            "break_threshold_minutes": 10.0,
            "min_loss_streak": 2,
            "is_baseline": False,
        },
        {
            "configuration": "break_15m_streak_2",
            "threshold_description": "break <= 15 min, streak >= 2",
            "break_threshold_minutes": 15.0,
            "min_loss_streak": 2,
            "is_baseline": False,
        },
        {
            "configuration": "break_5m_streak_3",
            "threshold_description": "break <= 5 min, streak >= 3",
            "break_threshold_minutes": 5.0,
            "min_loss_streak": 3,
            "is_baseline": False,
        },
        {
            "configuration": "break_5m_streak_4",
            "threshold_description": "break <= 5 min, streak >= 4",
            "break_threshold_minutes": 5.0,
            "min_loss_streak": 4,
            "is_baseline": False,
        },
    ]


def analyze_tilt_sensitivity(
    df: pd.DataFrame,
    settings: Settings,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Re-run the existing tilt analysis across the predefined thresholds."""
    rows: list[dict[str, Any]] = []
    sensitivity_settings = settings.model_copy(
        update={"analysis": settings.analysis.model_copy(update={"min_group_size": 1})}
    )

    for config in build_tilt_sensitivity_configs(settings):
        data = df.copy()
        data["tilt_proxy"] = build_tilt_proxy_mask(
            data,
            config["break_threshold_minutes"],
            config["min_loss_streak"],
        )
        decisive = data.loc[data["is_decisive"].astype(bool)].copy()
        n_tilt = int(decisive["tilt_proxy"].sum())
        n_control = int((~decisive["tilt_proxy"]).sum())
        small_sample = n_tilt < settings.analysis.min_group_size

        row: dict[str, Any] = {
            **config,
            "n_tilt_observations": n_tilt,
            "n_control_observations": n_control,
            "small_sample": small_sample,
            "stability_note": (
                f"Small tilt group: {n_tilt} observations, below the configured minimum of "
                f"{settings.analysis.min_group_size}; estimate should be treated as unstable."
                if small_sample
                else ""
            ),
            "tilt_loss_rate": np.nan,
            "control_loss_rate": np.nan,
            "difference_pp": np.nan,
            "difference_ci_low_pp": np.nan,
            "difference_ci_high_pp": np.nan,
            "odds_ratio": np.nan,
            "odds_ratio_ci_low": np.nan,
            "odds_ratio_ci_high": np.nan,
            "p_value": np.nan,
            "statistical_test": "not estimable",
        }

        if n_tilt == 0 or n_control == 0:
            row["stability_note"] = (
                "The configuration creates an empty comparison group, so inferential statistics are not estimable."
            )
            rows.append(row)
            continue

        result = analyze_tilt_proxy(data, sensitivity_settings, logger)
        row.update(
            {
                "tilt_loss_rate": result["loss_rate_tilt_proxy"],
                "control_loss_rate": result["loss_rate_control"],
                "difference_pp": 100.0 * result["risk_difference"],
                "difference_ci_low_pp": 100.0 * result["risk_difference_ci"][0],
                "difference_ci_high_pp": 100.0 * result["risk_difference_ci"][1],
                "odds_ratio": result["odds_ratio"],
                "odds_ratio_ci_low": result["odds_ratio_ci"][0],
                "odds_ratio_ci_high": result["odds_ratio_ci"][1],
                "p_value": result["p_value"],
                "statistical_test": result["test"],
            }
        )
        rows.append(row)

    sensitivity = pd.DataFrame(rows)

    # Keep the existing configured minimum-group rule unchanged. In addition,
    # the stricter streak definitions are explicitly flagged when they produce
    # substantially smaller tilt groups than the unchanged baseline. This is
    # descriptive metadata, not a new inferential sample-size threshold.
    if not sensitivity.empty:
        baseline_row = sensitivity.loc[sensitivity["is_baseline"]].iloc[0]
        baseline_n = int(baseline_row["n_tilt_observations"])
        stricter_than_baseline = sensitivity["min_loss_streak"] > int(
            baseline_row["min_loss_streak"]
        )
        smaller_than_baseline = sensitivity["n_tilt_observations"] < baseline_n
        estimable = sensitivity["difference_pp"].notna()
        relative_small = stricter_than_baseline & smaller_than_baseline & estimable

        sensitivity.loc[relative_small, "small_sample"] = True
        sensitivity.loc[relative_small, "stability_note"] = sensitivity.loc[
            relative_small, "n_tilt_observations"
        ].map(
            lambda n: (
                f"Smaller tilt group than baseline ({int(n)} vs {baseline_n} observations); "
                "point estimate has greater statistical uncertainty."
            )
        )

    return sensitivity


def save_tilt_sensitivity(
    sensitivity: pd.DataFrame,
    csv_path: Path,
    markdown_path: Path,
) -> None:
    """Persist the sensitivity table and a compact interpretation summary."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(csv_path, index=False)
    markdown_path.write_text(render_tilt_sensitivity_summary(sensitivity), encoding="utf-8")


def render_tilt_sensitivity_summary(sensitivity: pd.DataFrame) -> str:
    """Render a cautious summary of threshold sensitivity."""
    usable = sensitivity["difference_pp"].dropna()
    positive = int((usable > 0).sum())
    negative = int((usable < 0).sum())
    zero = int((usable == 0).sum())
    baseline = sensitivity.loc[sensitivity["is_baseline"]].iloc[0]

    if len(usable) == 0:
        direction = "No configuration produced an estimable effect difference."
    elif positive == len(usable):
        direction = f"The estimated difference is positive for all {len(usable)} estimable configurations."
    elif negative == len(usable):
        direction = f"The estimated difference is negative for all {len(usable)} estimable configurations."
    else:
        direction = (
            f"The estimated difference changes direction across configurations: "
            f"{positive} positive, {negative} negative, and {zero} exactly zero."
        )

    range_note = ""
    if len(usable) > 0:
        range_note = (
            f"Across estimable configurations, the difference ranges from {usable.min():.1f} to "
            f"{usable.max():.1f} percentage points."
        )

    rows = [
        "| Configuration | Tilt n | Control n | Tilt loss rate | Control loss rate | Difference (pp) | 95% CI (pp) | Odds ratio | p-value | Stability |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in sensitivity.itertuples(index=False):
        ci = (
            f"[{row.difference_ci_low_pp:.1f}, {row.difference_ci_high_pp:.1f}]"
            if pd.notna(row.difference_ci_low_pp)
            else "NA"
        )
        stability = (
            row.stability_note
            if pd.notna(row.stability_note) and row.stability_note
            else "ok"
        )
        rows.append(
            f"| {row.threshold_description}{' (baseline)' if row.is_baseline else ''} | "
            f"{row.n_tilt_observations} | {row.n_control_observations} | "
            f"{row.tilt_loss_rate:.3f} | {row.control_loss_rate:.3f} | "
            f"{row.difference_pp:.1f} | {ci} | {row.odds_ratio:.2f} | "
            f"{row.p_value:.4g} | {stability} |"
        )

    smaller_group_rows = sensitivity.loc[sensitivity["small_sample"]].copy()
    smaller_group_note = ""
    if not smaller_group_rows.empty:
        details = "; ".join(
            f"{row.threshold_description} ({int(row.n_tilt_observations)} vs {int(baseline.n_tilt_observations)} baseline observations)"
            for row in smaller_group_rows.itertuples(index=False)
        )
        smaller_group_note = (
            f"The configurations marked as smaller tilt groups are: {details}. "
            "Their point estimates have greater statistical uncertainty because they are based on substantially fewer tilt observations than the baseline. "
            "This is a descriptive comparison with the baseline group size, not a new statistical exclusion threshold.\n"
        )

    return (
        "# Tilt Proxy Sensitivity Analysis\n\n"
        "The existing tilt-proxy analysis was repeated with alternative break and loss-streak thresholds. "
        "The baseline definition is retained unchanged and is included in the table for direct comparison.\n\n"
        "## Thresholds\n\n"
        "The six configurations are 2/2, 5/2 (baseline), 10/2, 15/2, 5/3 and 5/4, where the first number is the maximum break in minutes and the second is the minimum previous loss streak.\n\n"
        "## Results\n\n"
        + "\n".join(rows)
        + "\n\n"
        f"Baseline: **{baseline.threshold_description}**, with {int(baseline.n_tilt_observations)} tilt observations and an estimated difference of {baseline.difference_pp:.1f} percentage points.\n\n"
        f"{direction} {range_note}\n\n"
        "The table uses the same statistical logic as the primary analysis: the existing Fisher/two-proportion test choice, odds-ratio confidence interval, Wilson group-rate intervals, and the existing risk-difference bootstrap interval. No new inferential method is introduced here.\n\n"
        "## Interpretation\n\n"
        "Sensitivity analysis describes how the estimated association changes when the behavioral rule is defined with nearby thresholds. It does not establish causation and should not be interpreted as a direct measurement of psychological tilt. "
        + smaller_group_note
    )


def save_stats(result: dict[str, Any], path: Path) -> None:
    """Persist statistical results as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def render_summary(result: dict[str, Any]) -> str:
    """Render a cautious Markdown summary."""
    quality_section = ""
    if "quality_test" in result:
        q = result["quality_test"]
        quality_section = (
            "## Optional engine-evaluation quality analysis\n\n"
            f"- Metric: **{q['metric']}**\n"
            f"- Test: **{q['test']}**\n"
            f"- Tilt-proxy median: **{q['median_tilt_proxy']:.3f}**\n"
            f"- Control median: **{q['median_control']:.3f}**\n"
            f"- Rank-biserial effect: **{q['rank_biserial_effect']:.3f}**\n"
            f"- p-value: **{q['p_value']:.4g}**\n\n"
        )
    bootstrap_section = ""
    if "bootstrap_comparison" in result:
        comparison = result["bootstrap_comparison"]
        row_low, row_high = comparison["row_level_bootstrap_ci_pp"]
        session_low, session_high = comparison["session_bootstrap_ci_pp"]
        bootstrap_section = (
            "## Dependence-aware bootstrap\n\n"
            f"The existing row-level bootstrap CI is **[{row_low:.2f}, {row_high:.2f}]** percentage points. "
            f"The session-aware bootstrap CI is **[{session_low:.2f}, {session_high:.2f}]** percentage points across "
            f"{comparison['n_sessions']} sessions. Both methods use {comparison['bootstrap_iterations']} iterations with seed {comparison['random_seed']}.\n\n"
            "The observed point estimate is unchanged because only the resampling unit changes. The session-level interval accounts for dependence among games inside the same existing session.\n\n"
        )

    return (
        "# Tilt Proxy Statistical Analysis\n\n"
        "## Hypotheses\n\n"
        f"- H0: {result['hypotheses']['H0']}\n"
        f"- H1: {result['hypotheses']['H1']}\n\n"
        "## Results\n\n"
        f"- Test: **{result['test']}**\n"
        f"- Tilt-proxy group: **{result['n_tilt_proxy']}** decisive games\n"
        f"- Control group: **{result['n_control']}** decisive games\n"
        f"- Loss rate, tilt proxy: **{result['loss_rate_tilt_proxy']:.3f}**, 95% CI [{result['loss_rate_tilt_proxy_ci'][0]:.3f}, {result['loss_rate_tilt_proxy_ci'][1]:.3f}]\n"
        f"- Loss rate, control: **{result['loss_rate_control']:.3f}**, 95% CI [{result['loss_rate_control_ci'][0]:.3f}, {result['loss_rate_control_ci'][1]:.3f}]\n"
        f"- Risk difference: **{result['risk_difference']:.3f}**, bootstrap 95% CI [{result['risk_difference_ci'][0]:.3f}, {result['risk_difference_ci'][1]:.3f}]\n"
        f"- Odds ratio: **{result['odds_ratio']:.3f}**, 95% CI [{result['odds_ratio_ci'][0]:.3f}, {result['odds_ratio_ci'][1]:.3f}]\n"
        f"- p-value: **{result['p_value']:.4g}**\n"
        f"- Decision at α={result['alpha']:.3f}: **{result['decision']}**\n\n"
        + quality_section
        + bootstrap_section
        + "## Interpretation\n\n"
        "The observed relationship is an **association**, not evidence of causation. The tilt proxy is a constructed behavioral rule based on game history, not a direct measurement of psychological state.\n\n"
        "## Limitations\n\n"
        "- The dataset represents one player's Lichess history.\n"
        "- The proxy can be sensitive to the chosen break and streak thresholds.\n"
        "- Engine evaluation coverage is incomplete and may be non-random.\n"
        "- Time-control, rating, opponent strength, and other confounders may remain.\n"
    )
