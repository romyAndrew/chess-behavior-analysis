"""Statistical analysis of the tilt proxy."""

from __future__ import annotations

import logging
from math import exp, log, sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from .config import Settings
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
# predefined tilt rule cannot silently drift away from the analysis definition.
