"""Static visualizations for the analysis and report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_result_by_hour(df: pd.DataFrame, output_dir: Path, timezone_name: str = "UTC") -> None:
    """Plot win rate by hour in the configured project timezone."""
    summary = df.groupby("hour", dropna=False)["is_win"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(summary, x="hour", y="is_win", marker="o", ax=ax)
    ax.set(xlabel=f"{timezone_name} hour", ylabel="Win rate", title="Win rate by hour")
    _save(fig, output_dir / "win_rate_by_hour.png")


def plot_break_by_tilt(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot break duration by tilt-proxy state."""
    plot_df = df.loc[df["break_after_previous"].notna()].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(plot_df, x="tilt_proxy", y="break_after_previous", ax=ax)
    ax.set(
        xlabel="Tilt proxy",
        ylabel="Break after previous game (minutes)",
        title="Inter-game break by tilt-proxy state",
    )
    _save(fig, output_dir / "break_by_tilt_proxy.png")


def plot_heatmap(df: pd.DataFrame, output_dir: Path, timezone_name: str = "UTC") -> None:
    """Plot win-rate heatmap across local hour and speed."""
    pivot = df.pivot_table(index="hour", columns="speed", values="is_win", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set(title="Win-rate heatmap: hour × speed", xlabel="Speed", ylabel=f"{timezone_name} hour")
    _save(fig, output_dir / "win_rate_heatmap.png")


def plot_tilt_ci(stats_result: dict[str, Any], output_dir: Path) -> None:
    """Plot Wilson confidence intervals for group-level loss rates."""
    labels = ["Tilt proxy", "Control"]
    values = [stats_result["loss_rate_tilt_proxy"], stats_result["loss_rate_control"]]
    cis = [stats_result["loss_rate_tilt_proxy_ci"], stats_result["loss_rate_control_ci"]]
    lower = [max(0.0, values[i] - cis[i][0]) for i in range(2)]
    upper = [max(0.0, cis[i][1] - values[i]) for i in range(2)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(labels, values, yerr=[lower, upper], fmt="o", capsize=5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Loss rate")
    ax.set_title("Observed loss rate by tilt-proxy state")
    _save(fig, output_dir / "tilt_loss_rate_ci.png")


def plot_correlations(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot a compact correlation matrix for numeric behavioral features."""
    cols = [
        "rating_diff",
        "break_after_previous",
        "loss_streak_before",
        "win_streak_before",
        "session_game_number",
        "avg_move_time",
        "time_pressure",
        "is_win",
    ]
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Behavioral feature correlations")
    _save(fig, output_dir / "correlation_heatmap.png")


def plot_tilt_sensitivity(sensitivity: pd.DataFrame, output_path: Path) -> None:
    """Plot estimated loss-rate differences with confidence intervals by threshold."""
    plot_df = sensitivity.copy()
    plot_df = plot_df.loc[plot_df["difference_pp"].notna()].copy()
    if plot_df.empty:
        return

    labels = plot_df["threshold_description"].tolist()
    labels = [
        f"{label} (baseline)" if baseline else label
        for label, baseline in zip(labels, plot_df["is_baseline"].tolist(), strict=True)
    ]
    y = list(range(len(plot_df)))[::-1]
    x = plot_df["difference_pp"].to_numpy(dtype=float)
    lower = x - plot_df["difference_ci_low_pp"].to_numpy(dtype=float)
    upper = plot_df["difference_ci_high_pp"].to_numpy(dtype=float) - x

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.errorbar(x, y, xerr=[lower, upper], fmt="o", capsize=5)
    ax.axvline(0.0, linewidth=1, linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Estimated difference in loss rate (percentage points)")
    ax.set_ylabel("Threshold configuration")
    ax.set_title("Tilt-proxy sensitivity to break and streak thresholds")
    _save(fig, output_path)


def plot_tilt_bootstrap_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot row-level and session-level bootstrap confidence intervals."""
    if comparison.empty:
        return

    plot_df = comparison.copy()
    y = list(range(len(plot_df)))[::-1]
    x = plot_df["observed_difference_pp"].to_numpy(dtype=float)
    lower = x - plot_df["ci_low"].to_numpy(dtype=float)
    upper = plot_df["ci_high"].to_numpy(dtype=float) - x
    labels = plot_df["method"].str.replace("_", " ").str.title().tolist()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.errorbar(x, y, xerr=[lower, upper], fmt="o", capsize=5)
    ax.axvline(0.0, linewidth=1, linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Estimated difference in loss rate (percentage points)")
    ax.set_ylabel("Inference method")
    ax.set_title("Tilt association: row-level vs session-aware bootstrap")
    _save(fig, output_path)



def plot_walk_forward_roc_auc(fold_metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot ROC-AUC across expanding-window temporal folds for all models."""
    if fold_metrics.empty or "roc_auc" not in fold_metrics.columns:
        return

    plot_df = fold_metrics.loc[fold_metrics["roc_auc"].notna()].copy()
    if plot_df.empty:
        return
    plot_df["fold"] = plot_df["fold"].astype(int)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.lineplot(
        plot_df,
        x="fold",
        y="roc_auc",
        hue="model",
        marker="o",
        ax=ax,
    )
    ax.axhline(0.5, linewidth=1, linestyle="--")
    ax.set_xlabel("Walk-forward test fold")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Walk-forward ROC-AUC across temporal folds")
    _save(fig, output_path)


def generate_all(
    df: pd.DataFrame,
    stats_result: dict[str, Any],
    output_dir: Path,
    timezone_name: str = "UTC",
    sensitivity: pd.DataFrame | None = None,
    bootstrap_comparison: pd.DataFrame | None = None,
    walk_forward_folds: pd.DataFrame | None = None,
) -> None:
    """Generate the full figure set, including inferential comparisons."""
    plot_result_by_hour(df, output_dir, timezone_name)
    plot_break_by_tilt(df, output_dir)
    plot_heatmap(df, output_dir, timezone_name)
    plot_tilt_ci(stats_result, output_dir)
    plot_correlations(df, output_dir)
    if sensitivity is not None:
        plot_tilt_sensitivity(sensitivity, output_dir / "tilt_sensitivity.png")
    if bootstrap_comparison is not None:
        plot_tilt_bootstrap_comparison(
            bootstrap_comparison,
            output_dir / "tilt_bootstrap_comparison.png",
        )
    if walk_forward_folds is not None:
        plot_walk_forward_roc_auc(
            walk_forward_folds,
            output_dir / "walk_forward_roc_auc.png",
        )


def plot_multi_player_tilt_effects(player_effects: pd.DataFrame, output_path: Path) -> None:
    """Plot player-level tilt associations in selection order, not rank order."""
    plot_df = player_effects.sort_values("selection_order").copy()
    if plot_df.empty:
        return
    y = np.arange(len(plot_df))
    x = plot_df["difference_pp"].to_numpy(dtype=float)
    ci_low = plot_df["ci_low_pp"].to_numpy(dtype=float)
    ci_high = plot_df["ci_high_pp"].to_numpy(dtype=float)
    lower = np.where(np.isfinite(ci_low), x - ci_low, 0.0)
    upper = np.where(np.isfinite(ci_high), ci_high - x, 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, y, xerr=[lower, upper], fmt="o", capsize=4)
    ax.axvline(0.0, linewidth=1, linestyle="--")
    ax.set_yticks(y, plot_df["player_id"].astype(str).tolist())
    ax.set_xlabel("Observed difference in loss rate (percentage points)")
    ax.set_ylabel("Selected player")
    ax.set_title("Player-level tilt-proxy associations")
    _save(fig, output_path)


def plot_multi_player_sample_sizes(player_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot total, decisive and tilt observations by selection order."""
    plot_df = player_summary.sort_values("player_id").copy()
    if plot_df.empty:
        return
    x = np.arange(len(plot_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, plot_df["games"], width, label="Total games")
    ax.bar(x, plot_df["decisive_games"], width, label="Decisive games")
    ax.bar(x + width, plot_df["tilt_observations"], width, label="Tilt observations")
    ax.set_xticks(x, plot_df["player_id"].astype(str).tolist(), rotation=45, ha="right")
    ax.set_ylabel("Number of games / observations")
    ax.set_title("Multi-player sample sizes")
    ax.legend()
    _save(fig, output_path)


def plot_player_heterogeneity(results: pd.DataFrame, output_path: Path) -> None:
    """Plot player-level association estimates in selection order, not rank order."""
    plot_df = results.sort_values("selection_order").copy()
    plot_df = plot_df.loc[plot_df["difference_pp"].notna()].copy()
    if plot_df.empty:
        return

    y = np.arange(len(plot_df))
    x = plot_df["difference_pp"].to_numpy(dtype=float)
    ci_low = plot_df["difference_ci_low"].to_numpy(dtype=float)
    ci_high = plot_df["difference_ci_high"].to_numpy(dtype=float)
    valid_ci = np.isfinite(ci_low) & np.isfinite(ci_high)
    lower = np.where(valid_ci, x - ci_low, 0.0)
    upper = np.where(valid_ci, ci_high - x, 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, y, xerr=[lower, upper], fmt="o", capsize=4)
    ax.axvline(0.0, linewidth=1, linestyle="--")
    ax.set_yticks(y, plot_df["player_id"].astype(str).tolist())
    ax.set_xlabel("Estimated difference in loss rate (percentage points)")
    ax.set_ylabel("Selected player (selection order)")
    ax.set_title("Player-level heterogeneity of the tilt-proxy association")
    _save(fig, output_path)
