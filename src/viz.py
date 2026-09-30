"""Final visualizations retained for the portfolio release."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_walk_forward_roc_auc(fold_metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot ROC-AUC across expanding-window temporal folds."""
    if fold_metrics.empty or "roc_auc" not in fold_metrics.columns:
        return
    plot_df = fold_metrics.loc[fold_metrics["roc_auc"].notna()].copy()
    if plot_df.empty:
        return
    plot_df["fold"] = plot_df["fold"].astype(int)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model, group in plot_df.groupby("model", sort=False):
        ax.plot(group["fold"], group["roc_auc"], marker="o", label=str(model))
    ax.legend(title="Model")
    ax.axhline(0.5, linewidth=1, linestyle="--")
    ax.set_xlabel("Walk-forward test fold")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Walk-forward ROC-AUC across temporal folds")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_tilt_effect(pooled: pd.DataFrame, output_path: Path) -> None:
    """Plot the primary pooled loss-rate comparison."""
    if pooled.empty:
        return
    row = pooled.iloc[0]
    tilt_rate = float(row["tilt_loss_rate"]) * 100.0
    control_rate = float(row["control_loss_rate"]) * 100.0
    difference = float(row["difference_pp"])
    tilt_n = int(row["tilt_observations"])
    control_n = int(row["control_observations"])
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"After tilt-proxy (n={tilt_n})", f"Control (n={control_n})"]
    values = [tilt_rate, control_rate]
    bars = ax.barh(labels, values)
    ax.set_xlabel("Loss rate (%)")
    ax.set_title("Observed loss rate after tilt-proxy vs control")
    ax.set_xlim(0, max(values) + 10)
    for bar, value in zip(bars, values, strict=True):
        ax.text(value + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontsize=11)
    sign = "+" if difference >= 0 else ""
    ax.text(0.98, 0.05, f"Difference: {sign}{difference:.2f} pp", transform=ax.transAxes, ha="right", va="bottom", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
