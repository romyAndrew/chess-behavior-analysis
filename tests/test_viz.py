from pathlib import Path

import pandas as pd

from src.viz import plot_pooled_tilt_effect, plot_walk_forward_roc_auc


def test_retained_figures_are_written(tmp_path: Path) -> None:
    pooled = pd.DataFrame([{
        "tilt_loss_rate": 0.4446,
        "control_loss_rate": 0.3180,
        "difference_pp": 12.66,
        "tilt_observations": 641,
        "control_observations": 6859,
    }])
    folds = pd.DataFrame([
        {"fold": 1, "model": "rating_only", "roc_auc": 0.51},
        {"fold": 1, "model": "baseline", "roc_auc": 0.52},
        {"fold": 2, "model": "rating_only", "roc_auc": 0.49},
        {"fold": 2, "model": "baseline", "roc_auc": 0.53},
    ])
    plot_pooled_tilt_effect(pooled, tmp_path / "pooled_tilt_effect.png")
    plot_walk_forward_roc_auc(folds, tmp_path / "walk_forward_roc_auc.png")
    assert (tmp_path / "pooled_tilt_effect.png").exists()
    assert (tmp_path / "walk_forward_roc_auc.png").exists()
