import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_settings
from src.heterogeneity import analyze_player_heterogeneity, summarize_player_heterogeneity


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _settings_with_sample(tmp_path: Path):
    base = load_settings(Path("config/config.yaml"))
    sample_path = tmp_path / "sample_players.csv"
    pd.DataFrame(
        {
            "player_id": ["p1", "p2", "p3"],
            "selection_order": [1, 2, 3],
            "selected": [True, True, True],
        }
    ).to_csv(sample_path, index=False)
    paths = base.paths.model_copy(update={"sample_players_csv": sample_path})
    return base.model_copy(update={"paths": paths})


def _fixture() -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    specs = {
        "p1": (12, 24, 6, 4),
        "p2": (2, 3, 1, 2),
        "p3": (0, 5, 0, 0),
    }
    game_index = 0
    for player, (tilt_n, control_n, tilt_losses, control_losses) in specs.items():
        for idx in range(tilt_n + control_n):
            is_tilt = idx < tilt_n
            is_loss = (
                idx < tilt_losses if is_tilt else idx - tilt_n < control_losses
            )
            rows.append(
                {
                    "player_id": player,
                    "game_id": f"g{game_index}",
                    "created_at": base + pd.Timedelta(minutes=game_index),
                    "session_id": game_index // 4,
                    "is_decisive": True,
                    "tilt_proxy": is_tilt,
                    "is_loss": is_loss,
                    "user_result": "loss" if is_loss else "win",
                }
            )
            game_index += 1
    return pd.DataFrame(rows)


def test_player_level_results_have_one_record_per_player(tmp_path: Path) -> None:
    settings = _settings_with_sample(tmp_path)
    result = analyze_player_heterogeneity(_fixture(), settings, _Logger())
    assert result["player_id"].nunique() == 3
    assert len(result) == 3


def test_tilt_and_control_counts_equal_analyzable_observations(tmp_path: Path) -> None:
    settings = _settings_with_sample(tmp_path)
    df = _fixture()
    result = analyze_player_heterogeneity(df, settings, _Logger())
    for row in result.itertuples(index=False):
        source = df.loc[df["player_id"] == row.player_id]
        assert row.tilt_n + row.control_n == int(source["is_decisive"].sum())


def test_point_estimate_is_difference_between_group_loss_rates(tmp_path: Path) -> None:
    settings = _settings_with_sample(tmp_path)
    result = analyze_player_heterogeneity(_fixture(), settings, _Logger())
    p1 = result.loc[result["player_id"] == "p1"].iloc[0]
    expected = (6 / 12 - 4 / 24) * 100
    assert np.isclose(p1["difference_pp"], expected)


def test_zero_tilt_count_does_not_crash_or_get_dropped(tmp_path: Path) -> None:
    settings = _settings_with_sample(tmp_path)
    result = analyze_player_heterogeneity(_fixture(), settings, _Logger())
    p3 = result.loc[result["player_id"] == "p3"].iloc[0]
    assert int(p3["tilt_n"]) == 0
    assert np.isnan(p3["difference_pp"])
    assert p3["inference_status"] == "not_estimable_empty_comparison_group"


def test_small_tilt_groups_are_retained_and_flagged(tmp_path: Path) -> None:
    settings = _settings_with_sample(tmp_path)
    result = analyze_player_heterogeneity(_fixture(), settings, _Logger())
    p2 = result.loc[result["player_id"] == "p2"].iloc[0]
    assert int(p2["tilt_n"]) == 2
    assert bool(p2["small_tilt_group"])
    assert p2["player_id"] in result["player_id"].tolist()


def test_summary_counts_signs_and_range(tmp_path: Path) -> None:
    settings = _settings_with_sample(tmp_path)
    result = analyze_player_heterogeneity(_fixture(), settings, _Logger())
    summary = summarize_player_heterogeneity(result)
    assert summary["players"] == 3
    assert summary["positive_estimates"] + summary["negative_estimates"] == 2
    assert summary["minimum_difference_pp"] <= summary["maximum_difference_pp"]
    assert "p2" in summary["players_below_existing_min_group_size"]
