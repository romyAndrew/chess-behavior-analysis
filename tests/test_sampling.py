from pathlib import Path

import pandas as pd

from src.config import load_settings
from src.research import filter_research_games, select_latest_decisive_window
from src.sampling import LichessSamplingFrame


class _Logger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def _sampler(target_players: int = 3, seed: int = 42) -> LichessSamplingFrame:
    base = load_settings(Path("config/config.yaml"))
    settings = base.model_copy(
        update={
            "sampling": base.sampling.model_copy(
                update={"target_players": target_players, "random_seed": seed}
            )
        }
    )
    return LichessSamplingFrame(settings, _Logger())


def _screen_frame(
    player_id: str,
    decisive: int = 500,
    draws: int = 1,
    *,
    start: str = "2026-01-01T00:00:00Z",
) -> pd.DataFrame:
    rows = []
    current = pd.Timestamp(start)
    for idx in range(decisive + draws):
        result = "win" if idx % 2 == 0 else "loss"
        if idx >= decisive:
            result = "draw"
        rows.append(
            {
                "game_id": f"{player_id}-{idx}",
                "created_at": current,
                "username": player_id,
                "user_result": result,
                "time_control": "60+0",
            }
        )
        current += pd.Timedelta(minutes=2)
    return pd.DataFrame(rows)


def _patch_frame(sampler: LichessSamplingFrame, frames: dict[str, pd.DataFrame]) -> None:
    pool = pd.DataFrame(
        {
            "player_id": list(frames),
            "source_perftypes": ["bullet"] * len(frames),
        }
    )
    sampler._leaderboard_players = lambda: pool.copy()  # type: ignore[method-assign]
    sampler._collect_and_parse_candidate = lambda username: (
        frames[username].copy(),
        {"games_fetched": 0, "games_in_raw_artifact": len(frames[username]), "raw_data_reused": True, "additional_fetch_required": False},
    )  # type: ignore[method-assign]
    sampler.settings = sampler.settings.model_copy(
        update={
            "api": sampler.settings.api.model_copy(update={"request_interval_seconds": 0.0}),
        }
    )


def test_header_stop_condition_respects_cutoff() -> None:
    pgn = (
        '[Event "Rated Bullet"]\n[Site "https://lichess.org/BEFORE01"]\n'
        '[UTCDate "2026.09.28"]\n[UTCTime "23:59:00"]\n'
        '[TimeControl "60+0"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n\n'
        '[Event "Rated Bullet"]\n[Site "https://lichess.org/AFTER001"]\n'
        '[UTCDate "2026.09.29"]\n[UTCTime "00:00:00"]\n'
        '[TimeControl "60+0"]\n[Result "0-1"]\n\n1. d4 d5 0-1\n'
    )
    assert not LichessSamplingFrame._header_stop_condition(
        pgn, 2, "60+0", "2026-09-28T23:59:59Z"
    )
    assert LichessSamplingFrame._header_stop_condition(
        pgn, 1, "60+0", "2026-09-28T23:59:59Z"
    )


def test_exact_eligibility_threshold_is_inclusive() -> None:
    sampler = _sampler()
    assert sampler._is_eligible_from_decisive_count(500, sampler.settings)


def test_499_decisive_games_is_not_eligible() -> None:
    sampler = _sampler()
    assert not sampler._is_eligible_from_decisive_count(499, sampler.settings)


def test_cutoff_and_exact_time_control_filtering() -> None:
    base = pd.Timestamp("2026-09-28T23:59:00Z")
    df = pd.DataFrame(
        [
            {"game_id": "g1", "created_at": base, "user_result": "win", "time_control": "60+0"},
            {"game_id": "g2", "created_at": base + pd.Timedelta(minutes=1), "user_result": "loss", "time_control": "60+1"},
            {"game_id": "g3", "created_at": pd.Timestamp("2026-09-29T00:00:00Z"), "user_result": "win", "time_control": "60+0"},
        ]
    )
    filtered = filter_research_games(df, load_settings(Path("config/config.yaml")))
    assert filtered["game_id"].tolist() == ["g1"]


def test_latest_500_decisive_window_retains_draw_inside_window() -> None:
    settings = load_settings(Path("config/config.yaml"))
    rows = []
    start = pd.Timestamp("2026-09-01T00:00:00Z")
    for idx in range(500):
        rows.append(
            {
                "game_id": f"d{idx}",
                "created_at": start + pd.Timedelta(minutes=idx * 2),
                "user_result": "win" if idx % 2 == 0 else "loss",
                "time_control": "60+0",
            }
        )
    insert_at = 250
    rows.insert(
        insert_at,
        {
            "game_id": "draw",
            "created_at": start + pd.Timedelta(minutes=insert_at * 2 + 1),
            "user_result": "draw",
            "time_control": "60+0",
        },
    )
    df = pd.DataFrame(rows)
    window, summary = select_latest_decisive_window(df, settings)
    assert summary is not None
    assert summary["selected_decisive_games"] == 500
    assert summary["total_games_in_analysis_window"] == 501
    assert summary["draws_in_analysis_window"] == 1
    assert "draw" in window["game_id"].tolist()


def test_same_seed_produces_same_selection() -> None:
    frames = {f"p{i}": _screen_frame(f"p{i}") for i in range(10)}
    first = _sampler(seed=42)
    second = _sampler(seed=42)
    _patch_frame(first, frames)
    _patch_frame(second, frames)

    selected_first, metadata_first, _ = first.select()
    selected_second, metadata_second, _ = second.select()

    assert selected_first["player_id"].tolist() == selected_second["player_id"].tolist()
    assert selected_first["selection_order"].tolist() == [1, 2, 3]
    assert metadata_first["candidate_pool_sha256"] == metadata_second["candidate_pool_sha256"]


def test_different_seed_can_change_selection() -> None:
    frames = {f"p{i}": _screen_frame(f"p{i}") for i in range(10)}
    first = _sampler(seed=42)
    second = _sampler(seed=7)
    _patch_frame(first, frames)
    _patch_frame(second, frames)

    selected_first, _, _ = first.select()
    selected_second, _, _ = second.select()

    assert selected_first["player_id"].tolist() != selected_second["player_id"].tolist()


def test_selection_records_window_columns_and_ignores_external_counts() -> None:
    sampler = _sampler(target_players=2, seed=42)
    frames = {
        "p0": _screen_frame("p0", decisive=500, draws=2),
        "p1": _screen_frame("p1", decisive=500, draws=3),
        "p2": _screen_frame("p2", decisive=499, draws=20),
    }
    _patch_frame(sampler, frames)

    selected, metadata, _ = sampler.select()

    required = {
        "player_id",
        "selection_order",
        "selected",
        "eligible",
        "available_decisive_games_before_cutoff",
        "selected_decisive_games",
        "analysis_window_start",
        "analysis_window_end",
        "total_games_in_analysis_window",
        "decisive_games_in_analysis_window",
        "draws_in_analysis_window",
        "eligibility_basis",
        "time_control",
        "cutoff_datetime",
        "sampling_seed",
    }
    assert required.issubset(selected.columns)
    assert (selected["selected_decisive_games"] == 500).all()
    assert (selected["decisive_games_in_analysis_window"] >= 500).all()
    assert metadata["time_control"] == "60+0"
    assert metadata["minimum_decisive_games"] == 500
    assert metadata["external_profile_counts_used_for_eligibility"] is False


def test_player_with_500_decisive_passes() -> None:
    frame = _screen_frame("pass", decisive=500, draws=0)
    window, summary = select_latest_decisive_window(frame, load_settings(Path("config/config.yaml")))
    assert summary is not None
    assert summary["selected_decisive_games"] == 500
    assert len(window) == 500


def test_header_screening_reads_metadata_without_parsing_moves(tmp_path) -> None:
    path = tmp_path / "sample.pgn"
    path.write_text(
        '[Event "Rated Bullet"]\n[Site "https://lichess.org/ABCDEF12"]\n'
        '[UTCDate "2026.09.28"]\n[UTCTime "23:00:00"]\n'
        '[White "alice"]\n[Black "bob"]\n[TimeControl "60+0"]\n[Result "1-0"]\n\n'
        '1. e4 e5 1-0\n',
        encoding="utf-8",
    )
    frame = LichessSamplingFrame._screen_pgn_headers(path, "alice")
    assert frame.loc[0, "game_id"] == "ABCDEF12"
    assert frame.loc[0, "time_control"] == "60+0"
    assert frame.loc[0, "user_result"] == "win"


def test_sampling_uses_configured_screen_batch_size() -> None:
    sampler = _sampler()
    assert sampler.settings.sampling.screen_batch_size == 1000
