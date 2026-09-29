from src.collect import _extract_game_ids, _parse_oldest_timestamp_ms


def test_extract_game_ids_and_timestamp() -> None:
    pgn = '''[Event "Rated Blitz game"]\n[Site "https://lichess.org/ABCDEFGH"]\n[UTCDate "2026.01.01"]\n[UTCTime "12:00:00"]\n\n1. e4 e5 1/2-1/2\n\n[Event "Rated Blitz game"]\n[Site "https://lichess.org/IJKLMNOP"]\n[UTCDate "2025.12.31"]\n[UTCTime "23:00:00"]\n\n1. d4 d5 1-0\n'''
    assert _extract_game_ids(pgn) == ["ABCDEFGH", "IJKLMNOP"]
    assert _parse_oldest_timestamp_ms(pgn) is not None



def test_build_params_accepts_multi_player_batch_size() -> None:
    from src.config import load_settings
    import logging
    from src.collect import LichessCollector

    settings = load_settings("config/config.yaml")
    collector = LichessCollector(settings, logging.getLogger("test"))
    params = collector._build_params(batch_size=500)
    assert params["max"] == 500


def test_build_params_supports_research_overrides_and_cutoff() -> None:
    from src.config import load_settings
    import logging
    from src.collect import LichessCollector

    settings = load_settings("config/config.yaml")
    collector = LichessCollector(settings, logging.getLogger("test"))
    params = collector._build_params(
        until_ms=123456,
        batch_size=3000,
        params_override={"perfType": "bullet", "variant": "standard"},
    )
    assert params["max"] == 3000
    assert params["until"] == 123456
    assert params["perfType"] == "bullet"
    assert params["variant"] == "standard"


def test_multi_collect_skips_complete_players_and_caps_batch_size(tmp_path, monkeypatch) -> None:
    from src.config import load_settings
    from src.pipeline import _run_multi_collect

    settings = load_settings("config/config.yaml")
    paths = settings.paths.model_copy(update={
        "sample_players_csv": tmp_path / "sample_players.csv",
        "multi_raw_dir": tmp_path / "raw",
        "log_file": tmp_path / "pipeline.log",
    })
    settings = settings.model_copy(update={"paths": paths})
    settings.resolve_path(paths.sample_players_csv).parent.mkdir(parents=True, exist_ok=True)
    settings.resolve_path(paths.sample_players_csv).write_text(
    "player_id,selection_order,eligible,selected,games_available,decisive_games,sampling_seed\n"
    "complete,1,True,True,500,500,42\n"
    "needs_more,2,True,True,500,500,42\n",
    encoding="utf-8",
    )

    raw_dir = settings.resolve_path(paths.multi_raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    complete_pgn = "\n\n".join(
        f'[Event "Rated"]\n[Site "https://lichess.org/{idx:08d}"]\n'
        '[UTCDate "2026.01.01"]\n[UTCTime "12:00:00"]\n\n1. e4 e5 1-0'
        for idx in range(settings.sampling.collection_max_games)
    ) + "\n"
    (raw_dir / "complete.pgn").write_text(complete_pgn, encoding="utf-8")

    calls = []

    class DummyCollector:
        def __init__(self, settings, logger):
            pass

        def collect_user(self, username, output, max_games=None, *, batch_size=None, until_ms=None, params_override=None, stop_condition=None, reuse_existing=False):
            calls.append((username, max_games, batch_size, until_ms, params_override, reuse_existing))

    monkeypatch.setattr("src.pipeline.LichessCollector", DummyCollector)

    class Logger:
        def info(self, *args, **kwargs):
            pass

    _run_multi_collect(settings, Logger())
    assert len(calls) == 2
    assert calls[0][0:3] == ("complete", 10000, 1000)
    assert calls[1][0:3] == ("needs_more", 10000, 1000)
    assert calls[0][3] is not None and calls[1][3] is not None
    assert calls[0][4] == {"perfType": "bullet", "variant": "standard"}
    assert calls[1][4] == {"perfType": "bullet", "variant": "standard"}
    assert calls[0][5] is True and calls[1][5] is True


def test_single_player_collect_keeps_existing_target(monkeypatch) -> None:
    from src.config import load_settings
    from src.collect import LichessCollector

    settings = load_settings("config/config.yaml")
    collector = LichessCollector(settings, None)
    calls = []

    def fake_collect_user(username, output, max_games=None, *, batch_size=None):
        calls.append((username, output, max_games, batch_size))
        return output

    monkeypatch.setattr(collector, "collect_user", fake_collect_user)
    result = collector.collect()

    assert result == settings.resolve_path(settings.paths.raw_pgn)
    assert calls == [(
        settings.user.username,
        settings.resolve_path(settings.paths.raw_pgn),
        settings.api.max_games,
        None,
    )]


def test_collect_user_reuses_sufficient_raw_without_network(tmp_path, monkeypatch) -> None:
    from src.config import load_settings
    from src.collect import LichessCollector
    import logging

    settings = load_settings("config/config.yaml")
    collector = LichessCollector(settings, logging.getLogger("test"))
    raw_path = tmp_path / "player.pgn"
    raw_path.write_text(
        '[Event "Rated Bullet"]\n'
        '[Site "https://lichess.org/ABCDEFGH"]\n'
        '[UTCDate "2026.01.01"]\n'
        '[UTCTime "12:00:00"]\n'
        '[TimeControl "60+0"]\n'
        '[Result "1-0"]\n\n'
        '1. e4 e5 1-0\n',
        encoding="utf-8",
    )

    calls = []
    monkeypatch.setattr(collector, "_download_chunk", lambda *args, **kwargs: calls.append(True))
    stop = lambda text: 'TimeControl "60+0"' in text and '[Result "1-0"]' in text

    collector.collect_user(
        "player",
        raw_path,
        10000,
        batch_size=3000,
        until_ms=123,
        params_override={"perfType": "bullet", "variant": "standard"},
        stop_condition=stop,
        reuse_existing=True,
    )

    assert calls == []
    assert "ABCDEFGH" in raw_path.read_text(encoding="utf-8")


def test_collect_user_fetches_additional_history_from_existing_raw(tmp_path, monkeypatch) -> None:
    from src.config import load_settings
    from src.collect import LichessCollector, PgnChunk
    import logging

    settings = load_settings("config/config.yaml")
    collector = LichessCollector(settings, logging.getLogger("test"))
    raw_path = tmp_path / "player.pgn"
    raw_path.write_text(
        '[Event "Rated Bullet"]\n'
        '[Site "https://lichess.org/OLDER123"]\n'
        '[UTCDate "2026.01.02"]\n'
        '[UTCTime "12:00:00"]\n'
        '[TimeControl "60+0"]\n'
        '[Result "1-0"]\n\n'
        '1. e4 e5 1-0\n',
        encoding="utf-8",
    )

    seen = []
    def fake_download(username, until_ms, batch_size=None, params_override=None):
        seen.append((username, until_ms, batch_size, params_override))
        return PgnChunk(
            text=(
                '[Event "Rated Bullet"]\n'
                '[Site "https://lichess.org/NEW12345"]\n'
                '[UTCDate "2026.01.01"]\n'
                '[UTCTime "11:00:00"]\n'
                '[TimeControl "60+0"]\n'
                '[Result "0-1"]\n\n'
                '1. d4 d5 0-1\n'
            ),
            source_url="test",
        )
    monkeypatch.setattr(collector, "_download_chunk", fake_download)

    stop = lambda text: text.count('[TimeControl "60+0"]') >= 2
    collector.collect_user(
        "player",
        raw_path,
        3,
        batch_size=3,
        stop_condition=stop,
        reuse_existing=True,
    )

    assert len(seen) == 1
    assert isinstance(seen[0][1], int)
    text = raw_path.read_text(encoding="utf-8")
    assert "OLDER123" in text
    assert "NEW12345" in text
