from pathlib import Path
from types import SimpleNamespace

from src.config import load_settings
from src.parse import infer_speed, parse_clock_comment, parse_eval_comment, parse_game, parse_time_control


def test_clock_parser() -> None:
    assert parse_clock_comment("{ [%clk 0:01:23.50] }") == 83.5
    assert parse_clock_comment("no clock") is None


def test_eval_parser() -> None:
    assert parse_eval_comment("{ [%eval 0.42] }") == 0.42
    assert parse_eval_comment("{ [%eval #-4] }") == -10.0
    assert parse_eval_comment("no eval") is None


def test_time_control_parser() -> None:
    assert parse_time_control("300+3") == (300.0, 3.0)
    assert parse_time_control("-") == (None, 0.0)
    assert infer_speed("180+2") == "blitz"
    assert infer_speed("600+0") == "rapid"


def test_parse_game_with_mocked_nodes() -> None:
    settings = load_settings(Path("config/config.yaml"))
    nodes = [
        SimpleNamespace(comment="[%clk 0:04:59] [%eval 0.1]", turn=True),
        SimpleNamespace(comment="[%clk 0:05:00] [%eval 0.2]", turn=False),
        SimpleNamespace(comment="[%clk 0:04:56] [%eval -0.3]", turn=True),
        SimpleNamespace(comment="[%clk 0:04:59] [%eval 0.0]", turn=False),
    ]
    fake_game = SimpleNamespace(
        headers={
            "Site": "https://lichess.org/ABCDEFGH",
            "UTCDate": "2026.01.01",
            "UTCTime": "12:00:00",
            "White": "bat1skaf",
            "Black": "opponent",
            "WhiteElo": "1500",
            "BlackElo": "1490",
            "Result": "1-0",
            "Rated": "True",
            "Speed": "blitz",
            "Perf": "blitz",
            "Variant": "standard",
            "TimeControl": "300+3",
        },
        errors=[],
        mainline=lambda: nodes,
    )

    class DummyLogger:
        def warning(self, *args, **kwargs):
            return None
        def exception(self, *args, **kwargs):
            return None

    record = parse_game(fake_game, "bat1skaf", settings, DummyLogger())
    assert record is not None
    assert record.game_id == "ABCDEFGH"
    assert record.user_result == "win"
    assert record.user_move_count == 2
    assert record.avg_move_time is not None


def test_user_move_count_is_not_clock_coverage() -> None:
    settings = load_settings(Path("config/config.yaml"))
    nodes = [
        SimpleNamespace(comment="", turn=True),
        SimpleNamespace(comment="", turn=False),
        SimpleNamespace(comment="", turn=True),
        SimpleNamespace(comment="", turn=False),
    ]
    fake_game = SimpleNamespace(
        headers={
            "Site": "https://lichess.org/IJKLMNOP",
            "UTCDate": "2026.01.01",
            "UTCTime": "12:00:00",
            "White": "bat1skaf",
            "Black": "opponent",
            "Result": "1/2-1/2",
            "TimeControl": "180+2",
        },
        errors=[],
        mainline=lambda: nodes,
    )

    class DummyLogger:
        def warning(self, *args, **kwargs):
            return None
        def exception(self, *args, **kwargs):
            return None

    record = parse_game(fake_game, "bat1skaf", settings, DummyLogger())
    assert record is not None
    assert record.user_move_count == 2
    assert record.clock_coverage == 0.0
    assert record.avg_move_time is None


def test_clock_durations_do_not_use_initial_clock_when_first_annotation_is_missing() -> None:
    settings = load_settings(Path("config/config.yaml"))
    nodes = [
        SimpleNamespace(comment="", turn=True),
        SimpleNamespace(comment="[%clk 0:02:59]", turn=False),
        SimpleNamespace(comment="[%clk 0:01:59]", turn=True),
        SimpleNamespace(comment="[%clk 0:02:56]", turn=False),
        SimpleNamespace(comment="[%clk 0:01:56]", turn=True),
    ]
    fake_game = SimpleNamespace(
        headers={
            "Site": "https://lichess.org/QRSTUVWX",
            "UTCDate": "2026.01.01",
            "UTCTime": "12:00:00",
            "White": "bat1skaf",
            "Black": "opponent",
            "Result": "1-0",
            "TimeControl": "180+2",
        },
        errors=[],
        mainline=lambda: nodes,
    )

    class DummyLogger:
        def warning(self, *args, **kwargs):
            return None
        def exception(self, *args, **kwargs):
            return None

    record = parse_game(fake_game, "bat1skaf", settings, DummyLogger())
    assert record is not None
    # Only the second white clock annotation can be used to measure a white move.
    assert record.user_move_count == 3
    assert record.clock_coverage == 1 / 3
    assert record.avg_move_time is not None
    assert record.avg_move_time < 10


def test_real_style_node_turn_method_uses_parent_board_for_mover() -> None:
    settings = load_settings(Path("config/config.yaml"))

    class DummyBoard:
        def __init__(self, turn: bool):
            self.turn = turn

    class DummyParent:
        def board(self):
            return DummyBoard(True)

    class DummyNode:
        parent = DummyParent()
        comment = "[%clk 0:02:59]"

        def turn(self):
            # python-chess GameNode.turn() is the side to move at this node,
            # i.e. the opponent after the move represented by this node.
            return False

    nodes = [DummyNode()]
    fake_game = SimpleNamespace(
        headers={
            "Site": "https://lichess.org/REALTURN1",
            "UTCDate": "2026.01.01",
            "UTCTime": "12:00:00",
            "White": "bat1skaf",
            "Black": "opponent",
            "Result": "1-0",
            "TimeControl": "180+2",
        },
        errors=[],
        mainline=lambda: nodes,
    )

    class DummyLogger:
        def warning(self, *args, **kwargs):
            return None
        def exception(self, *args, **kwargs):
            return None

    record = parse_game(fake_game, "bat1skaf", settings, DummyLogger())
    assert record is not None
    assert record.user_move_count == 1
    assert record.avg_move_time is not None
