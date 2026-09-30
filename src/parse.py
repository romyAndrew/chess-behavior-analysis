"""PGN parsing and validation.

The module uses python-chess at runtime but keeps import-time dependencies light,
which makes the parsing helpers straightforward to unit-test with mocks.
"""

from __future__ import annotations

import io
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings

_CLOCK_RE = re.compile(r"\[%clk\s+([0-9]+(?::[0-9]{1,2}){1,2}(?:\.[0-9]+)?)\]")
_EVAL_RE = re.compile(r"\[%eval\s+([^\]\s]+)\]")


class GameRecord(BaseModel):
    """Validated game-level observation."""

    model_config = ConfigDict(extra="forbid")

    game_id: str
    created_at: datetime
    username: str
    white_player: str
    black_player: str
    user_color: str
    result: str
    user_result: str
    rated: bool | None = None
    speed: str | None = None
    perf: str | None = None
    variant: str | None = None
    time_control: str | None = None
    initial_time_seconds: float | None = Field(default=None, ge=0)
    increment_seconds: float = Field(default=0.0, ge=0)
    white_rating: float | None = None
    black_rating: float | None = None
    opponent_rating: float | None = None
    rating_diff: float | None = None
    ply_count: int = Field(ge=0)
    move_count: int = Field(ge=0)
    user_move_count: int = Field(ge=0)
    total_thinking_time_seconds: float | None = None
    avg_move_time: float | None = None
    median_move_time: float | None = None
    time_pressure: float | None = None
    mean_eval_loss: float | None = None
    eval_coverage: float | None = None
    clock_coverage: float | None = None
    opening: str | None = None
    termination: str | None = None


def parse_clock_value(value: str) -> float:
    """Parse a PGN clock string into seconds.

    Args:
        value: Clock value such as ``0:01:23.50`` or ``1:23``.

    Returns:
        Clock time in seconds.
    """
    parts = value.strip().split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    return float(value)


def parse_clock_comment(comment: str) -> float | None:
    """Extract remaining clock time from a PGN comment."""
    match = _CLOCK_RE.search(comment or "")
    return parse_clock_value(match.group(1)) if match else None


def parse_eval_comment(comment: str) -> float | None:
    """Extract a white-perspective evaluation in pawns from a PGN comment.

    Mate scores are converted to a capped +/-10 pawn-equivalent value.
    """
    match = _EVAL_RE.search(comment or "")
    if not match:
        return None
    raw = match.group(1).strip()
    if raw.startswith("#"):
        tail = raw[1:]
        sign = -1.0 if tail.startswith("-") else 1.0
        return sign * 10.0
    try:
        return float(raw)
    except ValueError:
        return None


def parse_time_control(time_control: str | None) -> tuple[float | None, float]:
    """Parse common Lichess time-control strings.

    Args:
        time_control: PGN TimeControl value such as ``300+3``.

    Returns:
        Tuple of initial seconds and increment seconds.
    """
    if not time_control or time_control in {"-", "?"}:
        return None, 0.0
    last_stage = time_control.split(":")[-1]
    increment = 0.0
    if "+" in last_stage:
        base, inc = last_stage.split("+", 1)
        try:
            increment = float(inc)
        except ValueError:
            increment = 0.0
    else:
        base = last_stage
    if "/" in base:
        _, seconds = base.split("/", 1)
    else:
        seconds = base
    try:
        return float(seconds), increment
    except ValueError:
        return None, increment


def infer_speed(time_control: str | None) -> str | None:
    """Infer the Lichess time-control category from a PGN TimeControl value.

    Lichess defines the estimated duration as initial seconds + 40 * increment.
    Categories are UltraBullet <= 29s, Bullet <= 179s, Blitz <= 479s,
    Rapid <= 1499s, and Classical >= 1500s.
    """
    initial, increment = parse_time_control(time_control)
    if initial is None:
        return None
    estimated = initial + 40.0 * increment
    if estimated <= 29:
        return "ultrabullet"
    if estimated <= 179:
        return "bullet"
    if estimated <= 479:
        return "blitz"
    if estimated <= 1499:
        return "rapid"
    return "classical"


def normalize_result(result: str, user_color: str) -> str:
    """Convert PGN result to the user's perspective."""
    if result == "1/2-1/2":
        return "draw"
    if result == "1-0":
        return "win" if user_color == "white" else "loss"
    if result == "0-1":
        return "loss" if user_color == "white" else "win"
    return "unknown"


def _node_turn(node: Any) -> str:
    """Get the side that made the move represented by a PGN node.

    python-chess exposes ``GameNode.turn()`` as a *method* describing the
    side to move at the node. For a child node that is the side to move
    *after* the move, so it is not the mover. The parent board is therefore
    the authoritative source for real python-chess nodes. Test doubles may
    expose a boolean ``turn`` attribute directly.
    """
    parent = getattr(node, "parent", None)
    if parent is not None and hasattr(parent, "board"):
        return "white" if parent.board().turn else "black"

    turn_attr = getattr(node, "turn", None)
    if turn_attr is not None and not callable(turn_attr):
        return "white" if bool(turn_attr) else "black"

    board = node.board()
    return "white" if board.turn else "black"


def _game_id_from_headers(headers: Any) -> str:
    site = str(headers.get("Site", ""))
    if site:
        return site.rstrip("/").split("/")[-1][:8]
    return str(headers.get("GameId", "unknown"))


def _header_datetime(headers: Any) -> datetime:
    date_text = headers.get("UTCDate") or headers.get("Date")
    time_text = headers.get("UTCTime") or headers.get("Time")
    if date_text and time_text:
        try:
            return datetime.strptime(
                f"{date_text} {time_text}", "%Y.%m.%d %H:%M:%S"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError("PGN does not contain a parseable UTCDate/UTCTime pair")


def parse_game(game: Any, username: str, settings: Settings, logger: logging.Logger) -> GameRecord | None:
    """Convert one python-chess game object to a validated record.

    Args:
        game: Parsed python-chess PGN game or a compatible test double.
        username: Target Lichess username.
        settings: Validated project settings.
        logger: Project logger.

    Returns:
        A validated ``GameRecord`` or ``None`` for an invalid game.
    """
    headers = game.headers
    white = str(headers.get("White", ""))
    black = str(headers.get("Black", ""))
    if username.lower() == white.lower():
        user_color = "white"
        opponent = black
        user_rating = _to_float(headers.get("WhiteElo"))
        opponent_rating = _to_float(headers.get("BlackElo"))
    elif username.lower() == black.lower():
        user_color = "black"
        opponent = white
        user_rating = _to_float(headers.get("BlackElo"))
        opponent_rating = _to_float(headers.get("WhiteElo"))
    else:
        logger.warning("Skipping game not containing target username: %s vs %s", white, black)
        return None

    created_at = _header_datetime(headers)
    result = str(headers.get("Result", "*"))
    user_result = normalize_result(result, user_color)
    initial, increment = parse_time_control(headers.get("TimeControl"))

    # We intentionally start clock history as unknown. If the export omits the
    # first clock annotation for a side, treating the first observed clock as if
    # it followed the initial time creates a fake multi-minute move.
    previous_clock: dict[str, float | None] = {"white": None, "black": None}
    side_move_number: dict[str, int] = {"white": 0, "black": 0}
    previous_eval: float | None = None
    user_move_times: list[float] = []
    user_clock_after: list[float] = []
    user_eval_losses: list[float] = []
    ply_count = 0
    user_move_count = 0

    try:
        nodes: Iterable[Any] = game.mainline()
        for node in nodes:
            ply_count += 1
            turn = _node_turn(node)
            side_move_number[turn] += 1
            if turn == user_color:
                user_move_count += 1
            comment = str(getattr(node, "comment", "") or "")
            clock_after = parse_clock_comment(comment)
            eval_after = parse_eval_comment(comment)

            if clock_after is not None:
                prev = previous_clock.get(turn)
                spent: float | None = None
                if prev is not None:
                    spent = prev + increment - clock_after
                elif side_move_number[turn] == 1 and initial is not None:
                    # The first clock annotation can be converted to move time
                    # because the initial clock is known from TimeControl.
                    spent = initial + increment - clock_after

                if spent is not None:
                    # A clock drop larger than the available clock budget is
                    # treated as missing rather than becoming a bogus duration.
                    budget = (prev if prev is not None else initial)
                    if spent < -0.25 or (budget is not None and spent > budget + increment + 60.0):
                        spent = None

                if spent is not None and spent >= 0 and turn == user_color:
                    user_move_times.append(spent)
                previous_clock[turn] = clock_after
                if turn == user_color:
                    user_clock_after.append(clock_after)

            if eval_after is not None:
                if previous_eval is not None and turn == user_color:
                    before = previous_eval if user_color == "white" else -previous_eval
                    after = eval_after if user_color == "white" else -eval_after
                    user_eval_losses.append(max(0.0, before - after))
                previous_eval = eval_after
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        logger.exception("Broken PGN mainline for game %s: %s", _game_id_from_headers(headers), exc)
        return None

    total_user_time = sum(user_move_times) if user_move_times else None
    avg_move_time = float(total_user_time / len(user_move_times)) if user_move_times else None
    median_move_time = float(pd.Series(user_move_times).median()) if user_move_times else None
    clock_coverage = (
        float(len(user_move_times) / user_move_count)
        if user_move_count
        else None
    )

    time_pressure_fraction = None
    if user_clock_after:
        threshold = settings.features.time_pressure_seconds
        if initial is not None:
            threshold = max(threshold, initial * settings.features.time_pressure_fraction_of_initial)
        time_pressure_fraction = float(sum(x <= threshold for x in user_clock_after) / len(user_clock_after))

    eval_coverage = float(len(user_eval_losses) / max(user_move_count, 1)) if user_move_count else None
    mean_eval_loss = float(sum(user_eval_losses) / len(user_eval_losses)) if user_eval_losses else None

    rating_diff = None
    if user_rating is not None and opponent_rating is not None:
        rating_diff = user_rating - opponent_rating

    return GameRecord(
        game_id=_game_id_from_headers(headers),
        created_at=created_at,
        username=username,
        white_player=white,
        black_player=black,
        user_color=user_color,
        result=result,
        user_result=user_result,
        rated=_to_bool(headers.get("Rated")),
        speed=str(headers.get("Speed", "")) or infer_speed(headers.get("TimeControl")),
        perf=str(headers.get("Perf", "")) or None,
        variant=str(headers.get("Variant", "standard")) or None,
        time_control=str(headers.get("TimeControl", "")) or None,
        initial_time_seconds=initial,
        increment_seconds=increment,
        white_rating=_to_float(headers.get("WhiteElo")),
        black_rating=_to_float(headers.get("BlackElo")),
        opponent_rating=opponent_rating,
        rating_diff=rating_diff,
        ply_count=ply_count,
        move_count=math.ceil(ply_count / 2),
        user_move_count=user_move_count,
        total_thinking_time_seconds=total_user_time,
        avg_move_time=avg_move_time,
        median_move_time=median_move_time,
        time_pressure=time_pressure_fraction,
        mean_eval_loss=mean_eval_loss,
        eval_coverage=eval_coverage,
        clock_coverage=clock_coverage,
        opening=str(headers.get("Opening", "")) or None,
        termination=str(headers.get("Termination", "")) or None,
    )


def _to_float(value: Any) -> float | None:
    try:
        if value in {None, "", "-", "?"}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    return str(value).lower() in {"true", "1", "yes"}


def parse_pgn_file(
    path: Path,
    settings: Settings,
    logger: logging.Logger,
    *,
    username: str,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Parse a PGN file into a validated DataFrame."""
    try:
        import chess.pgn
    except ImportError as exc:
        raise RuntimeError(
            "python-chess is required. Install pinned dependencies with `make install`."
        ) from exc

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            offset = handle.tell()
            try:
                game = chess.pgn.read_game(handle)
            except Exception as exc:
                logger.exception("PGN parse failed near byte offset %s: %s", offset, exc)
                continue
            if game is None:
                break
            if getattr(game, "errors", None):
                logger.warning("Game contained parser errors near byte offset %s: %s", offset, game.errors)
            record = parse_game(game, username, settings, logger)
            if record is not None:
                records.append(record.model_dump())

    df = pd.DataFrame(records)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df = df.sort_values("created_at").drop_duplicates("game_id", keep="first").reset_index(drop=True)
    if output_path is None:
        raise ValueError("output_path is required for the final multi-player pipeline")
    output = output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Parsed %d games into %s", len(df), output)
    return df
