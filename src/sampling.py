"""Reproducible multi-player sampling and analysis helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import Settings
from .research import filter_research_games, select_latest_decisive_window
from .stats import analyze_tilt_proxy
from .parse import normalize_result
from .viz import plot_multi_player_tilt_effects


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_filename(value: str) -> str:
    """Return a filesystem-safe player filename."""
    return _SAFE_NAME_RE.sub("_", value).strip("._") or "player"


def _response_json(response: requests.Response) -> Any:
    response.raise_for_status()
    return response.json()


def _candidate_pool_hash(players: pd.DataFrame) -> str:
    """Return a stable hash for the sampling-frame contents."""
    ordered = players.sort_values(["player_id", "source_perftypes"])[
        ["player_id", "source_perftypes"]
    ]
    payload = "\n".join(
        f"{row.player_id}|{row.source_perftypes}"
        for row in ordered.itertuples(index=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LichessSamplingFrame:
    """Build a candidate pool from public Lichess leaderboards."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": settings.api.user_agent,
            }
        )

    def _get(self, url: str, params: dict[str, object] | None = None) -> requests.Response:
        """Issue one public API request with the project's retry policy."""
        last_error: Exception | None = None
        for attempt in range(self.settings.api.max_retries + 1):
            if attempt > 0:
                delay = self.settings.api.backoff_seconds * (2 ** (attempt - 1))
                time.sleep(delay)
            try:
                response = self.session.get(
                    url,
                    params=params or {},
                    timeout=60,
                    allow_redirects=self.settings.api.follow_redirects,
                )
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = max(
                            int(retry_after),
                            self.settings.api.retry_after_default_seconds,
                        ) if retry_after else self.settings.api.retry_after_default_seconds
                    except ValueError:
                        delay = self.settings.api.retry_after_default_seconds
                    self.logger.warning(
                        "Sampling API rate limit (429). Waiting %ss before retry.", delay
                    )
                    time.sleep(delay)
                    continue
                if response.status_code in {500, 502, 503, 504}:
                    last_error = requests.HTTPError(
                        f"Sampling API server error {response.status_code}"
                    )
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.settings.api.max_retries:
                    raise
                self.logger.warning("Sampling API request failed: %s", exc)
        raise RuntimeError("Sampling API request failed after retries") from last_error

    def _leaderboard_players(self) -> pd.DataFrame:
        """Fetch and union players from the configured public leaderboards."""
        rows: dict[str, set[str]] = {}
        base_url = self.settings.api.base_url.rstrip("/")
        for perf_type in self.settings.sampling.leaderboard_perf_types:
            endpoint = self.settings.sampling.leaderboard_endpoint.format(
                nb=self.settings.sampling.leaderboard_size,
                perf_type=perf_type,
            )
            response = self._get(base_url + endpoint)
            payload = _response_json(response)
            users = payload.get("users", []) if isinstance(payload, dict) else payload
            if not isinstance(users, list):
                raise ValueError(
                    f"Unexpected leaderboard response for {perf_type}: expected a user list"
                )
            for user in users:
                if not isinstance(user, dict):
                    continue
                username = user.get("username") or user.get("id")
                if username:
                    key = str(username).casefold()
                    rows.setdefault(key, set()).add(perf_type)
            time.sleep(self.settings.api.request_interval_seconds)

        pool = pd.DataFrame(
            [
                {
                    "player_id": key_value,
                    "source_perftypes": ",".join(sorted(perf_types)),
                }
                for key_value, perf_types in rows.items()
            ]
        )
        if pool.empty:
            raise ValueError("The sampling frame returned no candidates")
        return pool.sort_values("player_id").reset_index(drop=True)

    def _profile_counts(self, username: str) -> tuple[int, int] | None:
        """Return total and decisive game counts from public user data."""
        url = (
            self.settings.api.base_url.rstrip("/")
            + self.settings.sampling.user_endpoint.format(username=username)
        )
        try:
            response = self._get(url)
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                return None
            raise
        payload = _response_json(response)
        count = payload.get("count", {}) if isinstance(payload, dict) else {}
        if not isinstance(count, dict):
            return None
        wins = int(count.get("win", 0) or 0)
        losses = int(count.get("loss", 0) or 0)
        draws = int(count.get("draw", 0) or 0)
        games = int(count.get("all", wins + losses + draws) or 0)
        decisive = wins + losses
        return games, decisive

    def _candidate_paths(self, player_id: str) -> tuple[Path, Path]:
        """Return v11 bullet raw and processed paths used to screen one candidate."""
        safe = _safe_filename(player_id)
        raw = self.settings.resolve_path(self.settings.paths.multi_raw_dir) / f"{safe}.pgn"
        processed = (
            self.settings.resolve_path(self.settings.paths.multi_processed_dir)
            / f"{safe}.csv"
        )
        return raw, processed

    @staticmethod
    def _header_stop_condition(
        pgn: str,
        required: int,
        time_control: str,
        cutoff_datetime: str | None = None,
    ) -> bool:
        """Stop history collection once enough qualifying decisive headers are present."""
        cutoff = pd.Timestamp(cutoff_datetime) if cutoff_datetime else None
        blocks = re.split(r"(?=^\[Event )", pgn, flags=re.MULTILINE)
        decisive = 0
        for block in blocks:
            if not block.strip():
                continue
            headers = dict(
                re.findall(
                    r'^\[([A-Za-z0-9_]+) "([^"]*)"\]$',
                    block,
                    flags=re.MULTILINE,
                )
            )
            if headers.get("TimeControl", "").strip() != time_control:
                continue
            if headers.get("Result") not in {"1-0", "0-1"}:
                continue
            if cutoff is not None:
                date_text = headers.get("UTCDate") or headers.get("Date")
                time_text = headers.get("UTCTime") or headers.get("Time")
                if not date_text or not time_text:
                    continue
                try:
                    created_at = pd.Timestamp(
                        f"{date_text} {time_text}", tz="UTC"
                    )
                except (TypeError, ValueError):
                    continue
                if created_at > cutoff:
                    continue
            decisive += 1
            if decisive >= required:
                return True
        return False

    @staticmethod
    def _screen_pgn_headers(path: Path, username: str) -> pd.DataFrame:
        """Parse only PGN headers for fast sampling eligibility screening."""
        if not path.exists():
            return pd.DataFrame(
                columns=["game_id", "created_at", "username", "user_result", "time_control"]
            )

        text = path.read_text(encoding="utf-8", errors="replace")
        blocks = re.split(r"(?=^\[Event )", text, flags=re.MULTILINE)
        rows: list[dict[str, Any]] = []
        username_lower = username.casefold()
        for block in blocks:
            if not block.strip():
                continue
            headers = dict(
                re.findall(
                    r'^\[([A-Za-z0-9_]+) "([^"]*)"\]$',
                    block,
                    flags=re.MULTILINE,
                )
            )
            white = str(headers.get("White", ""))
            black = str(headers.get("Black", ""))
            if username_lower == white.casefold():
                user_color = "white"
            elif username_lower == black.casefold():
                user_color = "black"
            else:
                continue

            date_text = headers.get("UTCDate") or headers.get("Date")
            time_text = headers.get("UTCTime") or headers.get("Time")
            if not date_text or not time_text:
                continue
            try:
                created_at = pd.Timestamp(f"{date_text} {time_text}", tz="UTC")
            except (TypeError, ValueError):
                continue

            site = str(headers.get("Site", ""))
            game_id = site.rstrip("/").split("/")[-1][:8] if site else str(headers.get("GameId", "unknown"))
            result = str(headers.get("Result", "*"))
            user_result = normalize_result(result, user_color)
            rows.append(
                {
                    "game_id": game_id,
                    "created_at": created_at,
                    "username": username,
                    "user_result": user_result,
                    "time_control": str(headers.get("TimeControl", "")),
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=["game_id", "created_at", "username", "user_result", "time_control"]
            )
        frame = pd.DataFrame(rows)
        frame = (
            frame.sort_values("created_at")
            .drop_duplicates("game_id", keep="first")
            .reset_index(drop=True)
        )
        return frame

    def _collect_and_parse_candidate(self, player_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Collect/reuse bullet games and return parsed candidate rows plus collection metadata."""
        from .collect import LichessCollector, _extract_game_ids
        from .parse import parse_pgn_file

        raw_path, processed_path = self._candidate_paths(player_id)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        target = self.settings.sampling.collection_max_games
        batch_size = min(target, self.settings.sampling.screen_batch_size)
        cutoff_ms = int(pd.Timestamp(self.settings.research.cutoff_datetime).timestamp() * 1000)
        stop_condition = lambda text: self._header_stop_condition(
            text,
            self.settings.sampling.min_decisive_games,
            self.settings.research.time_control,
            self.settings.research.cutoff_datetime,
        )

        raw_existed_before = raw_path.exists()
        existing_text = raw_path.read_text(encoding="utf-8") if raw_existed_before else ""
        existing_ids_before = set(_extract_game_ids(existing_text))
        raw_was_sufficient_before = bool(existing_text) and stop_condition(existing_text)

        LichessCollector(self.settings, self.logger).collect_user(
            player_id,
            raw_path,
            target,
            batch_size=batch_size,
            until_ms=cutoff_ms,
            params_override={
                "perfType": "bullet",
                "variant": "standard",
            },
            stop_condition=stop_condition,
            reuse_existing=True,
        )

        raw_text = raw_path.read_text(encoding="utf-8")
        final_ids = set(_extract_game_ids(raw_text))
        games_in_raw_artifact = len(final_ids)
        games_fetched = max(0, len(final_ids - existing_ids_before))
        parsed = self._screen_pgn_headers(raw_path, player_id)
        collection_meta = {
            "games_fetched": int(games_fetched),
            "games_in_raw_artifact": int(games_in_raw_artifact),
            "raw_data_reused": bool(raw_was_sufficient_before),
            "additional_fetch_required": bool(raw_existed_before and not raw_was_sufficient_before),
        }
        return parsed, collection_meta

    @staticmethod
    def _is_eligible_from_decisive_count(
        analyzable_decisive_games: int,
        settings: Settings,
    ) -> bool:
        """Apply the v11 eligibility rule to analyzable decisive bullet games."""
        return analyzable_decisive_games >= settings.sampling.min_decisive_games

    @staticmethod
    def _final_player_counts(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Count games after the same cross-player dedup used by the analytical dataset."""
        if not frames:
            return pd.DataFrame(
                columns=[
                    "player_id",
                    "total_games_in_analysis_window",
                    "decisive_games_in_analysis_window",
                    "draws_in_analysis_window",
                ]
            )

        combined_parts: list[pd.DataFrame] = []
        for player_id, frame in frames.items():
            part = frame[["game_id", "user_result"]].copy()
            part["player_id"] = player_id
            combined_parts.append(part)
        combined = pd.concat(combined_parts, ignore_index=True)

        shared_ids = set(
            combined.groupby("game_id")["player_id"].nunique().loc[lambda values: values > 1].index.astype(str)
        )
        if shared_ids:
            combined = combined.loc[~combined["game_id"].astype(str).isin(shared_ids)].copy()

        combined["is_decisive"] = combined["user_result"].isin(["win", "loss"])
        combined["is_draw"] = combined["user_result"].eq("draw")
        counts = (
            combined.groupby("player_id", as_index=False)
            .agg(
                total_games_in_analysis_window=("game_id", "size"),
                decisive_games_in_analysis_window=("is_decisive", "sum"),
                draws_in_analysis_window=("is_draw", "sum"),
            )
        )
        for col in ["total_games_in_analysis_window", "decisive_games_in_analysis_window", "draws_in_analysis_window"]:
            counts[col] = counts[col].astype(int)
        return counts

    def select(self) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
        """Build, screen and reproducibly select players for the v11 bullet sample."""
        pool = self._leaderboard_players()
        pool_hash = _candidate_pool_hash(pool)
        candidate_ids = pool["player_id"].tolist()
        random.Random(self.settings.sampling.random_seed).shuffle(candidate_ids)

        selected: list[dict[str, Any]] = []
        selected_frames: dict[str, pd.DataFrame] = {}
        selected_windows: dict[str, dict[str, Any]] = {}
        candidates_checked = 0
        individually_eligible = 0

        for player_id in candidate_ids:
            candidates_checked += 1
            parsed, collection_meta = self._collect_and_parse_candidate(player_id)
            filtered = filter_research_games(parsed, self.settings)
            window, window_meta = select_latest_decisive_window(filtered, self.settings)

            if window_meta is None or not self._is_eligible_from_decisive_count(
                int(window_meta.get("available_decisive_games_before_cutoff", 0)),
                self.settings,
            ):
                available = 0 if window_meta is None else int(window_meta.get("available_decisive_games_before_cutoff", 0))
                self.logger.info(
                    "Rejecting %s: available analyzable decisive 60+0 games before cutoff=%d; threshold=%d",
                    player_id,
                    available,
                    self.settings.sampling.min_decisive_games,
                )
                continue

            individually_eligible += 1
            trial_frames = dict(selected_frames)
            trial_frames[player_id] = window.copy()
            final_counts = self._final_player_counts(trial_frames)
            final_by_player = final_counts.set_index("player_id")
            final_ok = all(
                self._is_eligible_from_decisive_count(
                    int(final_by_player.loc[pid, "decisive_games_in_analysis_window"]),
                    self.settings,
                )
                for pid in trial_frames
            )
            if not final_ok:
                self.logger.info(
                    "Rejecting %s: cross-player duplicate removal would reduce a selected player's decisive window below %d",
                    player_id,
                    self.settings.sampling.min_decisive_games,
                )
                continue

            selected_frames = trial_frames
            selected_windows[player_id] = dict(window_meta)
            selection_order = len(selected) + 1
            candidate_final = final_by_player.loc[player_id]
            selected.append(
                {
                    "player_id": player_id,
                    "selection_order": selection_order,
                    "selected": True,
                    "eligible": True,
                    "available_decisive_games_before_cutoff": int(window_meta["available_decisive_games_before_cutoff"]),
                    "selected_decisive_games": int(self.settings.sampling.min_decisive_games),
                    "games_fetched": int(collection_meta["games_fetched"]),
                    "games_in_raw_artifact": int(collection_meta["games_in_raw_artifact"]),
                    "games_after_time_filter": int(len(filtered)),
                    "decisive_60plus0_games": int(window_meta["available_decisive_games_before_cutoff"]),
                    "raw_data_reused": bool(collection_meta["raw_data_reused"]),
                    "additional_fetch_required": bool(collection_meta["additional_fetch_required"]),
                    "analysis_window_start": window_meta["analysis_window_start"],
                    "analysis_window_end": window_meta["analysis_window_end"],
                    "total_games_in_analysis_window": int(candidate_final["total_games_in_analysis_window"]),
                    "decisive_games_in_analysis_window": int(candidate_final["decisive_games_in_analysis_window"]),
                    "draws_in_analysis_window": int(candidate_final["draws_in_analysis_window"]),
                    "window_duration_days": float(window_meta["window_duration_days"]),
                    "eligibility_basis": "latest_500_decisive_60+0_before_cutoff",
                    "time_control": self.settings.research.time_control,
                    "cutoff_datetime": self.settings.research.cutoff_datetime,
                    "sampling_seed": self.settings.sampling.random_seed,
                }
            )
            self.logger.info(
                "Selected %s: decisive=%d, total_window=%d, draws=%d (%d/%d)",
                player_id,
                int(candidate_final["decisive_games_in_analysis_window"]),
                int(candidate_final["total_games_in_analysis_window"]),
                int(candidate_final["draws_in_analysis_window"]),
                len(selected),
                self.settings.sampling.target_players,
            )

            if len(selected) == self.settings.sampling.target_players:
                break

        if len(selected) < self.settings.sampling.target_players:
            raise RuntimeError(
                "Could not find enough eligible players in the bullet sampling frame: "
                f"requested={self.settings.sampling.target_players}, selected={len(selected)}, "
                f"candidates_checked={candidates_checked}, candidate_pool={len(pool)}"
            )

        selected_df = pd.DataFrame(selected)[
            [
                "player_id",
                "selection_order",
                "selected",
                "eligible",
                "available_decisive_games_before_cutoff",
                "selected_decisive_games",
                "games_fetched",
                "games_in_raw_artifact",
                "games_after_time_filter",
                "decisive_60plus0_games",
                "analysis_window_start",
                "analysis_window_end",
                "total_games_in_analysis_window",
                "decisive_games_in_analysis_window",
                "draws_in_analysis_window",
                "window_duration_days",
                "raw_data_reused",
                "additional_fetch_required",
                "eligibility_basis",
                "time_control",
                "cutoff_datetime",
                "sampling_seed",
            ]
        ]
        metadata = {
            "sampling_method": "random_selection_from_public_bullet_leaderboard_with_latest_500_decisive_window",
            "sampling_frame_source": "Lichess public bullet leaderboard",
            "sampling_frame_endpoint": self.settings.sampling.leaderboard_endpoint,
            "candidate_pool_size": int(len(pool)),
            "candidate_pool_sha256": pool_hash,
            "random_seed": int(self.settings.sampling.random_seed),
            "target_players": int(self.settings.sampling.target_players),
            "time_control": self.settings.research.time_control,
            "cutoff_datetime": self.settings.research.cutoff_datetime,
            "minimum_decisive_games": int(self.settings.sampling.min_decisive_games),
            "selection_unit": "latest_500_decisive_games",
            "selected_players": selected_df["player_id"].tolist(),
            "candidates_checked": int(candidates_checked),
            "individually_eligible_candidates": int(individually_eligible),
            "leaderboard_size": int(self.settings.sampling.leaderboard_size),
            "leaderboard_perf_types": list(self.settings.sampling.leaderboard_perf_types),
            "collection_max_games": int(self.settings.sampling.collection_max_games),
            "sampling_timestamp": datetime.now(timezone.utc).isoformat(),
            "selection_is_analysis_independent": True,
            "external_profile_counts_used_for_eligibility": False,
            "analysis_window_definition": "all analyzable 60+0 games from the earliest of the latest 500 decisive games through cutoff",
            "collection_strategy": "adaptive_pagination_with_raw_artifact_reuse",
            "raw_data_reuse_enabled": True,
            "selected_players_reused_raw_data": int(selected_df["raw_data_reused"].sum()),
            "selected_players_requiring_additional_fetch": int(selected_df["additional_fetch_required"].sum()),
            "rejected_candidates": int(candidates_checked - len(selected)),
        }
        return selected_df, metadata, pool


def save_sampling_artifacts(
    selected: pd.DataFrame,
    metadata: dict[str, Any],
    candidate_pool: pd.DataFrame,
    settings: Settings,
) -> None:
    """Save selected players, sampling metadata and the candidate pool."""
    selected_path = settings.resolve_path(settings.paths.sample_players_csv)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(selected_path, index=False)

    pool_path = settings.resolve_path(settings.paths.sampling_candidate_pool_csv)
    candidate_pool.to_csv(pool_path, index=False)

    metadata_path = settings.resolve_path(settings.paths.sampling_metadata_json)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_selected_players(settings: Settings) -> list[str]:
    """Load the selected player IDs in selection order."""
    path = settings.resolve_path(settings.paths.sample_players_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"Selected player file not found: {path}. Run the sampling stage first."
        )
    data = pd.read_csv(path)
    if "selected" in data.columns:
        data = data.loc[data["selected"].astype(bool)]
    data = data.sort_values("selection_order")
    return data["player_id"].astype(str).tolist()


def _pooled_descriptive_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build pooled descriptive tilt statistics without pooled inference."""
    decisive = df.loc[df["is_decisive"].astype(bool)].copy()
    tilt = decisive.loc[decisive["tilt_proxy"].astype(bool), "is_loss"].astype(float)
    control = decisive.loc[~decisive["tilt_proxy"].astype(bool), "is_loss"].astype(float)
    row = {
        "players": int(df["player_id"].nunique()),
        "games": int(len(df)),
        "decisive_games": int(len(decisive)),
        "sessions": int(df["session_id"].nunique()),
        "tilt_observations": int(len(tilt)),
        "control_observations": int(len(control)),
        "tilt_loss_rate": float(tilt.mean()) if len(tilt) else np.nan,
        "control_loss_rate": float(control.mean()) if len(control) else np.nan,
        "difference_pp": (
            float((tilt.mean() - control.mean()) * 100.0)
            if len(tilt) and len(control)
            else np.nan
        ),
        "inference": "descriptive_only; no player-clustered pooled inference introduced in v11",
    }
    return pd.DataFrame([row])


def analyze_multi_player(
    df: pd.DataFrame,
    settings: Settings,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate player-level and pooled descriptive statistics for the v11 windows."""
    required = {"player_id", "session_id", "is_decisive", "tilt_proxy", "is_loss", "created_at"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Multi-player analysis requires columns: {sorted(missing)}")

    working = df.copy().sort_values(["player_id", "created_at"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    cutoff = pd.Timestamp(settings.research.cutoff_datetime)

    for player_id, group in working.groupby("player_id", sort=False):
        decisive = group.loc[group["is_decisive"].astype(bool)].copy()
        tilt_mask = decisive["tilt_proxy"].astype(bool)
        tilt = decisive.loc[tilt_mask, "is_loss"].astype(float)
        control = decisive.loc[~tilt_mask, "is_loss"].astype(float)
        difference_pp = (
            float((tilt.mean() - control.mean()) * 100.0)
            if len(tilt) and len(control)
            else np.nan
        )
        ci_low_pp = np.nan
        ci_high_pp = np.nan
        inference_status = "descriptive_only"
        if len(tilt) >= settings.analysis.min_group_size and len(control) >= settings.analysis.min_group_size:
            result = analyze_tilt_proxy(group, settings, logger)
            ci_low_pp = 100.0 * float(result["risk_difference_ci"][0])
            ci_high_pp = 100.0 * float(result["risk_difference_ci"][1])
            inference_status = "existing_player_level_bootstrap_ci"

        total_games = int(len(group))
        decisive_games = int(len(decisive))
        draws = int(group["user_result"].eq("draw").sum())
        window_start = pd.Timestamp(group["created_at"].min())
        window_duration_days = float((cutoff - window_start).total_seconds() / 86400.0)

        rows.append(
            {
                "player_id": str(player_id),
                "selected_decisive_games": int(settings.sampling.min_decisive_games),
                "total_games_in_window": total_games,
                "decisive_games_in_window": decisive_games,
                "draws_in_window": draws,
                "games": total_games,
                "decisive_games": decisive_games,
                "sessions": int(group["session_id"].nunique()),
                "analysis_window_start": window_start.isoformat(),
                "analysis_window_end": cutoff.isoformat(),
                "window_duration_days": window_duration_days,
                "tilt_observations": int(len(tilt)),
                "control_observations": int(len(control)),
                "tilt_loss_rate": float(tilt.mean()) if len(tilt) else np.nan,
                "control_loss_rate": float(control.mean()) if len(control) else np.nan,
                "difference_pp": difference_pp,
            }
        )
        effects.append(
            {
                "player_id": str(player_id),
                "difference_pp": difference_pp,
                "ci_low_pp": ci_low_pp,
                "ci_high_pp": ci_high_pp,
                "tilt_observations": int(len(tilt)),
                "selection_order": np.nan,
                "inference_status": inference_status,
            }
        )
        quality_rows.append(
            {
                "player_id": str(player_id),
                "duplicate_player_game_ids": int(group.duplicated(["player_id", "game_id"]).sum()),
                "missing_session_ids": int(group["session_id"].isna().sum()),
                "games": total_games,
            }
        )

    player_summary = pd.DataFrame(rows)
    player_effects = pd.DataFrame(effects)
    pooled = _pooled_descriptive_summary(working)
    quality = pd.DataFrame(quality_rows)
    return player_summary, player_effects, pooled, quality


def render_multi_player_summary(
    player_summary: pd.DataFrame,
    pooled: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    """Render the v11 bullet-population and observation-window report."""
    pooled_row = pooled.iloc[0]
    lines = [
        "# Multi-Player Bullet Analysis",
        "",
        "The primary multi-player analysis uses a reproducibly selected random sample of active Lichess bullet players from the configured bullet leaderboard sampling frame.",
        "Player selection is independent of tilt outcomes, streaks, breaks, loss rates and model performance.",
        "",
        "## Population and sampling",
        "",
        f"- Candidate pool: {metadata['candidate_pool_size']}",
        f"- Target players: {metadata['target_players']}",
        f"- Selected players: {len(metadata['selected_players'])}",
        f"- Time control: **{metadata['time_control']}**",
        f"- Cutoff: **{metadata['cutoff_datetime']}**",
        f"- Eligibility: at least **{metadata['minimum_decisive_games']} analyzable decisive {metadata['time_control']} games before cutoff**",
        f"- Selection unit: **{metadata['selection_unit']}**",
        f"- Candidates checked: {metadata['candidates_checked']}",
        f"- Random seed: {metadata['random_seed']}",
        "",
        "The sampling frame is the public Lichess bullet leaderboard. It is a convenience sampling frame and should not be interpreted as representative of all Lichess users.",
        "Eligibility is determined from the actual parsed dataset after the fixed cutoff and exact 60+0 time-control filter; external profile game counts are not used as the final eligibility rule.",
        "",
        "## Observation window",
        "",
        "For each selected player, the fixed sample size is the latest 500 decisive 60+0 games before the cutoff. Decisive games define the sample size, but all analyzable 60+0 games between the timestamp of the earliest of those 500 decisive games and the cutoff are retained for sequential feature construction.",
        "This preserves draws and the complete temporal sequence needed for previous result, loss/win streaks, breaks, sessions and games-in-session.",
        "",
        "## Player-level window summary",
        "",
        player_summary.to_markdown(index=False),
        "",
        "## Pooled descriptive result",
        "",
        f"Total games in windows: {int(pooled_row['games'])}; decisive games: {int(pooled_row['decisive_games'])}; sessions: {int(pooled_row['sessions'])}.",
        f"Tilt observations: {int(pooled_row['tilt_observations'])}; control observations: {int(pooled_row['control_observations'])}.",
        f"Observed pooled difference: {pooled_row['difference_pp']:.2f} percentage points.",
        "",
        "The pooled result is descriptive only. Observations from the same player are not treated as fully independent in a new pooled inferential model.",
        "",
        "## Interpretation",
        "",
        "The baseline tilt definition remains the existing rule: previous game was a loss, break <= 5 minutes, and previous loss streak >= 2. Player-level variation is retained without ranking players.",
    ]
    return "\n".join(lines) + "\n"


def save_multi_player_analysis(
    player_summary: pd.DataFrame,
    player_effects: pd.DataFrame,
    pooled: pd.DataFrame,
    quality: pd.DataFrame,
    metadata: dict[str, Any],
    settings: Settings,
) -> None:
    """Persist multi-player analytical artifacts."""
    player_summary.to_csv(settings.resolve_path(settings.paths.player_summary_csv), index=False)
    pooled.to_csv(settings.resolve_path(settings.paths.multi_player_pooled_csv), index=False)
    quality_path = settings.resolve_path(settings.paths.multi_player_data_quality_csv)
    if quality_path.exists():
        existing_quality = pd.read_csv(quality_path)
        existing_quality.insert(0, "quality_scope", "global_parse")
        player_quality = quality.copy()
        player_quality.insert(0, "quality_scope", "player_features")
        combined_quality = pd.concat([existing_quality, player_quality], ignore_index=True, sort=False)
        combined_quality.to_csv(quality_path, index=False)
    else:
        quality = quality.copy()
        quality.insert(0, "quality_scope", "player_features")
        quality.to_csv(quality_path, index=False)
    metadata = dict(metadata)
    metadata["analysis_summary_created_at"] = datetime.now(timezone.utc).isoformat()
    report_path = settings.resolve_path(settings.paths.multi_player_analysis_md)
    report_path.write_text(render_multi_player_summary(player_summary, pooled, metadata), encoding="utf-8")

    effects = player_effects.copy()
    selection = pd.read_csv(settings.resolve_path(settings.paths.sample_players_csv))
    selection = selection.loc[selection["selected"].astype(bool), ["player_id", "selection_order"]]
    effects = effects.drop(columns=["selection_order"], errors="ignore").merge(
        selection, on="player_id", how="left", validate="one_to_one"
    )
    effects.to_csv(settings.resolve_path(settings.paths.player_tilt_effects_csv), index=False)
    plot_multi_player_tilt_effects(effects, settings.resolve_path(settings.paths.figures_dir) / "player_tilt_effects.png")
