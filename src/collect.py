"""Lichess PGN collection with retry and pagination support."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import requests

from .config import Settings


_GAME_ID_RE = re.compile(r'^\[Site "https://lichess\.org/([^"/]+)"\]$', re.MULTILINE)
_UTC_DATE_RE = re.compile(r'^\[UTCDate "(\d{4}\.\d{2}\.\d{2})"\]$', re.MULTILINE)
_UTC_TIME_RE = re.compile(r'^\[UTCTime "(\d{2}:\d{2}:\d{2})"\]$', re.MULTILINE)


class EndpointNotFoundError(RuntimeError):
    """Raised when a Lichess endpoint returns HTTP 404."""


@dataclass(frozen=True)
class PgnChunk:
    """Single downloaded PGN chunk."""

    text: str
    source_url: str

    @property
    def content_hash(self) -> str:
        """Return a stable content hash."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _parse_oldest_timestamp_ms(pgn: str) -> int | None:
    """Return the oldest UTC timestamp found in a PGN chunk."""
    dates = _UTC_DATE_RE.findall(pgn)
    times = _UTC_TIME_RE.findall(pgn)
    if not dates or not times or len(dates) != len(times):
        return None
    values: list[int] = []
    for date_text, time_text in zip(dates, times, strict=False):
        try:
            dt = datetime.strptime(
                f"{date_text} {time_text}", "%Y.%m.%d %H:%M:%S"
            ).replace(tzinfo=timezone.utc)
            values.append(int(dt.timestamp() * 1000))
        except ValueError:
            continue
    return min(values) if values else None


def _extract_game_ids(pgn: str) -> list[str]:
    """Extract Lichess game IDs from PGN headers."""
    return _GAME_ID_RE.findall(pgn)


class LichessCollector:
    """Download a user's Lichess games as PGN."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": settings.api.accept,
                "User-Agent": settings.api.user_agent,
            }
        )

        token = __import__("os").getenv("LICHESS_TOKEN")
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def _request(self, url: str, params: dict[str, object]) -> requests.Response:
        """Perform one GET request with backoff for rate limits and server errors."""
        last_error: Exception | None = None
        for attempt in range(self.settings.api.max_retries + 1):
            if attempt > 0:
                delay = self.settings.api.backoff_seconds * (2 ** (attempt - 1))
                self.logger.warning("Retrying request in %.1fs", delay)
                time.sleep(delay)
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=120,
                    allow_redirects=self.settings.api.follow_redirects,
                )
                if response.status_code == 404:
                    raise EndpointNotFoundError(f"Endpoint not found: {url}")
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
                        "Lichess rate limit (429). Waiting %ss before retry.", delay
                    )
                    time.sleep(delay)
                    continue
                if response.status_code in {500, 502, 503, 504}:
                    last_error = requests.HTTPError(
                        f"Server error {response.status_code}: {response.text[:200]}"
                    )
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                self.logger.warning("Request failed: %s", exc)
        raise RuntimeError("Lichess request failed after retries") from last_error

    def _build_params(
        self,
        until_ms: int | None = None,
        batch_size: int | None = None,
        params_override: dict[str, object] | None = None,
    ) -> dict[str, object]:
        params = dict(self.settings.api.params)
        if params_override:
            params.update(params_override)
        params["max"] = batch_size if batch_size is not None else self.settings.api.batch_size
        if until_ms is not None:
            params["until"] = until_ms
        params = {k: v for k, v in params.items() if v is not None}
        return params

    def _download_chunk(
        self,
        username: str,
        until_ms: int | None,
        batch_size: int | None = None,
        params_override: dict[str, object] | None = None,
    ) -> PgnChunk:
        api_url = self.settings.api.base_url.rstrip("/") + self.settings.api.games_endpoint.format(
            username=username
        )
        fallback_url = self.settings.api.base_url.rstrip("/") + self.settings.api.fallback_endpoint.format(
            username=username
        )
        params = self._build_params(
            until_ms, batch_size=batch_size, params_override=params_override
        )
        try:
            response = self._request(api_url, params)
            return PgnChunk(text=response.text, source_url=response.url)
        except EndpointNotFoundError:
            if not self.settings.api.use_legacy_fallback_on_404:
                raise
            self.logger.warning(
                "Primary games endpoint failed; attempting fallback export endpoint: %s",
                fallback_url,
            )
            response = self._request(fallback_url, params)
            return PgnChunk(text=response.text, source_url=response.url)

    def collect_user(
        self,
        username: str,
        output: Path,
        max_games: int | None = None,
        *,
        batch_size: int | None = None,
        until_ms: int | None = None,
        params_override: dict[str, object] | None = None,
        stop_condition: Callable[[str], bool] | None = None,
        reuse_existing: bool = False,
    ) -> Path:
        """Collect one user's games, optionally reusing an existing raw artifact.

        When ``reuse_existing`` is enabled, an existing PGN becomes the starting
        point for pagination. If it already satisfies ``stop_condition`` no
        network request is made; otherwise only older history is fetched.
        The method is used by the multi-player collection stage and can also
        reuse an existing per-player raw artifact.
        """
        target = max_games if max_games is not None else self.settings.api.max_games
        seen_ids: set[str] = set()
        chunks: list[str] = []
        until_cursor_ms = until_ms
        total = 0

        if reuse_existing and output.exists():
            existing_text = output.read_text(encoding="utf-8")
            existing_ids = _extract_game_ids(existing_text)
            if existing_ids:
                seen_ids.update(existing_ids)
                chunks.append(existing_text.strip())
                total = len(seen_ids)
                if stop_condition is not None and stop_condition(existing_text):
                    self.logger.info(
                        "REUSE existing raw data for %s: %d games already satisfy research stop condition",
                        username,
                        total,
                    )
                    return output
                if total >= target:
                    self.logger.info(
                        "REUSE existing raw data for %s: %d games already reach collection target=%d",
                        username,
                        total,
                        target,
                    )
                    return output
                oldest_ms = _parse_oldest_timestamp_ms(existing_text)
                if oldest_ms is not None:
                    until_cursor_ms = oldest_ms - 1
                    self.logger.info(
                        "FETCH additional history for %s: existing=%d games, target=%d",
                        username,
                        total,
                        target,
                    )

        while total < target:
            self.logger.info(
                "Downloading PGN chunk for %s (total=%d/%d)",
                username,
                total,
                target,
            )
            effective_batch_size = (
                min(batch_size, target) if batch_size is not None else None
            )
            chunk = self._download_chunk(
                username, until_cursor_ms, batch_size=effective_batch_size, params_override=params_override
            )
            if not chunk.text.strip():
                self.logger.info("Empty response for %s, stopping collection", username)
                break

            game_ids = _extract_game_ids(chunk.text)
            new_ids = [gid for gid in game_ids if gid not in seen_ids]
            if not new_ids:
                self.logger.info("No new game IDs for %s, stopping collection", username)
                break

            seen_ids.update(new_ids)
            chunks.append(chunk.text.strip())
            total += len(new_ids)

            combined_so_far = "\n\n".join(chunks)
            if stop_condition is not None and stop_condition(combined_so_far):
                self.logger.info("Stopping collection for %s after research stop condition", username)
                break

            oldest_ms = _parse_oldest_timestamp_ms(chunk.text)
            if oldest_ms is None:
                self.logger.warning(
                    "Could not parse oldest game timestamp for %s; stopping pagination",
                    username,
                )
                break
            until_cursor_ms = oldest_ms - 1
            effective_batch_size = (
                min(batch_size, target) if batch_size is not None else self.settings.api.batch_size
            )
            if len(game_ids) < effective_batch_size:
                break
            time.sleep(self.settings.api.request_interval_seconds)

        combined = "\n\n".join(chunks) + ("\n" if chunks else "")
        if total > target:
            combined = self._trim_to_game_count(combined, target)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(combined, encoding="utf-8")
        self.logger.info("Saved %d games to %s", min(total, target), output)
        return output

    @staticmethod
    def _trim_to_game_count(pgn: str, target: int) -> str:
        """Trim a combined PGN to the first target games."""
        separators = list(re.finditer(r"(?=^\[Event )", pgn, flags=re.MULTILINE))
        if len(separators) <= target:
            return pgn
        cut = separators[target].start()
        return pgn[:cut].rstrip() + "\n"
