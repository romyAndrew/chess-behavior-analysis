"""Configuration loading and validation with Pydantic."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    timezone: str = "UTC"

    @field_validator("timezone")
    @classmethod
    def timezone_must_be_valid(cls, value: str) -> str:
        value = value.strip()
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone: {value}") from exc
        return value


class UserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str

    @field_validator("username")
    @classmethod
    def username_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("username must not be empty")
        return value


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw_pgn: Path
    games_csv: Path
    features_csv: Path
    stats_json: Path
    model_metrics_json: Path
    model_coefficients_csv: Path
    model_comparison_csv: Path
    multi_model_metrics_json: Path
    multi_model_coefficients_csv: Path
    multi_model_comparison_csv: Path
    analysis_summary_md: Path
    tilt_sensitivity_csv: Path
    tilt_sensitivity_md: Path
    tilt_bootstrap_comparison_csv: Path
    tilt_bootstrap_comparison_md: Path
    tilt_sensitivity_session_bootstrap_csv: Path
    walk_forward_metrics_csv: Path
    walk_forward_fold_metrics_csv: Path
    walk_forward_metrics_md: Path
    multi_walk_forward_metrics_csv: Path
    multi_walk_forward_fold_metrics_csv: Path
    multi_walk_forward_metrics_md: Path
    sample_players_csv: Path
    sampling_metadata_json: Path
    sampling_candidate_pool_csv: Path
    multi_raw_dir: Path
    multi_processed_dir: Path
    multi_games_csv: Path
    multi_features_csv: Path
    player_summary_csv: Path
    player_tilt_effects_csv: Path
    multi_player_pooled_csv: Path
    multi_player_data_quality_csv: Path
    multi_player_analysis_md: Path
    player_heterogeneity_csv: Path
    player_heterogeneity_summary_json: Path
    log_file: Path
    figures_dir: Path
    synthetic_null_distribution_csv: Path
    synthetic_null_summary_csv: Path
    synthetic_null_summary_json: Path
    opponent_context_csv: Path
    opponent_adjusted_models_csv: Path
    opponent_adjusted_coefficients_csv: Path
    v13_analysis_md: Path


class ApiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str
    games_endpoint: str
    fallback_endpoint: str
    accept: str
    user_agent: str
    max_games: int = Field(gt=0, le=100_000)
    batch_size: int = Field(gt=0, le=3000)
    request_interval_seconds: float = Field(ge=0.0)
    max_retries: int = Field(ge=0, le=20)
    backoff_seconds: float = Field(ge=0.0)
    retry_after_default_seconds: int = Field(ge=1)
    follow_redirects: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    use_legacy_fallback_on_404: bool = True


class FeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_gap_minutes: float = Field(gt=0)
    short_break_minutes: float = Field(gt=0)
    min_loss_streak: int = Field(ge=1)
    time_pressure_seconds: float = Field(gt=0)
    time_pressure_fraction_of_initial: float = Field(gt=0, lt=1)
    min_eval_coverage_for_quality: float = Field(gt=0, le=1)


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confidence_level: float = Field(gt=0, lt=1)
    min_group_size: int = Field(ge=2)
    bootstrap_iterations: int = Field(ge=500)
    bootstrap_seed: int


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    test_size: float = Field(gt=0, lt=0.5)
    random_state: int
    C: float = Field(gt=0)
    max_iter: int = Field(gt=100)
    class_weight: str | None = None


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    initial_train_size: int = Field(gt=1)
    test_window_size: int = Field(gt=0)
    min_training_size: int = Field(gt=1)


class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cutoff_datetime: str
    time_control: str

    @field_validator("cutoff_datetime")
    @classmethod
    def cutoff_must_be_utc(cls, value: str) -> str:
        value = value.strip()
        if not value.endswith("Z"):
            raise ValueError("cutoff_datetime must be an ISO-8601 UTC timestamp ending in Z")
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid cutoff_datetime: {value}") from exc
        return value

    @field_validator("time_control")
    @classmethod
    def time_control_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("time_control must not be empty")
        return value


class V13Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    null_simulations: int = Field(default=5000, ge=2000)
    simulation_seed: int = 42


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_players: int = Field(default=15, gt=0)
    min_decisive_games: int = Field(default=500, gt=0)
    random_seed: int = 42
    leaderboard_size: int = Field(default=200, gt=0, le=200)
    leaderboard_perf_types: list[str] = Field(
        default_factory=lambda: ["bullet", "blitz", "rapid", "classical"]
    )
    collection_max_games: int = Field(default=500, gt=0, le=100_000)
    screen_batch_size: int = Field(default=1000, gt=0, le=3000)
    leaderboard_endpoint: str = "/api/player/top/{nb}/{perf_type}"
    user_endpoint: str = "/api/user/{username}"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: ProjectConfig
    user: UserConfig
    paths: PathsConfig
    api: ApiConfig
    features: FeaturesConfig
    analysis: AnalysisConfig
    model: ModelConfig
    walk_forward: WalkForwardConfig
    research: ResearchConfig
    sampling: SamplingConfig = SamplingConfig()
    v13: V13Config = V13Config()

    repo_root: Path

    def resolve_path(self, value: Path) -> Path:
        """Resolve a configured path relative to the repository root."""
        return value if value.is_absolute() else self.repo_root / value

    def ensure_directories(self) -> None:
        """Create all configured parent directories."""
        for path in (
            self.resolve_path(self.paths.raw_pgn),
            self.resolve_path(self.paths.games_csv),
            self.resolve_path(self.paths.features_csv),
            self.resolve_path(self.paths.stats_json),
            self.resolve_path(self.paths.model_metrics_json),
            self.resolve_path(self.paths.model_coefficients_csv),
            self.resolve_path(self.paths.model_comparison_csv),
            self.resolve_path(self.paths.multi_model_metrics_json),
            self.resolve_path(self.paths.multi_model_coefficients_csv),
            self.resolve_path(self.paths.multi_model_comparison_csv),
            self.resolve_path(self.paths.analysis_summary_md),
            self.resolve_path(self.paths.tilt_sensitivity_csv),
            self.resolve_path(self.paths.tilt_sensitivity_md),
            self.resolve_path(self.paths.tilt_bootstrap_comparison_csv),
            self.resolve_path(self.paths.tilt_bootstrap_comparison_md),
            self.resolve_path(self.paths.tilt_sensitivity_session_bootstrap_csv),
            self.resolve_path(self.paths.walk_forward_metrics_csv),
            self.resolve_path(self.paths.walk_forward_fold_metrics_csv),
            self.resolve_path(self.paths.walk_forward_metrics_md),
            self.resolve_path(self.paths.multi_walk_forward_metrics_csv),
            self.resolve_path(self.paths.multi_walk_forward_fold_metrics_csv),
            self.resolve_path(self.paths.multi_walk_forward_metrics_md),
            self.resolve_path(self.paths.sample_players_csv),
            self.resolve_path(self.paths.sampling_metadata_json),
            self.resolve_path(self.paths.sampling_candidate_pool_csv),
            self.resolve_path(self.paths.multi_games_csv),
            self.resolve_path(self.paths.multi_processed_dir),
            self.resolve_path(self.paths.multi_features_csv),
            self.resolve_path(self.paths.player_summary_csv),
            self.resolve_path(self.paths.player_tilt_effects_csv),
            self.resolve_path(self.paths.multi_player_pooled_csv),
            self.resolve_path(self.paths.multi_player_data_quality_csv),
            self.resolve_path(self.paths.multi_player_analysis_md),
            self.resolve_path(self.paths.player_heterogeneity_csv),
            self.resolve_path(self.paths.player_heterogeneity_summary_json),
            self.resolve_path(self.paths.log_file),
            self.resolve_path(self.paths.synthetic_null_distribution_csv),
            self.resolve_path(self.paths.synthetic_null_summary_csv),
            self.resolve_path(self.paths.synthetic_null_summary_json),
            self.resolve_path(self.paths.opponent_context_csv),
            self.resolve_path(self.paths.opponent_adjusted_models_csv),
            self.resolve_path(self.paths.opponent_adjusted_coefficients_csv),
            self.resolve_path(self.paths.v13_analysis_md),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.resolve_path(self.paths.figures_dir).mkdir(parents=True, exist_ok=True)


def load_settings(config_path: str | Path) -> Settings:
    """Load and validate YAML configuration.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Validated settings object.
    """
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    payload["repo_root"] = path.parent.parent
    settings = Settings.model_validate(payload)
    settings.ensure_directories()
    return settings
