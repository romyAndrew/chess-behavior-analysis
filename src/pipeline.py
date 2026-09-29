"""Command-line orchestration for the chess behavior analysis pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .collect import LichessCollector
from .config import load_settings
from .features import add_behavioral_features, build_features_from_csv
from .logging_utils import configure_logging
from .model import fit_model_suite, save_model_results, save_walk_forward_results, walk_forward_validate
from .parse import parse_pgn_file
from .research import select_latest_decisive_window
from .stats import (
    analyze_tilt_proxy,
    analyze_tilt_sensitivity,
    analyze_tilt_sensitivity_session_bootstrap,
    compare_tilt_bootstrap_methods,
    render_summary,
    save_stats,
    save_tilt_bootstrap_comparison,
    save_tilt_sensitivity,
)
from .viz import generate_all
from .heterogeneity import analyze_player_heterogeneity, save_player_heterogeneity
from .sampling import (
    LichessSamplingFrame,
    analyze_multi_player,
    load_selected_players,
    save_multi_player_analysis,
    save_sampling_artifacts,
)


def _settings_and_logger(config_path: str) -> tuple[object, logging.Logger]:
    settings = load_settings(config_path)
    logger = configure_logging(settings.resolve_path(settings.paths.log_file))
    return settings, logger


def _selected_player_paths(settings, player_id: str) -> tuple[Path, Path]:
    safe = "".join(ch if ch.isalnum() or ch in {".", "_", "-"} else "_" for ch in player_id)
    raw = settings.resolve_path(settings.paths.multi_raw_dir) / f"{safe}.pgn"
    processed = settings.resolve_path(settings.paths.multi_processed_dir) / f"{safe}.csv"
    return raw, processed


def _run_multi_collect(settings, logger) -> None:
    players = load_selected_players(settings)
    collector = LichessCollector(settings, logger)
    raw_dir = settings.resolve_path(settings.paths.multi_raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = settings.sampling.collection_max_games
    batch_size = min(target, settings.sampling.screen_batch_size)
    cutoff_ms = int(pd.Timestamp(settings.research.cutoff_datetime).timestamp() * 1000)
    stop_condition = lambda text: LichessSamplingFrame._header_stop_condition(
        text,
        settings.sampling.min_decisive_games,
        settings.research.time_control,
        settings.research.cutoff_datetime,
    )
    for player_id in players:
        raw_path, _ = _selected_player_paths(settings, player_id)
        collector.collect_user(
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


def _run_multi_parse(settings, logger) -> pd.DataFrame:
    sample_path = settings.resolve_path(settings.paths.sample_players_csv)
    sample = pd.read_csv(sample_path)
    sample = sample.loc[sample["selected"].astype(bool)].sort_values("selection_order")
    frames: list[pd.DataFrame] = []
    for row in sample.itertuples(index=False):
        player_id = str(row.player_id)
        raw_path, processed_path = _selected_player_paths(settings, player_id)
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw PGN for selected player: {raw_path}")
        if processed_path.exists():
            logger.info("REUSE existing parsed data for %s", player_id)
            parsed = pd.read_csv(processed_path, parse_dates=["created_at"])
        else:
            parsed = parse_pgn_file(
                raw_path,
                settings,
                logger,
                username=player_id,
                output_path=processed_path,
            )
        if parsed.empty:
            continue
        parsed["player_id"] = player_id
        window, window_meta = select_latest_decisive_window(parsed, settings)
        if window_meta is None or int(window_meta["selected_decisive_games"]) != int(settings.sampling.min_decisive_games):
            raise ValueError(
                f"Selected player {player_id} no longer has the configured {settings.sampling.min_decisive_games} decisive {settings.research.time_control} games before cutoff"
            )
        # Reuse the fixed sampling window start to guarantee that parse_multi uses
        # exactly the observation window selected during the sampling stage.
        expected_start = pd.Timestamp(row.analysis_window_start)
        window = window.loc[window["created_at"] >= expected_start].copy()
        window = window.loc[window["created_at"] <= pd.Timestamp(row.analysis_window_end)].copy()
        frames.append(window)

    if not frames:
        raise ValueError("No player games were parsed from the selected sample")

    combined = pd.concat(frames, ignore_index=True)
    within_player_duplicates = int(combined.duplicated(["player_id", "game_id"]).sum())
    global_counts = combined.groupby("game_id")["player_id"].nunique()
    cross_player_ids = set(global_counts.loc[global_counts > 1].index.astype(str))
    if cross_player_ids:
        combined = combined.loc[~combined["game_id"].astype(str).isin(cross_player_ids)].copy()
    combined = combined.drop_duplicates(["player_id", "game_id"], keep="first")
    combined = combined.sort_values(["player_id", "created_at"]).reset_index(drop=True)

    # Re-validate final eligibility on the actual analysis dataset after global dedup.
    decisive_counts = (
        combined.assign(is_decisive=combined["user_result"].isin(["win", "loss"]))
        .groupby("player_id")["is_decisive"]
        .sum()
    )
    if (decisive_counts < settings.sampling.min_decisive_games).any():
        bad = decisive_counts[decisive_counts < settings.sampling.min_decisive_games].to_dict()
        raise ValueError(f"Final analysis window violates decisive eligibility after dedup: {bad}")

    output = settings.resolve_path(settings.paths.multi_games_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)

    quality = pd.DataFrame(
        [
            {
                "selected_players": len(sample),
                "parsed_rows_before_global_dedup": int(sum(len(frame) for frame in frames)),
                "within_player_duplicate_rows": within_player_duplicates,
                "cross_player_duplicate_game_ids_removed": len(cross_player_ids),
                "final_unique_game_rows": int(len(combined)),
            }
        ]
    )
    quality.to_csv(
        settings.resolve_path(settings.paths.multi_player_data_quality_csv),
        index=False,
    )
    return combined


def _run_multi_features(settings, logger) -> pd.DataFrame:
    path = settings.resolve_path(settings.paths.multi_games_csv)
    df = pd.read_csv(path, parse_dates=["created_at"])
    features = add_behavioral_features(df, settings)
    output = settings.resolve_path(settings.paths.multi_features_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    logger.info("Multi-player feature engineering produced %d rows", len(features))
    return features


def _run_multi_analysis(settings, logger) -> None:
    path = settings.resolve_path(settings.paths.multi_features_csv)
    df = pd.read_csv(path, parse_dates=["created_at"])
    player_summary, player_effects, pooled, quality = analyze_multi_player(df, settings, logger)
    metadata_path = settings.resolve_path(settings.paths.sampling_metadata_json)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    save_multi_player_analysis(
        player_summary,
        player_effects,
        pooled,
        quality,
        metadata,
        settings,
    )


def _run_heterogeneity(settings, logger) -> None:
    path = settings.resolve_path(settings.paths.multi_features_csv)
    df = pd.read_csv(path, parse_dates=["created_at"])
    results = analyze_player_heterogeneity(df, settings, logger)
    pooled_path = settings.resolve_path(settings.paths.multi_player_pooled_csv)
    if not pooled_path.exists():
        raise FileNotFoundError(
            f"Pooled multi-player summary not found: {pooled_path}. Run analyze_multi first."
        )
    pooled = pd.read_csv(pooled_path)
    summary = save_player_heterogeneity(results, pooled, settings)
    logger.info(
        "Player heterogeneity: %d players, positive=%d, negative=%d, range=[%.2f, %.2f] pp",
        summary["players"],
        summary["positive_estimates"],
        summary["negative_estimates"],
        summary["minimum_difference_pp"],
        summary["maximum_difference_pp"],
    )


def _run_multi_model(settings, logger) -> None:
    path = settings.resolve_path(settings.paths.multi_features_csv)
    df = pd.read_csv(path, parse_dates=["created_at"])
    metrics, comparison, coefficients = fit_model_suite(df, settings, logger)
    save_model_results(
        metrics,
        comparison,
        coefficients,
        settings,
        metrics_path=settings.paths.multi_model_metrics_json,
        comparison_path=settings.paths.multi_model_comparison_csv,
        coefficients_path=settings.paths.multi_model_coefficients_csv,
    )
    aggregate, folds, baseline = walk_forward_validate(df, settings, logger)
    save_walk_forward_results(
        aggregate,
        folds,
        baseline,
        settings,
        aggregate_path=settings.paths.multi_walk_forward_metrics_csv,
        folds_path=settings.paths.multi_walk_forward_fold_metrics_csv,
        markdown_path=settings.paths.multi_walk_forward_metrics_md,
    )


def run(command: str, config_path: str) -> None:
    """Run one pipeline stage."""
    settings, logger = _settings_and_logger(config_path)
    logger.info("Starting stage: %s", command)

    raw_pgn = settings.resolve_path(settings.paths.raw_pgn)
    games_csv = settings.resolve_path(settings.paths.games_csv)
    features_csv = settings.resolve_path(settings.paths.features_csv)

    if command == "collect":
        LichessCollector(settings, logger).collect()
        return

    if command == "parse":
        parse_pgn_file(raw_pgn, settings, logger)
        return

    if command == "features":
        build_features_from_csv(games_csv, settings, logger)
        return

    if command == "analyze":
        df = pd.read_csv(features_csv, parse_dates=["created_at"])
        result = analyze_tilt_proxy(df, settings, logger)
        bootstrap_comparison = compare_tilt_bootstrap_methods(df, result, settings)
        save_stats(result, settings.resolve_path(settings.paths.stats_json))
        summary_path = settings.resolve_path(settings.paths.analysis_summary_md)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(render_summary(result), encoding="utf-8")

        save_tilt_bootstrap_comparison(
            bootstrap_comparison,
            settings.resolve_path(settings.paths.tilt_bootstrap_comparison_csv),
            settings.resolve_path(settings.paths.tilt_bootstrap_comparison_md),
        )

        sensitivity = analyze_tilt_sensitivity(df, settings, logger)
        save_tilt_sensitivity(
            sensitivity,
            settings.resolve_path(settings.paths.tilt_sensitivity_csv),
            settings.resolve_path(settings.paths.tilt_sensitivity_md),
        )
        sensitivity_session_bootstrap = analyze_tilt_sensitivity_session_bootstrap(
            df, settings, logger
        )
        sensitivity_session_bootstrap.to_csv(
            settings.resolve_path(settings.paths.tilt_sensitivity_session_bootstrap_csv),
            index=False,
        )
        return

    if command == "model":
        df = pd.read_csv(features_csv, parse_dates=["created_at"])
        metrics, comparison, coefficients = fit_model_suite(df, settings, logger)
        save_model_results(metrics, comparison, coefficients, settings)
        walk_forward_aggregate, walk_forward_folds, chronological_baseline = walk_forward_validate(
            df, settings, logger
        )
        save_walk_forward_results(
            walk_forward_aggregate,
            walk_forward_folds,
            chronological_baseline,
            settings,
        )
        return

    if command == "viz":
        df = pd.read_csv(features_csv, parse_dates=["created_at"])
        stats_path = settings.resolve_path(settings.paths.stats_json)
        stats_result = json.loads(stats_path.read_text(encoding="utf-8"))
        sensitivity_path = settings.resolve_path(settings.paths.tilt_sensitivity_csv)
        sensitivity = pd.read_csv(sensitivity_path)
        bootstrap_path = settings.resolve_path(settings.paths.tilt_bootstrap_comparison_csv)
        bootstrap_comparison = pd.read_csv(bootstrap_path)
        walk_forward_path = settings.resolve_path(settings.paths.walk_forward_fold_metrics_csv)
        walk_forward_folds = pd.read_csv(walk_forward_path) if walk_forward_path.exists() else None
        generate_all(
            df,
            stats_result,
            settings.resolve_path(settings.paths.figures_dir),
            settings.project.timezone,
            sensitivity=sensitivity,
            bootstrap_comparison=bootstrap_comparison,
            walk_forward_folds=walk_forward_folds,
        )
        return

    if command == "sample":
        sampler = LichessSamplingFrame(settings, logger)
        selected, metadata, candidate_pool = sampler.select()
        save_sampling_artifacts(selected, metadata, candidate_pool, settings)
        logger.info("Selected %d eligible players from candidate pool of %d", len(selected), len(candidate_pool))
        return

    if command == "collect_multi":
        _run_multi_collect(settings, logger)
        return

    if command == "parse_multi":
        _run_multi_parse(settings, logger)
        return

    if command == "features_multi":
        _run_multi_features(settings, logger)
        return

    if command == "analyze_multi":
        _run_multi_analysis(settings, logger)
        return

    if command == "heterogeneity":
        _run_heterogeneity(settings, logger)
        return

    if command == "model_multi":
        _run_multi_model(settings, logger)
        return

    if command == "multi":
        run("sample", config_path)
        run("collect_multi", config_path)
        run("parse_multi", config_path)
        run("features_multi", config_path)
        run("analyze_multi", config_path)
        run("heterogeneity", config_path)
        run("model_multi", config_path)
        return

    if command == "all":
        run("collect", config_path)
        run("parse", config_path)
        run("features", config_path)
        run("analyze", config_path)
        run("model", config_path)
        run("viz", config_path)
        return

    raise ValueError(f"Unknown command: {command}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Lichess chess behavior analysis")
    parser.add_argument(
        "command",
        choices=["collect", "parse", "features", "analyze", "model", "viz", "all", "sample", "collect_multi", "parse_multi", "features_multi", "analyze_multi", "heterogeneity", "model_multi", "multi"],
    )
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.command, args.config)


if __name__ == "__main__":
    main()
