# Chess Behavior Analysis

## Project question

This project studies whether a short return to play after a loss and losing streak is associated with the result of the next **60+0 bullet** chess game on Lichess.

The main behavioral construct is a **tilt proxy**:

```text
previous game = loss
AND
break <= 5 minutes
AND
previous loss streak >= 2
```

This is an operational behavioral proxy, not a direct measure of psychological tilt.

## Final research design

The final research analysis uses a reproducibly selected multi-player sample:

| Item | Final design |
|---|---|
| Population | Active bullet players from the predefined public Lichess bullet leaderboard sampling frame |
| Sampling | Random candidate order, seed = 42, screening until 15 eligible players are selected |
| Time control | `60+0` only |
| Cutoff | `2026-09-28T23:59:59Z` |
| Eligibility | At least 500 analyzable decisive `60+0` games before the cutoff |
| Fixed sample anchor | Latest 500 decisive `60+0` games per selected player |
| Analysis window | All analyzable `60+0` games from the earliest selected decisive game through the cutoff |
| Draws | Retained inside the analysis window |

The sampling frame is a convenience frame from the public bullet leaderboard. It should not be treated as representative of all Lichess users or as an average-player sample.

Decisive games define the fixed sample size, but all analyzable games inside the corresponding time window are retained for sequential feature construction. Sequential features are isolated by player, and duplicate game IDs shared across selected players are removed from the combined dataset.

## Final dataset

The verified v12 multi-player dataset contains:

- **15 selected players**;
- **8,043 unique game rows** after global deduplication;
- **7,500 decisive games**;
- **543 draws**;
- **1,059 sessions**;
- **641 tilt-proxy observations**;
- **6,859 control observations**.

The pooled descriptive loss rates are:

| Group | Loss rate |
|---|---:|
| Tilt proxy | 44.46% |
| Control | 31.80% |

Observed pooled difference: **+12.66 percentage points**.

The pooled result is descriptive only. No player-clustered or hierarchical pooled inference was introduced in the final analysis.

## Player-level heterogeneity

The v12 analysis estimates the same baseline association separately for each selected player. Confidence intervals use the existing session-aware bootstrap infrastructure. Small tilt groups are retained rather than filtered out.

Across the 15 players:

- point estimates range from **-13.31 to +28.38 percentage points**;
- mean estimate: **+10.15 pp**;
- median estimate: **+12.01 pp**;
- **11** estimates are positive;
- **4** estimates are negative.

One player has only **9 tilt observations**, below the configured reference size of 10. That player remains in the analysis and is flagged as having greater statistical uncertainty.

The individual estimates are descriptive and uncertain. They should not be interpreted as measures of which players are psychologically more or less prone to tilt. Several individual p-values are below 0.05, but these are exploratory player-level results and no multiple-comparison correction was applied.

Primary outputs:

```text
results/player_heterogeneity.csv
results/player_heterogeneity_summary.json
figures/player_heterogeneity.png
```

The heterogeneity figure is ordered by selection order for readability, not as a ranking.

## Predictive modeling

Predictive modeling is secondary to the association analysis. Three logistic-regression specifications are evaluated:

1. `rating_only`: `rating_diff`;
2. `baseline`: rating difference, speed and player color;
3. `behavioral`: baseline variables plus breaks, streaks, session position, the tilt proxy and cyclic time-of-day features.

The final multi-player chronological holdout contains 1,500 test games:

| Model | ROC-AUC | Balanced accuracy | Log loss |
|---|---:|---:|---:|
| rating_only | 0.657 | 0.610 | 0.630 |
| baseline | 0.655 | 0.604 | 0.632 |
| behavioral | 0.646 | 0.603 | 0.629 |

The walk-forward evaluation uses **90 temporal folds**:

| Model | Mean ROC-AUC | Mean log loss | Mean Brier |
|---|---:|---:|---:|
| rating_only | 0.644 | 0.642 | 0.227 |
| baseline | 0.655 | 0.642 | 0.226 |
| behavioral | 0.645 | 0.650 | 0.230 |

These are predictive performance measurements, not evidence that any feature causes game outcomes. The model results are not the main research finding.

## Methods

The project combines:

- reproducible sampling and fixed temporal windows;
- feature engineering for sessions, breaks and streaks;
- threshold sensitivity for the tilt proxy;
- row-level and session-aware bootstrap comparisons in the original single-player baseline;
- player-level session-aware confidence intervals in v12;
- chronological logistic regression;
- expanding-window walk-forward validation.

The analysis deliberately does not introduce hierarchical or mixed-effects models, new behavioral predictors, or new machine-learning models in v12.

## Historical single-player baseline

The repository keeps the original `bat1skaf` single-player analysis as a development and regression baseline. It is **not** the final research population.

That baseline used 652 parsed games. Its baseline tilt-proxy result was:

- tilt group: 50 decisive observations, loss rate 56.0%;
- control group: 570 decisive observations, loss rate 47.2%;
- observed difference: +8.8 pp;
- 95% row-bootstrap CI: [-5.2, +22.9] pp;
- odds ratio: 1.42;
- p-value: 0.232.

Its multi-fold temporal metrics are retained in `results/` for regression and development history. They should not be presented as the final multi-player research result.

## Limitations

- The design is observational. Associations should not be read as causal effects.
- The tilt proxy is a constructed behavioral rule and is not a direct psychological measurement.
- The population is restricted to `60+0` bullet games from a public leaderboard-derived sampling frame.
- The sample does not represent all Lichess users.
- Player-level estimates can be imprecise when the tilt group is small.
- The pooled multi-player association is descriptive because no player-clustered pooled inference was introduced.
- Rating, opponent strength, playing context, time of day and other unmeasured factors may confound observed associations.
- Lichess is a live external source, so a new collection may not reproduce the same game snapshot.

## Reproducibility

Create a Python environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows
python -m pip install -r requirements.txt
```

Run the tests:

```bash
python -m pytest -q
```

Run the **final multi-player research pipeline**:

```bash
python -m src.pipeline multi --config config/config.yaml
```

The multi-player pipeline runs:

```text
sample → collect_multi → parse_multi → features_multi
       → analyze_multi → heterogeneity → model_multi
```

The raw multi-player collection is cached locally and can be reused on later runs when the existing artifacts are sufficient.

The original single-player baseline is still available with:

```bash
python run.py
```

Use it for regression checking of the legacy `bat1skaf` analysis, not as the final research sample.

## Key artifacts

```text
results/
├── sample_players.csv
├── sampling_candidate_pool.csv
├── sampling_metadata.json
├── player_summary.csv
├── player_tilt_effects.csv
├── multi_player_pooled_summary.csv
├── multi_player_data_quality.csv
├── multi_player_analysis.md
├── player_heterogeneity.csv
├── player_heterogeneity_summary.json
├── multi_player_model_metrics.json
├── multi_player_model_comparison.csv
├── multi_player_model_coefficients.csv
├── multi_player_walk_forward_metrics.csv
├── multi_player_walk_forward_fold_metrics.csv
└── multi_player_walk_forward_metrics.md

figures/
├── player_tilt_effects.png
├── player_sample_sizes.png
├── player_heterogeneity.png
└── walk_forward_roc_auc.png
```

Additional legacy single-player statistical artifacts remain in `results/` for reproducibility and historical comparison.
