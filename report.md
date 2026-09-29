# Chess Behavior Analysis Report

## 1. Research question

The final analysis asks whether a short return to play after a loss and losing streak is associated with the result of the next `60+0` bullet game, and whether the size and direction of that observed association vary across players.

The main behavioral construct is an operational **tilt proxy**, not a direct measure of psychological state.

## 2. Final research design

### 2.1 Population and sampling frame

The final population is defined as active bullet players from the predefined public Lichess bullet leaderboard sampling frame. The sampling frame is a convenience frame and should not be interpreted as representative of all Lichess users.

Candidates are processed in a reproducible random order using seed **42**. The pipeline screens candidates against the actual parsed research data and selects the first 15 players who satisfy the predefined eligibility rule. Selection is independent of tilt outcomes, loss rates, streaks, breaks, player-level effects and model performance.

Eligibility requires at least **500 analyzable decisive `60+0` games before `2026-09-28T23:59:59Z`**. External profile game counts are not used as the final eligibility criterion.

The verified sampling metadata records a candidate pool of 100 players, 27 candidates checked and 15 selected players.

### 2.2 Time control and cutoff

Only exact `60+0` games are eligible for the final research sample.

The fixed cutoff is:

```text
2026-09-28T23:59:59Z
```

### 2.3 Observation window

For each selected player, the fixed sample-size anchor is the **latest 500 decisive `60+0` games before the cutoff**.

Decisive games define the fixed sample size, but **all analyzable `60+0` games between the timestamp of the earliest of those 500 decisive games and the cutoff are retained**. Draws are therefore preserved for sequential feature construction.

The resulting window can cover very different calendar durations across players. That duration is retained explicitly in the player summary rather than being treated as a fixed quantity.

### 2.4 Data integrity

The verified final dataset contains:

- 15 selected players;
- 8,043 unique game rows after global deduplication;
- 7,500 decisive games;
- 543 draws;
- 1,059 sessions;
- no within-player duplicate game IDs;
- no missing session IDs;
- 2 cross-player duplicate game IDs removed before the final pooled dataset.

Games shared by more than one selected player are excluded from the combined analytical dataset so that pooled descriptive totals do not count the same game twice.

## 3. Architecture

The repository separates collection, parsing, feature engineering, statistical analysis, predictive modeling and visualization under `src/`. Configuration is stored in `config/config.yaml`, and tests cover the core parsing, feature, statistics, modeling, sampling, visualization and heterogeneity layers.

The final multi-player pipeline is:

```text
sample
  ↓
collect_multi
  ↓
parse_multi
  ↓
features_multi
  ↓
analyze_multi
  ↓
heterogeneity
  ↓
model_multi
```

The original `run.py` pipeline remains as a single-player regression baseline for `bat1skaf`. It is not the final research population.

## 4. Feature engineering

The analysis constructs temporal, session and behavioral features, including:

- `hour` and `day_of_week`;
- cyclic `hour_sin` and `hour_cos` features;
- `session_id`, `session_game_number` and `session_length`;
- `break_after_previous`;
- `loss_streak_before` and `win_streak_before`;
- `rating_diff`;
- clock-derived timing features when usable clock annotations are present.

Sequential features are constructed separately within each player so that a previous result or losing streak cannot cross player boundaries.

A new session starts when the gap between consecutive games exceeds the configured 30-minute session threshold.

## 5. Tilt proxy

The baseline rule is unchanged from the earlier project versions:

```text
previous game = loss
AND
break before current game <= 5 minutes
AND
previous loss streak >= 2
```

This proxy is an operational behavioral definition. It is not a psychological measurement, and no result in this project establishes whether a player was actually experiencing a mental or emotional state called tilt.

### 5.1 Sensitivity definitions

The existing sensitivity analysis retains six definitions:

- break <= 2 min, streak >= 2;
- break <= 5 min, streak >= 2 (baseline);
- break <= 10 min, streak >= 2;
- break <= 15 min, streak >= 2;
- break <= 5 min, streak >= 3;
- break <= 5 min, streak >= 4.

These definitions are retained from the original single-player analysis. The observed association kept a positive point estimate across all six tested definitions in that baseline, while the estimated magnitude changed more when the required losing-streak length increased and the tilt group became smaller. This is treated as threshold sensitivity, not as proof of robustness.

The associated artifacts are:

```text
results/tilt_sensitivity.csv
results/tilt_sensitivity.md
results/tilt_sensitivity_session_bootstrap.csv
figures/tilt_sensitivity.png
```

## 6. Statistical analysis

The primary outcome is the current game's loss indicator among decisive observations.

The hypotheses are:

- **H0:** loss proportions are equal in the tilt-proxy and control groups;
- **H1:** loss proportions differ.

The existing statistical logic uses Fisher's exact test for small expected counts and a two-proportion z-test otherwise. Effect sizes include odds ratio and risk difference. The project also retains Wilson confidence intervals for group loss rates.

The original single-player baseline compares row-level bootstrap and session-aware bootstrap intervals. The session-aware version resamples complete existing sessions to account for within-session dependence. Both use 5,000 iterations and seed 42.

The v12 player-level analysis reuses that session-aware bootstrap infrastructure for player-specific risk-difference intervals. No new bootstrap framework was introduced.

## 7. Final multi-player association results

### 7.1 Pooled descriptive result

The verified final multi-player dataset contains:

- **641 tilt-proxy observations**;
- **6,859 control observations**.

Observed loss rates:

| Group | Loss rate |
|---|---:|
| Tilt proxy | 44.46% |
| Control | 31.80% |

The pooled observed difference is **+12.66 percentage points**.

This pooled result is explicitly **descriptive**. Observations from the same player are not treated as independent in a new pooled inferential model, and no player-clustered or hierarchical model was introduced.

### 7.2 Player-level heterogeneity

The v12 extension estimates the unchanged baseline tilt association separately for each selected player.

Across 15 players:

- minimum point estimate: **-13.31 pp**;
- maximum point estimate: **+28.38 pp**;
- mean: **+10.15 pp**;
- median: **+12.01 pp**;
- positive estimates: **11**;
- negative estimates: **4**;
- zero estimates: **0**.

The estimates therefore vary in both magnitude and direction across the sampled players, while the individual confidence intervals are often wide. The correct interpretation is descriptive heterogeneity with substantial player-level uncertainty, not evidence that particular players are more or less psychologically prone to tilt.

One player has only **9 tilt observations**, below the configured reference size of 10. The player is retained in the analysis and explicitly flagged rather than removed.

Several player-level p-values are below 0.05. These results are exploratory because multiple player-level tests were performed and no multiple-comparison correction was introduced in v12.

The canonical v12 outputs are:

```text
results/player_heterogeneity.csv
results/player_heterogeneity_summary.json
figures/player_heterogeneity.png
```

The primary visualization uses selection order for readability and includes point estimates, confidence intervals and a zero reference line. It is not a player ranking.

## 8. Predictive modeling

Predictive modeling is secondary to the association analysis. The implementation compares three logistic-regression specifications using pre-result information:

### Model A: rating-only

`rating_diff` only.

### Model B: baseline

Rating difference plus game speed and player color.

### Model C: behavioral

The baseline model plus break duration, previous streaks, session position, the tilt proxy and cyclic time-of-day features.

No new models or behavioral predictors were introduced in v12.

### 8.1 Chronological holdout

The final multi-player model evaluation contains 7,500 decisive observations, with 6,000 in training and 1,500 in the chronological test period.

| Model | ROC-AUC | Balanced accuracy | Log loss |
|---|---:|---:|---:|
| rating_only | 0.657 | 0.610 | 0.630 |
| baseline | 0.655 | 0.604 | 0.632 |
| behavioral | 0.646 | 0.603 | 0.629 |

These values describe predictive performance on the observed sample. They do not establish causal effects and no model is labeled as a universal or population-level winner.

### 8.2 Walk-forward validation

The existing expanding-window procedure produces **90 temporal folds**. The training and test windows remain strictly ordered in time.

| Model | Mean ROC-AUC | Mean log loss | Mean Brier |
|---|---:|---:|---:|
| rating_only | 0.644 | 0.642 | 0.227 |
| baseline | 0.655 | 0.642 | 0.226 |
| behavioral | 0.645 | 0.650 | 0.230 |

Variation across temporal folds is part of the result. These metrics should be read as temporal validation measurements, not as proof that a model will generalize to all chess games or users.

The detailed fold-level artifacts are stored in:

```text
results/multi_player_walk_forward_metrics.csv
results/multi_player_walk_forward_fold_metrics.csv
results/multi_player_walk_forward_metrics.md
```

## 9. Historical single-player baseline

The repository retains the original `bat1skaf` single-player analysis for regression and development history.

That baseline contained **652 parsed games**, with 50 tilt-proxy observations and 570 controls. The observed risk difference was **+8.8 pp**, the row-bootstrap 95% CI was **[-5.2, +22.9] pp**, the odds ratio was **1.42**, and the two-proportion z-test gave **p = 0.232**.

The original single-player model outputs and four-fold walk-forward results remain in `results/`. They are not the final multi-player research findings.

This historical baseline is useful for regression testing because v12 must not silently change its behavior when only the documentation and heterogeneity layer are modified.

## 10. Limitations

### Observational design

The analysis identifies associations in observational game histories. It does not establish causation.

### Behavioral proxy

The tilt proxy is constructed from observable sequence features. It is not a direct psychological measurement.

### Sampling frame

The sample is drawn from a public Lichess bullet leaderboard frame and is not representative of all Lichess users.

### Player-level uncertainty

Some players have relatively few tilt observations. Their point estimates can therefore have wide confidence intervals and should not be treated as equally precise.

### Confounding

Rating difference, opponent strength, playing context, time of day and other unmeasured variables may be associated with both the behavioral proxy and the outcome.

### Pooled inference

The pooled multi-player association is descriptive because no player-clustered inferential model was introduced.

### Live external source

Lichess data can change over time. The documented results correspond to the verified dataset and sampling metadata used for this analysis run.

## 11. Reproducibility

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

Run the final research pipeline:

```bash
python -m src.pipeline multi --config config/config.yaml
```

The original single-player regression baseline is available with:

```bash
python run.py
```

Because the project uses a live Lichess API, a fresh collection may not reproduce the same raw game snapshot. The repository records the research cutoff, sampling seed, candidate-pool hash, selected players and eligibility rule in `results/sampling_metadata.json`.

## 12. Key artifacts

```text
results/
├── sample_players.csv
├── sampling_candidate_pool.csv
├── sampling_metadata.json
├── player_summary.csv
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

Legacy single-player statistical artifacts remain in `results/` for regression testing and historical comparison. They are not the final multi-player research population.

## 13. Final interpretation

The final analysis shows a positive pooled descriptive association between the baseline tilt proxy and subsequent loss, with an observed difference of **+12.66 percentage points**. Player-level estimates vary from **-13.31 to +28.38 pp**, so the observed association is not uniform across the sampled players. At the same time, individual uncertainty is substantial, and the analysis does not establish psychological tilt or causality.

The main result is therefore an observed, heterogeneous behavioral association in the specified Lichess bullet sample, not a universal effect for chess players.
