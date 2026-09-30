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

## 13. V13: synthetic null model

V13 is an analytical layer over the fixed v12 dataset. The sampling frame, 15 selected players, cutoff, `60+0` restriction, observation windows, baseline tilt proxy, sensitivity analysis, player-level heterogeneity and predictive models are unchanged.

### 13.1 Baseline sanity check

The observed dataset contains **8,043** games, **7,500** decisive games and **543** draws across **15** players. Recomputing the baseline from the same feature file gives **641** tilt-proxy observations and an observed pooled difference of **+12.66 percentage points**.

### 13.2 Null data-generating process

For each simulation, player identity, opponent identity, ratings, rating differences, color, timestamps, breaks, sessions, draws, repeated-opponent structure and game counts are preserved. Only decisive win/loss outcomes are regenerated.

For the raw Elo null:

```text
p(win) = 1 / (1 + 10^(-rating_diff / 400))
```

For the calibrated null, the Elo loss probability is adjusted by an expanding player-specific residual correction computed only from prior decisive games:

```text
p_cal = clip(p_elo + mean(prior observed_loss - prior p_elo), 1e-4, 1 - 1e-4)
```

The current game's outcome is never used to construct its own probability, and draws do not update the calibration term.

Both null scenarios contain no explicit tilt effect. After synthetic outcomes are generated, the same baseline tilt-proxy rule is recomputed from the synthetic sequence.

### 13.3 Null simulation results

The analysis uses **2,000 simulations per scenario** with seed **42**.

| Scenario | Mean difference | Median | SD | 95% simulation interval | Share >= observed |
|---|---:|---:|---:|---|---:|
| Raw Elo | +11.62 pp | +11.67 pp | 1.98 pp | [7.65, 15.37] pp | 31.6% |
| Player-calibrated Elo | +12.18 pp | +12.15 pp | 2.08 pp | [8.28, 16.11] pp | 40.5% |

The interval above is the central 95% interval of the **synthetic null distribution**. It is not a confidence interval for the observed effect.

The upper-tail proportion is the share of null simulations at least as large as the observed +12.66 pp association. It is reported descriptively and is not relabelled as a conventional p-value.

The observed association therefore has a non-negligible occurrence rate under both specified null data-generating processes, even though no explicit tilt effect is simulated.

## 14. Opponent context and adjusted models

A first diagnostic shows that games following a previous loss tend to occur against relatively stronger opponents than games following a previous non-loss. The mean rating difference is **84.25** after a previous loss versus **151.10** after a previous non-loss; lower rating difference means the opponent is stronger relative to the focal player.

After a previous loss and short break, the mean rating difference is **71.58**, and the same-opponent rate is **53.4%**. The broader short-break group has a same-opponent rate of **51.0%**, while the longer-break group has a rate of **3.5%**. These are descriptive context measures rather than causal mechanisms.

### 14.1 Logistic specifications

All three models use current-game loss as the target. Standard errors are cluster-robust by `player_id`.

**Model A**

```text
loss ~ tilt_proxy
```

**Model B**

```text
loss ~ tilt_proxy + rating_difference + color + player_fixed_effect
```

**Model C**

```text
loss ~ tilt_proxy + rating_difference + color + same_opponent + player_fixed_effect
```

The estimated tilt-proxy odds ratios are:

| Model | OR | 95% CI | p-value |
|---|---:|---|---:|
| A: unadjusted | 1.717 | [1.228, 2.400] | 0.0016 |
| B: opponent/player adjusted | 1.122 | [0.901, 1.397] | 0.3028 |
| C: + same opponent | 1.120 | [0.899, 1.395] | 0.3122 |

The raw association is therefore materially smaller after adjustment for rating difference, color and player fixed effects. Adding same-opponent status changes the estimate only slightly.

These models remain observational. With only 15 player clusters, the cluster-robust p-values should be treated as approximate rather than as definitive population-level evidence.

## 15. Player heterogeneity after v13

The v12 player-level estimates are retained unchanged. They range from **-13.31 to +28.38 percentage points**, with mean **+10.15 pp**, median **+12.01 pp**, **11 positive** estimates and **4 negative** estimates.

The heterogeneity layer remains descriptive. Its confidence intervals quantify uncertainty in the player-level estimates, while the synthetic null and opponent-adjusted analyses provide additional context for interpreting the pooled association.

The combined evidence does not justify treating the raw player-level differences as direct measurements of psychological tilt.

## 16. Related literature

Gee et al. (2025) use a **hierarchical Bayesian logistic regression** to study experiential winner/loser effects in online chess, explicitly modelling population-level and player-level variation. Their paper reports little evidence for a strong, consistent global experiential effect, while allowing for some player-specific variability. urlcitehttps://pmc.ncbi.nlm.nih.gov/articles/PMC13265758/

The present project does **not** reproduce their hierarchical Bayesian methodology. The paper is used as methodological context and motivation for accounting for player-level variation and match context. In this project, the v13 opponent-adjusted models are simpler frequentist logistic regressions, while the synthetic null analysis asks a different question: how often can a raw association of the observed scale arise under explicitly specified no-tilt data-generating processes?

## 17. Updated interpretation

The original +12.66 pp pooled association remains a real descriptive feature of the observed v12 data. However, v13 shows that a difference of comparable magnitude can also arise under null simulations that preserve the observed sequence structure while removing any explicit tilt effect. The exact null distribution depends on how the outcome probabilities are specified: the raw Elo scenario places the observed value in 31.6% of simulations at or above the observed level, while the player-calibrated scenario gives 40.5%.

The opponent-adjusted models provide a second check. Once rating difference, color and player fixed effects are included, the estimated tilt-proxy odds ratio falls from 1.72 in the unadjusted model to about 1.12, with confidence intervals that include 1. Adding same-opponent status changes the estimate only slightly.

The appropriate conclusion is therefore not that tilt has been disproved or confirmed. Rather, the raw association is compatible with a broader data-generating process driven by player performance, opponent strength and sequential match context, and the current observational design does not isolate an independent psychological tilt effect.
