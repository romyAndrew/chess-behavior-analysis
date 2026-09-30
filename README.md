# Chess Behavior Analysis

A reproducible analysis of behavioral patterns in `60+0` Lichess bullet chess, focused on what can and cannot be inferred from a short return-to-play after losses.

## Question

Does a predefined behavioral proxy around losses and short breaks associate with the outcome of the next game, and how much of the observed association remains after accounting for player and match context?

The baseline **tilt proxy** is:

```text
previous game = loss
AND break <= 5 minutes
AND previous loss streak >= 2
```

This is an operational behavioral proxy, not a direct measure of psychological tilt.

## Data

- **15** randomly selected active bullet players from the predefined public Lichess bullet leaderboard sampling frame;
- exact time control: **60+0**;
- fixed cutoff: **2026-09-28T23:59:59Z**;
- **8,043** unique game rows;
- **7,500** decisive games and **543** draws;
- random seed: **42**;
- latest **500 decisive games per player** define the sample-size anchor, while all analyzable games in the corresponding time window are retained.

The sampling frame is not representative of all Lichess users.

## Method

The project combines temporal feature engineering, player-level heterogeneity, synthetic null simulation, opponent-adjusted logistic regression, and walk-forward validation.

The v13 extension keeps the final multi-player dataset and baseline analysis fixed, then adds two checks:

1. a **synthetic null model** that preserves the observed game structure but regenerates decisive outcomes from Elo-based probabilities, with a second time-respecting player-specific calibration scenario;
2. **opponent-adjusted models** that control for rating difference, color, player fixed effects, and, in the third specification, whether the current opponent is the same as in the previous game.

## Key findings

The raw pooled association is:

| Group | Loss rate |
|---|---:|
| Tilt proxy | **44.46%** |
| Control | **31.80%** |

Observed difference: **+12.66 pp**.

The raw association is not uniquely identified by the observed sequence alone. Under the synthetic null with no explicit tilt effect:

| Null scenario | Mean difference | 95% simulation interval | Share of simulations >= observed |
|---|---:|---:|---:|
| Raw Elo | **+11.62 pp** | [7.65, 15.37] pp | **31.6%** |
| Player-calibrated Elo | **+12.18 pp** | [8.28, 16.11] pp | **40.5%** |

These are simulation-distribution summaries, not confidence intervals or automatically interpreted p-values.

The player-level estimates remain heterogeneous and uncertain: **-13.31 to +28.38 pp**, with **11 positive** and **4 negative** estimates.

In the opponent-adjusted logistic models, the tilt-proxy odds ratio falls from **1.72** in the unadjusted specification to:

- **1.12** in the model with rating difference, color and player fixed effects (95% CI 0.90вЂ“1.40; p = 0.303);
- **1.12** after additionally controlling for same-opponent status (95% CI 0.90вЂ“1.39; p = 0.312).

These are observational associations, with cluster-robust standard errors by player. They do not establish a psychological or causal tilt effect.

## Main figures

![Observed pooled effect](figures/pooled_tilt_effect.png)

![Synthetic null distribution](figures/synthetic_null_distribution.png)

![Opponent-adjusted association](figures/opponent_adjusted_effect.png)

![Walk-forward ROC-AUC](figures/walk_forward_roc_auc.png)

## Limitations

- observational design and no causal identification;
- tilt proxy is not a direct psychological measurement;
- bullet `60+0` only;
- limited sampling frame and 15 selected players;
- substantial player-level uncertainty;
- synthetic null results depend on the specified probability model;
- opponent-adjusted inference uses only 15 player clusters;
- unmeasured match and player context may still confound the association.

## Reproducibility

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows
python -m pip install -r requirements.txt
python -m pytest -q
```

Run the verified multi-player data pipeline only when new collection is actually required:

```bash
python -m src.pipeline multi --config config/config.yaml
```

Run the v13 analytical layer on the existing verified multi-player features:

```bash
python -m src.pipeline v13 --config config/config.yaml
```

The v13 stage does not recollect Lichess games. It uses the existing `data/processed/multi_player_games_features.csv` artifact.

## Detailed results

- [Research report](report.md)
- [Multi-player analysis](results/multi_player_analysis.md)
- [V13 results](results/v13_analysis.md)
