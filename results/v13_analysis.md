# V13: Synthetic Null and Opponent-Adjusted Analysis

V13 is an analytical layer built on the fixed v12 research dataset. The v12 sampling design, data, tilt proxy, sensitivity analysis, heterogeneity analysis and predictive models are preserved.

## Observed baseline sanity check

- Players: **15**
- Total games: **8043**
- Decisive games: **7500**
- Draws retained: **543**
- Observed tilt-proxy observations among decisive games: **641**
- Observed control observations: **6859**
- Observed pooled difference: **+12.66 pp**

The observed baseline is recomputed from the same multi-player feature dataset before any synthetic data are generated.

## Synthetic null model

The null simulation preserves player identities, opponents, ratings, rating differences, colors, timestamps, breaks, sessions, draws, repeated-opponent structure and the number of games. Only decisive win/loss outcomes are regenerated from a specified probability model, after which the same baseline tilt-proxy logic is recomputed.

### Probability specifications

1. **Raw Elo:** `p(win) = 1 / (1 + 10^(-rating_diff / 400))`.
2. **Player-calibrated Elo:** expanding player-specific residual correction using only prior decisive games; p_cal = clip(p_elo + mean(prior observed_loss - p_elo), 1e-4, 1-1e-4).

These are null data-generating processes. They contain no explicit tilt effect.

### Raw Elo

- Simulations: **2000**
- Mean difference: **+11.62 pp**
- Median difference: **+11.67 pp**
- Standard deviation: **1.98 pp**
- 95% simulation interval: **[+7.65, +15.37] pp**
- Share of null simulations at least as large as observed: **31.6%**
- Mean simulated tilt-proxy count: **776.9**

### Player Calibrated Elo

- Simulations: **2000**
- Mean difference: **+12.18 pp**
- Median difference: **+12.15 pp**
- Standard deviation: **2.08 pp**
- 95% simulation interval: **[+8.28, +16.11] pp**
- Share of null simulations at least as large as observed: **40.5%**
- Mean simulated tilt-proxy count: **686.1**

The simulation interval describes the spread of the synthetic null distribution; it is not a confidence interval for the observed effect. The empirical upper-tail proportion is not labelled as a p-value.

## Opponent context

| Group | Observations | Mean rating difference | Mean opponent rating | Same-opponent rate |
|---|---:|---:|---:|---:|
| after_previous_loss | 2294 | 84.25 | 2953.01 | 43.5% |
| after_previous_nonloss | 5191 | 151.10 | 2906.91 | 42.8% |
| short_break | 6216 | 128.87 | 2925.27 | 51.0% |
| longer_break | 1269 | 139.14 | 2900.31 | 3.5% |
| after_previous_loss_and_short_break | 1834 | 71.58 | 2976.02 | 53.4% |

## Opponent-adjusted logistic models

The target is the current-game loss among decisive games. Standard errors are cluster-robust by `player_id`. Model B adds rating difference, color and player fixed effects. Model C additionally controls for whether the current opponent is the same as in the previous game.

| Model | Tilt OR | 95% CI | p-value | N | Player clusters |
|---|---:|---|---:|---:|---:|
| A_unadjusted | 1.717 | [1.228, 2.400] | 0.0016 | 7500 | 15 |
| B_opponent_player_adjusted | 1.122 | [0.901, 1.397] | 0.3028 | 7500 | 15 |
| C_same_opponent_adjusted | 1.120 | [0.899, 1.395] | 0.3122 | 7500 | 15 |

## Interpretation

The raw pooled association is compatible with a null data-generating process that preserves the observed structure of the games while regenerating outcomes without an explicit tilt effect. The player-calibrated null gives a similar result, with its distribution depending on the chosen probability specification.

After adjusting for rating difference, color and player fixed effects, the estimated tilt-proxy association is materially smaller than the raw pooled association. Adding same-opponent status produces a closely related adjusted estimate. These models describe observational associations; they do not establish a psychological mechanism or causality.

The v13 result therefore shifts the main interpretation from treating the raw +12.66 pp difference as a direct behavioral effect toward treating it as an association that can arise from the broader game-generating process and match context.
