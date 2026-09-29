# Tilt Proxy Statistical Analysis

## Hypotheses

- H0: Current-game loss proportion is equal for tilt-proxy and control groups.
- H1: Current-game loss proportion differs between tilt-proxy and control groups.

## Results

- Test: **two-proportion z-test**
- Tilt-proxy group: **50** decisive games
- Control group: **570** decisive games
- Loss rate, tilt proxy: **0.560**, 95% CI [0.423, 0.688]
- Loss rate, control: **0.472**, 95% CI [0.431, 0.513]
- Risk difference: **0.088**, bootstrap 95% CI [-0.052, 0.229]
- Odds ratio: **1.417**, 95% CI [0.796, 2.522]
- p-value: **0.232**
- Decision at α=0.050: **do not reject H0**

## Dependence-aware bootstrap

The existing row-level bootstrap CI is **[-5.23, 22.95]** percentage points. The session-aware bootstrap CI is **[-6.88, 21.93]** percentage points across 171 sessions. Both methods use 5000 iterations with seed 42.

The observed point estimate is unchanged because only the resampling unit changes. The session-level interval accounts for dependence among games inside the same existing session.

## Interpretation

The observed relationship is an **association**, not evidence of causation. The tilt proxy is a constructed behavioral rule based on game history, not a direct measurement of psychological state.

## Limitations

- The dataset represents one player's Lichess history.
- The proxy can be sensitive to the chosen break and streak thresholds.
- Engine evaluation coverage is incomplete and may be non-random.
- Time-control, rating, opponent strength, and other confounders may remain.
