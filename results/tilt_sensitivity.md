# Tilt Proxy Sensitivity Analysis

The existing tilt-proxy analysis was repeated with alternative break and loss-streak thresholds. The baseline definition is retained unchanged and is included in the table for direct comparison.

## Thresholds

The six configurations are 2/2, 5/2 (baseline), 10/2, 15/2, 5/3 and 5/4, where the first number is the maximum break in minutes and the second is the minimum previous loss streak.

## Results

| Configuration | Tilt n | Control n | Tilt loss rate | Control loss rate | Difference (pp) | 95% CI (pp) | Odds ratio | p-value | Stability |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| break <= 2 min, streak >= 2 | 24 | 596 | 0.583 | 0.475 | 10.9 | [-9.3, 30.2] | 1.53 | 0.2968 | ok |
| break <= 5 min, streak >= 2 (baseline) | 50 | 570 | 0.560 | 0.472 | 8.8 | [-5.2, 22.9] | 1.42 | 0.232 | ok |
| break <= 10 min, streak >= 2 | 61 | 559 | 0.557 | 0.470 | 8.7 | [-4.5, 21.8] | 1.41 | 0.1971 | ok |
| break <= 15 min, streak >= 2 | 83 | 537 | 0.542 | 0.469 | 7.3 | [-4.4, 18.8] | 1.34 | 0.216 | ok |
| break <= 5 min, streak >= 3 | 24 | 596 | 0.667 | 0.471 | 19.5 | [-0.5, 38.0] | 2.18 | 0.06055 | Smaller tilt group than baseline (24 vs 50 observations); point estimate has greater statistical uncertainty. |
| break <= 5 min, streak >= 4 | 12 | 608 | 0.667 | 0.475 | 19.1 | [-8.2, 44.3] | 2.08 | 0.1889 | Smaller tilt group than baseline (12 vs 50 observations); point estimate has greater statistical uncertainty. |

Baseline: **break <= 5 min, streak >= 2**, with 50 tilt observations and an estimated difference of 8.8 percentage points.

The estimated difference is positive for all 6 estimable configurations. Across estimable configurations, the difference ranges from 7.3 to 19.5 percentage points.

The table uses the same statistical logic as the primary analysis: the existing Fisher/two-proportion test choice, odds-ratio confidence interval, Wilson group-rate intervals, and the existing risk-difference bootstrap interval. No new inferential method is introduced here.

## Interpretation

Sensitivity analysis describes how the estimated association changes when the behavioral rule is defined with nearby thresholds. It does not establish causation and should not be interpreted as a direct measurement of psychological tilt. The configurations marked as smaller tilt groups are: break <= 5 min, streak >= 3 (24 vs 50 baseline observations); break <= 5 min, streak >= 4 (12 vs 50 baseline observations). Their point estimates have greater statistical uncertainty because they are based on substantially fewer tilt observations than the baseline. This is a descriptive comparison with the baseline group size, not a new statistical exclusion threshold.
