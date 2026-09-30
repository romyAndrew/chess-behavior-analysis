# Results

## Observed association

- Tilt-proxy loss rate: **44.46%**
- Control loss rate: **31.80%**
- Difference: **+12.66 pp**

## Synthetic null

| Scenario | Mean | Simulation interval | Upper-tail share |
|---|---:|---|---:|
| Raw Elo | +11.62 pp | [7.65, 15.37] pp | 31.6% |
| Player-calibrated Elo | +12.18 pp | [8.28, 16.11] pp | 40.5% |

## Opponent-adjusted models

| Model | Tilt OR | 95% CI | p-value |
|---|---:|---|---:|
| Unadjusted | 1.717 | [1.228, 2.400] | 0.0016 |
| Rating + color + player effects | 1.122 | [0.901, 1.397] | 0.3028 |
| + same opponent | 1.120 | [0.899, 1.395] | 0.3122 |

The v13 evidence supports a more cautious interpretation of the raw association.
