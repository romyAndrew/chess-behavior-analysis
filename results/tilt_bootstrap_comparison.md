# Tilt Bootstrap Inference Comparison

The primary tilt estimate is unchanged. The comparison only changes the resampling unit used to quantify uncertainty.

## Methods

- **row_level_bootstrap** resamples individual decisive games independently and preserves the existing v7 inference procedure.
- **session_bootstrap** resamples complete existing sessions with replacement. All relevant games from a selected session stay together.

The session-aware method is used to account for dependence among games played close together in the same session. It does not alter the observed point estimate.

## Results

| Method | Observed difference (pp) | 95% CI (pp) | Observations | Sessions | Iterations | Seed |
|---|---:|---|---:|---:|---:|---:|
| row-level bootstrap | 8.81 | [-5.23, 22.95] | 620 | 171 | 5000 | 42 |
| session bootstrap | 8.81 | [-6.88, 21.93] | 620 | 171 | 5000 | 42 |

The session-bootstrap CI lower bound differs from the row-level CI by -1.66 pp, and the upper bound differs by -1.02 pp. A wider session-level interval indicates less independent information after accounting for within-session dependence; a similar interval indicates that the dependence adjustment has limited impact for this dataset.

## Interpretation

The two methods estimate the same observed association. Only the uncertainty calculation changes. Neither method establishes causation or measures psychological tilt directly.
