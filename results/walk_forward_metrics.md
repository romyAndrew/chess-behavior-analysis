# Walk-Forward Temporal Validation

The existing chronological 80/20 split remains the predictive baseline. This analysis adds expanding-window validation without changing the three model specifications.

## Method

Initial training size: **300** decisive games.
Test window: **80** games per fold.
Minimum training size: **200** games.
No random shuffle is used. Preprocessing and logistic regression are fit separately inside each fold using training data only.
Every fold satisfies `max(train timestamp) < min(test timestamp)` and has disjoint train/test indices.

## Aggregate results

| model       |   mean_roc_auc |   std_roc_auc |   mean_log_loss |   std_log_loss |   mean_brier_score |   std_brier_score |   n_folds |   valid_auc_folds |
|:------------|---------------:|--------------:|----------------:|---------------:|-------------------:|------------------:|----------:|------------------:|
| rating_only |       0.546896 |     0.039291  |        0.691379 |     0.00160282 |           0.249114 |       0.000796571 |         4 |                 4 |
| baseline    |       0.539685 |     0.0542812 |        0.690363 |     0.00124562 |           0.248611 |       0.000620892 |         4 |                 4 |
| behavioral  |       0.541397 |     0.0752146 |        0.705944 |     0.0267518  |           0.255557 |       0.0120725   |         4 |                 4 |

## 80/20 comparison

| model       |   80/20 ROC-AUC |   80/20 log-loss |   80/20 Brier |   80/20 test n |
|:------------|----------------:|-----------------:|--------------:|---------------:|
| rating_only |        0.537601 |         0.69198  |      0.249418 |            124 |
| baseline    |        0.54879  |         0.691334 |      0.249101 |            124 |
| behavioral  |        0.529794 |         0.699045 |      0.252901 |            124 |

## Fold-level results

|   fold | model       |   n_train |   n_test | train_end                 | test_start                |   roc_auc |   log_loss |   brier_score |
|-------:|:------------|----------:|---------:|:--------------------------|:--------------------------|----------:|-----------:|--------------:|
|      1 | rating_only |       300 |       80 | 2026-05-20 20:20:32+00:00 | 2026-05-20 20:28:00+00:00 |  0.495301 |   0.691328 |      0.249098 |
|      1 | baseline    |       300 |       80 | 2026-05-20 20:20:32+00:00 | 2026-05-20 20:28:00+00:00 |  0.470551 |   0.691689 |      0.24927  |
|      1 | behavioral  |       300 |       80 | 2026-05-20 20:20:32+00:00 | 2026-05-20 20:28:00+00:00 |  0.516917 |   0.738319 |      0.26957  |
|      2 | rating_only |       380 |       80 | 2026-05-22 21:42:52+00:00 | 2026-05-22 21:45:45+00:00 |  0.585038 |   0.69024  |      0.248543 |
|      2 | baseline    |       380 |       80 | 2026-05-22 21:42:52+00:00 | 2026-05-22 21:45:45+00:00 |  0.560742 |   0.69111  |      0.248984 |
|      2 | behavioral  |       380 |       80 | 2026-05-22 21:42:52+00:00 | 2026-05-22 21:45:45+00:00 |  0.512148 |   0.702521 |      0.254307 |
|      3 | rating_only |       460 |       80 | 2026-05-31 11:31:39+00:00 | 2026-05-31 11:44:33+00:00 |  0.539062 |   0.693662 |      0.250246 |
|      3 | baseline    |       460 |       80 | 2026-05-31 11:31:39+00:00 | 2026-05-31 11:44:33+00:00 |  0.528646 |   0.689017 |      0.247936 |
|      3 | behavioral  |       460 |       80 | 2026-05-31 11:31:39+00:00 | 2026-05-31 11:44:33+00:00 |  0.484375 |   0.709742 |      0.258058 |
|      4 | rating_only |       540 |       80 | 2026-08-09 20:56:36+00:00 | 2026-08-09 21:09:59+00:00 |  0.568182 |   0.690285 |      0.248571 |
|      4 | baseline    |       540 |       80 | 2026-08-09 20:56:36+00:00 | 2026-08-09 21:09:59+00:00 |  0.598801 |   0.689637 |      0.248254 |
|      4 | behavioral  |       540 |       80 | 2026-08-09 20:56:36+00:00 | 2026-08-09 21:09:59+00:00 |  0.652146 |   0.673192 |      0.240291 |

## Interpretation

Walk-forward validation describes out-of-sample performance across several future time windows instead of relying on one holdout period. Variation between folds is evidence about temporal stability, not a reason to select a more favorable period.
A missing ROC-AUC is kept as `NA` when a test fold contains only one class; it is not replaced with a guessed value.
