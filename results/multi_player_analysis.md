# Multi-Player Bullet Analysis

The primary multi-player analysis uses a reproducibly selected random sample of active Lichess bullet players from the configured bullet leaderboard sampling frame.
Player selection is independent of tilt outcomes, streaks, breaks, loss rates and model performance.

## Population and sampling

- Candidate pool: 100
- Target players: 15
- Selected players: 15
- Time control: **60+0**
- Cutoff: **2026-09-28T23:59:59Z**
- Eligibility: at least **500 analyzable decisive 60+0 games before cutoff**
- Selection unit: **latest_500_decisive_games**
- Candidates checked: 27
- Random seed: 42

The sampling frame is the public Lichess bullet leaderboard. It is a convenience sampling frame and should not be interpreted as representative of all Lichess users.
Eligibility is determined from the actual parsed dataset after the fixed cutoff and exact 60+0 time-control filter; external profile game counts are not used as the final eligibility rule.

## Observation window

For each selected player, the fixed sample size is the latest 500 decisive 60+0 games before the cutoff. Decisive games define the sample size, but all analyzable 60+0 games between the timestamp of the earliest of those 500 decisive games and the cutoff are retained for sequential feature construction.
This preserves draws and the complete temporal sequence needed for previous result, loss/win streaks, breaks, sessions and games-in-session.

## Player-level window summary

| player_id            |   selected_decisive_games |   total_games_in_window |   decisive_games_in_window |   draws_in_window |   games |   decisive_games |   sessions | analysis_window_start     | analysis_window_end       |   window_duration_days |   tilt_observations |   control_observations |   tilt_loss_rate |   control_loss_rate |   difference_pp |
|:---------------------|--------------------------:|------------------------:|---------------------------:|------------------:|--------:|-----------------:|-----------:|:--------------------------|:--------------------------|-----------------------:|--------------------:|-----------------------:|-----------------:|--------------------:|----------------:|
| albornoz00           |                       500 |                     542 |                        500 |                42 |     542 |              500 |        170 | 2025-08-09T19:10:35+00:00 | 2026-09-28T23:59:59+00:00 |               415.201  |                  35 |                    465 |         0.485714 |            0.365591 |       12.0123   |
| arseniii_nesterov    |                       500 |                     549 |                        500 |                49 |     549 |              500 |         76 | 2026-01-06T14:14:45+00:00 | 2026-09-28T23:59:59+00:00 |               265.406  |                  53 |                    447 |         0.471698 |            0.33557  |       13.6128   |
| black_knight22       |                       500 |                     533 |                        500 |                33 |     533 |              500 |        233 | 2025-10-22T08:29:45+00:00 | 2026-09-28T23:59:59+00:00 |               341.646  |                   9 |                    491 |         0.555556 |            0.303462 |       25.2093   |
| gepard2014           |                       500 |                     526 |                        500 |                26 |     526 |              500 |         67 | 2026-01-31T20:17:21+00:00 | 2026-09-28T23:59:59+00:00 |               240.155  |                  88 |                    412 |         0.681818 |            0.398058 |       28.376    |
| gmadham              |                       500 |                     533 |                        500 |                33 |     533 |              500 |         30 | 2026-09-01T17:29:07+00:00 | 2026-09-28T23:59:59+00:00 |                27.2714 |                  53 |                    447 |         0.433962 |            0.299776 |       13.4186   |
| kayra_kamer          |                       500 |                     527 |                        500 |                27 |     527 |              500 |        112 | 2025-12-18T09:26:57+00:00 | 2026-09-28T23:59:59+00:00 |               284.606  |                  24 |                    476 |         0.291667 |            0.344538 |       -5.28711  |
| kingdomino1          |                       500 |                     539 |                        500 |                39 |     539 |              500 |         52 | 2020-09-13T16:23:56+00:00 | 2026-09-28T23:59:59+00:00 |              2206.32   |                  15 |                    485 |         0.266667 |            0.22268  |        4.39863  |
| kontrajako           |                       500 |                     532 |                        500 |                32 |     532 |              500 |         15 | 2025-03-22T19:53:23+00:00 | 2026-09-28T23:59:59+00:00 |               555.171  |                  42 |                    458 |         0.261905 |            0.349345 |       -8.74402  |
| maratgilfanovyoutube |                       500 |                     535 |                        500 |                35 |     535 |              500 |         80 | 2026-06-05T13:54:54+00:00 | 2026-09-28T23:59:59+00:00 |               115.42   |                  47 |                    453 |         0.425532 |            0.362031 |        6.3501   |
| onepoundrook         |                       500 |                     546 |                        500 |                46 |     546 |              500 |         42 | 2024-03-10T18:15:10+00:00 | 2026-09-28T23:59:59+00:00 |               932.239  |                  61 |                    439 |         0.52459  |            0.307517 |       21.7073   |
| roadto18thworldchamp |                       500 |                     532 |                        500 |                32 |     532 |              500 |         54 | 2026-08-14T13:31:20+00:00 | 2026-09-28T23:59:59+00:00 |                45.4366 |                  42 |                    458 |         0.238095 |            0.371179 |      -13.3084   |
| sergiooliva64        |                       500 |                     528 |                        500 |                28 |     528 |              500 |         37 | 2026-01-16T16:30:47+00:00 | 2026-09-28T23:59:59+00:00 |               255.312  |                  34 |                    466 |         0.441176 |            0.212446 |       22.873    |
| tonygazzo            |                       500 |                     547 |                        500 |                47 |     547 |              500 |         25 | 2026-09-17T12:56:47+00:00 | 2026-09-28T23:59:59+00:00 |                11.4606 |                  53 |                    447 |         0.358491 |            0.364653 |       -0.616268 |
| vladimirovich9000    |                       500 |                     555 |                        500 |                55 |     555 |              500 |         55 | 2025-10-22T16:38:14+00:00 | 2026-09-28T23:59:59+00:00 |               341.307  |                  42 |                    458 |         0.428571 |            0.331878 |        9.66937  |
| yoseph2013           |                       500 |                     519 |                        500 |                19 |     519 |              500 |         11 | 2026-05-12T19:35:29+00:00 | 2026-09-28T23:59:59+00:00 |               139.184  |                  43 |                    457 |         0.44186  |            0.21663  |       22.523    |

## Pooled descriptive result

Total games in windows: 8043; decisive games: 7500; sessions: 1059.
Tilt observations: 641; control observations: 6859.
Observed pooled difference: 12.66 percentage points.

The pooled result is descriptive only. Observations from the same player are not treated as fully independent in a new pooled inferential model.

## Interpretation

The baseline tilt definition remains the existing rule: previous game was a loss, break <= 5 minutes, and previous loss streak >= 2. Player-level variation is retained without ranking players.
## Player-level heterogeneity

### Method

The baseline tilt association is estimated separately for each selected player using the predefined baseline tilt proxy: previous loss, break <= 5 minutes, and previous loss streak >= 2. The observed risk difference, odds ratio and p-value reuse the project's existing statistical logic. Difference confidence intervals use the existing session-aware bootstrap with the configured bootstrap iterations and seed.

No player is removed because of a small tilt group. Groups below the project's existing `min_group_size` are retained and explicitly flagged as having greater statistical uncertainty.

### Results

Pooled descriptive association: **12.66 percentage points**.
Player-level point estimates range from **-13.31** to **28.38 pp**; mean = **10.15 pp**, median = **12.01 pp**.
Positive estimates: **11**; negative estimates: **4**; zero estimates: **0**.

The pooled association is not treated as a typical player-specific association. The player-specific estimates describe variation across the sampled players and retain their individual uncertainty.

### Small tilt groups

Players below the existing configured `min_group_size` reference: black_knight22. They remain in the analysis and are not filtered out.

### Interpretation

The estimated association varies across players to the extent shown by the observed point-estimate range and confidence intervals. This is descriptive evidence about heterogeneity in the sampled players, not evidence that any individual player is psychologically more or less prone to tilt.

The sample represents randomly selected active bullet players from the specified Lichess bullet leaderboard sampling frame who met the predefined data-availability criteria. It should not be interpreted as an average effect for all Lichess users.
