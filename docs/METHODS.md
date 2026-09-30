# Methods

## Final population

Active bullet players from the predefined public Lichess bullet leaderboard sampling frame; random seed 42; 15 selected players; exact `60+0`; fixed cutoff `2026-09-28T23:59:59Z`.

For each player, the latest 500 decisive `60+0` games define the fixed sample-size anchor. All analyzable `60+0` games in the corresponding time window are retained, including draws.

## Behavioral proxy

```text
previous game = loss
AND break <= 5 minutes
AND previous loss streak >= 2
```

The proxy is an operational behavioral rule, not a direct psychological measurement.

## V13 synthetic null

Decisive outcomes are regenerated while preserving player, opponent, rating, color, time, break, session, draw and repeated-opponent structure. The raw null uses Elo probabilities. The calibrated null applies an expanding player-specific residual correction using only prior decisive games. The same baseline proxy is then recomputed on every simulated sequence.

## Opponent-adjusted analysis

Current-game loss is modelled with logistic regression. The adjusted specifications include rating difference, color, player fixed effects and, in Model C, same-opponent status. Standard errors are cluster-robust by player.

## Uncertainty terminology

Synthetic null intervals are described as **simulation intervals**. Player-level bootstrap intervals are **confidence intervals**. Empirical upper-tail proportions are reported descriptively and are not relabelled as conventional p-values.
