# Speaking Basketball: A Play-by-Play Grammar and a Generative Model for Unified In-Game Forecasting

## Introduction

Basketball's play-by-play serves as a transcript of the game.
It contains hundreds of ordered events with a grammar of their own,
building up the game sequence by sequence.
Yet prediction research rarely touches this piece. Pregame models
consume season aggregates, in-game models reduce it to score and clock,
and possession simulators are memoryless by construction. Transformers read event streams in
baseball (SSAC) and soccer, but none reads, or writes, a basketball game whole. 
We take it to its generative conclusion: the Basketball Language Model,
a decoder-only transformer that treats the game as a document and each
event as a token, then generates the rest event by event. Repeated
generations form one joint distribution over endings: win probability,
margin, total, rotations, and lineup what-ifs.

## Methods

We tokenize play-by-play from 11,896 games across ten seasons (2016-2025)
into a 60-symbol vocabulary spanning every recorded event type;
shots are refined by zone, outcome, and assist. A compact decoder-only transformer reads the entire game so far and
predicts what happens next: the event, its elapsed time, and the acting
player, chosen by pointing at the ten on court. On substitutions, a
second pointer picks who enters from the bench. Player identity
enters through learned embeddings plus a "knowledge card" of
season-to-date statistics computed before each game; new players
and seasons slot in without retraining. Pregame, before any events
exist, a head over the pooled rosters gives the point forecast; in-game
forecasts come from simulation. Rollouts run in parallel from any
real game state, using only information available at that moment.

## Results

On its native task, next-event prediction, the model leads every
baseline at every level of predictability, all on the same 500 held-out
games (Table 1). Across both horizons it matches or beats every margin
baseline (Table 2). Correlation gains over Elo and halftime-lead are
statistically significant; winner accuracy and error are ties, and
gradient-boosted trees tie every pointwise number, so nothing is
sacrificed for generation. Beyond pointwise models, the rollouts write the full run of play. The
model drives every substitution itself, never consulting real rotations, and
per-player minutes and totals (0.707 correlation) fall out of the
simulation. Calibration holds as sampled: 50/80/90% margin intervals
cover within ±3 percentage points, totals within ±7.

Table 1. Next-event prediction, semantic event-class top-1 (16 classes),
stratified by transition difficulty: a median split on the entropy of the
next class given the prior event. Constrained transitions are the
play-by-play grammar's near-forced rules (a miss is followed by a
rebound); Open transitions are genuine basketball decisions.
Sixteen-class perplexity:
4.01 (LM), 4.61 (GB), 5.44 (bigram), 12.71 (historical). Three-bucket and
per-class results appear in the paper.

| Model (context used) | All | Constrained | Open |
|---|---|---|---|
| Historical average (none) | 14.2% | 24.0% | 2.5% |
| Bigram (last event) | 37.6% | 48.8% | 24.4% |
| GB trees (recent events + state) | 42.7% | 53.1% | 30.2% |
| Basketball LM (entire game) | 45.3% | 55.3% | 33.3% |

Table 2. Standard baseline = point-spread Elo with margin-of-victory
updates (K=20, home edge 100, 25% season regression) for pregame rows; a
train-fit regression of final margin on halftime lead for halftime rows.
Gradient-boosted trees receive the same roster inputs as the model.
Correlation gains over the standard baselines are significant (paired 95%
CI excludes zero); all other gaps are not.

| Metric (500 games) | Standard baseline | GB trees | Basketball LM |
|---|---|---|---|
| Pregame margin corr | 0.414 | 0.460 | 0.509 |
| Pregame winner accuracy | 64.8% | 65.4% | 65.0% |
| Pregame margin MAE | 10.82 | 10.63 | 10.54 |
| Halftime margin corr | 0.648 | 0.684 | 0.690 |
| Halftime winner accuracy | 73.6% | 75.4% | 74.6% |
| Halftime margin MAE | 9.21 | 8.98 | 8.84 |

## Conclusion

One model does what the separate models did. It predicts the next
event better than every baseline, and because it can simulate the rest of
the game from any point, it also gives calibrated forecasts the others
cannot: margin, minutes, win probability, and lineup what-ifs all come
from the same simulations, so they never contradict each other. A
gradient-boosted tree can match its point predictions, but cannot produce
a game. The approach needs nothing but play-by-play, so it works for any
sport that keeps one. An announcer calls the game once. The model calls it
over and over, and reads the forecast off the endings.
Model and code are openly released.
