# Speaking Basketball: A Play-by-Play Grammar and a Generative Model for Unified In-Game Forecasting

## Introduction

A basketball game is a sequence of discrete events shaped by who is on
the floor. Forecasting it is normally split across a chain of specialized
models: team rating, minutes projector, possession simulator, whose
links cannot share information or agree on uncertainty. Transformer sequence
models have reached sport (SSAC baseball pitch-outcome work, soccer
event models) but stop at predicting the next event. We take it to its
generative conclusion: the Basketball Language Model, a decoder-only
transformer that treats the game as a document, each event a
token, and generates the rest, event by event: one calibrated joint distribution
over endings, answering win probability, margin, total, rotations, and
lineup what-ifs.

## Methods

We tokenize play-by-play from 11,896 games across ten seasons (2016-2025)
into a 60-symbol vocabulary spanning every recorded event type, with
shots refined by zone, outcome, and assist. A compact decoder-only transformer, conditioned on the entire game so
far, predicts, through parallel heads, the next event, its
elapsed time, the acting player, and, on substitutions, the entrant, via
pointers over the on-court ten and the 13-man bench. Player identity
enters through learned embeddings plus a "knowledge card" of
season-to-date statistics computed strictly before each game; new players
and seasons slot in without retraining. Simulation runs rollouts in
parallel; forecasts are the distributions; calibration is
verified by coverage tests.

## Results

All baselines are paired on the identical 699 held-out games; captions
define them. On its native task, next-event prediction, the model leads
every baseline at every level of predictability, from near-forced
transitions to open decisions (Table 1). Rolled forward into full-game
simulation, the same model matches or beats every baseline on the margin
tasks (Table 2): correlation gains over the Elo and halftime-lead
baselines are statistically significant, winner accuracy and error are
ties, and gradient-boosted trees on the same inputs tie every pointwise
number: nothing is sacrificed for the generative machinery. Beyond any
pointwise model, the same rollouts yield per-player minutes (5.21 MAE vs
a dedicated rotation model's 5.33 and a naive habit's 5.47), game totals
(0.719 correlation), and a calibrated joint distribution with coverage
within ±5 percentage points.

Table 1. Next-event prediction, semantic event-class top-1 (16 classes),
stratified by transition difficulty: a median split on the entropy of the
next class given the prior event. Constrained transitions are the
play-by-play grammar's near-forced rules (a miss is followed by a
rebound); Open transitions are genuine basketball decisions.
Sixteen-class perplexity:
4.00 (LM), 4.89 (GB), 5.45 (bigram), 12.70 (historical). Three-bucket and
per-class results appear in the paper.

| Model | All | Constrained | Open |
|---|---|---|---|
| Historical average (context-free) | 14.3% | 24.1% | 2.5% |
| Bigram (last event) | 37.7% | 48.8% | 24.4% |
| GB trees (context features) | 42.6% | 53.0% | 30.2% |
| Basketball LM (full game) | 45.5% | 55.4% | 33.6% |

Table 2. Margin prediction on the identical 699 held-out games; standard
baseline = point-spread Elo (pregame rows), halftime-lead regression
(halftime rows); gradient-boosted trees receive the same roster inputs as
the model. Correlation gains over the standard baselines are significant
(paired 95% CI excludes zero); all other gaps, including every
model-vs-tree difference, are not.

| Metric (699 games) | Standard baseline | GB trees | Basketball LM |
|---|---|---|---|
| Pregame margin corr | 0.411 | 0.475 | 0.490 |
| Pregame winner accuracy | 65.2% | 67.1% | 68.0% |
| Pregame margin MAE | 10.68 | 10.40 | 10.47 |
| Halftime margin corr | 0.643 | 0.685 | 0.686 |
| Halftime winner accuracy | 73.5% | 74.7% | 73.7% |
| Halftime margin MAE | 9.15 | 8.76 | 8.75 |

## Conclusion

One transformer replaces the forecasting chain. It wins next-event
prediction outright and adds what no specialist can: a single calibrated
distribution over how the game ends, queryable from any game state. The
same rollouts that forecast the margin also produce each player's
minutes, the run of play, and the box score, so win probability, rotation
forecasts, and lineup what-ifs come from one model instead of three that
disagree. A gradient-boosted tree matches its point predictions and
produces nothing else; the model calls the whole game. The method needs
only play-by-play, not tracking data, and transfers to any league that
records it. An announcer
calls the game once; the model calls it again and again, and the
distribution of endings is the forecast. Data and model will be released
openly upon publication.
