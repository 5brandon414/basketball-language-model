# Speaking Basketball: A Play-by-Play Grammar and a Generative Model for Unified In-Game Forecasting

## Introduction

Play-by-play is basketball's transcript: hundreds of ordered events per
game, with a grammar of its own, building up the game sequence by sequence.
Yet forecasting never reads it whole. The game is split across
specialized models: team rating, minutes projector, possession simulator,
none sharing information or agreeing on uncertainty. Transformer sequence
models have reached sport (SSAC baseball pitch-outcome work, soccer
event models) but stop at predicting the next event. We take it to its
generative conclusion: the Basketball Language Model, a decoder-only
transformer that treats the game as a document, each event a token, and
generates the rest, event by event: one joint distribution over endings, answering win probability, margin, total, rotations, and
lineup what-ifs.

## Methods

We tokenize play-by-play from 11,896 games across ten seasons (2016-2025)
into a 60-symbol vocabulary spanning every recorded event type, with
shots refined by zone, outcome, and assist. A compact decoder-only transformer, conditioned on the entire game so
far, predicts, through parallel heads, the next event, its
elapsed time, the acting player, and, on substitutions, the entrant, via
pointers over the on-court ten and each bench. Player identity
enters through learned embeddings plus a "knowledge card" of
season-to-date statistics computed strictly before each game; new players
and seasons slot in without retraining. Pregame the model sees only public
information: roster, starters, prior statistics, never who will enter or
for how long. Simulation runs rollouts in
parallel; forecasts are the distributions.

## Results

All baselines are paired on the identical 500 held-out games; captions
define them. On its native task, next-event prediction, the model leads
every baseline at every level of predictability (Table 1). Rolled forward into full-game
simulation, the same model matches or beats every margin baseline
(Table 2): correlation gains over Elo and halftime-lead are
statistically significant; winner accuracy and error are ties, and gradient-boosted trees on the same inputs tie every pointwise
number: nothing is sacrificed for generation. Beyond any
pointwise model, the same rollouts yield per-player minutes, the run of
play, game totals (0.707 correlation), and a calibrated joint distribution:
raw sampling covers 50/80/90% margin intervals within ±3 percentage
points (totals ±7), no post-hoc correction. The model drives every substitution
itself; real rotations are never consulted.

Table 1. Next-event prediction, semantic event-class top-1 (16 classes),
stratified by transition difficulty: a median split on the entropy of the
next class given the prior event. Constrained transitions are the
play-by-play grammar's near-forced rules (a miss is followed by a
rebound); Open transitions are genuine basketball decisions.
Sixteen-class perplexity:
4.01 (LM), 4.61 (GB), 5.44 (bigram), 12.71 (historical). Three-bucket and
per-class results appear in the paper.

| Model | All | Constrained | Open |
|---|---|---|---|
| Historical average (context-free) | 14.2% | 24.0% | 2.5% |
| Bigram (last event) | 37.6% | 48.8% | 24.4% |
| GB trees (context features) | 42.7% | 53.1% | 30.2% |
| Basketball LM (full game) | 45.3% | 55.3% | 33.3% |

Table 2. Margin prediction on the identical 500 held-out games; standard
baseline = point-spread Elo (pregame rows), halftime-lead regression
(halftime rows); gradient-boosted trees receive the same roster inputs as
the model. Correlation gains over the standard baselines are significant
(paired 95% CI excludes zero); all other gaps, including every
model-vs-tree difference, are not.

| Metric (500 games) | Standard baseline | GB trees | Basketball LM |
|---|---|---|---|
| Pregame margin corr | 0.414 | 0.460 | 0.509 |
| Pregame winner accuracy | 64.8% | 65.4% | 65.0% |
| Pregame margin MAE | 10.82 | 10.63 | 10.54 |
| Halftime margin corr | 0.648 | 0.684 | 0.690 |
| Halftime winner accuracy | 73.6% | 75.4% | 74.6% |
| Halftime margin MAE | 9.21 | 8.98 | 8.84 |

## Conclusion

One transformer replaces the forecasting chain. It wins next-event
prediction outright and adds what no specialist can: a single calibrated
distribution over how the game ends, queryable from any game state. The
same rollouts that forecast the margin also produce each player's
minutes and the box score, so win probability, rotation
forecasts, and lineup what-ifs come from one model instead of three that
disagree. A gradient-boosted tree matches its point predictions and
produces nothing else; the model calls the whole game. It needs only play-by-play, not tracking, and
transfers to any league that records it. An announcer
calls the game once; the model calls it again and again, and the
distribution of endings is the forecast. Model and code are openly released.
