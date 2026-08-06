# Speaking Basketball: Generative Play-by-Play for Unified Game Forecasting

## Introduction

Basketball already writes itself down as an ordered transcript.
Prediction rarely reads it. Instead, work splits across separate models,
among them pregame ratings, live win probability and possession
simulators, and the live model often reads the pregame forecast rather
than the game itself. Event-stream transformers write soccer and
baseball, where substitutions are few and permanent, but basketball
rotations turn over all game, the same players leaving and returning, so
writing basketball means choosing who is on the floor. We ask whether
writing the game whole can replace rating it in pieces.

## Methods

We tokenize 5.79 million public NBA.com play-by-play events from 11,896
games into a 60-symbol vocabulary. A 1.75M-parameter decoder-only
transformer (Basketball LM) reads an entire game at once and predicts
the next event, which of the ten players on the floor performs it, and,
on substitutions, who checks in from the bench. Identity arrives as a
fixed learned embedding plus a 49-dial card of each player's state
strictly before that night, rebuilt per fold, so form and role stay
current while the embedding does not.
The pregame head reads that card alone over the dressed roster and
starters, embeddings zeroed, after we measured it over-trusting stale
embeddings. In-game forecasts come from 24 rollouts. In three
walk-forward folds we train from scratch on everything before each cutoff
and test on the next 180 games, refitting every baseline per fold.

## Results

The model leads every baseline at next-event prediction in all three
folds, beating gradient-boosted trees by 2.4 accuracy points pooled
(Table 1). On margin it is level with the strongest baseline (Table 2),
only the pregame correlation gain over Elo clearing a paired bootstrap.
Driving every substitution itself and never consulting real rotations,
the model still ties the GB trees on sampled halftime margin. Halftime
intervals cover nominal 50, 80 and 90 percent within five percentage
points, uncorrected.

Table 1. Next-event top-1 over 16 classes, pooled across folds
(n=269,881). Constrained and Open are a median split on next-class
entropy, separating near-forced transitions, like a rebound after a
miss, from genuine decisions.

| Model (context used) | All | Constrained | Open |
|---|---|---|---|
| Bigram (last event) | 37.8% | 49.3% | 24.1% |
| GB trees (recent events + state) | 42.6% | 53.2% | 29.9% |
| Basketball LM (entire game) | 45.0% | 55.3% | 32.8% |

Table 2. Margins, fold-averaged over 540 test games. Standard is
point-spread Elo before the game and a halftime-lead regression at
halftime.

| Metric | Standard (Elo) | GB trees (same roster inputs) | Basketball LM |
|---|---|---|---|
| Pregame margin corr | 0.382 | 0.426 | 0.426 |
| Pregame winner accuracy | 62.0% | 63.5% | 63.0% |
| Pregame margin MAE | 10.83 | 10.55 | 10.58 |
| Halftime margin corr | 0.639 | 0.672 | 0.660 |
| Halftime winner accuracy | 70.7% | 71.5% | 72.2% |
| Halftime margin MAE | 9.07 | 8.78 | 8.88 |

## Conclusion

Because it writes the game rather than rating it, one run answers what
happens if a starter sits, and gives margin, total and win probability
that agree. It needs only play-by-play, so any league that logs events
can run the same recipe, and we release the code.
