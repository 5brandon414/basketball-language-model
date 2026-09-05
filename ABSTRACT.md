# Speaking Basketball: Generative Play-by-Play That Picks Its Own Lineups

## Introduction

A basketball game writes itself down as a document, each possession a
sentence and each event a word. Forecasting rarely reads it whole, and standard live models move with score, clock and a pregame rating,
not with the hundreds of events on the page. Event-stream models already
exist for soccer and baseball, and they hold the lineup fixed.
Basketball needs a model to keep choosing, and un-choosing, who is on
the floor. We ask whether one model writing the game out can match the
specialists.

## Methods

We turn 11,896 games of public NBA.com play-by-play, ten regular seasons
through 2025-26, into 5.79 million tokens over a 60-symbol vocabulary. A
1.76M-parameter decoder-only transformer reads a whole game and predicts
each event, its clock, its actor on the floor and its entrant off the
bench. Three walk-forward folds train from scratch to a cutoff and test
on the next 180 games. Identity is a learned embedding plus a 49-dial
card of each player's state strictly before that game, rebuilt per fold
from pre-cutoff games. The model knows who is available and who starts,
not who enters or for how long. Pregame forecasts come from a margin
head reading the cards alone, and 200 raw rollouts of the remaining game
give halftime forecasts and all intervals.

## Results

The model leads every baseline at next-event prediction in all three
folds, beating gradient-boosted trees by 2.4 accuracy points pooled,
on near-forced and open transitions alike (Table 1). On pregame margin no gap
between model and trees clears a paired bootstrap (Table 2) while the
gain over Elo does. At halftime the box-score
trees significantly edge the sampled margin while the rollouts, choosing
every substitution themselves, clear the lead baseline. Uncorrected margin
and total intervals cover nominal 50, 80 and 90 percent within 2.1
percentage points, pregame and at halftime alike.

Table 1. Next-event top-1 accuracy, 16 classes, 269,881 pooled
positions, split at median next-class entropy into near-forced
transitions and genuine decisions.

| Model | All | Constrained | Open |
|---|---|---|---|
| Historical average (none) | 14.1% | 23.9% | 2.5% |
| Bigram (last event) | 37.8% | 49.3% | 24.1% |
| GB trees (recent events + state) | 43.7% | 54.0% | 31.3% |
| Model (entire game) | 46.0% | 56.2% | 33.9% |

Table 2. Margins, fold-averaged over 540 test games. Trees pool the
same cards and availability; halftime trees add the first-half box
score. Baselines refit per fold.

| Pregame | Corr | Winner % | MAE |
|---|---|---|---|
| Elo | 0.382 | 61.7 | 10.83 |
| GB trees | 0.442 | 65.2 | 10.48 |
| Model head | 0.426 | 63.0 | 10.58 |
| **Halftime** | | | |
| Lead regression | 0.639 | 70.7 | 9.07 |
| GB trees | 0.697 | 73.0 | 8.42 |
| Model rollouts | 0.678 | 71.5 | 8.72 |

## Conclusion

A model that picks its own lineups cedes a 0.02 halftime edge to
margin specialists sharing its inputs, and leads every baseline event
by event. Margin, total, winner and
counterfactuals like a starter sitting out come from the same
sampled second halves, one forecast rather than three numbers. Near-nominal coverage turns its margin forecasts into
odds a staff can plan against.
