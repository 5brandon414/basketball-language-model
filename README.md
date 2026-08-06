# Basketball Language Model

A game is a document; an event is a token. A small decoder-only transformer
(~1.75M params, d=160 x 5) reads an NBA game as a sequence of 60 event
tokens conditioned on the ten players on the floor and predicts the next
event, its clock burn, the actor, the assister, and — on substitutions —
who checks in. Simulation = sampling the rest of the document many times
and reading score distributions off the rollouts.

## Dataset
Code consumes the companion Hugging Face dataset (link added on
publication; pin a dataset revision when citing): 11,896 games / 5.79M events, 2016-17..2025-26,
built from public NBA.com endpoints. See DATA.md for the complete token
grammar and schemas. Point every script at it with `--data-dir`.

## Setup
python >= 3.9; `pip install -r requirements.txt`.

## Evaluation design
The paper evaluates strictly forward in time. Three walk-forward folds cut
at 2024-12-15, 2025-03-01 and 2025-12-15; each fold trains a model from
scratch on every game before its cutoff (the last 400 held out for early
stopping) and tests on the 180 games that follow. Every baseline is refit
per fold on the same pre-cutoff record. `data/fold{1,2,3}_splits.json`
carry the assignment (games after a cutoff that are not that fold's test
games are labelled `exclude` and never trained on) and
`data/fold{1,2,3}_test_gids.txt` list the test games. Reported numbers
average the three folds. The released checkpoint is fold 3.

## Reproduce (seeds pinned; fold 3 shown, repeat with 1 and 2)
```
python3 train_bball_lm.py --epochs 40 --maxlen 688 \
    --card player_card_fold3.parquet --splits data/fold3_splits.json \
    --ast-w 0.3 --sr-w 0.3 --mh-w 1.5 --pw-w 0.3 --mh-wpool --mh-cardonly \
    --sd-drop 0.3 --seed 7 --data-dir <dataset> --out ckpt/
python3 generate_bball_lm.py --rollouts 24 --games 180 --split all \
    --gids-file data/fold3_test_gids.txt --splits data/fold3_splits.json \
    --state half --lm-subs --kv --seed 7 --data-dir <dataset> \
    --ckpt ckpt/best_model.pt        # the paper's halftime configuration
python3 experiments/table1_nextevent.py --data-dir <dataset> \
    --splits data/fold3_splits.json --fit-with-val --ckpt ckpt/best_model.pt
python3 experiments/table2_margins.py --data-dir <dataset> \
    --splits data/fold3_splits.json --fit-with-val --ckpt ckpt/best_model.pt \
    --lm-half-preds <dataset>/lm_half_preds.parquet
# optional variance-shrink tooling (paper coverage is raw); needs pregame
# rollouts first (--state pre):
python3 d_lite_calibrate.py ckpt/
```
These commands reproduce one fold. The paper's tables average the three,
and its significance tests resample folds jointly, so a single fold's
printed confidence interval is narrower in scope than the paper's. Run
all three folds and average to match the reported numbers.
`--fit-with-val` is the paper's fairness rule, giving every baseline the
full pre-cutoff record. `--lm-half-preds` supplies the halftime rollout
predictions shipped with the dataset, produced by the generate command
above from the fold 3 checkpoint.

`--mh-cardonly` is part of the paper's recipe, not an option: it makes the
pregame margin and win-probability heads read the knowledge card alone,
with player embeddings zeroed, so pregame forecasts carry no learned
player identity. Omitting it trains a different model.

## Expected metrics (three folds, 540 test games, fold-averaged)
| Metric | Standard baseline | GB trees | Basketball LM |
|---|---|---|---|
| Next-event class top-1 (pooled) | 37.8% (bigram) | 42.6% | 45.0% |
| Pregame margin corr / acc / MAE | 0.382 / 62.0% / 10.83 | 0.426 / 63.5% / 10.55 | 0.426 / 63.0% / 10.58 |
| Halftime margin corr / acc / MAE | 0.639 / 70.7% / 9.07 | 0.672 / 71.5% / 8.78 | 0.661 / 72.2% / 8.88 |
| Game totals corr (halftime) | — | — | 0.721 |
| Halftime coverage at 50/80/90 (raw) | — | — | 47.0 / 75.4 / 86.3 |

Pregame LM figures come from the card-only roster head; halftime figures
use raw sampling with the model driving its own substitutions
(`--lm-subs`), no steering, and the real second-half rotations are never
consulted. The model's information is the public pregame state (available
roster, starters, strictly-before statistics) plus, at halftime, the
observed first half.

MIT licensed. File-by-file provenance in MANIFEST.md.
