# Basketball Language Model

A game is a document; an event is a token. A small decoder-only
transformer reads an NBA game as a sequence of event tokens conditioned
on the ten players on the floor and predicts the next event, its clock
burn, the actor, the assister, and — on substitutions — who checks in.
Simulation is sampling the rest of the document many times and reading
score distributions off the rollouts.

The submission itself, including the result tables, is ABSTRACT.md.

## Data
The play-by-play comes from public NBA.com endpoints and is **not
redistributed here**. DATA.md is the build specification — the token
grammar and the schema of every file the code expects — so the corpus can
be rebuilt from those endpoints. Point the scripts at your build with
`--data-dir`.

The evaluation splits **are** published, and they are the part that
matters for comparability: `data/fold{1,2,3}_splits.json` and
`data/fold{1,2,3}_test_gids.txt` are game-id lists, so a corpus rebuilt
independently can be trained and scored on exactly the games behind every
number in the paper.

Trained weights are not published with this repository; they come in a
later phase of the work.

## Setup
python >= 3.9; `pip install -r requirements.txt`.

## Evaluation design
The paper evaluates strictly forward in time. Three walk-forward folds cut
at 2024-12-15, 2025-03-01 and 2025-12-15; each fold trains a model from
scratch on every game before its cutoff (the last 400 held out for early
stopping) and tests on the 180 games that follow. Every baseline is refit
per fold on the same pre-cutoff record. In
`data/fold{1,2,3}_splits.json`, games after a cutoff that are not that
fold's test games are labelled `exclude` and never trained on. Reported
numbers average the three folds.

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
    --splits data/fold3_splits.json --card player_card_fold3.parquet \
    --fit-with-val --ckpt ckpt/best_model.pt --lm-half-preds <half_preds>.parquet
```

Two flags are part of the recipe, not options. `--mh-cardonly` makes the
pregame margin and win-probability heads read the knowledge card alone,
with player embeddings zeroed, so pregame forecasts carry no learned
player identity. `--fit-with-val` gives every baseline the full
pre-cutoff record. Omitting either runs a different experiment.

The generate step writes `gen_all_half_lmsubs.csv` beside the checkpoint.
`--lm-half-preds` reads a parquet with columns `pred` and `actual`
(optionally `pred_total`, `actual_total`) and scores its rows as they
stand, so convert that CSV first: `pm` and `am` are the predicted and
actual margins, `pt` and `at` the totals.

## Reading the output
Each run reports one fold. The paper averages three, and its significance
tests resample folds jointly, so a single fold's printed confidence
interval is narrower in scope than the paper's, and a single fold will not
reproduce the fold-averaged tables in ABSTRACT.md exactly. Run all three
folds and average to compare.

The LM pregame column is the card-only head, not a rollout. The LM
halftime column is raw sampling with the model driving its own
substitutions (`--lm-subs`), no steering, and the real second-half
rotations are never consulted. The model's information is the public
pregame state (available roster, starters, strictly-before statistics)
plus, at halftime, the observed first half.

Halftime interval coverage at the nominal 50, 80 and 90 percent levels is
47.0 / 75.4 / 86.3 percent, uncorrected. The generate step writes each
game's per-rollout margins, so that is checkable directly rather than
through a normal approximation.

MIT licensed. File inventory in MANIFEST.md.
