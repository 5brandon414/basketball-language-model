# Basketball Language Model

A game is a document; an event is a token. A small decoder-only
transformer reads an NBA game as a sequence of event tokens conditioned
on the ten players on the floor and predicts the next event, its clock
burn, the actor, the assister, and — on substitutions — who checks in.
Simulation is sampling the rest of the document many times and reading
score distributions off the rollouts.

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

Trained checkpoints are not tracked here; see Reproduce.

## Setup
python 3.9-3.12 (the pinned numpy and torch publish no 3.13 wheels);
`pip install -r requirements.txt`.

## Evaluation design
The paper evaluates strictly forward in time. Three walk-forward folds cut
at 2024-12-15, 2025-03-01 and 2025-12-15; each fold trains a model from
scratch on every game before its cutoff (the last 400 held out for early
stopping) and tests on the 180 games that follow. Every baseline is refit
per fold on the same pre-cutoff record. In
`data/fold{1,2,3}_splits.json`, games after a cutoff that are not that
fold's test games are labelled `exclude` and never trained on. Reported
numbers average the three folds.

## Data availability

Nothing in this repository touches a database or the network; the corpus
is read from `--data-dir`. It derives entirely from public NBA.com
play-by-play endpoints.
NBA.com's Terms of Use do not permit redistribution of that data, so the
tokenized parquet files are not included in this repository. DATA.md
specifies the full schema, token grammar, provenance, and
validation checks (token-stream score reconstruction matches official
box-score finals for 99.6 percent of games), so the corpus can be
rebuilt from the same public endpoints. The derived artifacts that are
our own outputs (the paper's halftime rollout predictions, fold split
assignments, and vocabulary) ship in this repository, and trained
checkpoints are available from the authors, so every table row that does
not require retraining can be reproduced without the raw data.

## Reproduce (seeds pinned; fold 3 shown, repeat with 1 and 2)

Training takes 4-5 hours per fold on `--device mps` or `cuda` (cpu is
impractical); `--smoke` verifies the setup in about a minute. Every
invocation pays a 10-25 minute corpus load. Trained fold checkpoints
(~140 MB each: weights plus the as-of rating table) are available from
the authors and load strict with `--ckpt`.

### Train

```
python3 train_bball_lm.py --epochs 40 --device mps \
    --card player_card_fold3.parquet --splits data/fold3_splits.json \
    --mh-w 1.5 --ast-w 0.3 --sr-w 0.3 --pw-w 0.3 --sd-drop 0.1 \
    --mh-wpool --mh-cardonly --fix-sub-boundary --mask-unlearnable \
    --budget-ch --prior <dataset>/prior_minutes.parquet \
    --pace-ch --pace-file <dataset>/team_pace.parquet --period-ch \
    --mtp-w 0.3 --tt-w 0.3 --seed 7 --data-dir <dataset> --out ckpt/
```

Every flag is part of the recipe; the checkpoint's `config.json` records
the full set and the evaluation scripts read it, so train and eval cannot
silently disagree. `--mh-cardonly`: pregame margin and win-probability
heads read the knowledge card alone, embeddings zeroed. `--fix-sub-boundary`:
substitution rows condition on the pre-substitution five. The sidecar
parquets are as-of, shifted one game, so nothing from the predicted game
leaks in.

### Rollouts

```
python3 generate_bball_lm.py --rollouts 200 --games 200 --split all \
    --gids-file data/fold3_test_gids.txt --splits data/fold3_splits.json \
    --state half --lm-subs --kv --seed 7 --data-dir <dataset> \
    --ckpt ckpt/best_model.pt        # the paper's halftime configuration
python3 generate_bball_lm.py ... --state pre ...   # same flags, pregame
```

Each run writes `gen_all_{half,pre}_lmsubs.csv` beside the checkpoint.
The paper's own halftime predictions ship as
`data/lm_half_preds_fold{1,2,3}.parquet`; to score your own run instead,
convert its CSV with `experiments/halfpreds_from_csv.py`.

### Tables

```
python3 experiments/table1_nextevent.py --data-dir <dataset> \
    --splits data/fold3_splits.json --fit-with-val --ckpt ckpt/best_model.pt
python3 experiments/table2_margins.py --data-dir <dataset> \
    --splits data/fold3_splits.json --card player_card_fold3.parquet \
    --fit-with-val --ckpt ckpt/best_model.pt \
    --lm-half-preds data/lm_half_preds_fold3.parquet
```

`--fit-with-val` gives every baseline the full pre-cutoff record. Each
run reports one fold and dumps its per-fold arrays;
`experiments/assemble_final_tables.py` pools the three folds into the
paper's tables and bootstrap CIs (see `experiments/README.md`).

## Counterfactual validation

A pre-registered star re-insertion experiment ships here: 346 candidate
cases (held-out games missing a 28-plus-minute player), 344 simulated
twice, identically seeded — star absent vs restored to roster and starting
five; 317 pass the primary validity gate, 201 the stricter pre-registered
one. Restoring the player moves the forecast toward their team by +0.72
points (95 percent CI +0.56 to +0.89; the strict gate gives +0.81 with a
negative fold-2 mean, traced to fold 2's sequence-length truncation). The
shift scales more with scoring quality (Spearman +0.34) than minutes
(+0.23); All-Stars move the line 4.6 times more than role players (Jokic
restored at Boston 2025-01-08: minus 7.5 to minus 2.2). The magnitude
sits below the 1.5-3 point consensus value of a star absence:
directionally validated, not a calibrated pricing model.
`experiments/reinsertion_analysis.py` reproduces every number from
`data/reinsert_results.csv` alone; `experiments/counterfactual_reinsertion.py`
regenerates the raw results from corpus and checkpoints (`--ckpt-pattern`
names the per-fold checkpoint directories).

## Reading the output
Each run reports one fold; the paper fold-averages three with jointly
resampled significance tests, so run all three to reproduce ABSTRACT.md.
The LM pregame column is the card-only head, not a rollout; the halftime
column is raw sampling with the model driving its own substitutions
(`--lm-subs`), never consulting the real second-half rotations. Its
information is the public pregame state plus, at halftime, the observed
first half.

Interval coverage at nominal 50/80/90, uncorrected central-normal from
each game's rollout mean and spread, fold-averaged over 540 test games:
halftime margins 50.2/78.0/88.7 (totals 48.0/78.9/90.0), pregame
50.2/81.3/89.8 (totals 50.2/78.7/89.4); per-rollout margins are also
written for percentile checks. Simulated games play overtime under the
rules (tied at the horizon extends five minutes at a time, up to four);
actual margins and totals include real overtime.

MIT licensed — code and the `data/` files (our own outputs) alike.
