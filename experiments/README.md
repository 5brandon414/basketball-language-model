# experiments/ — baseline comparisons (paper Tables 1-2)

Both scripts read a corpus you built (see ../DATA.md) through
`loader.load_corpus`, with the fold assignment supplied by `--splits`.
`table2_margins.py` also reads `game_meta.parquet` (for Elo) and
the player card named by `--card` (for the trees) from the same directory. Requires
scikit-learn.

## Scripts

| script | paper table | baselines (refit here) | LM column |
|---|---|---|---|
| `table1_nextevent.py` | Table 1 (next-event) | historical average (unigram), bigram, gradient-boosted trees | forward pass from `--ckpt` |
| `table2_margins.py` | Table 2 (margins) + totals | point-spread Elo, halftime-lead regression, gradient-boosted trees | pregame from `--ckpt`; halftime from `--lm-half-preds` |
| `_common.py` | — | shared loaders, class map, bootstrap | — |

## Run

From the repo root, one fold at a time; pass the fold whose checkpoint you
are scoring. `--fit-with-val` is the paper's fairness rule, giving every
baseline the full pre-cutoff record.

Baselines only (no checkpoint needed):

```
python3 experiments/table1_nextevent.py --data-dir <dataset> \
    --splits data/fold3_splits.json --fit-with-val
python3 experiments/table2_margins.py  --data-dir <dataset> \
    --splits data/fold3_splits.json --card player_card_fold3.parquet --fit-with-val
```

Add the LM column with `--ckpt ckpt/best_model.pt`. Train that checkpoint
with the full recipe in ../README.md; the bare defaults train a different
model.

The halftime-LM row of Table 2 comes from rollouts rather than `--ckpt`:
run `generate_bball_lm.py --state half --lm-subs`, convert the CSV it
writes beside the checkpoint as ../README.md describes, and pass the
result as `--lm-half-preds`.

## Output

- **Table 2**: correlation, winner accuracy and MAE over the fold's test
  games (n is printed per row). With `--ckpt`, the pregame LM line is
  followed by paired-bootstrap 95% CIs on its correlation difference
  against Elo and against the trees.
- **Table 1**: top-1 accuracy over the 16 semantic event classes, overall
  and stratified by transition difficulty — the entropy of
  P(class | prior event), which separates near-forced transitions (rebound
  after a miss) from genuine decisions — as a median split and as terciles.
- **Totals**: winner accuracy is meaningless (a total's sign is always
  positive); read correlation and MAE.

Each run reports one fold. The paper averages three and resamples them
jointly, so a single fold's printed interval is narrower in scope than the
published one.

## Notes

- The pregame LM column is the card-only head, not a rollout. If the
  checkpoint's `config.json` records `mh_cardonly`, `table2_margins.py`
  zeroes the same embedding slice the model was trained with, so training
  and evaluation agree.
- The halftime LM column is raw rollout output; nothing in these scripts
  re-weights or corrects it.
