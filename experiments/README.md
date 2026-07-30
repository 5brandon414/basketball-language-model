# experiments/ — baseline comparisons (paper Tables 1–2)

Reproduces the model-vs-baseline comparisons in the paper, entirely from
the public dataset. All scripts read data through `loader.load_corpus`
plus the shipped sidecars (`player_card.parquet`, `game_meta.parquet`,
`vocab.json`, `splits.json`) — no database, no external services.

## Scripts

| script | paper table | baselines (computed fresh) | LM column |
|---|---|---|---|
| `table1_nextevent.py` | Table 1 (next-event) | historical-average (unigram), bigram, gradient-boosted trees | forward pass from `--ckpt` |
| `table2_margins.py` | Table 2 (margins) + totals | point-spread Elo, halftime-lead regression, gradient-boosted trees | pregame from `--ckpt`; halftime from `--lm-half-preds` |
| `_common.py` | — | shared loaders, class map, significance tests | — |

## Run

Baselines only (no model needed — reproduces every baseline column):

```
python3 experiments/table2_margins.py  --data-dir ../sloan_hf_dataset
python3 experiments/table1_nextevent.py --data-dir ../sloan_hf_dataset
```

With the LM column, first train a checkpoint from the released data using
the full paper recipe (see the top-level README for the exact command —
the bare defaults train a smaller-headed model), then pass it:

```
python3 experiments/table1_nextevent.py --data-dir ../sloan_hf_dataset --ckpt ckpt/best_model.pt
python3 experiments/table2_margins.py  --data-dir ../sloan_hf_dataset --ckpt ckpt/best_model.pt
```

The halftime-LM row of Table 2 needs a rollout run; produce predictions
with `generate_bball_lm.py --state half --lm-subs --kv` (the model drives
its own second-half rotations; `--kv` is an exact-math cache, ~10x
faster), then rename columns to `game_id,pred,actual` (optionally
`pred_total,actual_total`) and pass `--lm-half-preds`.

## Metrics

- **Table 2** margins/totals: correlation, winner accuracy, MAE on the
  frozen `test` split. Correlation-difference significance via a paired
  bootstrap 95% CI (`_common.boot_corr_diff`).
- **Table 1** next-event: scored at the 16 semantic event classes
  (top-1), overall and stratified by transition difficulty — the entropy
  of P(class | prior event), which separates near-forced transitions
  (rebound after a miss) from genuine decisions. Perplexity reported at
  the 16-class level.

## Notes / caveats

- **Elo** uses `game_meta.parquet` (game_id, team abbreviations, date) —
  shipped alongside the dataset; team/date are public.
- **Halftime LM is raw sampling** — the paper's numbers come from plain
  rollouts of this model with no steering, re-weighting, or post-hoc
  correction anywhere in the pipeline; the reported calibration coverage
  is the raw sampling distribution's.
- Totals report a trivial 100% "accuracy" (a total's sign is always
  positive); read correlation and MAE for totals.
- Requires `scikit-learn` (in `requirements.txt`).
