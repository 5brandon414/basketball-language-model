# experiments/ — baseline comparisons (paper Tables 1–2)

Reproduces the model-vs-baseline comparisons in the paper, entirely from
the public dataset. All scripts read data through `loader.load_corpus`
plus the shipped sidecars (`player_card.parquet`, `game_meta.parquet`,
`vocab.json`, `splits.json`) — no database, no external services.

## Scripts

| script | paper table | baselines (computed fresh) | LM column |
|---|---|---|---|
| `table1_margins.py` | Table 1 (margins) + totals | point-spread Elo, halftime-lead regression, gradient-boosted trees | pregame from `--ckpt`; halftime from `--lm-half-preds` |
| `table2_nextevent.py` | Table 2 (next-event) | historical-average (unigram), bigram, gradient-boosted trees | forward pass from `--ckpt` |
| `_common.py` | — | shared loaders, class map, significance tests | — |

## Run

Baselines only (no model needed — reproduces every baseline column):

```
python3 experiments/table1_margins.py  --data-dir ../sloan_hf_dataset
python3 experiments/table2_nextevent.py --data-dir ../sloan_hf_dataset
```

With the LM column, first train a checkpoint from the released data, then
pass it:

```
python3 train_bball_lm.py --data-dir ../sloan_hf_dataset --out ckpt   # trains the model
python3 experiments/table2_nextevent.py --data-dir ../sloan_hf_dataset --ckpt ckpt/best_model.pt
python3 experiments/table1_margins.py  --data-dir ../sloan_hf_dataset --ckpt ckpt/best_model.pt
```

The halftime-LM row of Table 1 needs a rollout run; produce predictions
with `generate_bball_lm.py --state half` (columns `game_id,pred,actual`,
optionally `pred_total,actual_total`) and pass `--lm-half-preds`.

## Metrics

- **Table 1** margins/totals: correlation, winner accuracy, MAE on the
  frozen `test` split. Correlation-difference significance via a paired
  bootstrap 95% CI (`_common.boot_corr_diff`).
- **Table 2** next-event: scored at the 16 semantic event classes
  (top-1), overall and stratified by transition difficulty — the entropy
  of P(class | prior event), which separates near-forced transitions
  (rebound after a miss) from genuine decisions. Perplexity reported at
  the 16-class level.

## Notes / caveats

- **Elo** uses `game_meta.parquet` (game_id, team abbreviations, date) —
  shipped alongside the dataset; team/date are public.
- **Halftime LM uses raw sampling** in this release (the paper's
  generation-time steering is not part of the open release). Measured on
  the full 699-game test set, raw sampling *reproduces* the steered
  halftime numbers within noise — margin corr 0.686 vs 0.684, totals corr
  0.719 vs 0.721 — with winner accuracy ~2 points lower (73.4% vs 75.1%).
  So the release reproduces the paper's halftime results; the steering was
  a pregame-totals device that adds ~nothing once the real first half is
  in hand.
- Totals report a trivial 100% "accuracy" (a total's sign is always
  positive); read correlation and MAE for totals.
- Requires `scikit-learn` (in `requirements.txt`).
