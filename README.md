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

## Reproduce (seeds pinned; defaults shown)
```
python3 train_bball_lm.py --epochs 40 --maxlen 688 --card player_card.parquet \
    --ast-w 0.3 --sr-w 0.3 --mh-w 1.5 --pw-w 0.3 --mh-wpool --sd-drop 0.3 \
    --seed 7 --data-dir <dataset> --out ckpt/
python3 generate_bball_lm.py --games 500 --rollouts 24 --split test \
    --state half --lm-subs --kv --seed 7 --data-dir <dataset> \
    --ckpt ckpt/best_model.pt        # the paper's halftime configuration
python3 eval_margin_head.py --ckpt ckpt/best_model.pt --data-dir <dataset>
# optional variance-shrink tooling (paper coverage is raw); needs pregame
# rollouts on val+test first (--state pre --split val / test):
python3 d_lite_calibrate.py ckpt/
```

## Expected metrics (frozen 500-game test split)
| Metric | Value |
|---|---|
| Next-event perplexity, 16 classes (LM / bigram) | 4.01 / 5.44 |
| Pregame margin corr / winner acc / MAE | 0.509 / 65.0% / 10.54 |
| Halftime margin corr / winner acc / MAE | 0.690 / 74.6% / 8.84 |
| Game totals corr (halftime) | 0.707 |
| Calibration coverage (raw sampling) | margins ±3pp, totals ±7pp |

Halftime figures use raw sampling with the model driving its own
substitutions (`--lm-subs`) — no steering, and the real second-half
rotations are never consulted. The model's information is the public
pregame state (available roster, starters, strictly-before statistics)
plus, at halftime, the observed first half.

MIT licensed. File-by-file provenance in MANIFEST.md.
