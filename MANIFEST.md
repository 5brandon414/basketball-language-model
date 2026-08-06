# MANIFEST — sloan_release

## Code
Nothing here touches a database or the network; the corpus is read from
`--data-dir`.

- `loader.py` — the only data-access layer: reconstructs tokens, per-token
  lineups, available rosters and cumulative minutes from the dataset
  files. `--splits` selects a fold assignment.
- `train_bball_lm.py` — model and training loop.
- `generate_bball_lm.py` — batched Monte-Carlo rollouts, pregame or from a
  real first half. Raw sampling only.
- `print_bball_game.py` — render a game's token stream as readable
  play-by-play.
- `experiments/table1_nextevent.py` — paper Table 1: next-event, against
  historical-average, bigram and gradient-boosted-tree baselines.
- `experiments/table2_margins.py` — paper Table 2: margins and totals,
  against Elo, halftime-lead regression and gradient-boosted trees.
- `experiments/_common.py` — shared loaders, class map, significance
  tests. `experiments/README.md` — how to run both tables.

## Data published here
Game-id lists only; no play-by-play content.

- `data/fold{1,2,3}_splits.json` — the walk-forward assignment behind
  every number in the paper (train / val / test / exclude per game).
- `data/fold{1,2,3}_test_gids.txt` — each fold's 180 test games.

## Data you build
The corpus and its sidecars — events, lineups, rosters, the player cards,
`game_meta.parquet`, `vocab.json`, `players.csv` — come from public
NBA.com endpoints and are never redistributed here. DATA.md specifies
every file and column. Trained weights are not published with
this repository either.

## Docs
`README.md` (start here), `DATA.md` (corpus build specification),
`ABSTRACT.md` (the submission), `REFERENCES.md` (related work), `LICENSE`
(MIT), `requirements.txt`.
