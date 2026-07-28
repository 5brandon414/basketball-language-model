# MANIFEST — sloan_release

## Code (all consume the public dataset via --data-dir; no database anywhere)
- `loader.py` — reconstructs the full training corpus (tokens, per-token lineups, rosters, cumulative minutes) from the dataset files. The only data-access layer.
- `train_bball_lm.py` — model + training loop (event / clock / actor / entrant / assister / rate / margin / win-prob heads). Seeded via --seed.
- `generate_bball_lm.py` — batched Monte-Carlo rollouts (pregame or from a real first half), raw sampling only. Seeded via --seed.
- `eval_margin_head.py` — pregame readout metrics (corr / winner acc / MAE) on held-out splits.
- `d_lite_calibrate.py` — variance-shrink fit + coverage report (script only; no fitted constants ship).
- `print_bball_game.py` — render any game's token stream as readable play-by-play (names via players.csv).

## Experiments (paper Tables 1-2 reproduction)
- `experiments/table1_margins.py` — margin/total: Elo, halftime-lead, GB, LM.
- `experiments/table2_nextevent.py` — next-event: unigram, bigram, GB, LM (class-level, difficulty-stratified).
- `experiments/_common.py` — shared loaders + significance. `experiments/README.md` — how to run.

## Data (small, git-appropriate)
- `data/splits.json`, `data/games_{train,val,test,ttest}.txt` — frozen split assignment.
- `game_meta.parquet` (in the dataset) — game_id, team abbreviations, date (public; used by the Elo baseline).

## Docs
- `README.md` — project overview, dataset link placeholder, reproduction steps.
- `DATA.md` — standalone token-grammar and schema specification.
- `LICENSE` (MIT), `requirements.txt` (pinned, DB-free), this file.

## Excluded-but-referenced check
None — no included file imports or references any excluded module
(extraction/DB tooling, steering/pricing code, checkpoints, fitted
constants). Verified by grep sweep; see release report.
