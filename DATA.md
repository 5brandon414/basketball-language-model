# DATA.md — corpus specification

## Provenance
All data derives from public NBA.com endpoints (play-by-play and box-score
APIs), regular seasons 2016-17 through 2025-26: 11,896 games, 5,793,945
event tokens. Extraction was validation-gated: reconstructed final scores
from the token stream must match official box-score finals (99.6% exact;
known edge cases documented). No proprietary feeds.

## Token grammar (60-token vocabulary; vocab.json is authoritative)
Structural: `PAD`, `BOS`, `EOS`, `PERIOD`.
Per side S in {H (home), A (away)}:
- Shots, by class x outcome: `S_<CLASS>_MISS`, `S_<CLASS>_MAKE`,
  `S_<CLASS>_MAKE_AST` (assisted make), for CLASS in:
  `RIM` (dunk/layup), `JS` (jumper <10ft), `JM` (10-17ft), `JL` (17+ 2pt),
  `3PTC` (corner three, <=23ft), `3PTA` (above-the-break three).
- Free throws: `S_FT_MAKE`, `S_FT_MISS`.
- Rebounds: `S_OREB`, `S_DREB`.
- `S_TOV`, `S_STEAL`, `S_BLOCK`, `S_FOUL`, `S_TIMEOUT`.
- `S_SUB` — one token per outgoing player at a lineup change.

## Per-event fields (events_XX.parquet)
| column | meaning |
|---|---|
| game_id | 10-char NBA game id |
| idx | event index within game (0-based, includes BOS row) |
| token | vocabulary string (above) |
| dt | seconds to the next event (float; last event 0) |
| clock | elapsed game time / 2880 (OT clipped at 3600s) |
| sdiff | running score diff (home-away)/20, clipped to [-2, 2] |
| actor | on-court slot of the acting player: 0-4 home, 5-9 away, -100 n/a |
| entrant | roster index (0-12) of the incoming player on SUB rows, -100 n/a |
| assister | on-court slot of the assisting teammate on _MAKE_AST rows, -100 n/a |

## Model clock bins (dt discretization used in training)
Bin edges (seconds): [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 24, 28, 33];
de-binned at generation with empirical per-bin medians fit on train.

## Conditioning fields (reconstructed by loader.py)
Per token: the ten on-court player ids (5 home + 5 away, from
lineups.parquet by elapsed time), each player's cumulative minutes so far,
13-man roster ids per side (first 13 distinct players by id order), roster
cumulative minutes, and the score/clock scalars above.

## Other files
- `lineups.parquet`: game_id, t_start_sec, duration_sec, home_lineup,
  away_lineup (arrays of player-id strings) — lineup-stint timeline.
- `player_card.parquet`: per (player_id, key) rows; key = game id (as-of
  values strictly before that game), season ("16".."25"), or "*" (career).
  15 z-scored dials (scoring, playmaking, rebounding, shooting, size,
  minutes, experience, recent form, availability, defense, ball security,
  impact). Lookup chain: game -> season -> career.
- `players.csv`: player_id -> display name.
- `splits.json` / games_*.txt: train / val / test (frozen 699) / ttest
  (temporal holdout: chronologically last 200 train-era games of 2025-26).

License: MIT (code); data derives from publicly available NBA.com endpoints.
