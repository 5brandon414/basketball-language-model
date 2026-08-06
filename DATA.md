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
| entrant | roster index (0-16, into the sorted available roster) of the incoming player on SUB rows, -100 n/a |
| assister | on-court slot of the assisting teammate on _MAKE_AST rows, -100 n/a |

## Model clock bins (dt discretization used in training)
Bin edges (seconds): [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 24, 28, 33];
de-binned at generation with empirical per-bin medians fit on train.

## Information contract (what the model may know)
Pregame the model knows (1) the set of players AVAILABLE to play (the
dressed active list, fixed before tip), (2) the starters, and (3) player
statistics computed strictly before that game. It never observes who
actually enters off the bench or anyone's minutes — those are generated.
The available roster is outcome-independent (its size is uncorrelated with
the final margin, r = -0.01, versus r = 0.58 for the appeared roster).

## Conditioning fields (reconstructed by loader.py)
Per token: the ten on-court player ids (5 home + 5 away, from
lineups.parquet by elapsed time), each player's cumulative minutes so far,
the available roster per side (up to 17 ids, sorted; from rosters.parquet),
roster cumulative minutes, and the score/clock scalars above.

## Other files
- `lineups.parquet`: game_id, t_start_sec, duration_sec, home_lineup,
  away_lineup (arrays of player-id strings) — lineup-stint timeline.
- `rosters.parquet`: game_id, home_available, away_available — the pregame
  active (dressed) lists: everyone who played plus everyone in uniform who
  did not, excluding inactive/did-not-dress/not-with-team designations.
  Reconstructed per game from the league's own inactive and did-not-play
  designations; a game whose box record is empty is never silently reduced
  to the players who appeared, because that list depends on the outcome.
- `player_card_fold{1,2,3}.parquet`: the knowledge card, per (player_id,
  game_id) rows only; 49 z-scored dials covering scoring, playmaking,
  rebounding, shooting, size, minutes, experience, recent form,
  availability, defense, ball security and impact, plus shot diet,
  assisted share, foul and free-throw rates, substitution habits, age,
  prior-season play-type shares and rim protection. Each dial is the
  player's state strictly before that game night, and each fold's file is
  built and z-scored on that fold's pre-cutoff games alone, so a card for
  a test game never reflects the fold's test era. A missing row means no
  prior NBA appearances; consumers use zeros (the z-scored league mean).
- `player_card.parquet`: the 15-dial card of the earlier single-split
  release, kept for compatibility; no paper number uses it.
- `players.csv`: player_id -> display name.
- `splits.json`: the earlier single-split assignment (train / val / test /
  ttest). The paper's numbers come from the walk-forward fold files in the
  code repository's `data/` directory; pass one with `--splits`.

License: MIT (code); data derives from publicly available NBA.com endpoints.
