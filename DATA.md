# DATA.md — corpus build specification

The play-by-play is not redistributed here. This file specifies the corpus
the code expects, so it can be rebuilt from the public NBA.com play-by-play
and box-score endpoints; point every script at your build with `--data-dir`.

Scope: regular-season games of the 2016-17 through 2025-26 seasons. The
exact game ids are the keys of `data/fold{1,2,3}_splits.json`.

## Token grammar (60 tokens)

`vocab.json` is authoritative: a JSON array of the 60 token strings, where a
token's id is its index in the array.

Structural: `BOS` (row 0 of every game), `EOS` (last row), `PERIOD` (one row
at the start and one at the end of each period, so eight in a regulation
game), `PAD` (never appears in the data; id 0, used to pad batches).

Per side S in {H (home), A (away)}:
- Shots, by class x outcome: `S_<CLASS>_MISS`, `S_<CLASS>_MAKE`,
  `S_<CLASS>_MAKE_AST` (assisted make), for CLASS in:
  `RIM` (dunk/layup), `JS` (jumper <10ft), `JM` (10-17ft), `JL` (17+ 2pt),
  `3PTC` (corner three, <=23ft), `3PTA` (above-the-break three).
- Free throws: `S_FT_MAKE`, `S_FT_MISS`.
- Rebounds: `S_OREB`, `S_DREB`.
- `S_TOV`, `S_STEAL`, `S_BLOCK`, `S_FOUL`, `S_TIMEOUT`.
- `S_SUB` — one row per outgoing player at a lineup change.

Scores are derived from token strings alone (`3PT*_MAKE` = 3, `FT_MAKE` = 1,
any other `MAKE` = 2), so the stream must reproduce official final scores by
itself.

## `events_*.parquet` — the event stream

Any number of shards, globbed as `events_*.parquet`; all rows of one game
must sit in one shard (the loader groups per file and later files win).

| column | meaning |
|---|---|
| game_id | 10-char NBA game id |
| idx | event index within the game, 0-based, BOS row included |
| token | vocabulary string (above) |
| clock | elapsed game time / 2880 (OT capped at 3600s, so clock <= 1.25) |
| sdiff | running score diff (home-away) / 20, clipped to [-2, 2] |
| actor | on-court slot of the acting player: 0-4 home, 5-9 away; -100 if none |
| entrant | on `S_SUB` rows, the roster index of the incoming player; -100 elsewhere |
| assister | on `S_<CLASS>_MAKE_AST` rows, the on-court slot of the assister; -100 elsewhere, and on assisted makes whose assister does not resolve to a slot |

## `lineups.parquet` — stint timeline

`game_id`, `t_start_sec`, `duration_sec`, `home_lineup`, `away_lineup`
(five player-id strings each), `home_pts`, `away_pts` (points each side
scored during the stint; `generate_bball_lm.py` sums them for the game's
actual final score). Stints must be ordered and contiguous, each starting
where the previous ended: `loader.py` reads every event's five and every
player's cumulative minutes off this timeline.

## `rosters.parquet` — pregame available lists

`game_id`, `home_available`, `away_available`: the dressed active list per
side, everyone in uniform, excluding inactive / did-not-dress /
not-with-team designations. It must not be the list of players who
appeared — that set depends on the outcome. This is the whole pregame
roster contract: the model may know who is available and who starts, never
who actually enters or for how long.

`loader.py` sorts each list as strings and keeps the first 17; the
`entrant` column of `events_*.parquet` indexes into that sorted, truncated
list.

## `player_card_fold{1,2,3}.parquet`, `player_card.parquet` — knowledge card

Columns `player_id`, `key` (the game id the row is as-of), and `card_0` …
`card_{N-1}`: one row per (player, game) holding that player's z-scored
state strictly before that game. Each fold's file is built and normalized
on that fold's pre-cutoff games alone, so a card for a test game never
reflects the fold's test era. A missing (player, game) row feeds zeros, the
z-scored league mean.

N is read from the file, so any width works; `card_7` must be a
recent-minutes dial, because `--mh-wpool` and the trees baseline pool a
roster by softmax over it.

`--card` selects the training file (the paper uses
`player_card_fold{k}.parquet`, 49 dials). `experiments/_common.py` always
reads `player_card.parquet` (15 dials) for its trees baseline.

## `game_meta.parquet`

`game_id`, `home_team`, `away_team` (three-letter abbreviations),
`game_date` (`YYYY-MM-DD`). Used by the Elo baseline in
`experiments/table2_margins.py`.

## `players.csv`

`player_id`, `player_name`. Display only, for `print_bball_game.py`.

## `splits.json`

game_id -> split label. Games with no entry are skipped by the loader.
`<data-dir>/splits.json` is the default assignment; `--splits` overrides it
with the walk-forward fold files shipped in `data/`, which label every game
`train`, `val`, `test`, or `exclude`.

License: MIT (code); the data derives from publicly available NBA.com
endpoints and is not redistributed here.
