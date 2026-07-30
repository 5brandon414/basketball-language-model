#!/usr/bin/env python3
"""Reconstruct the training corpus from the public dataset files.
All scripts consume data via load_corpus(data_dir)."""
import glob, json, os
import numpy as np
import pandas as pd


def load_corpus(data_dir):
    V = json.load(open(os.path.join(data_dir, "vocab.json")))
    VIDX = {t: i for i, t in enumerate(V)}
    splits = json.load(open(os.path.join(data_dir, "splits.json")))
    lu = pd.read_parquet(os.path.join(data_dir, "lineups.parquet"))
    lu_by_game = {g: d.sort_values("t_start_sec") for g, d in lu.groupby("game_id")}
    # pregame available rosters (dressed lists) — the model may know who is
    # available and who starts, never who actually enters or their minutes
    ros = pd.read_parquet(os.path.join(data_dir, "rosters.parquet"))
    avail = {r.game_id: (sorted(str(p) for p in r.home_available),
                         sorted(str(p) for p in r.away_available))
             for r in ros.itertuples(index=False)}
    games = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "events_*.parquet"))):
        for gid, g in pd.read_parquet(f).groupby("game_id"):
            st = lu_by_game.get(gid)
            if st is None or gid not in splits:
                continue
            g = g.sort_values("idx")
            ts = st.t_start_sec.to_numpy(float)
            dur = st.duration_sec.to_numpy(float)
            HL = [np.array([str(p) for p in x]) for x in st.home_lineup]
            AL = [np.array([str(p) for p in x]) for x in st.away_lineup]
            el = g.clock.to_numpy(float) * 2880.0
            idx = np.clip(np.searchsorted(ts, el, side="right") - 1, 0, len(ts) - 1)
            R = 17
            hro, aro = avail[gid]
            hro, aro = list(hro)[:R], list(aro)[:R]
            base, stint_base = {}, []
            for i in range(len(ts)):
                stint_base.append({p: base.get(p, 0.0)
                                   for p in np.concatenate([HL[i], AL[i]])})
                for p in np.concatenate([HL[i], AL[i]]):
                    base[p] = base.get(p, 0.0) + dur[i]
            n = len(g)
            hmin = np.zeros((n, 5), np.float32); amin = np.zeros((n, 5), np.float32)
            hm13 = np.zeros((n, R), np.float32); am13 = np.zeros((n, R), np.float32)
            for j in range(n):
                i = idx[j]
                on = max(min(el[j] - ts[i], dur[i]), 0.0)
                for k, p in enumerate(HL[i]):
                    hmin[j, k] = (stint_base[i].get(p, 0.0) + on) / 2880.0
                for k, p in enumerate(AL[i]):
                    amin[j, k] = (stint_base[i].get(p, 0.0) + on) / 2880.0
                for k, p in enumerate(hro):
                    hm13[j, k] = (stint_base[i].get(p, 0.0)
                                  + (on if p in HL[i] else 0.0)) / 2880.0
                for k, p in enumerate(aro):
                    am13[j, k] = (stint_base[i].get(p, 0.0)
                                  + (on if p in AL[i] else 0.0)) / 2880.0
            games[gid] = {
                "tok": np.array([VIDX[t] for t in g.token], np.int16),
                "h5": np.stack([HL[i] for i in idx]),
                "a5": np.stack([AL[i] for i in idx]),
                "clock": g.clock.to_numpy(np.float32),
                "sdiff": g.sdiff.to_numpy(np.float32),
                "actor": g.actor.to_numpy(np.int64),
                "entrant": g.entrant.to_numpy(np.int64),
                "assister": g.assister.to_numpy(np.int64),
                "hro": np.array(hro), "aro": np.array(aro),
                "hmin": hmin, "amin": amin, "hm13": hm13, "am13": am13,
                "split": splits[gid],
            }
    return {"games": games, "vocab": V}


def load_lineup_timeline(data_dir, split=None):
    """Per-game stint timeline for generation (t_start, duration, fives)."""
    lu = pd.read_parquet(os.path.join(data_dir, "lineups.parquet"))
    splits = json.load(open(os.path.join(data_dir, "splits.json")))
    lu["split"] = lu.game_id.map(splits)
    if split:
        lu = lu[lu.split == split]
    return lu
