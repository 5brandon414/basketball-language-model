#!/usr/bin/env python3
"""Shared helpers for the paper-table experiments (loaders, significance)."""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)                 # sloan_release/
sys.path.insert(0, REL)                     # so `import loader`, `import train_bball_lm` work
from loader import load_corpus              # noqa: E402


def signed_make_points(vocab):
    """PH>0 on home make tokens, PA>0 on away make tokens (points scored)."""
    PH = np.array([(3 if "3PT" in t else 1 if "FT_" in t else 2)
                   if ("MAKE" in t and t.startswith("H_")) else 0 for t in vocab])
    PA = np.array([(3 if "3PT" in t else 1 if "FT_" in t else 2)
                   if ("MAKE" in t and t.startswith("A_")) else 0 for t in vocab])
    return PH, PA


def class_map(vocab):
    """16 semantic event classes; returns (cls_fn, CLS, tok2cls, NC)."""
    def cls(t):
        n = vocab[t]
        if "MAKE" in n: return "3PTmk" if "3PT" in n else ("FTmk" if "FT_" in n else "2PTmk")
        if "MISS" in n: return "3PTms" if "3PT" in n else ("FTms" if "FT_" in n else "2PTms")
        for k in ("DREB", "OREB", "TOV", "STEAL", "BLOCK", "FOUL", "SUB", "TIMEOUT", "PERIOD"):
            if k in n: return k
        return "other"
    CLS = sorted(set(cls(t) for t in range(len(vocab))))
    CI = {c: i for i, c in enumerate(CLS)}
    tok2cls = np.array([CI[cls(t)] for t in range(len(vocab))])
    return cls, CLS, tok2cls, len(CLS)


def to_class_dist(p, tok2cls, NC):
    o = np.zeros(NC); np.add.at(o, tok2cls, p); return o


def load_card(data_dir, card="player_card.parquet"):
    """As-of player card, per-(pid,game) rows only; a miss returns zeros (the
    z-scored league mean). Returns (rate(pid, gid) -> dials, n_dials)."""
    pc = pd.read_parquet(os.path.join(data_dir, card))
    dcols = [c for c in pc.columns if c.startswith("card_")]
    lut = {(str(r.player_id), str(r.key)): np.array([getattr(r, c) for c in dcols])
           for r in pc.itertuples(index=False)}
    P = len(dcols)
    def rate(pid, gid):
        v = lut.get((str(pid), str(gid)))
        return v if v is not None else np.zeros(P)
    return rate, P


def load_game_meta(data_dir):
    return pd.read_parquet(os.path.join(data_dir, "game_meta.parquet")).set_index("game_id")


# ---- significance ----
def boot_corr_diff(a, b, y, n=4000, seed=0):
    """Paired bootstrap over games on corr(a,y) - corr(b,y).
    Returns (mean diff, 2.5%, 97.5%, whether the CI excludes zero)."""
    rng = np.random.default_rng(seed); N = len(y); d = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        d.append(np.corrcoef(a[i], y[i])[0, 1] - np.corrcoef(b[i], y[i])[0, 1])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi), bool(lo > 0 or hi < 0)
