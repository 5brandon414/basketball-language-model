#!/usr/bin/env python3
"""Render a tokenized game as readable play-by-play.

  python3 print_bball_game.py --data-dir <dataset> [--gid <game_id>] [--split test]
"""
import argparse
from loader import load_corpus
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", default=None,
                    help="game_id (default: first game in --split)")
    ap.add_argument("--data-dir", default="../sloan_hf_dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0, help="max events (0 = all)")
    a = ap.parse_args()

    blob = load_corpus(a.data_dir)
    vocab = blob["vocab"]
    if a.gid:
        gid, g = a.gid, blob["games"][a.gid]
    else:
        gid, g = next((k, v) for k, v in blob["games"].items()
                      if v["split"] == a.split)

    import pandas as pd
    names = {}
    try:
        npdf = pd.read_csv(f"{a.data_dir}/players.csv")
        names = dict(zip(npdf.player_id.astype(str), npdf.player_name))
    except Exception:
        pass

    def nm(pid):
        return names.get(str(pid), str(pid))

    PTS = np.zeros(len(vocab), int)
    for i, t in enumerate(vocab):
        if "MAKE" in t:
            PTS[i] = 3 if "3PT" in t else (1 if "FT" in t else 2)

    hs = as_ = 0
    print(f"game {gid} | {len(g['tok'])} tokens | split {g['split']}\n")
    n_shown = 0
    for j, ti in enumerate(g["tok"]):
        t = vocab[ti]
        if t in ("PAD", "BOS"):
            continue
        el = g["clock"][j] * 2880.0
        q = min(int(el // 720) + 1, 5)
        rem = 720 - (el % 720) if el < 2880 else 300 - ((el - 2880) % 300)
        who = ""
        slot = int(g["actor"][j])
        if slot >= 0:
            five = g["h5"][j] if slot < 5 else g["a5"][j]
            who = nm(five[slot % 5])
        ast = ""
        asl = int(g.get("assister", np.full(len(g["tok"]), -100))[j])
        if asl >= 0:
            five = g["h5"][j] if asl < 5 else g["a5"][j]
            ast = f"  (ast: {nm(five[asl % 5])})"
        if t.startswith("H_") and PTS[ti]: hs += PTS[ti]
        if t.startswith("A_") and PTS[ti]: as_ += PTS[ti]
        score = f"{hs:3d}-{as_:<3d}"
        print(f"Q{q} {int(rem)//60:2d}:{int(rem)%60:02d}  {score}  "
              f"{t:<16} {who}{ast}")
        n_shown += 1
        if a.n and n_shown >= a.n:
            print(f"... ({len(g['tok']) - j - 1} more tokens)")
            break
    fh = sum(int(PTS[t]) for t in g["tok"] if vocab[t].startswith("H_"))
    fa = sum(int(PTS[t]) for t in g["tok"] if vocab[t].startswith("A_"))
    print(f"\nfinal (reconstructed): {fh}-{fa}")


if __name__ == "__main__":
    main()
