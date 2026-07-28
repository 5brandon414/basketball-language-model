#!/usr/bin/env python3
"""Pregame margin-head readout on held-out games (corr / winner acc / MAE).

  python3 eval_margin_head.py --ckpt <best_model.pt> --data-dir <dataset>
"""
import argparse
from loader import load_corpus
import numpy as np
import torch

import train_bball_lm as tb

DATA = "data"


def lut_rate(rlut, port, pid, season, gid):
    r = (rlut.get((pid, gid)) or rlut.get((pid, season))
         or rlut.get((pid, "*")) or ())
    return (tuple(r) + (0.0,) * port)[:port]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="../sloan_hf_dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--games", type=int, default=0, help="0 = all in split")
    ap.add_argument("--save", default=None, help="parquet path for per-game preds")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    vocab, pvocab = ck["vocab"], ck["pvocab"]
    rlut = ck.get("rlut") or {}
    sd = ck["model"]
    port = sd["pfuse.weight"].shape[1] - 33
    d = sd["tok.weight"].shape[1]
    nlayers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("tr.layers."))
    tb.MAXLEN = sd["pos.weight"].shape[0]
    model = tb.BasketballLM(len(vocab), len(pvocab), d=d, nlayers=nlayers, port=port)
    missing, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    import json, os
    cfgp = os.path.join(os.path.dirname(a.ckpt), "config.json")
    wpool = False
    if os.path.exists(cfgp):
        wpool = bool(json.load(open(cfgp)).get("mh_wpool", False))
    print(f"ckpt: d={d} layers={nlayers} port={port} vocab={len(vocab)} "
          f"| wpool={wpool} | missing keys: {missing or 'none'}")

    PTSD = np.zeros(len(vocab))
    for i, nm in enumerate(vocab):
        if "MAKE" in nm:
            p = 3 if "3PT" in nm else (1 if "FT" in nm else 2)
            PTSD[i] = p if nm.startswith("H") else -p

    blob = load_corpus(a.data_dir)
    games = [(gid, g) for gid, g in blob["games"].items() if g["split"] == a.split]
    if a.games:
        games = games[: a.games]
    print(f"{len(games)} {a.split} games")

    preds, actuals, gids = [], [], []
    with torch.no_grad():
        for gid, g in games:
            season = str(gid)[3:5]
            gv_sides = []
            for ro_ids in (g["hro"], g["aro"]):
                idx = torch.tensor([pvocab.get(str(p), 0) for p in ro_ids])
                rat = torch.tensor([lut_rate(rlut, port, str(p), season, str(gid))
                                    for p in ro_ids], dtype=torch.float32)
                f = model.pfuse(torch.cat(
                    [model.pemb(idx), torch.full((len(idx), 1), 0.5), rat], -1))
                if wpool:
                    wv = torch.softmax(torch.where(
                        idx > 0, rat[:, 7], torch.full_like(rat[:, 7], -1e9)), 0)
                    gv_sides.append((f * wv.unsqueeze(-1)).sum(0))
                else:
                    mask = (idx > 0).float().unsqueeze(-1)
                    gv_sides.append((f * mask).sum(0) / mask.sum().clamp(min=1))
            pred = float(model.mh(gv_sides[0] - gv_sides[1])) * 20.0
            actual = float(PTSD[g["tok"]].sum())
            preds.append(pred); actuals.append(actual); gids.append(str(gid))

    pr, ac = np.array(preds), np.array(actuals)
    corr = float(np.corrcoef(pr, ac)[0, 1])
    acc = float((np.sign(pr) == np.sign(ac)).mean())
    mae = float(np.abs(pr - ac).mean())
    bias = float((pr - ac).mean())
    print(f"\nMARGIN HEAD [{a.split}, n={len(pr)}]: "
          f"corr {corr:.3f} | winner acc {acc:.1%} | MAE {mae:.2f} | "
          f"bias {bias:+.2f}")

    if a.save:
        import pandas as pd
        pd.DataFrame({"game_id": gids, "pred_margin": pr,
                      "actual_margin": ac}).to_parquet(a.save)
        print(f"per-game preds -> {a.save}")


if __name__ == "__main__":
    main()
