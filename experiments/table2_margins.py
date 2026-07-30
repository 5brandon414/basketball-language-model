#!/usr/bin/env python3
"""Paper Table 1 — margin prediction: LM vs Elo, halftime-lead, and GB.

  python3 experiments/table2_margins.py --data-dir <dataset> --ckpt <best_model.pt>
"""
import argparse, json, os
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from _common import (load_corpus, signed_make_points, load_card, load_game_meta,
                     boot_corr_diff)

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="../sloan_hf_dataset")
ap.add_argument("--ckpt", default=None, help="checkpoint for the LM pregame column")
ap.add_argument("--lm-half-preds", default=None, help="parquet(game_id,pred,actual) from generate --state half")
a = ap.parse_args()

corpus = load_corpus(a.data_dir); vocab = corpus["vocab"]; games = corpus["games"]
PH, PA = signed_make_points(vocab)
rate, P = load_card(a.data_dir); meta = load_game_meta(a.data_dir)

# ---- per-game finals / halftime state ----
rows = []
for gid, g in games.items():
    tok = np.asarray(g["tok"]); ck = np.asarray(g["clock"]); k = int(np.searchsorted(ck, 0.5))
    hp, apn = int(PH[tok].sum()), int(PA[tok].sum())
    hm = int(PH[tok[:k]].sum() - PA[tok[:k]].sum()); ht = int(PH[tok[:k]].sum() + PA[tok[:k]].sum())
    rows.append((gid, g["split"], hp - apn, hp + apn, hm, ht))
df = pd.DataFrame(rows, columns=["gid", "split", "margin", "total", "half_margin", "half_total"])
df = df.join(meta, on="gid")

# ---- point-spread Elo ----
elo = df.dropna(subset=["home_team", "away_team", "game_date"]).sort_values(["game_date", "gid"]).copy()
R = {}; cur = None; K, HFA = 20.0, 100.0; pred = []
for r in elo.itertuples(index=False):
    s = r.gid[3:5]
    if s != cur: R = {t: 1500 + 0.75 * (v - 1500) for t, v in R.items()}; cur = s
    rh, ra = R.get(r.home_team, 1500.), R.get(r.away_team, 1500.); gd = rh - ra + HFA
    Eh = 1 / (1 + 10 ** (-gd / 400)); Sh = 1. if r.margin > 0 else (0.5 if r.margin == 0 else 0.)
    mov = np.log(abs(r.margin) + 1) * (2.2 / ((gd if Sh > 0.5 else -gd) * 0.001 + 2.2))
    R[r.home_team] = rh + K * mov * (Sh - Eh); R[r.away_team] = ra - K * mov * (Sh - Eh); pred.append(gd)
elo["ediff"] = pred
b1, b0 = np.polyfit(elo[elo.split == "train"].ediff, elo[elo.split == "train"].margin, 1)
elo["elo"] = b0 + b1 * elo.ediff
df = df.merge(elo[["gid", "elo"]], on="gid", how="left")

# ---- halftime-lead regression ----
tr = df[df.split == "train"]
hb1, hb0 = np.polyfit(tr.half_margin, tr.margin, 1)
df["lead"] = hb0 + hb1 * df.half_margin

# ---- gradient-boosted trees (pooled as-of card; +halftime state for halftime) ----
def pool(ros, gid):
    Vv = np.array([rate(p, gid) for p in ros])
    if len(Vv) == 0: return np.zeros(P)
    w = Vv[:, 7].copy(); w[[str(p) == "0" for p in ros]] = -1e9; w = np.exp(w - w.max()); w /= w.sum()
    return (Vv * w[:, None]).sum(0)
Xpre, Xhalf, gids = [], [], []
for gid, g in games.items():
    card = np.concatenate([pool(g["hro"], gid), pool(g["aro"], gid)])
    r = df[df.gid == gid].iloc[0]
    Xpre.append(card); Xhalf.append(np.concatenate([card, [r.half_margin, r.half_total]])); gids.append(gid)
Xpre, Xhalf = np.array(Xpre), np.array(Xhalf)
idx = df.set_index("gid"); sp = np.array([idx.loc[g, "split"] for g in gids])
ym = np.array([idx.loc[g, "margin"] for g in gids]); yt = np.array([idx.loc[g, "total"] for g in gids])
trm = sp == "train"
def gbfit(X, y): return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=4,
                                                      l2_regularization=1.0, random_state=0).fit(X[trm], y[trm]).predict(X)
gbmap = pd.DataFrame({"gid": gids, "gb_pre": gbfit(Xpre, ym), "gb_half": gbfit(Xhalf, ym),
                      "gb_tot_pre": gbfit(Xpre, yt), "gb_tot_half": gbfit(Xhalf, yt)})
df = df.merge(gbmap, on="gid", how="left")

# ---- LM pregame margin readout from --ckpt ----
if a.ckpt:
    import train_bball_lm as tb
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]; port = sd["pfuse.weight"].shape[1] - 33; rlut = ck.get("rlut") or {}
    d = sd["tok.weight"].shape[1]; nl = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("tr.layers."))
    tb.MAXLEN = sd["pos.weight"].shape[0]
    m = tb.BasketballLM(len(vocab), len(ck["pvocab"]), d=d, nlayers=nl, port=port)
    m.load_state_dict(sd, strict=False); m.eval()
    cfg = os.path.join(os.path.dirname(a.ckpt), "config.json")
    wpool = bool(json.load(open(cfg)).get("mh_wpool", False)) if os.path.exists(cfg) else False
    def lut(pid, gid):
        s = str(gid)[3:5]
        r = rlut.get((str(pid), str(gid))) or rlut.get((str(pid), s)) or rlut.get((str(pid), "*")) or ()
        return (tuple(r) + (0.,) * port)[:port]
    lm_pre = {}
    with torch.no_grad():
        for gid, g in games.items():
            gv = []
            for ros in (g["hro"], g["aro"]):
                pi = torch.tensor([ck["pvocab"].get(str(p), 0) for p in ros])
                rt = torch.tensor([lut(p, gid) for p in ros], dtype=torch.float32)
                f = m.pfuse(torch.cat([m.pemb(pi), torch.full((len(pi), 1), .5), rt], -1))
                if wpool:
                    wv = torch.softmax(torch.where(pi > 0, rt[:, 7], torch.full_like(rt[:, 7], -1e9)), 0)
                    gv.append((f * wv.unsqueeze(-1)).sum(0))
                else:
                    mk = (pi > 0).float().unsqueeze(-1); gv.append((f * mk).sum(0) / mk.sum().clamp(min=1))
            lm_pre[gid] = float(m.mh(gv[0] - gv[1])) * 20
    df["lm_pre"] = df.gid.map(lm_pre)

# ---- report (frozen test) ----
t = df[df.split == "test"]
def line(nm, p, y):
    p, y = np.asarray(p, float), np.asarray(y, float); ok = ~np.isnan(p)
    p, y = p[ok], y[ok]
    return f"  {nm:16s} corr {np.corrcoef(p,y)[0,1]:.3f} | acc {(np.sign(p)==np.sign(y)).mean():.1%} | MAE {np.abs(p-y).mean():.2f}  (n={len(p)})"
print("=== TABLE 2: PREGAME MARGIN (frozen test) ===")
print(line("Elo", t.elo, t.margin)); print(line("GB trees", t.gb_pre, t.margin))
if a.ckpt:
    print(line("LM", t.lm_pre, t.margin))
    j = t.dropna(subset=["lm_pre", "elo"])
    _, lo, hi, sig = boot_corr_diff(j.lm_pre.values, j.elo.values, j.margin.values)
    print(f"  LM-Elo corr diff 95%CI[{lo:+.3f},{hi:+.3f}] {'SIG' if sig else 'n.s.'}")
    j = t.dropna(subset=["lm_pre", "gb_pre"])
    _, lo, hi, sig = boot_corr_diff(j.lm_pre.values, j.gb_pre.values, j.margin.values)
    print(f"  LM-GB  corr diff 95%CI[{lo:+.3f},{hi:+.3f}] {'SIG' if sig else 'n.s.'}")

print("\n=== TABLE 2: HALFTIME MARGIN (frozen test) ===")
print(line("halftime-lead", t.lead, t.margin)); print(line("GB trees", t.gb_half, t.margin))
if a.lm_half_preds:
    h = pd.read_parquet(a.lm_half_preds)
    print(line("LM (raw sampling)", h.pred, h.actual))
else:
    print("  LM halftime: pass --lm-half-preds (from generate_bball_lm.py --state half); raw sampling only in this release.")

print("\n=== TOTALS (frozen test) ===")
print(line("GB pregame total", t.gb_tot_pre, t.total), " <- near-unpredictable for any model")
print(line("GB halftime total", t.gb_tot_half, t.total))
if a.lm_half_preds and "pred_total" in pd.read_parquet(a.lm_half_preds).columns:
    h = pd.read_parquet(a.lm_half_preds); print(line("LM halftime total", h.pred_total, h.actual_total))
