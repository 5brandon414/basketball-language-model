#!/usr/bin/env python3
"""Paper Table 2 — margin and total prediction: LM vs Elo, halftime-lead, GB trees.

  python3 experiments/table2_margins.py --data-dir <dataset> --splits data/fold3_splits.json --card player_card_fold3.parquet --fit-with-val --ckpt ckpt/best_model.pt
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from _common import (signed_make_points, load_card, load_game_meta,
                     boot_corr_diff)

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="../sloan_hf_dataset")
ap.add_argument("--splits", default=None,
                help="fold split JSON, e.g. data/fold3_splits.json; must "
                     "match the fold the checkpoint was trained on. Default "
                     "is the dataset's own splits.json")
ap.add_argument("--ckpt", default=None, help="checkpoint for the LM pregame column")
ap.add_argument("--lm-half-preds", default=None,
                help="parquet of halftime rollout predictions with columns "
                     "pred, actual (optionally pred_total, actual_total); "
                     "see experiments/README.md for how to build it")
ap.add_argument("--card", default=None,
                help="player card the trees baseline pools; use the fold card "
                     "matching --splits so the baseline sees the same as-of "
                     "normalization the model trained on")
ap.add_argument("--fit-with-val", action="store_true",
                help="fit every baseline on train+val, the full pre-cutoff "
                     "record, as the paper does")
a = ap.parse_args()
if a.card is None:
    if a.splits:
        raise SystemExit("--card is required with --splits: pass the fold card "
                         "(player_card_fold{1,2,3}.parquet) matching the fold; "
                         "player_card.parquet is normalized on a window that "
                         "overlaps the fold test eras.")
    a.card = "player_card.parquet"
print(f"trees-baseline card: {a.card}")

import loader as _ld0
_cfg0 = None
if a.ckpt:
    _cp0 = os.path.join(os.path.dirname(a.ckpt), "config.json")
    if not os.path.exists(_cp0):
        raise SystemExit(f"config.json not found beside {a.ckpt}")
    _cfg0 = json.load(open(_cp0))
corpus = _ld0.load_corpus(a.data_dir, a.splits,
                          fix_sub_boundary=bool(_cfg0 and _cfg0.get("fix_sub_boundary")))
vocab = corpus["vocab"]; games = corpus["games"]
print(f"corpus via lm24 loader (R=17 rosters), fix_sub_boundary={bool(_cfg0 and _cfg0.get('fix_sub_boundary'))}")
if a.fit_with_val:
    _n = sum(1 for g in games.values() if g["split"] == "val")
    for g in games.values():
        if g["split"] == "val":
            g["split"] = "train"
    print(f"fit-with-val: {_n} val games join train for baseline fits")
PH, PA = signed_make_points(vocab)
rate, P = load_card(a.data_dir, a.card); meta = load_game_meta(a.data_dir)

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

# ---- gradient-boosted trees: strongest fair configuration ----
# The trees get everything the LM's pregame head gets: the same pooled
# as-of cards, their difference, and prior-minutes availability summaries.
# The halftime trees additionally read a first-half box score (per-side
# event-class counts), i.e. the same first half the LM's rollouts condition
# on. Depth-2 with these features beat every deeper variant we tried; the
# audit that fixed this configuration is pre-registered in the dev log.
from _common import class_map as _cm
_cls, _CLS, _tok2cls, _NC = _cm(vocab)
_is_home = np.array([t.startswith("H_") for t in vocab])
_pl = pd.read_parquet(os.path.join(a.data_dir, "prior_minutes.parquet"))
_plut = {(str(r.player_id), str(r.game_id)): float(r.prior_min)
         for r in _pl.itertuples(index=False)}
def pool(ros, gid):
    Vv = np.array([rate(p, gid) for p in ros])
    if len(Vv) == 0: return np.zeros(P)
    w = Vv[:, 7].copy(); w[[str(p) == "0" for p in ros]] = -1e9; w = np.exp(w - w.max()); w /= w.sum()
    return (Vv * w[:, None]).sum(0)
def pri_feats(ros, gid):
    pri = np.array([_plut.get((str(p), str(gid)), 0.) for p in ros])
    top = np.sort(pri)[::-1]
    return np.array([pri.sum(), top[:5].sum(), top[:8].sum(), float((pri > 12).sum())])
Xpre, Xhalf, gids = [], [], []
for gid, g in games.items():
    ch, ca = pool(g["hro"], gid), pool(g["aro"], gid)
    card = np.concatenate([ch, ca, ch - ca, pri_feats(g["hro"], gid), pri_feats(g["aro"], gid)])
    tok = np.asarray(g["tok"]); k = int(np.searchsorted(np.asarray(g["clock"]), 0.5))
    t1 = tok[:k]
    bh = np.zeros(_NC); ba = np.zeros(_NC)
    np.add.at(bh, [_tok2cls[x] for x in t1[_is_home[t1]]], 1)
    np.add.at(ba, [_tok2cls[x] for x in t1[~_is_home[t1]]], 1)
    r = df[df.gid == gid].iloc[0]
    Xpre.append(card)
    Xhalf.append(np.concatenate([card, [r.half_margin, r.half_total], bh, ba, [float(k)]]))
    gids.append(gid)
Xpre, Xhalf = np.array(Xpre), np.array(Xhalf)
idx = df.set_index("gid"); sp = np.array([idx.loc[g, "split"] for g in gids])
ym = np.array([idx.loc[g, "margin"] for g in gids]); yt = np.array([idx.loc[g, "total"] for g in gids])
trm = sp == "train"
def gbfit(X, y): return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=2,
                                                      l2_regularization=1.0, random_state=0).fit(X[trm], y[trm]).predict(X)
gbmap = pd.DataFrame({"gid": gids, "gb_pre": gbfit(Xpre, ym), "gb_half": gbfit(Xhalf, ym),
                      "gb_tot_pre": gbfit(Xpre, yt), "gb_tot_half": gbfit(Xhalf, yt)})
df = df.merge(gbmap, on="gid", how="left")

# ---- LM pregame margin readout from --ckpt (LM-2.4 config-driven) ----
if a.ckpt:
    import train_bball_lm as tb
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfgp = os.path.join(os.path.dirname(a.ckpt), "config.json")
    if not os.path.exists(cfgp):
        raise SystemExit(f"config.json not found beside {a.ckpt}")
    _cfg = json.load(open(cfgp))
    sd = ck["model"]; rlut = ck.get("rlut") or {}
    port = _cfg["port"]
    tb.MAXLEN = sd["pos.weight"].shape[0]
    m = tb.BasketballLM(len(vocab), len(ck["pvocab"]), d=_cfg["d"], nlayers=_cfg["layers"],
                        port=port, budget_ch=_cfg["budget_ch"], pace_ch=_cfg["pace_ch"],
                        period_ch=_cfg["period_ch"], mtp=_cfg["mtp_w"] > 0, tt=_cfg["tt_w"] > 0)
    m.load_state_dict(sd, strict=True); m.eval()
    wpool = bool(_cfg.get("mh_wpool", False)); cardonly = bool(_cfg.get("mh_cardonly", False))
    print(f"LM pregame readout: mh_cardonly={cardonly} mh_wpool={wpool} port={port} "
          f"budget_ch={_cfg['budget_ch']} (from {cfgp})")
    plut = {}
    if _cfg["budget_ch"]:
        _pl = pd.read_parquet(os.path.join(a.data_dir, "prior_minutes.parquet"))
        plut = {(str(r.player_id), str(r.game_id)): float(r.prior_min)
                for r in _pl.itertuples(index=False)}
    def lut(pid, gid):
        return (tuple(rlut.get((str(pid), str(gid))) or ()) + (0.,) * port)[:port]
    lm_pre = {}
    with torch.no_grad():
        for gid, g in games.items():
            gv = []
            for ros in (g["hro"], g["aro"]):
                pi = torch.tensor([ck["pvocab"].get(str(p), 0) for p in ros])
                rt = torch.tensor([lut(p, gid) for p in ros], dtype=torch.float32)
                pe = torch.zeros(len(pi), 32) if cardonly else m.pemb(pi)
                xs = [pe, torch.full((len(pi), 1), .5), rt]
                if _cfg["budget_ch"]:
                    pri = torch.tensor([plut.get((str(p), str(gid)), 0.) / 48. for p in ros],
                                       dtype=torch.float32)
                    xs += [pri.unsqueeze(-1), torch.zeros(len(pi), 1)]
                f = m.pfuse(torch.cat(xs, -1))
                if wpool:
                    wv = torch.softmax(torch.where(pi > 0, rt[:, 7],
                                                   torch.full_like(rt[:, 7], -1e9)), 0)
                    gv.append((f * wv.unsqueeze(-1)).sum(0))
                else:
                    mk = (pi > 0).float().unsqueeze(-1)
                    gv.append((f * mk).sum(0) / mk.sum().clamp(min=1))
            lm_pre[gid] = float(m.mh(gv[0] - gv[1])) * 20
    df["lm_pre"] = df.gid.map(lm_pre)

# ---- report (frozen test) ----
t = df[df.split == "test"]
t.to_csv("fold_margins.csv", index=False)
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
    if "game_id" in h.columns:
        want, got = set(t.gid), set(h.game_id.astype(str))
        if got != want:
            raise SystemExit(
                f"--lm-half-preds covers {len(got & want)}/{len(want)} of this "
                "fold's test games; pass the file generated for this fold.")
    else:
        print("WARNING: --lm-half-preds has no game_id column; rows cannot be "
              "checked against this fold's test set.")
    print(line("LM (raw sampling)", h.pred, h.actual))
else:
    print("  LM halftime: pass --lm-half-preds (from generate_bball_lm.py --state half); raw sampling only in this release.")

print("\n=== TOTALS (frozen test) ===")
print(line("GB pregame total", t.gb_tot_pre, t.total), " <- near-unpredictable for any model")
print(line("GB halftime total", t.gb_tot_half, t.total))
if a.lm_half_preds and "pred_total" in pd.read_parquet(a.lm_half_preds).columns:
    h = pd.read_parquet(a.lm_half_preds); print(line("LM halftime total", h.pred_total, h.actual_total))
