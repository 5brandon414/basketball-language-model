#!/usr/bin/env python3
"""Paper Table 1 — next-event prediction: LM vs historical average, bigram, GB trees.

  python3 experiments/table1_nextevent.py --data-dir <dataset> --splits data/fold3_splits.json --fit-with-val --ckpt ckpt/best_model.pt
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sklearn.ensemble import HistGradientBoostingClassifier
from _common import class_map, to_class_dist
from loader import load_corpus as load_corpus24

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="../sloan_hf_dataset")
ap.add_argument("--splits", default=None,
                help="fold split JSON, e.g. data/fold3_splits.json; must "
                     "match the fold the checkpoint was trained on. Default "
                     "is the dataset's own splits.json")
ap.add_argument("--ckpt", default=None, help="trained checkpoint; omit to run baselines only")
ap.add_argument("--gb-sample", type=int, default=2000000,
                help="cap on GB training rows; above it, a random subsample is taken")
ap.add_argument("--fit-with-val", action="store_true",
                help="fit every baseline on train+val, the full pre-cutoff "
                     "record, as the paper does")
a = ap.parse_args()

_cfg = None
if a.ckpt:
    cfgp = os.path.join(os.path.dirname(a.ckpt), "config.json")
    if not os.path.exists(cfgp):
        raise SystemExit(f"config.json not found beside {a.ckpt}")
    _cfg = json.load(open(cfgp))
corpus = load_corpus24(a.data_dir, a.splits,
                       fix_sub_boundary=bool(_cfg and _cfg.get("fix_sub_boundary")))
vocab = corpus["vocab"]; V = len(vocab)
games = corpus["games"]
for _gid, _g in games.items():
    _g["gid"] = str(_gid); _g["season"] = str(_gid)[3:5]
print(f"corpus via lm24 loader, fix_sub_boundary={bool(_cfg and _cfg.get('fix_sub_boundary'))}")
if a.fit_with_val:
    _n = sum(1 for g in games.values() if g["split"] == "val")
    for g in games.values():
        if g["split"] == "val":
            g["split"] = "train"
    print(f"fit-with-val: {_n} val games join train for baseline fits")
tr = [g for g in games.values() if g["split"] == "train"]
te = [g for g in games.values() if g["split"] == "test"]
cls, CLS, tok2cls, NC = class_map(vocab)
IS_MAKE = np.array(["MAKE" in t for t in vocab]); IS_MISS = np.array(["MISS" in t for t in vocab])
IS_FOUL = np.array(["FOUL" in t for t in vocab])

def feats(g):
    t = np.asarray(g["tok"]); cl = np.asarray(g["clock"]); sd = np.asarray(g["sdiff"])
    X = []; Y = []; PR = []; nm = nmi = nf = 0
    for i in range(1, len(t)):
        dt_prev = cl[i-1] - cl[i-2] if i >= 2 else 0.0
        X.append([t[i-1], t[i-2] if i >= 2 else V, t[i-3] if i >= 3 else V,
                  cl[i-1], sd[i-1], nm, nmi, nf, i,
                  t[i-4] if i >= 4 else V, t[i-5] if i >= 5 else V,
                  dt_prev, min(int(cl[i-1] * 4), 4)]); Y.append(t[i]); PR.append(t[i-1])
        nm += IS_MAKE[t[i-1]]; nmi += IS_MISS[t[i-1]]; nf += IS_FOUL[t[i-1]]
    return X, Y, PR

# ---- unigram + class-level bigram (+ transition entropy) from train ----
uni = np.ones(V); bigc = np.ones((V, NC))
for g in tr:
    t = np.asarray(g["tok"])
    for i in range(len(t)):
        uni[t[i]] += 1
        if i: bigc[t[i-1], tok2cls[t[i]]] += 1
uni_cd = to_class_dist(uni / uni.sum(), tok2cls, NC)
bigc = bigc / bigc.sum(1, keepdims=True)
Hbits = -(bigc * np.log2(bigc)).sum(1)

# ---- GB context classifier ----
Xtr, Ytr = [], []
for g in tr:
    x, y, _ = feats(g); Xtr += x; Ytr += y
Xtr = np.array(Xtr, float); Ytr = np.array(Ytr); rng = np.random.default_rng(0)
if len(Xtr) > a.gb_sample:
    idx = rng.choice(len(Xtr), a.gb_sample, replace=False); Xtr = Xtr[idx]; Ytr = Ytr[idx]
print(f"training GB on {len(Xtr):,} rows...", flush=True)
# explicit patient early stopping: sklearn's 'auto' setting quietly stopped
# the previous config at 21 iterations; this one is stated, not silent
gb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=8,
                                    categorical_features=[0, 1, 2, 9, 10], random_state=0,
                                    early_stopping=True, validation_fraction=0.05,
                                    n_iter_no_change=20).fit(Xtr, Ytr)
gbc = gb.classes_

# ---- collect per-position class-level top-1 for every model ----
H = []; tc = []; U = []; B = []; G = []
for g in te:
    X, Y, PR = feats(g)
    if not X: continue
    pr = gb.predict_proba(np.array(X, float)); full = np.zeros((len(X), V)); full[:, gbc] = pr
    for r, (tru, pv) in enumerate(zip(Y, PR)):
        H.append(Hbits[pv]); tc.append(tok2cls[tru])
        U.append(int(uni_cd.argmax())); B.append(int(bigc[pv].argmax()))
        G.append(int(to_class_dist(full[r], tok2cls, NC).argmax()))
H = np.array(H); tc = np.array(tc); U = np.array(U); B = np.array(B); G = np.array(G)

# ---- LM forward (LM-2.4 config-driven) ----
LM = None
if a.ckpt:
    import train_bball_lm as tb
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]; port = _cfg["port"]
    tb.MAXLEN = sd["pos.weight"].shape[0]
    m = tb.BasketballLM(V, len(ck["pvocab"]), d=_cfg["d"], nlayers=_cfg["layers"],
                        port=port, budget_ch=_cfg["budget_ch"], pace_ch=_cfg["pace_ch"],
                        period_ch=_cfg["period_ch"], mtp=_cfg["mtp_w"] > 0, tt=_cfg["tt_w"] > 0)
    m.load_state_dict(sd, strict=True); m.eval()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    m = m.to(dev)
    plut = pz = None
    if _cfg["budget_ch"]:
        _pl = pd.read_parquet(os.path.join(a.data_dir, "prior_minutes.parquet"))
        plut = {(str(r.player_id), str(r.game_id)): float(r.prior_min)
                for r in _pl.itertuples(index=False)}
    if _cfg["pace_ch"]:
        pz = pd.read_parquet(os.path.join(a.data_dir, "team_pace.parquet")).set_index("game_id").pace_z
    ds = tb.GameDataset(te, ck["pvocab"], vocab, rlut=ck.get("rlut") or {}, port=port,
                        plut=plut,
                        pace={g["gid"]: float(pz.get(g["gid"], 0.)) for g in te}
                             if _cfg["pace_ch"] else None)
    dl = torch.utils.data.DataLoader(ds, batch_size=16)
    lm = []
    print(f"LM forward over test on {dev}...", flush=True)
    with torch.no_grad():
        for b in dl:
            (tok, h, aa, ckl, hm, am, sdf, hr, ar, ac, hro, aro, hr13, ar13, hm13, am13,
             hon, aon, en, asl, mask, hpri, apri, hpri13, apri13, pace_b) = [x.to(dev) for x in b]
            lg = m(tok, h, aa, ckl, hm, am, sdf, hr, ar, hro, aro, hr13, ar13,
                   hm13, am13, hon, aon, hpri, apri, hpri13, apri13, pace_b)[0]
            pr = torch.softmax(lg[:, :-1], -1).cpu().numpy(); tg = tok[:, 1:].cpu().numpy()
            w = mask[:, 1:].cpu().numpy().astype(bool)
            Bn, T = tg.shape
            for bi in range(Bn):
                for ti in range(T):
                    if w[bi, ti]:
                        lm.append(int(to_class_dist(pr[bi, ti], tok2cls, NC).argmax()))
    LM = np.array(lm)
    assert len(LM) == len(tc), f"{len(LM)} vs {len(tc)}"

# ---- report ----
def acc(p, mk=None): return (p[tc == tc] == tc).mean() if mk is None else (p[mk] == tc[mk]).mean()
mods = [("Historical avg", U), ("Bigram", B), ("GB (context)", G)] + ([("Basketball LM", LM)] if LM is not None else [])
med = np.median(H); con = H <= med; opn = H > med
print("\n=== Table 1: next-event class-level top-1 (stratified by transition difficulty) ===")
print(f"{'model':16s}  All     Constrained  Open")
for nm, p in mods:
    print(f"{nm:16s}  {acc(p):.1%}   {acc(p,con):.1%}        {acc(p,opn):.1%}")
q = np.quantile(H, [1/3, 2/3]); bk = np.digitize(H, q)
print(f"\n3-bucket (terciles):  {'model':16s}  All    Forced  Mixed  Open")
for nm, p in mods:
    print(f"                      {nm:16s}  {acc(p):.1%}  {acc(p,bk==0):.1%}  {acc(p,bk==1):.1%}  {acc(p,bk==2):.1%}")
print(f"\n(Constrained n={con.sum()}, Open n={opn.sum()}; difficulty = entropy of P(class|prior event).)")
if LM is None:
    print("LM column skipped (no --ckpt). Train via train_bball_lm.py then pass --ckpt.")

arrs = dict(tc=tc, H=H, U=U, B=B, G=G)
if LM is not None:
    arrs["LM"] = LM
np.savez("fold_nextevent.npz", **arrs)
print(f"fold_nextevent.npz written ({'with' if LM is not None else 'WITHOUT'} LM column)")
