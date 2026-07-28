#!/usr/bin/env python3
"""Paper Table 2 — next-event prediction: LM vs historical/bigram/GB.

  python3 experiments/table2_nextevent.py --data-dir <dataset> --ckpt <best_model.pt>
"""
import argparse
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from _common import (load_corpus, class_map, to_class_dist, load_card,
                     boot_corr_diff)

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="../sloan_hf_dataset")
ap.add_argument("--ckpt", default=None, help="trained checkpoint; omit to run baselines only")
ap.add_argument("--gb-sample", type=int, default=900000)
a = ap.parse_args()

corpus = load_corpus(a.data_dir); vocab = corpus["vocab"]; V = len(vocab)
games = corpus["games"]
tr = [g for g in games.values() if g["split"] == "train"]
te = [g for g in games.values() if g["split"] == "test"]
cls, CLS, tok2cls, NC = class_map(vocab)
IS_MAKE = np.array(["MAKE" in t for t in vocab]); IS_MISS = np.array(["MISS" in t for t in vocab])
IS_FOUL = np.array(["FOUL" in t for t in vocab])

def feats(g):
    t = np.asarray(g["tok"]); cl = np.asarray(g["clock"]); sd = np.asarray(g["sdiff"])
    X = []; Y = []; PR = []; nm = nmi = nf = 0
    for i in range(1, len(t)):
        X.append([t[i-1], t[i-2] if i >= 2 else V, t[i-3] if i >= 3 else V,
                  cl[i-1], sd[i-1], nm, nmi, nf, i]); Y.append(t[i]); PR.append(t[i-1])
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
Hbits = -(bigc * np.log2(bigc)).sum(1)      # forced-ness of transitions out of each prior event

# ---- GB context classifier ----
Xtr, Ytr = [], []
for g in tr:
    x, y, _ = feats(g); Xtr += x; Ytr += y
Xtr = np.array(Xtr, float); Ytr = np.array(Ytr); rng = np.random.default_rng(0)
if len(Xtr) > a.gb_sample:
    idx = rng.choice(len(Xtr), a.gb_sample, replace=False); Xtr = Xtr[idx]; Ytr = Ytr[idx]
print(f"training GB on {len(Xtr):,} rows...", flush=True)
gb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_depth=8,
                                    categorical_features=[0, 1, 2], random_state=0).fit(Xtr, Ytr)
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

# ---- LM forward (optional) ----
LM = None
if a.ckpt:
    import train_bball_lm as tb
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]; port = sd["pfuse.weight"].shape[1] - 33
    d = sd["tok.weight"].shape[1]
    nl = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("tr.layers."))
    tb.MAXLEN = sd["pos.weight"].shape[0]
    m = tb.BasketballLM(V, len(ck["pvocab"]), d=d, nlayers=nl, port=port)
    m.load_state_dict(sd, strict=False); m.eval()
    dl = torch.utils.data.DataLoader(
        tb.GameDataset(te, ck["pvocab"], vocab, rlut=ck.get("rlut") or {}, port=port), batch_size=16)
    lm = []
    print("LM forward over test...", flush=True)
    with torch.no_grad():
        for b in dl:
            tok, h, aa, ckl, hm, am, sdf, hr, ar, ac, hro, aro, hr13, ar13, hm13, am13, hon, aon, en, asl, mask = b
            lg = m(tok, h, aa, ckl, hm, am, sdf, hr, ar, hro, aro, hr13, ar13, hm13, am13, hon, aon)[0]
            pr = torch.softmax(lg[:, :-1], -1).numpy(); tg = tok[:, 1:].numpy(); w = mask[:, 1:].numpy().astype(bool)
            Bn, T = tg.shape
            for bi in range(Bn):
                for ti in range(T):
                    if w[bi, ti]: lm.append(int(to_class_dist(pr[bi, ti], tok2cls, NC).argmax()))
    LM = np.array(lm)
    assert len(LM) == len(tc), f"{len(LM)} vs {len(tc)}"

# ---- report ----
def acc(p, mk=None): return (p[tc == tc] == tc).mean() if mk is None else (p[mk] == tc[mk]).mean()
mods = [("Historical avg", U), ("Bigram", B), ("GB (context)", G)] + ([("Basketball LM", LM)] if LM is not None else [])
med = np.median(H); con = H <= med; opn = H > med
print("\n=== Table 2: next-event class-level top-1 (stratified by transition difficulty) ===")
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
