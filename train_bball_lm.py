#!/usr/bin/env python3
"""Train the Basketball LM: a causal transformer over play-by-play tokens.

  python3 train_bball_lm.py --data-dir <dataset> [--smoke]
"""
import argparse, json, time
from loader import load_corpus
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DATA = "data"
OUT = Path("ckpt")
MAXLEN = 688    # longest corpus game is 671 tokens (multi-OT); the margin
                # above it is headroom for generated rollouts. A learned
                # positional table makes this a checkpoint dimension; the
                # trainer aborts if any game exceeds it.
# time-to-next-event bins (seconds)
DT_BOUNDS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 24, 28, 33]
DT_MID = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 9, 11, 13, 15.5, 18.5,
                   22, 26, 30.5, 38])


def _mirror_map(vocab):
    m = np.arange(len(vocab))
    for i, t in enumerate(vocab):
        if t.startswith("H_"): m[i] = vocab.index("A_" + t[2:])
        elif t.startswith("A_"): m[i] = vocab.index("H_" + t[2:])
    return m


class GameDataset(torch.utils.data.Dataset):
    def __init__(self, games, vocab_p, vocab=None, aug=False, rlut=None, port=3):
        self.g = games
        self.vp = vocab_p
        self.aug = aug
        self.mm = _mirror_map(vocab) if vocab is not None else None
        self.rlut = rlut or {}
        self.port = port

    def _rate(self, pid, season, gid=None):
        # card lookup: per-(player, game) as-of row; a miss = zeros
        r = (self.rlut.get((pid, gid)) or self.rlut.get((pid, season))
             or self.rlut.get((pid, "*")) or ())
        return (tuple(r) + (0.0,) * self.port)[: self.port]

    def __len__(self): return len(self.g)

    def __getitem__(self, i):
        d = self.g[i]
        T = min(len(d["tok"]), MAXLEN)
        tok = np.zeros(MAXLEN, np.int64); tok[:T] = d["tok"][:T]
        h = np.zeros((MAXLEN, 5), np.int64); a = np.zeros((MAXLEN, 5), np.int64)
        for k in range(5):
            h[:T, k] = [self.vp.get(p, 0) for p in d["h5"][:T, k]]
            a[:T, k] = [self.vp.get(p, 0) for p in d["a5"][:T, k]]
        ck = np.zeros(MAXLEN, np.float32); ck[:T] = d["clock"][:T]
        hm = np.zeros((MAXLEN, 5), np.float32); hm[:T] = d["hmin"][:T]
        am = np.zeros((MAXLEN, 5), np.float32); am[:T] = d["amin"][:T]
        sd = np.zeros(MAXLEN, np.float32); sd[:T] = d["sdiff"][:T]
        ac = np.full(MAXLEN, -100, np.int64)
        if "actor" in d: ac[:T] = d["actor"][:T]
        asl = np.full(MAXLEN, -100, np.int64)
        if "assister" in d: asl[:T] = d["assister"][:T]
        R = 17    # available-roster width
        hro = np.zeros(R, np.int64); aro = np.zeros(R, np.int64)
        hm13 = np.zeros((MAXLEN, R), np.float32); am13 = np.zeros((MAXLEN, R), np.float32)
        hr13 = np.zeros((R, self.port), np.float32); ar13 = np.zeros((R, self.port), np.float32)
        hon = np.zeros((MAXLEN, R), np.float32); aon = np.zeros((MAXLEN, R), np.float32)
        en = np.full(MAXLEN, -100, np.int64)
        if "hro" in d:
            sea = d.get("season", "*"); gid = d.get("gid")
            nh, na = len(d["hro"]), len(d["aro"])
            hro[:nh] = [self.vp.get(p, 0) for p in d["hro"]]
            aro[:na] = [self.vp.get(p, 0) for p in d["aro"]]
            hr13[:nh] = [self._rate(str(p), sea, gid) for p in d["hro"]]
            ar13[:na] = [self._rate(str(p), sea, gid) for p in d["aro"]]
            hm13[:T, :nh] = d["hm13"][:T, :nh]; am13[:T, :na] = d["am13"][:T, :na]
            hon[:T] = (h[:T, None, :] == hro[None, :, None]).any(-1) & (hro > 0)[None, :]
            aon[:T] = (a[:T, None, :] == aro[None, :, None]).any(-1) & (aro > 0)[None, :]
            en[:T] = d["entrant"][:T]
        hr = np.zeros((MAXLEN, 5, self.port), np.float32); ar = np.zeros((MAXLEN, 5, self.port), np.float32)
        sea = d.get("season", "*"); gid = d.get("gid")
        for k in range(5):
            hr[:T, k] = self._rate(str(d["h5"][0, k]), sea, gid) if T else (0, 0)
            ar[:T, k] = self._rate(str(d["a5"][0, k]), sea, gid) if T else (0, 0)
        for tt in range(T):
            for k in range(5):
                hr[tt, k] = self._rate(str(d["h5"][tt, k]), sea, gid)
                ar[tt, k] = self._rate(str(d["a5"][tt, k]), sea, gid)
        if self.aug:   # card-noise augmentation
            if np.random.random() < 0.5:
                hr = hr + np.random.normal(0, 0.3, hr.shape).astype(np.float32)
                ar = ar + np.random.normal(0, 0.3, ar.shape).astype(np.float32)
            if np.random.random() < 0.1:      # rating dropout: survive blind
                hr[:] = 0; ar[:] = 0
        mask = np.zeros(MAXLEN, np.float32); mask[:T] = 1.0
        if self.aug and np.random.random() < 0.5:   # home/away mirror
            tok = self.mm[tok]
            h, a = a, h
            hm, am = am, hm
            hr, ar = ar, hr
            hro, aro = aro, hro
            hm13, am13 = am13, hm13
            hr13, ar13 = ar13, hr13
            hon, aon = aon, hon
            sd = -sd
            sw = ac >= 0
            ac = np.where(sw, np.where(ac < 5, ac + 5, ac - 5), ac)
            sw = asl >= 0
            asl = np.where(sw, np.where(asl < 5, asl + 5, asl - 5), asl)
        return (tok, h, a, ck, hm, am, sd, hr, ar, ac,
                hro, aro, hr13, ar13, hm13, am13, hon, aon, en, asl, mask)


class BasketballLM(nn.Module):
    def __init__(self, vocab, n_players, d=160, nlayers=5, nhead=4,
                 warm_emb=None, eft=None, port=3):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(MAXLEN, d)
        self.pemb = nn.Embedding(n_players + 1, 32, padding_idx=0)
        if warm_emb is not None:                # start knowing who's who
            with torch.no_grad():
                self.pemb.weight.copy_(torch.tensor(warm_emb, dtype=torch.float32))
        self.port = port
        self.pfuse = nn.Linear(33 + port, 32)   # [emb | minutes | knowledge port]
        self.lineup = nn.Linear(64, d)          # [home-sum(32) | away-sum(32)]
        self.ctx = nn.Linear(2, d)              # [clock, score diff]
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(d, vocab)         # next EVENT
        self.dt_head = nn.Linear(d, len(DT_BOUNDS) + 1)  # time to next event
        self.actor_q = nn.Linear(d, 32)         # WHO acts next (pointer)
        self.ent_q = nn.Linear(d, 32)           # WHO checks in (bench)
        self.ast_q = nn.Linear(d, 32)           # assister pointer
        self.th = nn.Linear(32, 1)              # totals head
        self.vh = nn.Linear(d, 1)               # live value head
        self.mh = nn.Linear(32, 1)              # game-margin head
        self.pw = nn.Linear(32, 1)              # pregame win-prob head
        self.sr = nn.Linear(32, 1)              # self-rating head
        cm = torch.triu(torch.ones(MAXLEN, MAXLEN), diagonal=1).bool()
        self.register_buffer("cmask", cm)

    def forward(self, tok, h5, a5, ck, hm, am, sd, hr=None, ar=None,
                hro=None, aro=None, hr13=None, ar13=None, hm13=None, am13=None,
                hon=None, aon=None, shuffle_lineups=False, pos_offset=0):
        B, T = tok.shape
        if hr is None: hr = torch.zeros(*h5.shape, self.port)
        if ar is None: ar = torch.zeros(*a5.shape, self.port)
        if shuffle_lineups:                     # eval probe: break identity
            perm = torch.randperm(B)
            h5, a5, hm, am, hr, ar = h5[perm], a5[perm], hm[perm], am[perm], hr[perm], ar[perm]
        ph = self.pfuse(torch.cat([self.pemb(h5), hm.unsqueeze(-1), hr], -1))
        pa = self.pfuse(torch.cat([self.pemb(a5), am.unsqueeze(-1), ar], -1))
        self.last_sr = self.sr(ph.sum(2) - pa.sum(2)).squeeze(-1)  # [B,T]
        lu = self.lineup(torch.cat([ph.sum(2), pa.sum(2)], -1))
        pos_idx = (torch.arange(T, device=tok.device) + pos_offset).clamp(max=MAXLEN - 1)
        x = (self.tok(tok) + self.pos(pos_idx)[None]
             + lu + self.ctx(torch.stack([ck, sd], -1)))
        x = self.tr(x, mask=self.cmask[:T, :T])
        self.last_vh = self.vh(x).squeeze(-1)
        # actor pointer (10 on-court)
        cands = torch.cat([ph, pa], 2)                        # [B,T,10,32]
        actor = torch.einsum("btd,btkd->btk", self.actor_q(x), cands)
        self.last_ast = torch.einsum("btd,btkd->btk", self.ast_q(x), cands)
        ent_h = ent_a = None
        if hro is not None:
            B_, T_ = tok.shape
            q = self.ent_q(x)
            def bench_logits(ro, r13, m13, on):
                pv = self.pfuse(torch.cat(
                    [self.pemb(ro).unsqueeze(1).expand(B_, T_, -1, -1),
                     m13.unsqueeze(-1),
                     r13.unsqueeze(1).expand(B_, T_, -1, -1)], -1))
                lg_ = torch.einsum("btd,btkd->btk", q, pv)
                pad = (ro == 0).unsqueeze(1).expand(B_, T_, -1)
                return lg_.masked_fill(pad | (on > 0.5), -1e9)
            ent_h = bench_logits(hro, hr13, hm13, hon)
            ent_a = bench_logits(aro, ar13, am13, aon)
        return self.head(x), self.dt_head(x), actor, ent_h, ent_a


def dt_targets(ck, m):
    """binned time (sec) between consecutive events; mask invalid tails."""
    dt = (ck[:, 1:] - ck[:, :-1]).clamp(min=0) * 2880.0
    bins = torch.bucketize(dt, torch.tensor(DT_BOUNDS, dtype=dt.dtype,
                                            device=dt.device))
    return bins, m[:, 1:]


def sr_targets(tok, h5, a5, ck, m, ptsd):
    """Per-token label = its lineup-segment's margin rate (pts/min).
    Predicting this from player fusions alone is the LM's joint-rate estimator —
    the de-confounded joint estimation E performs externally. Segments
    under 45s are masked (rate too noisy)."""
    B, T = tok.shape
    dev_ = tok.device
    chg = torch.zeros(B, T, dtype=torch.bool, device=dev_)
    chg[:, 1:] = ((h5[:, 1:] != h5[:, :-1]).any(-1)
                  | (a5[:, 1:] != a5[:, :-1]).any(-1))
    seg = chg.cumsum(1)
    nseg = int(seg.max()) + 1
    flat = (torch.arange(B, device=dev_)[:, None] * nseg + seg).reshape(-1)
    mf = (m > 0).reshape(-1)
    pts = torch.zeros(B * nseg, device=dev_)
    pts.scatter_add_(0, flat, ptsd[tok].reshape(-1) * mf.float())
    tmin = torch.full((B * nseg,), 1e9, device=dev_).scatter_reduce(
        0, flat, torch.where(mf, ck.reshape(-1),
                             torch.full_like(ck.reshape(-1), 1e9)), "amin")
    tmax = torch.zeros(B * nseg, device=dev_).scatter_reduce(
        0, flat, ck.reshape(-1) * mf.float(), "amax")
    dur = (tmax - tmin).clamp(min=0) * 2880.0
    rate = (pts / (dur / 60.0).clamp(min=0.75)).clamp(-6, 6)
    ok = dur >= 45.0
    return rate[flat].reshape(B, T), ok[flat].reshape(B, T) & (m > 0)


@torch.no_grad()
def evaluate(model, loader, shuffle=False):
    model.eval()
    dev = next(model.parameters()).device
    tot, n, dt_ae, ac_ok, ac_n = 0.0, 0, 0.0, 0, 0
    en_ok, en_n, as_ok, as_n = 0, 0, 0, 0
    ce = nn.CrossEntropyLoss(reduction="none")
    mid = torch.tensor(DT_MID, dtype=torch.float32, device=dev)
    for batch in loader:
        (tok, h, a, ck, hm, am, sd, hr, ar, ac,
         hro, aro, hr13, ar13, hm13, am13, hon, aon, en, ast_t, m) = [x.to(dev) for x in batch]
        logits, dtl, actl, enth, enta = model(
            tok, h, a, ck, hm, am, sd, hr, ar,
            hro, aro, hr13, ar13, hm13, am13, hon, aon, shuffle_lineups=shuffle)
        loss = ce(logits[:, :-1].reshape(-1, logits.shape[-1]),
                  tok[:, 1:].reshape(-1))
        w = m[:, 1:].reshape(-1)
        tot += (loss * w).sum().item(); n += w.sum().item()
        tb, tm = dt_targets(ck, m)
        pred_dt = (torch.softmax(dtl[:, :-1], -1) * mid).sum(-1)
        true_dt = (ck[:, 1:] - ck[:, :-1]).clamp(min=0) * 2880.0
        dt_ae += ((pred_dt - true_dt).abs() * tm).sum().item()
        av = ac[:, 1:] >= 0
        if av.any():
            pred_a = actl[:, :-1].argmax(-1)
            ac_ok += ((pred_a == ac[:, 1:]) & av).sum().item()
            ac_n += av.sum().item()
        sv = ast_t[:, 1:] >= 0
        if sv.any():
            pred_s = model.last_ast[:, :-1].argmax(-1)
            as_ok += ((pred_s == ast_t[:, 1:]) & sv).sum().item()
            as_n += sv.sum().item()
        if enth is not None:
            ev_ = en[:, 1:] >= 0
            if ev_.any():
                is_h = ac[:, 1:] < 5
                pe = torch.where(is_h, enth[:, :-1].argmax(-1), enta[:, :-1].argmax(-1))
                en_ok += ((pe == en[:, 1:].clamp(min=0)) & ev_).sum().item()
                en_n += ev_.sum().item()
    return (float(np.exp(tot / n)), dt_ae / n,
            ac_ok / ac_n if ac_n else float("nan"),
            en_ok / en_n if en_n else float("nan"),
            as_ok / as_n if as_n else float("nan"))


def bigram_ppl(train_games, test_games, vocab_n):
    C = np.ones((vocab_n, vocab_n))             # +1 smoothing
    for d in train_games:
        t = d["tok"]
        np.add.at(C, (t[:-1], t[1:]), 1)
    P = C / C.sum(1, keepdims=True)
    tot, n = 0.0, 0
    for d in test_games:
        t = d["tok"]
        tot += -np.log(P[t[:-1], t[1:]]).sum(); n += len(t) - 1
    return float(np.exp(tot / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--data-dir", default="../sloan_hf_dataset",
                    help="path to the public dataset directory")
    ap.add_argument("--d", type=int, default=160, help="model width")
    ap.add_argument("--layers", type=int, default=5, help="transformer layers")
    ap.add_argument("--maxlen", type=int, default=0,
                    help="override MAXLEN (0 = keep 688, the paper setting)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sr-w", type=float, default=0.0,
                    help="self-rating head loss weight (0 = off)")
    ap.add_argument("--mh-w", type=float, default=0.0,
                    help="game-margin head loss weight (0 = off)")
    ap.add_argument("--balance", action="store_true",
                    help="normalize each head loss by its own EMA so nominal "
                         "weights are true gradient shares")
    ap.add_argument("--ast-w", type=float, default=0.0,
                    help="assister pointer head loss weight (0 = off); "
                         "labels = on-court slot of assist_player_id")
    ap.add_argument("--ls", type=float, default=0.0,
                    help="label smoothing on the EVENT head CE only")
    ap.add_argument("--pw-w", type=float, default=0.0,
                    help="pregame win-prob head weight (BCE on winner from "
                         "roster fusion; winner trained in)")
    ap.add_argument("--mh-wpool", action="store_true",
                    help="pool mh/pw roster fusion by softmax(mpg_l10 dial) "
                         "instead of flat mean (needs the 15-dial card)")
    ap.add_argument("--sd-drop", type=float, default=0.0,
                    help="blank the score-diff input for this fraction "
                         "of training games (reduces lead-erosion during generation)")
    ap.add_argument("--card", nargs="?", const="player_card.parquet",
                    default="player_card.parquet",
                    help="player-card parquet in the dataset dir: per-"
                         "(player, game) as-of rows; a missing row feeds "
                         "zeros (league mean)")
    a = ap.parse_args()
    global OUT, MAXLEN
    if a.out: OUT = Path(a.out)
    if a.maxlen: MAXLEN = a.maxlen
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time()

    blob = load_corpus(a.data_dir)
    games, vocab = blob["games"], blob["vocab"]
    longest = max(len(g["tok"]) for g in games.values())
    if longest > MAXLEN:
        raise SystemExit(
            f"ABORT: longest game is {longest} tokens > MAXLEN {MAXLEN} — "
            f"training would silently truncate it. Re-run with "
            f"--maxlen {longest + 32} or higher.")
    print(f"longest game {longest} tokens (MAXLEN {MAXLEN}: fits, with "
          f"{MAXLEN - longest} tokens of rollout headroom)", flush=True)
    for gid, g in games.items():
        g["season"] = str(gid)[3:5]
        g["gid"] = str(gid)
    tr = [g for g in games.values() if g["split"] == "train"]
    va = [g for g in games.values() if g["split"] == "val"]
    te = [g for g in games.values() if g["split"] == "test"]
    import pandas as _pd
    rlut = {}
    if a.card:   # replace the ratings feed with the raw-stat card
        _c = _pd.read_parquet(f"{a.data_dir}/{a.card}")
        keycol = "key" if "key" in _c.columns else "season"
        ccols = sorted([c for c in _c.columns if c.startswith("card_")],
                       key=lambda c: int(c.split("_")[1]))
        rlut = {(str(r[0]), str(r[1])): tuple(float(x) for x in r[2:])
                for r in _c[["player_id", keycol] + ccols].itertuples(index=False)}
        print(f"CARD port [{a.card}, key={keycol}]: {len(rlut):,} rows x "
              f"{len(ccols)} dials (card feed)",
              flush=True)
    if a.smoke:   # spread the sample across the corpus
        tr = tr[:: max(len(tr) // 300, 1)][:300]
        va, a.epochs = va[:80], 2

    pvocab = {}
    for d in tr:
        for arr in (d["h5"], d["a5"]):
            for p in arr.ravel(): pvocab.setdefault(p, len(pvocab) + 1)
    print(f"games: train {len(tr)} val {len(va)} test {len(te)} | "
          f"players {len(pvocab)} | vocab {len(vocab)}", flush=True)

    # player embeddings are random-init and learned from the play-by-play
    # alone — the paper model uses no side features at initialization
    warm = None

    # empirical dt-bin means
    dts, dbs = [], []
    for d in tr:
        dt = np.clip(np.diff(d["clock"]) * 2880.0, 0, None)
        dts.append(dt); dbs.append(np.digitize(dt, DT_BOUNDS))
    dts, dbs = np.concatenate(dts), np.concatenate(dbs)
    dt_means = np.array([np.median(dts[dbs == b]) if (dbs == b).any() else DT_MID[b]
                         for b in range(len(DT_BOUNDS) + 1)], np.float32)
    print(f"dt bin means: {np.round(dt_means, 1).tolist()}", flush=True)

    bg = bigram_ppl(tr, te, len(vocab))
    print(f"BIGRAM baseline test perplexity: {bg:.3f}  (pre-registered bar)", flush=True)

    port_w = len(next(iter(rlut.values())))
    tl = torch.utils.data.DataLoader(
        GameDataset(tr, pvocab, vocab, aug=True, rlut=rlut, port=port_w),
        batch_size=16, shuffle=True)
    vl = torch.utils.data.DataLoader(GameDataset(va, pvocab, rlut=rlut, port=port_w),
                                     batch_size=32)
    el = torch.utils.data.DataLoader(GameDataset(te, pvocab, rlut=rlut, port=port_w),
                                     batch_size=32)

    model = BasketballLM(len(vocab), len(pvocab), d=a.d, nlayers=a.layers,
                         warm_emb=warm, port=port_w)
    dev = torch.device(a.device)
    model = model.to(dev)
    PTSD = torch.zeros(len(vocab), device=dev)
    for _i, _nm in enumerate(vocab):
        if "MAKE" in _nm:   # substring: covers _MAKE_AST + 3PTC/3PTA too
            _p = 3 if "3PT" in _nm else (1 if "FT" in _nm else 2)
            PTSD[_i] = _p if _nm.startswith("H") else -_p
    if a.sr_w > 0:
        print(f"SELF-RATING head ON (weight {a.sr_w}) — joint-rate estimator, "
              f"labels = segment margin rates", flush=True)
    if a.ast_w > 0:
        n_ast = sum(int((g["assister"] >= 0).sum()) for g in tr
                    if "assister" in g)
        print(f"ASSISTER head ON (weight {a.ast_w}) — pointer over 10 "
              f"on-court, {n_ast:,} train labels "
              f"(+{model.ast_q.weight.numel() + model.ast_q.bias.numel():,} "
              f"params)", flush=True)
        assert n_ast > 0, "ast-w set but data has no assister labels"
    print(f"device: {dev}", flush=True)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    # player embeddings exempt from weight decay
    emb_p = list(model.pemb.parameters())
    other_p = [p for p in model.parameters() if not any(p is e for e in emb_p)]
    opt = torch.optim.AdamW(
        [{"params": other_p, "weight_decay": 0.01},
         {"params": emb_p, "weight_decay": 0.0, "lr": 1e-3}], lr=3e-4)
    ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    ce_ev = nn.CrossEntropyLoss(reduction="none", ignore_index=-100,
                                label_smoothing=a.ls)
    if a.ls > 0:
        print(f"EVENT label smoothing ON ({a.ls})", flush=True)
    if a.pw_w > 0:
        print(f"PREGAME WIN-PROB head ON (weight {a.pw_w})", flush=True)
    if a.mh_wpool:
        print("mh/pw roster pooling: softmax(mpg_l10) WEIGHTED", flush=True)
    warm_ep = 2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warm_ep if e < warm_ep
        else 0.5 * (1 + np.cos(np.pi * (e - warm_ep) / max(a.epochs - warm_ep, 1))))

    OUT.mkdir(parents=True, exist_ok=True)
    best, stale = 1e9, 0
    ema = {}   # per-head loss EMAs for --balance
    def bal(name, x):
        if not a.balance: return x
        v = float(x.detach())
        ema[name] = 0.98 * ema.get(name, v) + 0.02 * v
        return x / max(ema[name], 1e-6)
    for ep in range(a.epochs):
        model.train()
        for batch in tl:
            (tok, h, a5_, ck, hm, am, sd, hr, ar, ac,
             hro, aro, hr13, ar13, hm13, am13, hon, aon, en, ast_t, m) = [x.to(dev) for x in batch]
            if a.sd_drop > 0:
                keep = (torch.rand(sd.shape[0], 1, device=dev)
                        >= a.sd_drop).float()
                sd = sd * keep
            logits, dtl, actl, enth, enta = model(
                tok, h, a5_, ck, hm, am, sd, hr, ar,
                hro, aro, hr13, ar13, hm13, am13, hon, aon)
            w = m[:, 1:].reshape(-1)
            loss_t = (ce_ev(logits[:, :-1].reshape(-1, logits.shape[-1]),
                            tok[:, 1:].reshape(-1)) * w).sum() / w.sum()
            tb, _ = dt_targets(ck, m)
            loss_d = (ce(dtl[:, :-1].reshape(-1, dtl.shape[-1]),
                         tb.reshape(-1)) * w).sum() / w.sum()
            # actor loss
            loss_a = ce(actl[:, :-1].reshape(-1, 10), ac[:, 1:].reshape(-1))
            loss_a = loss_a.sum() / max((ac[:, 1:] >= 0).sum(), 1)
            # entrant loss
            is_h = (ac[:, 1:] < 5) & (ac[:, 1:] >= 0)
            en_t = en[:, 1:].clone()
            eh_t = torch.where(is_h & (en_t >= 0), en_t, torch.full_like(en_t, -100))
            ea_t = torch.where(~is_h & (en_t >= 0), en_t, torch.full_like(en_t, -100))
            loss_e = (ce(enth[:, :-1].reshape(-1, 17), eh_t.reshape(-1)).sum()
                      + ce(enta[:, :-1].reshape(-1, 17), ea_t.reshape(-1)).sum())
            loss_e = loss_e / max((en_t >= 0).sum(), 1)
            loss = (bal("t", loss_t) + 0.5 * bal("d", loss_d)
                    + 0.5 * bal("a", loss_a) + 0.3 * bal("e", loss_e))
            if a.ast_w > 0:
                # assister loss
                loss_as = ce(model.last_ast[:, :-1].reshape(-1, 10),
                             ast_t[:, 1:].reshape(-1))
                loss_as = loss_as.sum() / max((ast_t[:, 1:] >= 0).sum(), 1)
                loss = loss + a.ast_w * bal("as", loss_as)
            if a.sr_w > 0:
                st_, sv_ = sr_targets(tok, h, a5_, ck, m, PTSD)
                if sv_.any():
                    loss = loss + a.sr_w * bal("sr", nn.functional.huber_loss(
                        model.last_sr[sv_], st_[sv_]))
            if a.mh_w > 0 or a.pw_w > 0:
                # margin / win-prob head loss
                fm = (PTSD[tok] * m).sum(1) / 20.0
                hf = model.pfuse(torch.cat([model.pemb(hro),
                     torch.full((*hro.shape, 1), 0.5, device=dev), hr13], -1))
                af = model.pfuse(torch.cat([model.pemb(aro),
                     torch.full((*aro.shape, 1), 0.5, device=dev), ar13], -1))
                hmask = (hro > 0).unsqueeze(-1); amask = (aro > 0).unsqueeze(-1)
                if a.mh_wpool:
                    # minutes-weighted pooling
                    wh_ = torch.softmax(torch.where(
                        hro > 0, hr13[..., 7], torch.full_like(hr13[..., 7], -1e9)), 1)
                    wa_ = torch.softmax(torch.where(
                        aro > 0, ar13[..., 7], torch.full_like(ar13[..., 7], -1e9)), 1)
                    gv = ((hf * wh_.unsqueeze(-1)).sum(1)
                          - (af * wa_.unsqueeze(-1)).sum(1))
                else:
                    gv = ((hf * hmask).sum(1) / hmask.sum(1).clamp(min=1)
                          - (af * amask).sum(1) / amask.sum(1).clamp(min=1))
                if a.pw_w > 0:
                    loss = loss + a.pw_w * bal("pw",
                        nn.functional.binary_cross_entropy_with_logits(
                            model.pw(gv).squeeze(-1), (fm > 0).float()))
            if a.mh_w > 0:
                loss = loss + a.mh_w * bal("mh", nn.functional.huber_loss(
                    model.mh(gv).squeeze(-1), fm))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        vp, vdt, vac, ven, vas = evaluate(model, vl)
        print(f"E{ep+1:3d} | val ppl {vp:.3f} | dt MAE {vdt:.1f}s | "
              f"actor acc {vac:.1%} | entrant acc {ven:.1%} | "
              f"assister acc {vas:.1%} ({time.time()-t0:.0f}s)", flush=True)
        if vp < best - 1e-3:
            best, stale = vp, 0
            torch.save({"model": model.state_dict(),
                        "pvocab": {str(k): int(v) for k, v in pvocab.items()},
                        "vocab": vocab, "dt_means": dt_means,
                        "rlut": rlut}, OUT / "best_model.pt")
        else:
            stale += 1
            if stale >= 3: print("early stop", flush=True); break

    model.load_state_dict(
        torch.load(OUT / "best_model.pt", weights_only=False)["model"])
    tp, tdt, tac, ten, tas = evaluate(model, el)
    tps, *_ = evaluate(model, el, shuffle=True)
    print(f"\nTEST perplexity: {tp:.3f} | bigram {bg:.3f} | "
          f"lineups-shuffled {tps:.3f} | dt MAE {tdt:.1f}s | "
          f"actor acc {tac:.1%} | entrant acc {ten:.1%} | "
          f"assister acc {tas:.1%}")
    print(f"pre-reg 1 (beat bigram): {'PASS' if tp < bg else '** FAIL **'}")
    print(f"pre-reg 2 (identity matters): {'PASS' if tps > tp * 1.01 else '** FAIL **'} "
          f"(shuffle degrades ppl by {(tps/tp-1)*100:.1f}%)")
    json.dump({"test_ppl": round(tp, 4), "bigram_ppl": round(bg, 4),
               "shuffled_ppl": round(tps, 4), "val_ppl": round(best, 4),
               "test_dt_mae_sec": round(tdt, 2),
               "test_actor_acc": round(tac, 4) if tac == tac else None,
               "test_entrant_acc": round(ten, 4) if ten == ten else None,
               "test_assister_acc": round(tas, 4) if tas == tas else None,
               "prereg1_beat_bigram": tp < bg,
               "prereg2_identity_matters": tps > tp * 1.01,
               "data": "hf-dataset", "d": a.d, "layers": a.layers,
               "maxlen": MAXLEN, "card": a.card, "ast_w": a.ast_w,
               "sr_w": a.sr_w, "mh_w": a.mh_w, "sd_drop": a.sd_drop,
               "ls": a.ls, "pw_w": a.pw_w, "mh_wpool": a.mh_wpool,
               "balance": a.balance},
              open(OUT / "config.json", "w"), indent=2)
    print(f"saved -> {OUT}/ ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
