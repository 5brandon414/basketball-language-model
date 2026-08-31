#!/usr/bin/env python3
"""Simulate games with the Basketball LM (batched Monte-Carlo rollouts).

Modes: --state pre (from tip-off) | half (real first half as prompt).
Vocabulary and dimensions are auto-detected from the checkpoint.
"""
import argparse, json, os, time
from loader import load_corpus, load_lineup_timeline
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_bball_lm import BasketballLM, DT_MID
import train_bball_lm as _tb

CKPT = Path("ckpt/best_model.pt")
DT_MEANS = np.asarray(DT_MID)
VIDX = {}
PTS_H = PTS_A = None
MAXLEN = _tb.MAXLEN


def init_tables(vocab):
    """Token tables built from the checkpoint's own vocab."""
    global VIDX, PTS_H, PTS_A
    NV = len(vocab)
    VIDX = {t: i for i, t in enumerate(vocab)}
    PTS_H = np.zeros(NV); PTS_A = np.zeros(NV)
    for i, t in enumerate(vocab):
        if "MAKE" in t:
            p = 3 if "3PT" in t else (1 if "FT_" in t else 2)
            (PTS_H if t.startswith("H_") else PTS_A)[i] = p


def make_rater(ck):
    rlut = ck.get("rlut")
    if rlut:
        k0 = next(iter(rlut)); pad = (0.0,) * len(rlut[k0])
        return lambda p, s, g=None: (rlut.get((p, g)) or rlut.get((p, s))
                                     or rlut.get((p, "*")) or pad)
    return lambda p, s, g=None: (0.0,) * 32


@torch.no_grad()
def kv_prefill(model, xseq):
    """Prefix pass: build the per-layer (K,V) caches. Hand-rolled replica of
    the encoder-layer math, so it must match forward() exactly."""
    x = xseq
    P = x.shape[1]
    mask = torch.triu(torch.full((P, P), float("-inf"), device=x.device), 1)
    caches = []
    for layer in model.tr.layers:
        sa = layer.self_attn
        d = x.shape[-1]; nh = sa.num_heads; hd = d // nh
        h = layer.norm1(x)
        qkv = torch.nn.functional.linear(h, sa.in_proj_weight, sa.in_proj_bias)
        q, k, v = qkv.chunk(3, -1)
        caches.append((k, v))
        B, T, _ = k.shape
        qh = q.view(B, T, nh, hd).transpose(1, 2)
        kh = k.view(B, T, nh, hd).transpose(1, 2)
        vh = v.view(B, T, nh, hd).transpose(1, 2)
        at = (torch.softmax(qh @ kh.transpose(-1, -2) / hd ** 0.5 + mask, -1) @ vh)
        x = x + sa.out_proj(at.transpose(1, 2).reshape(B, T, d))
        x = x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
    return caches


def kv_step(model, caches, xn):
    """One-token transformer step with projected K/V caching.
    xn: [K,1,d] input embedding for the new position."""
    x = xn
    for l, layer in enumerate(model.tr.layers):
        sa = layer.self_attn
        d = x.shape[-1]; nh = sa.num_heads; hd = d // nh
        h = layer.norm1(x)
        qkv = torch.nn.functional.linear(h, sa.in_proj_weight, sa.in_proj_bias)
        q, k, v = qkv.chunk(3, -1)
        if caches[l] is None:
            K_, V_ = k, v
        else:
            K_ = torch.cat([caches[l][0], k], 1); V_ = torch.cat([caches[l][1], v], 1)
        caches[l] = (K_, V_)
        B, T, _ = K_.shape
        qh = q.view(B, 1, nh, hd).transpose(1, 2)
        kh = K_.view(B, T, nh, hd).transpose(1, 2)
        vh = V_.view(B, T, nh, hd).transpose(1, 2)
        at = (torch.softmax(qh @ kh.transpose(-1, -2) / hd ** 0.5, -1) @ vh)
        x = x + sa.out_proj(at.transpose(1, 2).reshape(B, 1, d))
        x = x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
    return x


@torch.no_grad()
def rollout_lmsubs(model, pvocab, game, gen, K, port, dev, rate=None, season="*",
                   prefix=None, kv=False, gid="", plut=None, paceval=0.0):
    """The LM drives its own rotations. Lineups start at the real starters
    (or, with a prefix, the real halftime lineup); on every sampled SUB token
    the actor head picks who comes off and the entrant head picks who checks
    in. Returns scores + per-player simulated seconds."""
    R = 17    # available-roster slots per side
    def order(ids, starters):
        st = [str(p) for p in starters]
        rest = [str(p) for p in ids if str(p) not in st]
        return (st + rest)[:R]
    hro_ids = order(game["hro"], game["h5"][0])
    aro_ids = order(game["aro"], game["a5"][0])
    nh, na = len(hro_ids), len(aro_ids)
    rate = rate or (lambda p, s: (0.0,) * port)
    hro = torch.zeros(R, dtype=torch.long); aro = torch.zeros(R, dtype=torch.long)
    hro[:nh] = torch.tensor([pvocab.get(str(p), 0) for p in hro_ids])
    aro[:na] = torch.tensor([pvocab.get(str(p), 0) for p in aro_ids])
    hr13 = torch.zeros(R, port); ar13 = torch.zeros(R, port)
    hr13[:nh] = torch.tensor([rate(str(p), season) for p in hro_ids])
    ar13[:na] = torch.tensor([rate(str(p), season) for p in aro_ids])
    pri13h = torch.zeros(R); pri13a = torch.zeros(R)
    if plut:
        pri13h[:nh] = torch.tensor([plut.get((str(p), gid), 0.0) / 48. for p in hro_ids])
        pri13a[:na] = torch.tensor([plut.get((str(p), gid), 0.0) / 48. for p in aro_ids])

    def slots_of(five, ids):
        m = np.zeros(R, bool)
        for p in five:
            w = [k for k, q in enumerate(ids) if str(q) == str(p)]
            if w: m[w[0]] = True
        return m
    hon0 = slots_of(game["h5"][0], hro_ids); aon0 = slots_of(game["a5"][0], aro_ids)

    L = MAXLEN
    tokB = torch.full((K, L), VIDX["PAD"], dtype=torch.long, device=dev)
    tokB[:, 0] = VIDX["BOS"]
    h5B = torch.zeros(K, L, 5, dtype=torch.long, device=dev)
    a5B = torch.zeros(K, L, 5, dtype=torch.long, device=dev)
    hmB = torch.zeros(K, L, 5, device=dev); amB = torch.zeros(K, L, 5, device=dev)
    hrB = torch.zeros(K, L, 5, port, device=dev); arB = torch.zeros(K, L, 5, port, device=dev)
    ckB = torch.zeros(K, L, device=dev); sdB = torch.zeros(K, L, device=dev)
    hm13B = torch.zeros(K, L, R, device=dev); am13B = torch.zeros(K, L, R, device=dev)
    honB = torch.zeros(K, L, R, device=dev); aonB = torch.zeros(K, L, R, device=dev)
    hroB = hro.repeat(K, 1).to(dev); aroB = aro.repeat(K, 1).to(dev)
    hr13B = hr13.repeat(K, 1, 1).to(dev); ar13B = ar13.repeat(K, 1, 1).to(dev)
    hpriB = torch.zeros(K, L, 5, device=dev); apriB = torch.zeros(K, L, 5, device=dev)
    pri13hB = pri13h.repeat(K, 1).to(dev); pri13aB = pri13a.repeat(K, 1).to(dev)
    paceB = torch.full((K,), float(paceval), device=dev)

    t = np.zeros(K); hp = np.zeros(K); ap = np.zeros(K)
    done = np.zeros(K, bool)
    bound = np.full(K, 2880.0); OT_CAP = 2880.0 + 4 * 300.0
    hmin = np.zeros((K, R)); amin = np.zeros((K, R))
    hon = np.repeat(hon0[None], K, 0); aon = np.repeat(aon0[None], K, 0)
    T = 1
    if prefix is not None:
        # halftime start: condition on the real first half, then the LM
        # drives every second-half rotation itself — no oracle timeline
        P = len(prefix["tok"])
        tokB[:, :P] = torch.tensor(prefix["tok"], dtype=torch.long,
                                   device=dev)[None]
        h5B[:, :P] = torch.tensor(prefix["h5"], dtype=torch.long, device=dev)[None]
        a5B[:, :P] = torch.tensor(prefix["a5"], dtype=torch.long, device=dev)[None]
        hmB[:, :P] = torch.tensor(np.array(prefix["hm"]), dtype=torch.float32,
                                  device=dev)[None]
        amB[:, :P] = torch.tensor(np.array(prefix["am"]), dtype=torch.float32,
                                  device=dev)[None]
        hrB[:, :P] = torch.tensor(np.array([[rate(str(p), season) for p in row]
                                            for row in prefix["h5_raw"]]),
                                  dtype=torch.float32, device=dev)
        arB[:, :P] = torch.tensor(np.array([[rate(str(p), season) for p in row]
                                            for row in prefix["a5_raw"]]),
                                  dtype=torch.float32, device=dev)
        ckB[:, :P] = torch.tensor(prefix["ck"], dtype=torch.float32, device=dev)[None]
        sdB[:, :P] = torch.tensor(prefix["sd"], dtype=torch.float32, device=dev)[None]
        if plut:
            hpriB[:, :P] = torch.tensor(
                [[plut.get((str(p), gid), 0.0) / 48. for p in row]
                 for row in prefix["h5_raw"]], dtype=torch.float32, device=dev)[None]
            apriB[:, :P] = torch.tensor(
                [[plut.get((str(p), gid), 0.0) / 48. for p in row]
                 for row in prefix["a5_raw"]], dtype=torch.float32, device=dev)[None]
        # roster-space prefix rows, permuted sorted-order -> starters-first
        srt_h = [str(p) for p in game["hro"]]; srt_a = [str(p) for p in game["aro"]]
        perm_h = [srt_h.index(p) for p in hro_ids]
        perm_a = [srt_a.index(p) for p in aro_ids]
        hm13_p = np.zeros((P, R), np.float32); am13_p = np.zeros((P, R), np.float32)
        hm13_p[:, :nh] = game["hm13"][:P][:, perm_h]
        am13_p[:, :na] = game["am13"][:P][:, perm_a]
        hm13B[:, :P] = torch.tensor(hm13_p, device=dev)
        am13B[:, :P] = torch.tensor(am13_p, device=dev)
        hon_p = np.stack([slots_of(row, hro_ids) for row in prefix["h5_raw"]])
        aon_p = np.stack([slots_of(row, aro_ids) for row in prefix["a5_raw"]])
        honB[:, :P] = torch.tensor(hon_p, dtype=torch.float32, device=dev)
        aonB[:, :P] = torch.tensor(aon_p, dtype=torch.float32, device=dev)
        # sim state entering H2: real halftime lineup, banked minutes, score
        t[:] = prefix["t0"]; hp[:] = prefix["h0"]; ap[:] = prefix["a0"]
        hon[:] = hon_p[-1][None]; aon[:] = aon_p[-1][None]
        hmin[:] = hm13_p[-1][None] * 2880.0; amin[:] = am13_p[-1][None] * 2880.0
        T = P

    caches = None      # kv: built at j==0, or prefilled over the real prefix
    while (~done).any() and T < MAXLEN - 1:
        j = T - 1
        for k in range(K):
            hi = np.where(hon[k])[0][:5]; ai = np.where(aon[k])[0][:5]
            h5B[k, j] = hroB[k, hi]; a5B[k, j] = aroB[k, ai]
            hmB[k, j] = torch.tensor(hmin[k, hi] / 2880.0, dtype=torch.float32)
            amB[k, j] = torch.tensor(amin[k, ai] / 2880.0, dtype=torch.float32)
            hrB[k, j] = hr13B[k, hi]; arB[k, j] = ar13B[k, ai]
            hpriB[k, j] = pri13hB[k, hi]; apriB[k, j] = pri13aB[k, ai]
        hm13B[:, j] = torch.tensor(hmin / 2880.0, dtype=torch.float32, device=dev)
        am13B[:, j] = torch.tensor(amin / 2880.0, dtype=torch.float32, device=dev)
        honB[:, j] = torch.tensor(hon, dtype=torch.float32, device=dev)
        aonB[:, j] = torch.tensor(aon, dtype=torch.float32, device=dev)
        ckB[:, j] = torch.tensor(np.minimum(t, 3600.0) / 2880.0,
                                 dtype=torch.float32, device=dev)
        sdB[:, j] = torch.tensor(np.clip((hp - ap) / 20.0, -2, 2),
                                 dtype=torch.float32, device=dev)

        if kv:
            def emb_range(lo_, hi_):
                def _f5(embt, mnt, rtt, prit):
                    xs = [embt, mnt.unsqueeze(-1), rtt]
                    if getattr(model, "budget_ch", False):
                        dfc = prit * ckB[:, lo_:hi_].unsqueeze(-1) - mnt
                        xs += [prit.unsqueeze(-1), dfc.unsqueeze(-1)]
                    return model.pfuse(torch.cat(xs, -1))
                ph_ = _f5(model.pemb(h5B[:, lo_:hi_]), hmB[:, lo_:hi_],
                          hrB[:, lo_:hi_], hpriB[:, lo_:hi_])
                pa_ = _f5(model.pemb(a5B[:, lo_:hi_]), amB[:, lo_:hi_],
                          arB[:, lo_:hi_], apriB[:, lo_:hi_])
                lu_ = model.lineup(torch.cat([ph_.sum(2), pa_.sum(2)], -1))
                pos_ = model.pos.weight[
                    torch.arange(lo_, hi_, device=dev).clamp(max=MAXLEN - 1)]
                cf = [ckB[:, lo_:hi_], sdB[:, lo_:hi_]]
                if getattr(model, "pace_ch", False):
                    cf.append(paceB.unsqueeze(-1).expand(K, hi_ - lo_))
                if getattr(model, "period_ch", False):
                    el_ = ckB[:, lo_:hi_] * 2880.0
                    reg_ = (720.0 - torch.remainder(el_, 720.0)) / 720.0
                    ot_ = (300.0 - torch.remainder(el_ - 2880.0, 300.0)) / 720.0
                    cf.append(torch.where(el_ < 2880.0, reg_, ot_))
                return (model.tok(tokB[:, lo_:hi_]) + pos_[None] + lu_
                        + model.ctx(torch.stack(cf, -1))), ph_, pa_
            if j == 0:
                caches = [None] * len(model.tr.layers)
                xr, ph1, pa1 = emb_range(0, 1)
            elif caches is None:      # prefix run: build the cache once
                xall, _, _ = emb_range(0, j)
                caches = kv_prefill(model, xall)
                xr, ph1, pa1 = emb_range(j, j + 1)
            else:
                xr, ph1, pa1 = emb_range(j, j + 1)
            hid = kv_step(model, caches, xr)
            x1 = hid[:, 0]
            lg = model.head(hid)[:, 0].float().cpu()
            dl = model.dt_head(hid)[:, 0].float().cpu()
            # pointer heads at position j — exact replica of forward's math
            cands = torch.cat([ph1[:, 0], pa1[:, 0]], 1)
            actl_1 = torch.einsum("kd,knd->kn", model.actor_q(x1), cands)
            q1 = model.ent_q(x1)
            def bench1(ro, r13, m13_j, on_j, pri13_):
                xs = [model.pemb(ro), m13_j.unsqueeze(-1), r13]
                if getattr(model, "budget_ch", False):
                    dfc = pri13_ * ckB[:, j].unsqueeze(-1) - m13_j
                    xs += [pri13_.unsqueeze(-1), dfc.unsqueeze(-1)]
                pv = model.pfuse(torch.cat(xs, -1))
                lg_ = torch.einsum("kd,krd->kr", q1, pv)
                return lg_.masked_fill((ro == 0) | (on_j > 0.5), -1e9)
            act_lg_full = actl_1.float().cpu()
            eh_full = bench1(hroB, hr13B, hm13B[:, j], honB[:, j], pri13hB).float().cpu()
            ea_full = bench1(aroB, ar13B, am13B[:, j], aonB[:, j], pri13aB).float().cpu()
        else:
            lo = 0
            logits, dtl, actl, enth, enta = model(
                tokB[:, lo:T], h5B[:, lo:T], a5B[:, lo:T], ckB[:, lo:T],
                hmB[:, lo:T], amB[:, lo:T], sdB[:, lo:T], hrB[:, lo:T], arB[:, lo:T],
                hroB, aroB, hr13B, ar13B, hm13B[:, lo:T], am13B[:, lo:T],
                honB[:, lo:T], aonB[:, lo:T], hpriB[:, lo:T], apriB[:, lo:T],
                pri13hB, pri13aB, paceB, pos_offset=lo)
            lg = logits[:, -1].float().cpu()
            dl = dtl[:, -1].float().cpu()
        pt = torch.softmax(lg, -1)
        pt[:, VIDX["PAD"]] = 0; pt[:, VIDX["BOS"]] = 0
        # rules constraint, not steering: no NBA game ends level, so EOS
        # is illegal while the simulated score is tied
        _tied = torch.tensor(hp == ap, dtype=torch.bool)
        if _tied.any():
            pt[_tied, VIDX["EOS"]] = 0
        nxt = torch.multinomial(pt / pt.sum(1, keepdim=True), 1,
                                generator=gen)[:, 0].numpy()
        dtb = torch.multinomial(torch.softmax(dl, -1), 1, generator=gen)[:, 0].numpy()
        if kv:
            act_lg = act_lg_full; eh_lg = eh_full; ea_lg = ea_full
        else:
            act_lg = actl[:, -1].float().cpu()
            eh_lg = enth[:, -1].float().cpu(); ea_lg = enta[:, -1].float().cpu()

        nxt[done] = VIDX["PAD"]
        alive = ~done
        newly_eos = alive & (nxt == VIDX["EOS"])
        act = alive & ~newly_eos
        for k in np.where(act)[0]:
            tk = int(nxt[k])
            if tk == VIDX["H_SUB"] or tk == VIDX["A_SUB"]:
                home_side = tk == VIDX["H_SUB"]
                on = hon[k] if home_side else aon[k]
                onk = np.where(on)[0][:5]
                sl = slice(0, 5) if home_side else slice(5, 10)
                ex_local = int(torch.multinomial(
                    torch.softmax(act_lg[k][sl], -1), 1, generator=gen))
                ex_idx = onk[ex_local] if ex_local < len(onk) else onk[0]
                elg = (eh_lg if home_side else ea_lg)[k].clone()
                elg[torch.tensor(on, dtype=torch.bool)] = -1e9
                nlim = nh if home_side else na
                elg[nlim:] = -1e9
                in_idx = int(torch.multinomial(torch.softmax(elg, -1),
                                               1, generator=gen))
                if not on[in_idx]:
                    on[ex_idx] = False; on[in_idx] = True
        hp[act] += PTS_H[nxt[act]]; ap[act] += PTS_A[nxt[act]]
        dt_s = DT_MEANS[dtb]
        for k in np.where(act)[0]:
            hmin[k, hon[k]] += dt_s[k]; amin[k, aon[k]] += dt_s[k]
        t[act] += dt_s[act]
        # overtime: extend the horizon 5:00 at a time while tied (cap 4 OTs)
        ext = alive & (t >= bound) & (hp == ap) & (bound < OT_CAP)
        bound[ext] += 300.0
        done |= newly_eos | ((t >= bound) & (hp != ap)) | (t >= OT_CAP)
        tokB[:, T] = torch.tensor(nxt, dtype=torch.long, device=dev)
        T += 1
    # id orderings the minute columns are indexed by
    if (~done).any():
        print(f"  WARNING: {int((~done).sum())}/{K} rollouts truncated at "
              f"MAXLEN with clock {t[~done].min():.0f}s", flush=True)
    return hp, ap, hmin, amin, hro_ids, aro_ids, bound


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data-dir", default="../sloan_hf_dataset")
    ap.add_argument("--splits", default=None,
                    help="split-assignment JSON overriding splits.json "
                         "(use the fold file matching the checkpoint)")
    ap.add_argument("--rollouts", type=int, default=24)
    ap.add_argument("--state", default="pre", choices=["pre", "half"])
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lm-subs", action="store_true",
                    help="the LM drives its own rotations")
    ap.add_argument("--split", default="test",
                    choices=["test", "val", "all"],
                    help="'all' ignores the split label and takes the games "
                         "listed in --gids-file (how the fold tables are run)")
    ap.add_argument("--kv", action="store_true",
                    help="KV-cached generation (same math as the full forward)")
    ap.add_argument("--gids-file", default=None, help="restrict to game_ids listed in file")
    ap.add_argument("--prior", default=None,
                    help="prior-minutes parquet (default <data-dir>/prior_minutes.parquet)")
    ap.add_argument("--pace-file", default=None,
                    help="pace parquet (default <data-dir>/team_pace.parquet)")
    a = ap.parse_args()
    if not a.lm_subs:
        raise SystemExit("the oracle-timeline mode is not part of the "
                         "release; every paper number uses --lm-subs")
    if a.prior is None: a.prior = os.path.join(a.data_dir, "prior_minutes.parquet")
    if a.pace_file is None: a.pace_file = os.path.join(a.data_dir, "team_pace.parquet")
    t0 = time.time()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    _cfgp = os.path.join(os.path.dirname(a.ckpt) or ".", "config.json")
    _cfg = json.load(open(_cfgp)) if os.path.exists(_cfgp) else {}
    global DT_MEANS
    DT_MEANS = np.asarray(ck.get("dt_means", DT_MID))
    pvocab = ck["pvocab"]
    port = _cfg.get("port", ck["model"]["pfuse.weight"].shape[1] - 33)
    dev = torch.device(a.device)
    init_tables(ck["vocab"])
    _sd = ck["model"]
    _tb.MAXLEN = _sd["pos.weight"].shape[0]
    global MAXLEN; MAXLEN = _tb.MAXLEN
    _d = _sd["tok.weight"].shape[1]
    _nl = 1 + max(int(k.split(".")[2]) for k in _sd if k.startswith("tr.layers."))
    model = BasketballLM(len(ck["vocab"]), len(pvocab), d=_d, nlayers=_nl,
                         port=port, budget_ch=_cfg.get("budget_ch", False),
                         pace_ch=_cfg.get("pace_ch", False),
                         period_ch=_cfg.get("period_ch", False),
                         mtp=_cfg.get("mtp_w", 0) > 0,
                         tt=_cfg.get("tt_w", 0) > 0).to(dev)
    for _f in ("budget_ch", "pace_ch", "period_ch", "fix_sub_boundary"):
        if _cfg.get(_f): print(f"GEN config: {_f} ON", flush=True)
    model.load_state_dict(ck["model"], strict=True); model.eval()
    rate0 = make_rater(ck)
    gen = torch.Generator().manual_seed(a.seed)

    blob = load_corpus(a.data_dir, a.splits,
                       fix_sub_boundary=_cfg.get("fix_sub_boundary", False))["games"]
    plut, pmap = {}, {}
    if _cfg.get("budget_ch"):
        _p = pd.read_parquet(a.prior)
        plut = {(str(r.player_id), str(r.game_id)): float(r.prior_min)
                for r in _p.itertuples(index=False)}
        print(f"GEN prior-minutes sidecar: {len(plut):,} rows", flush=True)
    if _cfg.get("pace_ch"):
        _p = pd.read_parquet(a.pace_file)
        pmap = {str(r.game_id): float(r.pace_z) for r in _p.itertuples(index=False)}
        print(f"GEN pace sidecar: {len(pmap):,} games", flush=True)

    stints = load_lineup_timeline(a.data_dir,
                                  None if a.split == "all" else a.split,
                                  a.splits)
    if a.gids_file:
        keep = set(open(a.gids_file).read().split())
        stints = stints[stints.game_id.isin(keep)]
    rows = []
    for n_g, (gid, g) in enumerate(stints.groupby("game_id", sort=False)):
        if n_g >= a.games: break
        g = g.sort_values("t_start_sec")
        season = gid[3:5]
        # card lookup bound to this game id
        rate = lambda p, s, _g=str(gid): (tuple(rate0(p, s, _g))
                                          + (0.0,) * port)[:port]
        prefix = None
        if a.state == "half":
            gd = blob.get(gid)
            if gd is None: continue
            ck_arr = gd["clock"]
            k = int(np.searchsorted(ck_arr, 0.5))
            if k < 10 or k >= len(gd["tok"]) - 5: continue
            mp = lambda arr: [[pvocab.get(str(p), 0) for p in row] for row in arr]
            prefix = {"tok": [int(x) for x in gd["tok"][:k]],
                      "h5": mp(gd["h5"][:k]), "a5": mp(gd["a5"][:k]),
                      "h5_raw": gd["h5"][:k], "a5_raw": gd["a5"][:k],
                      "hm": gd["hmin"][:k].tolist(), "am": gd["amin"][:k].tolist(),
                      "ck": gd["clock"][:k].tolist(), "sd": gd["sdiff"][:k].tolist(),
                      "t0": float(ck_arr[k - 1] * 2880.0),
                      "h0": int(sum(PTS_H[int(x)] for x in gd["tok"][:k])),
                      "a0": int(sum(PTS_A[int(x)] for x in gd["tok"][:k]))}
        gd = blob.get(gid)
        if gd is None: continue
        hpv, apv, hmn, amn, hid, aid, bnd = rollout_lmsubs(
            model, pvocab, gd, gen, a.rollouts, port, dev,
            rate=rate, season=season, prefix=prefix, kv=a.kv,
            gid=str(gid), plut=plut, paceval=pmap.get(str(gid), 0.0))
        if prefix is None:      # minutes MAE only for full-game rollouts
            act_h = np.zeros(len(hid)); act_a = np.zeros(len(aid))
            for r_ in g.itertuples(index=False):
                for p in r_.home_lineup:
                    if str(p) in hid: act_h[hid.index(str(p))] += r_.duration_sec
                for p in r_.away_lineup:
                    if str(p) in aid: act_a[aid.index(str(p))] += r_.duration_sec
            mmae = (np.abs(hmn.mean(0)[:len(hid)] - act_h).mean()
                    + np.abs(amn.mean(0)[:len(aid)] - act_a).mean()) / 2 / 60.0
        else:
            mmae = np.nan
        if rows and len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(
                f"{os.path.dirname(a.ckpt) or '.'}/gen_partial.csv", index=False)
        rows.append({"game_id": gid, "mmae": mmae,
                     "pm": hpv.mean() - apv.mean(), "pt": hpv.mean() + apv.mean(),
                     # win probability is the share of rollouts the home side wins
                     "wp": float((hpv > apv).mean()),
                     "mstd": (hpv - apv).std(), "tstd": (hpv + apv).std(),
                     # per-rollout margins, so interval coverage is checkable
                     # without assuming the sampled spread is normal
                     "margins": " ".join(f"{v:.0f}" for v in (hpv - apv)),
                     "totals": " ".join(f"{v:.0f}" for v in (hpv + apv)),
                     "ot_share": float((bnd > 2880.0).mean()),
                     "hp": hpv.mean(), "ap": apv.mean(),
                     "am": float(g.home_pts.sum() - g.away_pts.sum()),
                     "at": float(g.home_pts.sum() + g.away_pts.sum())})
        if (n_g + 1) % 10 == 0:
            pm = np.array([r["pm"] for r in rows]); am_ = np.array([r["am"] for r in rows])
            pt_ = np.array([r["pt"] for r in rows]); at_ = np.array([r["at"] for r in rows])
            mc = np.corrcoef(pm, am_)[0, 1] if pm.std() > 0 and am_.std() > 0 else float("nan")
            tc = np.corrcoef(pt_, at_)[0, 1] if pt_.std() > 0 and at_.std() > 0 else float("nan")
            mm = np.nanmean([r["mmae"] for r in rows])
            mtxt = f" | min MAE {mm:.2f}" if not np.isnan(mm) else ""
            print(f"  {n_g+1} games ({time.time()-t0:.0f}s) | CUM margin corr "
                  f"{mc:.3f} acc {((pm>0)==(am_>0)).mean():.1%} "
                  f"MAE {np.abs(pm-am_).mean():.1f} | total corr {tc:.3f} "
                  f"bias {(pt_-at_).mean():+.1f}{mtxt}", flush=True)

    d = pd.DataFrame(rows)
    am, at_ = d["am"], d["at"]
    print(f"\nSANITY: pts/side H {d.hp.mean():.1f} A {d.ap.mean():.1f}")
    print(f"WIN PROB: mean home {d.wp.mean():.1%} | "
          f"acc {((d.wp > 0.5) == (am > 0)).mean():.1%}")
    print(f"MARGIN: corr {np.corrcoef(d.pm, am)[0,1]:.3f} "
          f"acc {(np.sign(d.pm) == np.sign(am)).mean():.1%} "
          f"MAE {(d.pm - am).abs().mean():.1f}")
    print(f"TOTAL:  corr {np.corrcoef(d.pt, at_)[0,1]:.3f} "
          f"MAE {(d.pt - at_).abs().mean():.1f} bias {d.pt.mean() - at_.mean():+.1f}")
    if d.mmae.notna().any():
        print(f"MINUTES: per-player MAE {d.mmae.mean():.2f} min")
    name = f"gen_{a.split}_{a.state}_lmsubs.csv"
    d.to_csv(f"{os.path.dirname(a.ckpt) or '.'}/{name}", index=False)
    print(f"saved {name} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
