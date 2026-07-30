#!/usr/bin/env python3
"""Simulate games with the Basketball LM (batched Monte-Carlo rollouts).

Modes: --state pre (from tip-off) | half (real first half as prompt).
Vocabulary and dimensions are auto-detected from the checkpoint.
"""
import argparse, os, time
from loader import load_corpus, load_lineup_timeline
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_bball_lm import BasketballLM, DT_MID
import train_bball_lm as _tb

DATA = "data"
CKPT = Path("ckpt/best_model.pt")
DT_MEANS = np.asarray(DT_MID)
VIDX = {}; VOCAB = None
NV = 0; PTS_H = PTS_A = IS_FT = None
MAKES_H = MAKES_A = MISSES_H = MISSES_A = None
MAXLEN = _tb.MAXLEN


def init_tables(vocab):
    """Vocab-agnostic token tables built from the CHECKPOINT's vocab
    (supports 36- and 60-token vocabularies)."""
    global VIDX, VOCAB, NV, PTS_H, PTS_A, IS_FT
    global MAKES_H, MAKES_A, MISSES_H, MISSES_A
    VOCAB = list(vocab); NV = len(vocab)
    VIDX = {t: i for i, t in enumerate(vocab)}
    PTS_H = np.zeros(NV); PTS_A = np.zeros(NV)
    IS_FT = np.zeros(NV, bool)
    MAKES_H, MAKES_A, MISSES_H, MISSES_A = [], [], [], []
    global SIDE, AST_TOKS
    SIDE = np.array([1 if t.startswith("H_") else (2 if t.startswith("A_") else 0)
                     for t in vocab])
    AST_TOKS = np.array([t.endswith("MAKE_AST") for t in vocab])
    for i, t in enumerate(vocab):
        if "FT_" in t:
            IS_FT[i] = True
        if "MAKE" in t:
            p = 3 if "3PT" in t else (1 if "FT_" in t else 2)
            (PTS_H if t.startswith("H_") else PTS_A)[i] = p
            (MAKES_H if t.startswith("H_") else MAKES_A).append(i)
        elif "MISS" in t:
            (MISSES_H if t.startswith("H_") else MISSES_A).append(i)


def make_rater(ck, pvocab):
    rlut = ck.get("rlut")
    if rlut:
        k0 = next(iter(rlut)); pad = (0.0,) * len(rlut[k0])
        return lambda p, s, g=None: (rlut.get((p, g)) or rlut.get((p, s))
                                     or rlut.get((p, "*")) or pad)
    return lambda p, s, g=None: (0.0,) * 32


@torch.no_grad()
def kv_prefill(model, xseq):
    """Multi-token cache build (the prefix pass): same math as the full
    causal forward, returns per-layer (K,V) caches + last hidden state."""
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
    return caches, x[:, -1:]


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
def rollout_batch(model, pvocab, timeline, gen, K, port, dev, window=0,
                  prefix=None, rate=None, season="*",
                  sd_scale=1.0, kv=False):
    _dump = bool(os.environ.get("DUMP_TOKS"))
    _dumpA, _dumpS = [], []
    ts_arr = np.array([s[0] for s in timeline])
    n_st = len(timeline)
    rate = rate or (lambda p, s: (0.0,) * port)


    L = MAXLEN
    tokB = torch.full((K, L), VIDX["PAD"], dtype=torch.long, device=dev)
    h5B = torch.zeros(K, L, 5, dtype=torch.long, device=dev)
    a5B = torch.zeros(K, L, 5, dtype=torch.long, device=dev)
    hmB = torch.zeros(K, L, 5, device=dev); amB = torch.zeros(K, L, 5, device=dev)
    hrB = torch.zeros(K, L, 5, port, device=dev); arB = torch.zeros(K, L, 5, port, device=dev)
    ckB = torch.zeros(K, L, device=dev); sdB = torch.zeros(K, L, device=dev)
    T = 0
    t = np.zeros(K); hp = np.zeros(K); ap = np.zeros(K)
    done = np.zeros(K, bool); early = np.zeros(K, bool)
    events = np.zeros(K); ftn = np.zeros(K)
    m3 = np.full(K, np.nan)   # margin at Q3 end (lead-erosion diagnostic)
    base = [dict() for _ in range(K)]
    seg_end = np.zeros(K); cur_i = np.zeros(K, int)

    stint_h5 = [torch.tensor([pvocab.get(str(p), 0) for p in s[2]]) for s in timeline]
    stint_a5 = [torch.tensor([pvocab.get(str(p), 0) for p in s[3]]) for s in timeline]
    stint_hr = [torch.tensor([rate(str(p), season) for p in s[2]], dtype=torch.float32)
                for s in timeline]
    stint_ar = [torch.tensor([rate(str(p), season) for p in s[3]], dtype=torch.float32)
                for s in timeline]

    if prefix is None:
        tokB[:, 0] = VIDX["BOS"]
        T = 1
    else:
        P = len(prefix["tok"])
        tokB[:, :P] = torch.tensor(prefix["tok"], dtype=torch.long, device=dev)[None]
        h5B[:, :P] = torch.tensor(prefix["h5"], dtype=torch.long, device=dev)[None]
        a5B[:, :P] = torch.tensor(prefix["a5"], dtype=torch.long, device=dev)[None]
        hmB[:, :P] = torch.tensor(np.array(prefix["hm"]), dtype=torch.float32, device=dev)[None]
        amB[:, :P] = torch.tensor(np.array(prefix["am"]), dtype=torch.float32, device=dev)[None]
        hrp = torch.tensor(np.array([[rate(str(p), season) for p in row]
                                     for row in prefix["h5_raw"]]), dtype=torch.float32)
        arp = torch.tensor(np.array([[rate(str(p), season) for p in row]
                                     for row in prefix["a5_raw"]]), dtype=torch.float32)
        hrB[:, :P] = hrp.to(dev)[None]; arB[:, :P] = arp.to(dev)[None]
        ckB[:, :P] = torch.tensor(prefix["ck"], dtype=torch.float32, device=dev)[None]
        sdB[:, :P] = torch.tensor(prefix["sd"], dtype=torch.float32, device=dev)[None]
        T = P
        t[:] = prefix["t0"]; hp[:] = prefix["h0"]; ap[:] = prefix["a0"]
        seg_end[:] = prefix["t0"]
        cur_i[:] = np.clip(np.searchsorted(ts_arr, prefix["t0"], side="right") - 1,
                           0, n_st - 1)

    caches = None      # kv: built at j==0, or prefilled over a real prefix
    while (~done).any() and T < MAXLEN - 1:
        j = T - 1        # conditioning row for the position being predicted
        idx = np.clip(np.searchsorted(ts_arr, t, side="right") - 1, 0, n_st - 1)
        h5r = torch.empty(K, 5, dtype=torch.long); a5r = torch.empty(K, 5, dtype=torch.long)
        hrr = torch.empty(K, 5, port); arr_ = torch.empty(K, 5, port)
        hmr = torch.zeros(K, 5); amr = torch.zeros(K, 5)
        for k in range(K):
            i = int(idx[k])
            if i != cur_i[k] and not done[k]:
                pi = int(cur_i[k])
                for p in np.concatenate([timeline[pi][2], timeline[pi][3]]):
                    base[k][p] = base[k].get(p, 0.0) + max(
                        min(t[k], timeline[pi][0] + timeline[pi][1]) - seg_end[k], 0)
                cur_i[k] = i; seg_end[k] = max(timeline[i][0], seg_end[k])
            i = int(cur_i[k])
            h5r[k] = stint_h5[i]; a5r[k] = stint_a5[i]
            hrr[k] = stint_hr[i]; arr_[k] = stint_ar[i]
            on = max(t[k] - max(timeline[i][0], seg_end[k]), 0.0)
            hmr[k] = torch.tensor([(base[k].get(p, 0.0) + on) / 2880.0
                                   for p in timeline[i][2]])
            amr[k] = torch.tensor([(base[k].get(p, 0.0) + on) / 2880.0
                                   for p in timeline[i][3]])
        h5B[:, j] = h5r.to(dev); a5B[:, j] = a5r.to(dev)
        hrB[:, j] = hrr.to(dev); arB[:, j] = arr_.to(dev)
        hmB[:, j] = hmr.to(dev); amB[:, j] = amr.to(dev)
        ckB[:, j] = torch.tensor(np.minimum(t, 3600.0) / 2880.0,
                                 dtype=torch.float32, device=dev)
        sdB[:, j] = torch.tensor(np.clip((hp - ap) / 20.0, -2, 2) * sd_scale,
                                 dtype=torch.float32, device=dev)

        if kv:
            def emb_range(lo_, hi_):
                ph_ = model.pfuse(torch.cat([model.pemb(h5B[:, lo_:hi_]),
                                             hmB[:, lo_:hi_].unsqueeze(-1),
                                             hrB[:, lo_:hi_]], -1))
                pa_ = model.pfuse(torch.cat([model.pemb(a5B[:, lo_:hi_]),
                                             amB[:, lo_:hi_].unsqueeze(-1),
                                             arB[:, lo_:hi_]], -1))
                lu_ = model.lineup(torch.cat([ph_.sum(2), pa_.sum(2)], -1))
                pos_ = model.pos.weight[
                    torch.arange(lo_, hi_, device=dev).clamp(max=MAXLEN - 1)]
                return (model.tok(tokB[:, lo_:hi_]) + pos_[None] + lu_
                        + model.ctx(torch.stack([ckB[:, lo_:hi_],
                                                 sdB[:, lo_:hi_]], -1)))
            if j == 0:
                caches = [None] * len(model.tr.layers)
            elif caches is None:      # prefix run: build the cache once
                caches, _ = kv_prefill(model, emb_range(0, j))
            hid = kv_step(model, caches, emb_range(j, j + 1))
            lg = model.head(hid)[:, 0].float().cpu()
            dl = model.dt_head(hid)[:, 0].float().cpu()
        else:
            lo = max(0, T - window) if window else 0
            logits, dtl, _actlg, *_ = model(
                tokB[:, lo:T], h5B[:, lo:T], a5B[:, lo:T], ckB[:, lo:T],
                hmB[:, lo:T], amB[:, lo:T], sdB[:, lo:T], hrB[:, lo:T], arB[:, lo:T],
                pos_offset=lo)
            lg = logits[:, -1].float().cpu()
            dl = dtl[:, -1].float().cpu()
            if _dump:
                _al = _actlg[:, -1].float().cpu()
                _as = model.last_ast[:, -1].float().cpu()
        pt = torch.softmax(lg, -1)
        pt[:, VIDX["PAD"]] = 0; pt[:, VIDX["BOS"]] = 0
        nxt = torch.multinomial(pt / pt.sum(1, keepdim=True), 1,
                                generator=gen)[:, 0].numpy()
        dtb = torch.multinomial(torch.softmax(dl, -1), 1, generator=gen)[:, 0].numpy()
        if _dump:
            K_ = len(nxt)
            actor_id = np.zeros(K_, np.int64); assist_id = np.zeros(K_, np.int64)
            sides = SIDE[nxt]
            for k in range(K_):
                if sides[k] == 0 or not kv and False: continue
                if sides[k] == 0: continue
                sl = slice(0, 5) if sides[k] == 1 else slice(5, 10)
                pa_ = torch.softmax(_al[k, sl], -1)
                slot = int(torch.multinomial(pa_, 1, generator=gen))
                five = (h5B if sides[k] == 1 else a5B)[k, j]
                actor_id[k] = int(five[slot])
                if AST_TOKS[nxt[k]]:
                    aw = _as[k, sl].clone(); aw[slot] = -1e9
                    ps_ = torch.softmax(aw, -1)
                    aslot = int(torch.multinomial(ps_, 1, generator=gen))
                    assist_id[k] = int(five[aslot])
            _dumpA.append(actor_id); _dumpS.append(assist_id)

        nxt[done] = VIDX["PAD"]
        alive = ~done
        newly_eos = alive & (nxt == VIDX["EOS"])
        early |= newly_eos & (t < 2700)
        act = alive & ~newly_eos
        hp[act] += PTS_H[nxt[act]]; ap[act] += PTS_A[nxt[act]]
        events[act] += 1; ftn[act] += IS_FT[nxt[act]]
        t[act] += DT_MEANS[dtb[act]]
        cross3 = np.isnan(m3) & (t > 2160.0)
        m3[cross3] = (hp - ap)[cross3]
        done |= newly_eos | (t >= 2880.0)
        tokB[:, T] = torch.tensor(nxt, dtype=torch.long, device=dev)
        T += 1
    m3[np.isnan(m3)] = (hp - ap)[np.isnan(m3)]
    if _dump:
        np.savez(os.environ["DUMP_TOKS"],
                 toks=tokB[:, :T].cpu().numpy(),
                 actors=np.stack(_dumpA, 1) if _dumpA else np.zeros((0, 0)),
                 assists=np.stack(_dumpS, 1) if _dumpS else np.zeros((0, 0)))
    return hp, ap, events, ftn, early, m3




@torch.no_grad()
def rollout_lmsubs(model, pvocab, game, gen, K, port, dev, window=0, rate=None, season="*",
                   exit_temp=1.0, ent_temp=1.0, prefix=None, kv=False):
    """The LM drives its own rotations. Lineups start at the real starters
    (or, with a prefix, the real halftime lineup); every sampled SUB token
    triggers the exit pointer (who comes off) and the entrant pointer (who
    checks in). Returns scores + per-player simulated seconds."""
    R = 17    # available-roster width
    _dump = False   # debug hook used only by rollout_batch
    # starters first
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

    # starters = first row of the tokenized game (roster indices)
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

    t = np.zeros(K); hp = np.zeros(K); ap = np.zeros(K)
    done = np.zeros(K, bool)
    hmin = np.zeros((K, R)); amin = np.zeros((K, R))
    hon = np.repeat(hon0[None], K, 0); aon = np.repeat(aon0[None], K, 0)
    T = 1
    if prefix is not None:
        # halftime start: condition on the REAL first half (observed), then
        # the LM drives every H2 rotation itself — no oracle timeline
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
                                  dtype=torch.float32)
        arB[:, :P] = torch.tensor(np.array([[rate(str(p), season) for p in row]
                                            for row in prefix["a5_raw"]]),
                                  dtype=torch.float32)
        ckB[:, :P] = torch.tensor(prefix["ck"], dtype=torch.float32, device=dev)[None]
        sdB[:, :P] = torch.tensor(prefix["sd"], dtype=torch.float32, device=dev)[None]
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
                ph_ = model.pfuse(torch.cat([model.pemb(h5B[:, lo_:hi_]),
                                             hmB[:, lo_:hi_].unsqueeze(-1),
                                             hrB[:, lo_:hi_]], -1))
                pa_ = model.pfuse(torch.cat([model.pemb(a5B[:, lo_:hi_]),
                                             amB[:, lo_:hi_].unsqueeze(-1),
                                             arB[:, lo_:hi_]], -1))
                lu_ = model.lineup(torch.cat([ph_.sum(2), pa_.sum(2)], -1))
                pos_ = model.pos.weight[
                    torch.arange(lo_, hi_, device=dev).clamp(max=MAXLEN - 1)]
                return (model.tok(tokB[:, lo_:hi_]) + pos_[None] + lu_
                        + model.ctx(torch.stack([ckB[:, lo_:hi_],
                                                 sdB[:, lo_:hi_]], -1))), ph_, pa_
            if j == 0:
                caches = [None] * len(model.tr.layers)
                xr, ph1, pa1 = emb_range(0, 1)
            elif caches is None:      # prefix run: build the cache once
                xall, _, _ = emb_range(0, j)
                caches, _ = kv_prefill(model, xall)
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
            def bench1(ro, r13, m13_j, on_j):
                pv = model.pfuse(torch.cat([model.pemb(ro),
                                            m13_j.unsqueeze(-1), r13], -1))
                lg_ = torch.einsum("kd,krd->kr", q1, pv)
                return lg_.masked_fill((ro == 0) | (on_j > 0.5), -1e9)
            act_lg_full = actl_1.float().cpu()
            eh_full = bench1(hroB, hr13B, hm13B[:, j], honB[:, j]).float().cpu()
            ea_full = bench1(aroB, ar13B, am13B[:, j], aonB[:, j]).float().cpu()
        else:
            lo = max(0, T - window) if window else 0
            logits, dtl, actl, enth, enta = model(
                tokB[:, lo:T], h5B[:, lo:T], a5B[:, lo:T], ckB[:, lo:T],
                hmB[:, lo:T], amB[:, lo:T], sdB[:, lo:T], hrB[:, lo:T], arB[:, lo:T],
                hroB, aroB, hr13B, ar13B, hm13B[:, lo:T], am13B[:, lo:T],
                honB[:, lo:T], aonB[:, lo:T], pos_offset=lo)
            lg = logits[:, -1].float().cpu()
            dl = dtl[:, -1].float().cpu()
        pt = torch.softmax(lg, -1)
        pt[:, VIDX["PAD"]] = 0; pt[:, VIDX["BOS"]] = 0
        nxt = torch.multinomial(pt / pt.sum(1, keepdim=True), 1,
                                generator=gen)[:, 0].numpy()
        dtb = torch.multinomial(torch.softmax(dl, -1), 1, generator=gen)[:, 0].numpy()
        if _dump:
            K_ = len(nxt)
            actor_id = np.zeros(K_, np.int64); assist_id = np.zeros(K_, np.int64)
            sides = SIDE[nxt]
            for k in range(K_):
                if sides[k] == 0 or not kv and False: continue
                if sides[k] == 0: continue
                sl = slice(0, 5) if sides[k] == 1 else slice(5, 10)
                pa_ = torch.softmax(_al[k, sl], -1)
                slot = int(torch.multinomial(pa_, 1, generator=gen))
                five = (h5B if sides[k] == 1 else a5B)[k, j]
                actor_id[k] = int(five[slot])
                if AST_TOKS[nxt[k]]:
                    aw = _as[k, sl].clone(); aw[slot] = -1e9
                    ps_ = torch.softmax(aw, -1)
                    aslot = int(torch.multinomial(ps_, 1, generator=gen))
                    assist_id[k] = int(five[aslot])
            _dumpA.append(actor_id); _dumpS.append(assist_id)
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
                # sample rotations (temperature-controlled)
                ex_local = int(torch.multinomial(
                    torch.softmax(act_lg[k][sl] / exit_temp, -1), 1,
                    generator=gen))
                ex_idx = onk[ex_local] if ex_local < len(onk) else onk[0]
                elg = (eh_lg if home_side else ea_lg)[k].clone()
                elg[torch.tensor(on, dtype=torch.bool)] = -1e9
                nlim = nh if home_side else na
                elg[nlim:] = -1e9
                in_idx = int(torch.multinomial(torch.softmax(elg / ent_temp, -1),
                                               1, generator=gen))
                if not on[in_idx]:
                    on[ex_idx] = False; on[in_idx] = True
                (price.clear if False else (lambda: None))()
        hp[act] += PTS_H[nxt[act]]; ap[act] += PTS_A[nxt[act]]
        dt_s = DT_MEANS[dtb]
        for k in np.where(act)[0]:
            hmin[k, hon[k]] += dt_s[k]; amin[k, aon[k]] += dt_s[k]
        t[act] += dt_s[act]
        done |= newly_eos | (t >= 2880.0)
        tokB[:, T] = torch.tensor(nxt, dtype=torch.long, device=dev)
        T += 1
    # id orderings the minute columns are indexed by
    return hp, ap, hmin, amin, hro_ids, aro_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data-dir", default="../sloan_hf_dataset")
    ap.add_argument("--rollouts", type=int, default=24)
    ap.add_argument("--state", default="pre", choices=["pre", "half"])
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lm-subs", action="store_true",
                    help="the LM drives its own rotations")
    ap.add_argument("--window", type=int, default=0,
                    help="attend to only the last W events (0 = full)")
    ap.add_argument("--split", default="test",
                    choices=["test", "val", "ttest"])
    ap.add_argument("--kv", action="store_true",
                    help="KV-cached generation (exact; pre-state only for now)")
    ap.add_argument("--gids-file", default=None, help="restrict to game_ids listed in file")
    ap.add_argument("--sd-scale", type=float, default=1.0,
                    help="scale the score-diff input at "
                         "generation (0 = model can't see the lead)")
    a = ap.parse_args()
    t0 = time.time()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    global DT_MEANS
    DT_MEANS = np.asarray(ck.get("dt_means", DT_MID))
    pvocab = ck["pvocab"]
    port = ck["model"]["pfuse.weight"].shape[1] - 33
    dev = torch.device(a.device)
    init_tables(ck["vocab"])
    _sd = ck["model"]
    _tb.MAXLEN = _sd["pos.weight"].shape[0]
    global MAXLEN; MAXLEN = _tb.MAXLEN
    _d = _sd["tok.weight"].shape[1]
    _nl = 1 + max(int(k.split(".")[2]) for k in _sd if k.startswith("tr.layers."))
    model = BasketballLM(len(ck["vocab"]), len(pvocab), d=_d, nlayers=_nl,
                         port=port).to(dev)
    model.load_state_dict(ck["model"], strict=False); model.eval()
    rate0 = make_rater(ck, pvocab)
    gen = torch.Generator().manual_seed(a.seed)


    blob = None
    if a.state == "half" or a.lm_subs:
        blob = load_corpus(a.data_dir)["games"]

    stints = load_lineup_timeline(a.data_dir, a.split)
    if a.gids_file:
        keep = set(open(a.gids_file).read().split())
        stints = stints[stints.game_id.isin(keep)]
    if a.state == "half" and not a.lm_subs:
        print("WARNING: oracle-timeline mode — simulated lineups replay the "
              "REAL second-half rotation (future information). Diagnostic "
              "only; the paper's halftime numbers use --lm-subs.", flush=True)
    rows = []
    all_m3, all_mf = [], []   # per-rollout Q3/final margins (lead erosion)
    for n_g, (gid, g) in enumerate(stints.groupby("game_id", sort=False)):
        if n_g >= a.games: break
        g = g.sort_values("t_start_sec")
        season = gid[3:5]
        # per-game card lookup (season/career fallback)
        rate = lambda p, s, _g=str(gid): (tuple(rate0(p, s, _g))
                                          + (0.0,) * port)[:port]
        timeline = []
        for r in g.itertuples(index=False):
            h5 = np.array([str(p) for p in r.home_lineup])
            a5 = np.array([str(p) for p in r.away_lineup])
            rh = ra = 0.0
            timeline.append((r.t_start_sec, r.duration_sec, h5, a5,
                             1, rh, ra))
        prefix = None
        if blob is not None:
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
        if a.lm_subs:
            gd = blob.get(gid)
            if gd is None: continue
            hpv, apv, hmn, amn, hid, aid = rollout_lmsubs(
                model, pvocab, gd, gen, a.rollouts, port, dev, window=a.window,
                rate=rate, season=season, prefix=prefix, kv=a.kv)
            if prefix is None:      # full-game minutes gate (pregame only)
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
            ev = np.zeros(1); ft = np.zeros(1); early = np.zeros(1)
        else:
            hpv, apv, ev, ft, early, m3s = rollout_batch(
                model, pvocab, timeline, gen, a.rollouts, port, dev,
                window=a.window, prefix=prefix,
                rate=rate, season=season, sd_scale=a.sd_scale, kv=a.kv)
            all_m3.extend(m3s.tolist()); all_mf.extend((hpv - apv).tolist())
            mmae = np.nan
        if rows and len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(
                f"{os.path.dirname(a.ckpt)}/gen_partial.csv", index=False)
        rows.append({"game_id": gid, "mmae": mmae,
                     "pm": hpv.mean() - apv.mean(), "pt": hpv.mean() + apv.mean(),
                     "mstd": (hpv - apv).std(), "tstd": (hpv + apv).std(),
                     "hp": hpv.mean(), "ap": apv.mean(), "ev": ev.mean(),
                     "ft_share": ft.sum() / max(ev.sum(), 1),
                     "eos": early.mean(),
                     "am": float(g.home_pts.sum() - g.away_pts.sum()),
                     "at": float(g.home_pts.sum() + g.away_pts.sum())})
        if (n_g + 1) % 10 == 0:
            # progress line
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
    print(f"\nSANITY: pts/side H {d.hp.mean():.1f} A {d.ap.mean():.1f} (real ~112) | "
          f"events/gm {d.ev.mean():.0f} (real ~360) | FT {d.ft_share.mean():.1%} | "
          f"early-EOS {d.eos.mean():.1%}")
    print(f"MARGIN: corr {np.corrcoef(d.pm, am)[0,1]:.3f} "
          f"acc {(np.sign(d.pm) == np.sign(am)).mean():.1%} "
          f"MAE {(d.pm - am).abs().mean():.1f}   (E oracle: 0.454/66%/10.6)")
    print(f"TOTAL:  corr {np.corrcoef(d.pt, at_)[0,1]:.3f} "
          f"MAE {(d.pt - at_).abs().mean():.1f} bias {d.pt.mean() - at_.mean():+.1f}")
    if a.lm_subs:
        print(f"MINUTES: per-player MAE {d.mmae.mean():.2f} min   "
              f"(naive last-5 baseline: 5.47)")
    if len(all_m3) > 20 and np.std(all_m3) > 0:
        print(f"LEAD EROSION: slope(final~Q3margin) "
              f"{np.polyfit(all_m3, all_mf, 1)[0]:.3f}   "
              f"(real ~0.95; sim baseline 0.79) | sd_scale {a.sd_scale}")
    suffix = ""
    name = f"gen_{a.split}_{a.state}{suffix}{'_lmsubs' if a.lm_subs else ''}.csv"
    d.to_csv(f"{os.path.dirname(a.ckpt)}/{name}", index=False)
    print(f"saved {name} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
