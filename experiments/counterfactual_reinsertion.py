#!/usr/bin/env python3
"""Star re-insertion natural experiment (pre-registered in GATES.md 2026-08-27).

Arm A: the production forecast, roster exactly as loaded (star genuinely absent).
Arm B: star re-inserted into roster + starting five, fed their real pre-absence
       card row and recomputed last-5 minutes. Everything else identical.
Paired, same seed, K rollouts each. Streams per-case progress + partial CSV.
"""
import argparse, copy, json, os, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import generate_bball_lm as G
import loader as l24
tb = G._tb

ap = argparse.ArgumentParser()
ap.add_argument("--folds", default="1,2,3")
ap.add_argument("--data-dir", default="../sloan_hf_dataset")
ap.add_argument("--ckpt-pattern", default="ckpt_probe_f{fold}",
                help="per-fold checkpoint dir containing best_model.pt + config.json")
ap.add_argument("--rollouts", type=int, default=200)
ap.add_argument("--limit", type=int, default=0, help="cases per fold (0=all)")
ap.add_argument("--gap-max", type=float, default=14.0)
ap.add_argument("--out", default="reinsert_results.csv")
a = ap.parse_args()

cases = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "reinsert_cases.csv"))
cases["game_id"] = cases.game_id.astype(str).str.zfill(10)
cases["prev_gid"] = cases.prev_gid.astype(str).str.zfill(10)
cases["player_id"] = cases.player_id.astype(str)
cases = cases[cases.gap_days <= a.gap_max]

# resume: skip cases already in the output CSV (keyed game_id+player_id)
done_keys = set()
rows = []
if os.path.exists(a.out):
    prev = pd.read_csv(a.out)
    prev["game_id"] = prev.game_id.astype(str).str.zfill(10)
    rows = prev.to_dict("records")
    done_keys = {(r["game_id"], str(r["player_id"])) for r in rows}
    print(f"RESUME: {len(done_keys)} cases already done in {a.out}", flush=True)

pl = pd.read_parquet(os.path.join(a.data_dir, "prior_minutes.parquet"))
plut_base = {(str(r.player_id), str(r.game_id)): float(r.prior_min) for r in pl.itertuples(index=False)}
# same pace file the checkpoints trained on; both arms share the value, so
# paired shifts are invariant to this input in any case
pz = pd.read_parquet(os.path.join(a.data_dir, "team_pace.parquet")).set_index("game_id").pace_z
gm = pd.read_parquet(os.path.join(a.data_dir, "game_meta.parquet"))
gm["game_id"] = gm.game_id.astype(str).str.zfill(10)
dev = torch.device("cpu")
t_start = pd.Timestamp.now()

for fold in [int(x) for x in a.folds.split(",")]:
    cfg = json.load(open(f"{a.ckpt_pattern.format(fold=fold)}/config.json"))
    ck = torch.load(f"{a.ckpt_pattern.format(fold=fold)}/best_model.pt", map_location="cpu", weights_only=False)
    blob = l24.load_corpus(a.data_dir, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", f"fold{fold}_splits.json"),
                           fix_sub_boundary=cfg["fix_sub_boundary"])
    games, vocab = blob["games"], blob["vocab"]
    G.init_tables(vocab)
    port = cfg["port"]; rlut = ck["rlut"]; pvocab = ck["pvocab"]
    tb.MAXLEN = ck["model"]["pos.weight"].shape[0]
    G.MAXLEN = tb.MAXLEN   # the rollout buffer reads the module-level copy
    model = tb.BasketballLM(len(vocab), len(pvocab), d=cfg["d"], nlayers=cfg["layers"], port=port,
                            budget_ch=cfg["budget_ch"], pace_ch=cfg["pace_ch"],
                            period_ch=cfg["period_ch"], mtp=cfg["mtp_w"] > 0, tt=cfg["tt_w"] > 0)
    model.load_state_dict(ck["model"], strict=True); model.eval()
    sub = cases[cases.fold == fold]
    if a.limit: sub = sub.head(a.limit)
    print(f"=== fold {fold}: {len(sub)} cases ===", flush=True)
    for n, c in enumerate(sub.itertuples(index=False)):
        if (c.game_id, str(c.player_id)) in done_keys:
            continue
        g = games.get(c.game_id)
        if g is None: continue
        g = dict(g); g["gid"] = c.game_id
        star = c.player_id
        home = str(g["hro"][0]) if False else None
        # which side is the star's team? the side whose roster he is NOT on but
        # whose opponent he is not either -> use scout's team vs meta home/away
        # (cases carry team/opp; identify by checking which roster lacks him and
        # matching against prev-game co-appearance is overkill: use both sides)
        star_v = pvocab.get(star, 0)
        if star_v == 0:
            print(f"  skip {c.player}: not in pvocab", flush=True); continue
        card = rlut.get((star, c.prev_gid))
        if not card or not any(abs(x) > 1e-9 for x in card):
            print(f"  skip {c.player}: no non-zero pre-absence card row", flush=True); continue
        # side assignment: the star's team is the one whose available roster we modify.
        # Determine by comparing team code in meta.
        meta = blob.get("meta")
        side = c.side if hasattr(c, "side") else None
        # fall back: put him on the side whose starters have lower total prior minutes? no --
        # use game_meta lookup
        side_is_home = (c.team == g.get("home_team")) if g.get("home_team") else None
        if side_is_home is None:
            row = gm[gm.game_id == c.game_id]
            if not len(row): continue
            side_is_home = (row.home_team.iloc[0] == c.team)
        rk, fk = ("hro", "h5") if side_is_home else ("aro", "a5")
        # ---- Arm B roster construction ----
        g2 = copy.deepcopy(g)
        ros = [str(p) for p in g2[rk]]
        if star in ros:
            print(f"  skip {c.player}: already on roster", flush=True); continue
        starters = [str(p) for p in g2[fk][0]]
        pri_of = {p: plut_base.get((p, c.game_id), 0.0) for p in ros}
        # displace the fill-in starter: lowest recomputed prior among actual starters
        drop_starter = min(starters, key=lambda p: pri_of.get(p, 0.0))
        ros_b = [star] + [p for p in ros]
        if len(ros_b) > 17:   # explicit lowest-prior non-starter drop (never rely on order())
            cand_drop = [p for p in ros_b if p not in starters and p != star]
            if cand_drop:
                ros_b.remove(min(cand_drop, key=lambda p: pri_of.get(p, 0.0)))
            else:
                ros_b = ros_b[:17]
        g2[rk] = [int(p) for p in ros_b]
        g2[fk] = copy.deepcopy(g[fk])
        g2[fk][0] = [int(star) if str(p) == drop_starter else p for p in g[fk][0]]
        # star inputs: real pre-absence card + recomputed last-5 minutes
        rlut_b = dict(rlut); rlut_b[(star, c.game_id)] = card
        plut_b = dict(plut_base); plut_b[(star, c.game_id)] = float(c.pri_min)
        def mkrate(RL):
            return lambda p, s, _g=c.game_id: (tuple(RL.get((str(p), _g)) or RL.get((str(p), s))
                                                     or RL.get((str(p), "*")) or (0.,) * port)
                                               + (0.,) * port)[:port]
        out = {}
        for arm, gg, RL, PL in (("A", g, rlut, plut_base), ("B", g2, rlut_b, plut_b)):
            gen = torch.Generator(device="cpu").manual_seed(7)
            hpv, apv, hmn, amn, hid, aid, bnd = G.rollout_lmsubs(
                model, pvocab, gg, gen, a.rollouts, port, dev, rate=mkrate(RL),
                season=c.game_id[3:5], prefix=None, kv=True, gid=c.game_id,
                plut=PL, paceval=float(pz.get(c.game_id, 0.0)))
            marg = (hpv - apv) if side_is_home else (apv - hpv)   # oriented to star's team
            ids = hid if side_is_home else aid
            mn = hmn if side_is_home else amn
            smin = float(mn.mean(0)[ids.index(star)] / 60.0) if star in ids else 0.0
            out[arm] = dict(mean=float(marg.mean()), q=np.percentile(marg, [5, 10, 25, 50, 75, 90, 95]),
                            teammin=float(mn.mean(0)[:len(ids)].sum() / 60.0), starmin=smin)
        shift = out["B"]["mean"] - out["A"]["mean"]
        # the two validity gates reinsertion_analysis.py reads:
        # `strict_tripwire` — the pre-registered gate: team sim minutes within
        #   240±2.5 in BOTH arms, star entering with >15 simulated minutes.
        # `valid` — the primary gate: same >15-minute entry, but paired-arm team
        #   minutes consistent within 2 of EACH OTHER (the absolute 240 window
        #   interacts with fold 2's known sequence-length truncation).
        strict = (abs(out["A"]["teammin"] - 240) <= 2.5 and abs(out["B"]["teammin"] - 240) <= 2.5
                  and out["B"]["starmin"] > 15)
        valid = (out["B"]["starmin"] > 15
                 and abs(out["B"]["teammin"] - out["A"]["teammin"]) < 2.0)
        rows.append(dict(game_id=c.game_id, fold=fold, player=c.player, player_id=star,
                         team=c.team, opp=c.opp, date=c.date, tier=c.tier, ppg=c.ppg,
                         pri_min=c.pri_min, gap_days=c.gap_days,
                         mean_A=out["A"]["mean"], mean_B=out["B"]["mean"], shift=shift,
                         starmin_B=out["B"]["starmin"], teammin_A=out["A"]["teammin"],
                         teammin_B=out["B"]["teammin"], actual=c.team_margin,
                         valid=valid, strict_tripwire=strict,
                         qA=" ".join(f"{x:.1f}" for x in out["A"]["q"]),
                         qB=" ".join(f"{x:.1f}" for x in out["B"]["q"])))
        pd.DataFrame(rows).to_csv(a.out, index=False)
        el = (pd.Timestamp.now() - t_start).total_seconds()
        print(f"  [{n+1}/{len(sub)}] f{fold} {c.player:22s} ({c.pri_min:.1f}m, gap {c.gap_days:.0f}d) "
              f"A {out['A']['mean']:+.2f} B {out['B']['mean']:+.2f} shift {shift:+.2f} "
              f"starmin {out['B']['starmin']:.1f} {'' if strict else 'TRIPWIRE-FAIL'} [{el:.0f}s]", flush=True)
print(f"\nwrote {a.out} ({len(rows)} cases)")
