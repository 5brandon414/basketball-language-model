#!/usr/bin/env python3
"""Paper tables from three fold runs: fold-averaged Table 2 rows, pooled
Table 1, raw coverage, and fold-stratified paired-bootstrap CIs.

Inputs are the per-fold dumps the two table scripts write (fold_margins.csv,
fold_nextevent.npz; rename per fold) plus generate_bball_lm.py's rollout
CSVs. All three folds are required, in fold order:

  python3 experiments/assemble_final_tables.py \
      --margins f1_margins.csv f2_margins.csv f3_margins.csv \
      --nextevent f1_nextevent.npz f2_nextevent.npz f3_nextevent.npz \
      --gen-half f1_half.csv f2_half.csv f3_half.csv \
      --gen-pre f1_pre.csv f2_pre.csv f3_pre.csv

The bootstrap is seeded; reruns reproduce the printed CIs exactly.
"""
import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--margins", nargs=3, required=True,
                help="three fold_margins.csv dumps from table2_margins.py, fold order")
ap.add_argument("--nextevent", nargs=3, default=None,
                help="three fold_nextevent.npz dumps from table1_nextevent.py")
ap.add_argument("--gen-half", nargs=3, default=None,
                help="three gen_all_half_lmsubs.csv from generate_bball_lm.py --state half --lm-subs")
ap.add_argument("--gen-pre", nargs=3, default=None,
                help="three gen_all_pre_lmsubs.csv from generate_bball_lm.py --state pre --lm-subs")
a = ap.parse_args()

rng = np.random.default_rng(0)
FOLDS = [1, 2, 3]


def strat_boot_corrdiff(dfs, ca, cb, y="margin", n=4000):
    """fold-stratified paired bootstrap of the FOLD-AVERAGE corr difference
    (matches the paper's 'reported numbers average the three folds')."""
    dl = []
    for _ in range(n):
        ds = []
        for d in dfs:
            s = d.sample(len(d), replace=True, random_state=rng)
            ds.append(np.corrcoef(s[ca], s[y])[0, 1]
                      - np.corrcoef(s[cb], s[y])[0, 1])
        dl.append(np.mean(ds))
    lo, hi = np.percentile(dl, [2.5, 97.5])
    sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
    print(f"    {ca}-{cb} fold-avg corr diff 95%CI[{lo:+.3f},{hi:+.3f}] {sig}")


def fold_avg_line(name, dfs, col, y="margin"):
    cs, accs, maes = [], [], []
    for d in dfs:
        p, yy = d[col].values, d[y].values
        cs.append(np.corrcoef(p, yy)[0, 1])
        accs.append((np.sign(p) == np.sign(yy))[yy != 0].mean())
        maes.append(np.abs(p - yy).mean())
    print(f"  {name:14s} fold-avg corr {np.mean(cs):.3f} | acc "
          f"{np.mean(accs):.1%} | MAE {np.mean(maes):.2f}  "
          f"(folds: {' '.join(f'{c:.3f}' for c in cs)})")


# ---------------- margins: readout table ----------------
tests = []
for k, path in zip(FOLDS, a.margins):
    d = pd.read_csv(path)
    if "split" in d.columns:
        d = d[d.split == "test"].copy()
    d["fold"] = k
    tests.append(d)
pool = pd.concat(tests)
print(f"=== PREGAME (readout head vs baselines), fold-avg over n={len(pool)} ===")
have_lm = "lm_pre" in pool.columns and pool.lm_pre.notna().any()
cols = [("elo", "Elo"), ("gb_pre", "GB trees")] + ([("lm_pre", "LM readout")] if have_lm else [])
for col, nm in cols:
    fold_avg_line(nm, tests, col)
if have_lm:
    strat_boot_corrdiff(tests, "lm_pre", "elo")
    strat_boot_corrdiff(tests, "lm_pre", "gb_pre")
else:
    print("  (lm_pre column absent: dumps came from a baselines-only run)")

print(f"\n=== HALFTIME (baselines; LM sim separate), fold-avg n={len(pool)} ===")
for col, nm in (("lead", "halftime-lead"), ("gb_half", "GB trees")):
    fold_avg_line(nm, tests, col)

# ---------------- sims from gen CSVs ----------------
for paths, label, state in ((a.gen_half, "HALFTIME SIM", "half"),
                            (a.gen_pre, "PREGAME SIM", "pre")):
    if not paths:
        continue
    frames = []
    for k, f in zip(FOLDS, paths):
        g = pd.read_csv(f)
        g["fold"] = k
        frames.append(g)
    g = pd.concat(frames)
    print(f"\n=== {label} (LM generation), fold-avg over n={len(g)} ===")
    # fold-average is the reported convention (same as the baseline rows);
    # pooled is printed alongside because the two differ in the 3rd decimal
    fold_avg_line("LM sim margin", frames, "pm", y="am")
    fold_avg_line("LM sim total", frames, "pt", y="at")
    print("   pooled for reference: margin "
          f"{np.corrcoef(g['pm'], g['am'])[0, 1]:.4f} | total "
          f"{np.corrcoef(g['pt'], g['at'])[0, 1]:.4f}")
    if state == "half":
        # merge with pooled baselines on the same games for paired tests
        m = g.merge(pool[["gid", "fold", "margin", "lead", "gb_half"]],
                    left_on=["game_id", "fold"], right_on=["gid", "fold"])
        assert len(m) == len(g), f"merge lost games: {len(m)} vs {len(g)}"
        parts = [m[m.fold == k] for k in FOLDS]
        strat_boot_corrdiff(parts, "pm", "lead")
        strat_boot_corrdiff(parts, "pm", "gb_half")
    # raw normal-interval coverage from rollout std
    print("  raw coverage (normal intervals from rollout std):")
    for lvl, z in ((50, 0.674), (80, 1.282), (90, 1.645)):
        cov_m = (np.abs(g["am"] - g["pm"]) <= z * g["mstd"]).mean()
        cov_t = (np.abs(g["at"] - g["pt"]) <= z * g["tstd"]).mean()
        print(f"    {lvl}%: margin {cov_m:.1%} | total {cov_t:.1%}")

# ---------------- next-event pooled ----------------
if a.nextevent:
    tcs, Us, Bs, Gs, LMs, cons = [], [], [], [], [], []
    for path in a.nextevent:
        z = np.load(path)
        tcs.append(z["tc"]); Us.append(z["U"]); Bs.append(z["B"])
        Gs.append(z["G"])
        if "LM" in z.files:
            LMs.append(z["LM"])
        cons.append(z["H"] <= np.median(z["H"]))   # per-fold median split
    tc = np.concatenate(tcs); con = np.concatenate(cons)
    print(f"\n=== NEXT-EVENT pooled, n_pos={len(tc):,} ===")
    print(f"{'model':16s}  All     Constrained  Open")
    arrs = {"Historical": np.concatenate(Us), "Bigram": np.concatenate(Bs),
            "GB (context)": np.concatenate(Gs)}
    if len(LMs) == len(a.nextevent):
        arrs["LM"] = np.concatenate(LMs)
    else:
        print("(LM row skipped: one or more npz dumps lack the LM column)")
    for nm, p in arrs.items():
        print(f"{nm:16s}  {(p==tc).mean():.1%}   {(p[con]==tc[con]).mean():.1%}"
              f"        {(p[~con]==tc[~con]).mean():.1%}")
    if "LM" not in arrs:
        raise SystemExit(0)
    lm_ok = (arrs["LM"] == tc).astype(int)
    gb_ok = (arrs["GB (context)"] == tc).astype(int)
    d = lm_ok - gb_ok
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"LM-GB pooled acc diff {d.mean():+.3%} 95%CI[{lo:+.3%},{hi:+.3%}] "
          f"{'SIG' if lo > 0 else 'n.s.'}")
