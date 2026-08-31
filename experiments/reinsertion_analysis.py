#!/usr/bin/env python3
"""Counterfactual star re-insertion: pre-registered analysis, fully reproducible
from data/reinsert_results.csv alone (no corpus or checkpoint needed).

Design: 346 candidate cases — held-out test games in which a 28-plus-minute
player was genuinely unavailable — of which 344 were simulated (the harness
skips two cases whose star, a 2025-26 rookie, has no player-vocabulary
entry); 317 pass the primary validity gate
below, 201 the stricter pre-registered gate. Arm A simulates the game as
loaded (star absent, the production
forecast); Arm B re-inserts the star into roster and starting five with their
real pre-absence card and recomputed prior minutes; K=200 rollouts per arm,
same seed, everything else identical. shift = mean margin(B) - mean margin(A),
oriented toward the star's team.

Both arms received the same per-game pace scalar, so paired shifts are
invariant to it. The `valid` column (star entered with >15 simulated minutes
and paired-arm team minutes consistent within 2) is the primary filter; the
stricter pre-registered `strict_tripwire` column is reported alongside because
its team-minutes window interacts with fold 2's known sequence-length
truncation. Claim ceiling: the absence response is directionally validated;
its magnitude sits below the 1.5-3 point consensus value of a star absence,
so this is not a calibrated absence-pricing model.
"""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "..", "data", "reinsert_results.csv"))
for label, mask in (("primary (valid)", d["valid"]), ("strict pre-registered", d["strict_tripwire"])):
    s = d[mask]
    x = s["shift"].values
    rng = np.random.default_rng(0)
    boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rho, p = spearmanr(s["ppg"], s["shift"])
    folds = [s[s["fold"] == k]["shift"].mean() for k in (1, 2, 3)]
    print(f"[{label}] n={len(s)}")
    print(f"  G1 mean shift {x.mean():+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] | Wilcoxon p={wilcoxon(x).pvalue:.1e}")
    print(f"  G2 dose-response Spearman(as-of PPG) {rho:+.3f} p={p:.5f}")
    print(f"  G3 per-fold means {[round(v, 2) for v in folds]}")
    for k in (1, 2, 3):
        tt = s[s["tier"] == k]["shift"]
        print(f"    tier {k} ({'All-Star' if k==1 else '20+ PPG' if k==2 else 'role'}): n={len(tt)} mean {tt.mean():+.2f}")
    print()
