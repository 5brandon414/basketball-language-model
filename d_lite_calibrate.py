#!/usr/bin/env python3
"""Fit the distribution variance-shrink on val rollouts; report test coverage.

  python3 d_lite_calibrate.py <ckpt_dir>
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

d = sys.argv[1]
va = pd.read_csv(f"{d}/gen_val_pre.csv")
te = pd.read_csv(f"{d}/gen_test_pre.csv")

lam_m = float(np.sqrt(((va.pm - va["am"]) ** 2).mean() / (va.mstd ** 2).mean()))
lam_t = float(np.sqrt(((va.pt - va["at"]) ** 2).mean() / (va.tstd ** 2).mean()))
print(f"fitted on {len(va)} val games: lambda_margin {lam_m:.3f} | "
      f"lambda_total {lam_t:.3f}")

for tag, pred, act, sd_, lam in (("margin", te.pm, te["am"], te.mstd, lam_m),
                                 ("total", te.pt, te["at"], te.tstd, lam_t)):
    z = (act - pred) / (sd_ * lam + 1e-9)
    cov = {q: float((z <= norm.ppf(q / 100)).mean()) for q in (5, 25, 50, 75, 95)}
    print(f"  {tag:6s} post-shrink coverage (test, normal approx): "
          + " ".join(f"p{q}:{cov[q]:.0%}" for q in (5, 25, 50, 75, 95)))
import json
json.dump({"lambda_margin": round(lam_m, 4), "lambda_total": round(lam_t, 4),
           "n_val": len(va)}, open(f"{d}/lm_shrink.json", "w"), indent=2)
print(f"saved {d}/lm_shrink.json")
