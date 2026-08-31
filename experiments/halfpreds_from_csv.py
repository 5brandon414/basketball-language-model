#!/usr/bin/env python3
"""Convert a generate_bball_lm.py rollout CSV to the --lm-half-preds parquet."""
import sys
import pandas as pd

g = pd.read_csv(sys.argv[1])
pd.DataFrame({"game_id": g.game_id.astype(str).str.zfill(10),
              "pred": g.pm, "actual": g.am,
              "pred_total": g.pt, "actual_total": g["at"]}
             ).to_parquet(sys.argv[2], index=False)
print(f"{sys.argv[2]}: {len(g)} games")
