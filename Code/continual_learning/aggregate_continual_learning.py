"""Aggregate continual-learning results across seeds.

Run in the folder that has the result files:
    python aggregate_continual_learning.py

Uses continual_learning_results_<source>_seed*.json, plus the original
continual_learning_results_<source>.json (seed 31415) if it is present.
"""
import glob
import json
import os

import numpy as np

for src in ("nigerian_fraud", "phishing_pot"):
    files = sorted(glob.glob(f"continual_learning_results_{src}_seed*.json"))
    orig = f"continual_learning_results_{src}.json"
    if os.path.exists(orig):
        files = [orig] + files
    else:
        print(f"[{src}] note: {orig} (seed 31415) not found, using seed files only")
    base, retention, adapt = {}, {}, {}
    for f in files:
        r = json.load(open(f))
        for m, v in r["methods"].items():
            retention.setdefault(m, []).append(v["on_base_test"]["accuracy"])
            adapt.setdefault(m, []).append(v["on_new_campaign_test"]["accuracy"])
    print(f"\n=== {src}: {len(files)} files ===")
    for title, d in (("RETENTION (base_test accuracy)", retention),
                     ("ADAPTATION (new_campaign_test accuracy)", adapt)):
        print(title)
        for m, v in d.items():
            print(f"  {m:<16}{np.mean(v):.4f} +/- {np.std(v):.4f}  "
                  f"n={len(v)}  {[round(x, 4) for x in v]}")
