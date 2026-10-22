
import json

combined = {}
files = ["bae_raw_results_10_20.json", "bae_raw_results_30_40.json", "bae_raw_results_50.json"]

for fname in files:
    with open(fname) as f:
        data = json.load(f)
    for key, val in data.items():
        if key in ("clean_full", "clean_phish_only"):
            combined[key] = val  # identical across all three, just keep one copy
        else:
            combined[key] = val

with open("bae_raw_results.json", "w") as f:
    json.dump(combined, f, indent=2)

print("Combined into bae_raw_results.json")
print(f"Keys: {list(combined.keys())}")

print("\nSummary:")
for pct in [10, 20, 30, 40, 50]:
    key = f"bae_{pct}"
    if key in combined:
        r = combined[key]
        print(f"  {pct}%: Evasion {r['evasion_success_rate']:.4f} ({r['n_evaded']}/{r['n_pre_correct']})")
