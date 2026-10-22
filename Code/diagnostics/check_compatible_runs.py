
import json
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    print("Usage: python check_compatible_runs.py <predictions1.json> <predictions2.json> ...")
    sys.exit(1)

groups = defaultdict(list)
for path in sys.argv[1:]:
    try:
        with open(path) as f:
            d = json.load(f)
    except FileNotFoundError:
        print(f"[skip] {path} not found")
        continue
    y_true_key = tuple(d['y_true'])  # exact sequence, including order
    groups[y_true_key].append((path, d.get('seed', '?'), len(d['y_true'])))

print("=" * 90)
print(f"Found {len(groups)} distinct test-set signature(s) across {sum(len(v) for v in groups.values())} files")
print("=" * 90)

for i, (sig, files) in enumerate(groups.items()):
    print(f"\nGroup {i+1} (test set size = {files[0][2]}):")
    for path, seed, n in files:
        print(f"  seed={seed:<10} {path}")

if len(groups) == 1:
    print("\nAll files share the SAME test set -- safe to ensemble together as-is, no re-run needed.")
elif len(groups) > 1:
    largest_group = max(groups.values(), key=len)
    print(f"\nLargest compatible group has {len(largest_group)} file(s):")
    for path, seed, n in largest_group:
        print(f"  seed={seed}: {path}")
    print(f"\nRECOMMENDATION: use only the files in the largest group above for ensemble_eval.py.")
    other_seeds = [seed for g in groups.values() if g != largest_group for _, seed, _ in g]
    if other_seeds:
        print(f"Only these seed(s) need re-running to join that group: {other_seeds}")
