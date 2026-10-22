
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python plot_reliability_diagrams.py <calibration_full_analysis_results.json> [attack_key1] [attack_key2] ...")
    print("attack_key values must match the filenames you passed to calibration_full_analysis.py")
    sys.exit(1)

with open(sys.argv[1]) as f:
    data = json.load(f)

attack_keys = sys.argv[2:] if len(sys.argv) > 2 else list(data["attacks"].keys())[:2]

conditions = [("Clean", data["clean"])]
for key in attack_keys:
    if key not in data["attacks"]:
        print(f"[WARNING] '{key}' not found in the calibration results. Available keys: {list(data['attacks'].keys())}")
        continue
    conditions.append((key, data["attacks"][key]))

n = len(conditions)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
if n == 1:
    axes = [axes]

for ax, (name, stats) in zip(axes, conditions):
    bins = stats["ece_bins"]
    confidences = [b["avg_confidence"] for b in bins]
    accuracies = [b["accuracy"] for b in bins]
    weights = [b["n"] for b in bins]

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.scatter(confidences, accuracies, s=[max(20, w) for w in weights], alpha=0.7, color="steelblue")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence (bin)")
    ax.set_ylabel("Empirical accuracy (bin)")
    ax.set_title(f"{name}\nECE={stats['ece']:.4f}, Brier={stats['brier']:.4f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("reliability_diagrams.png", dpi=200, bbox_inches="tight")
plt.savefig("reliability_diagrams.pdf", bbox_inches="tight")
print("Saved reliability_diagrams.png and reliability_diagrams.pdf")
print(f"Plotted: {[name for name, _ in conditions]}")
