
import json
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

if len(sys.argv) < 3:
    print("Usage: python ensemble_eval.py <predictions1.json> <predictions2.json> ...")
    print("(need at least 2 seed runs to be meaningful)")
    sys.exit(1)

runs = []
for path in sys.argv[1:]:
    with open(path) as f:
        runs.append(json.load(f))

y_true_ref = runs[0]['y_true']
for r in runs[1:]:
    if r['y_true'] != y_true_ref:
        print(f"[ERROR] y_true mismatch between runs -- these are NOT the same test set/order.")
        print("Ensembling across mismatched test sets is invalid. Check your split settings.")
        sys.exit(1)

print("=" * 90)
print(f"Combining {len(runs)} seed runs (seeds: {[r['seed'] for r in runs]})")
print("=" * 90)

print(f"\n{'Seed':<10}{'F1':<10}{'Accuracy':<12}{'MCC':<10}")
f1s, accs, mccs = [], [], []
for r in runs:
    print(f"{r['seed']:<10}{r['f1']:<10.4f}{r['accuracy']:<12.4f}{r['mcc']:<10.4f}")
    f1s.append(r['f1']); accs.append(r['accuracy']); mccs.append(r['mcc'])

print(f"\n--- Individual-seed statistics (the confidence-interval numbers) ---")
print(f"F1:       {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  (min {min(f1s):.4f}, max {max(f1s):.4f})")
print(f"Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}  (min {min(accs):.4f}, max {max(accs):.4f})")
print(f"MCC:      {np.mean(mccs):.4f} +/- {np.std(mccs):.4f}  (min {min(mccs):.4f}, max {max(mccs):.4f})")

all_probs = np.array([r['y_prob'] for r in runs])  
ensemble_probs = all_probs.mean(axis=0)
ensemble_pred = (ensemble_probs > 0.5).astype(int)

ens_f1 = f1_score(y_true_ref, ensemble_pred, zero_division=0)
ens_acc = accuracy_score(y_true_ref, ensemble_pred)
ens_mcc = matthews_corrcoef(y_true_ref, ensemble_pred)

print(f"\n--- Ensemble (probability-averaged across all {len(runs)} seeds) ---")
print(f"F1:       {ens_f1:.4f}")
print(f"Accuracy: {ens_acc:.4f}")
print(f"MCC:      {ens_mcc:.4f}")

print(f"\n--- Comparison ---")
print(f"Best single seed F1: {max(f1s):.4f}")
print(f"Ensemble F1:         {ens_f1:.4f}")
print(f"Ensemble vs mean of individual seeds: {ens_f1 - np.mean(f1s):+.4f}")

with open("ensemble_results.json", "w") as f:
    json.dump({
        "seeds": [r['seed'] for r in runs],
        "individual_f1": f1s, "individual_acc": accs, "individual_mcc": mccs,
        "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s)),
        "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
        "mean_mcc": float(np.mean(mccs)), "std_mcc": float(np.std(mccs)),
        "ensemble_f1": ens_f1, "ensemble_acc": ens_acc, "ensemble_mcc": ens_mcc,
        "ensemble_y_prob": ensemble_probs.tolist(),
    }, f, indent=2)
print("\nSaved ensemble_results.json")
print("\nFor the paper: report BOTH the mean+/-std (shows honest run-to-run")
print("variance) AND the ensemble number (if you choose to use the ensemble")
print("as your actual reported model) -- state clearly which one is 'the'")
print("headline result if they differ.")
