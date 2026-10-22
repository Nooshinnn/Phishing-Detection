"""
calibration_full_analysis.py

Addresses the supervisor's request: report confidence displacement,
Expected Calibration Error (ECE), Brier score, negative log-likelihood
(NLL), and reliability-diagram data, before and after each attack --
alongside the existing attack success rate, not instead of it.

Works entirely from EXISTING saved prediction JSONs -- no new training or
attack generation needed.

Usage:
    python calibration_full_analysis.py clean.json attack1.json attack2.json ...

The first file argument is treated as the CLEAN baseline; every
subsequent file is compared against it. Each JSON is expected to have
"y_true"/"y_prob" keys somewhere in it, matching the format every attack
script in this project already saves.
"""
import sys
import json
import numpy as np

if len(sys.argv) < 3:
    print("Usage: python calibration_full_analysis.py <clean.json> <attack1.json> [attack2.json ...]")
    sys.exit(1)


def load_probs(path, prefer_clean=False):
    """
    FIX: the original version searched for "clean_phish_only" as a
    candidate key BEFORE any actual attack-result key -- since every
    attack results file also stores its own internal clean baseline
    under that exact name (used to compute evasion rate), this caused
    every attack file to silently return the CLEAN data instead of the
    post-attack data. Confirmed by all 7 "attacks" returning identical
    stats matching the clean phishing-only accuracy (0.9938).

    Fix: explicitly exclude known clean-baseline keys from the search
    unless prefer_clean=True is passed (used only for loading the
    dedicated clean-baseline file itself). Among remaining candidate
    keys, prefer ones containing "50" (the 50%-budget headline result
    used throughout this project); otherwise fall back to the last
    matching key, since these scripts write the clean baseline first
    and the final/strongest attack result last.
    """
    with open(path) as f:
        d = json.load(f)

    CLEAN_KEYS = {"clean_full", "clean_phish_only"}

    if prefer_clean:
        if "y_true" in d and isinstance(d["y_true"], list):
            return np.array(d["y_true"]), np.array(d["y_prob"])
        for key in ["clean_phish_only", "clean_full"]:
            if key in d and isinstance(d[key], dict) and "y_true" in d[key]:
                return np.array(d[key]["y_true"]), np.array(d[key]["y_prob"])
        raise ValueError(f"Could not find a clean baseline in {path}")

    # Top-level y_true (e.g. coordinated-attack "clean" condition dicts) --
    # only valid here if this ISN'T a clean-only file, so leave as a low-priority fallback.
    candidate_keys = [
        k for k, v in d.items()
        if k not in CLEAN_KEYS and isinstance(v, dict) and "y_true" in v
    ]
    if not candidate_keys:
        if "y_true" in d and isinstance(d["y_true"], list):
            return np.array(d["y_true"]), np.array(d["y_prob"])
        raise ValueError(f"Could not find a non-clean attack result in {path} -- check its structure manually.")

    fifty_pct = [k for k in candidate_keys if "50" in k]
    chosen_key = fifty_pct[0] if fifty_pct else candidate_keys[-1]
    return np.array(d[chosen_key]["y_true"]), np.array(d[chosen_key]["y_prob"])


def brier_score(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))


def nll(y_true, y_prob, eps=1e-7):
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        if in_bin.sum() == 0:
            continue
        bin_conf = y_prob[in_bin].mean()
        bin_acc = ((y_prob[in_bin] > 0.5).astype(int) == y_true[in_bin]).mean()
        bin_weight = in_bin.sum() / len(y_prob)
        ece += bin_weight * abs(bin_conf - bin_acc)
        bin_details.append({"range": f"{lo:.1f}-{hi:.1f}", "n": int(in_bin.sum()),
                             "avg_confidence": float(bin_conf), "accuracy": float(bin_acc)})
    return float(ece), bin_details


def summarize(name, y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    acc = (y_pred == y_true).mean()
    ece, bins = expected_calibration_error(y_true, y_prob)
    print(f"\n{'='*90}\n{name}\n{'='*90}")
    print(f"  N: {len(y_true)} | Accuracy: {acc:.4f}")
    print(f"  Brier score: {brier_score(y_true, y_prob):.4f}")
    print(f"  NLL: {nll(y_true, y_prob):.4f}")
    print(f"  ECE ({len(bins)} populated bins): {ece:.4f}")
    return {"n": len(y_true), "accuracy": float(acc), "brier": brier_score(y_true, y_prob),
            "nll": nll(y_true, y_prob), "ece": ece, "ece_bins": bins}


clean_path = sys.argv[1]
attack_paths = sys.argv[2:]

print("Loading clean baseline...")
y_true_clean, y_prob_clean = load_probs(clean_path, prefer_clean=True)
clean_stats = summarize(f"CLEAN BASELINE ({clean_path})", y_true_clean, y_prob_clean)

all_results = {"clean": clean_stats, "attacks": {}}

print("\n" + "#" * 90)
print("PER-ATTACK: Delta-p (probability displacement) + calibration shift")
print("#" * 90)

summary_rows = []
for path in attack_paths:
    y_true_adv, y_prob_adv = load_probs(path)
    if len(y_true_adv) != len(y_true_clean):
        print(f"\n[WARNING] {path}: length {len(y_true_adv)} != clean baseline length {len(y_true_clean)}. "
              f"Delta-p requires matched, same-ordering examples -- skipping Delta-p for this file, "
              f"calibration stats only.")
        matched = False
    else:
        matched = True

    attack_stats = summarize(path, y_true_adv, y_prob_adv)

    if matched:
        delta_p = np.abs(y_prob_clean - y_prob_adv)
        mean_dp = float(delta_p.mean())
        max_dp = float(delta_p.max())
        median_dp = float(np.median(delta_p))
        n_moved_gt_01 = int((delta_p > 0.1).sum())
        n_moved_gt_03 = int((delta_p > 0.3).sum())

        pred_clean = (y_prob_clean > 0.5).astype(int)
        pred_adv = (y_prob_adv > 0.5).astype(int)
        flipped = pred_clean != pred_adv
        n_flipped = int(flipped.sum())
        n_displaced_not_flipped = int(((delta_p > 0.1) & ~flipped).sum())
        n_little_effect = int(((delta_p <= 0.1) & ~flipped).sum())

        print(f"  Delta-p: mean {mean_dp:.4f} | median {median_dp:.4f} | max {max_dp:.4f}")
        print(f"  Examples with |Delta-p| > 0.1: {n_moved_gt_01} | > 0.3: {n_moved_gt_03}")
        print(f"  3-way breakdown -- flipped (evaded): {n_flipped} | "
              f"significantly displaced but NOT flipped: {n_displaced_not_flipped} | "
              f"little effect: {n_little_effect}")

        attack_stats.update({
            "mean_delta_p": mean_dp, "median_delta_p": median_dp, "max_delta_p": max_dp,
            "n_delta_p_gt_0.1": n_moved_gt_01, "n_delta_p_gt_0.3": n_moved_gt_03,
            "n_flipped": n_flipped, "n_displaced_not_flipped": n_displaced_not_flipped,
            "n_little_effect": n_little_effect,
        })
        summary_rows.append((path, mean_dp, max_dp, n_flipped, n_displaced_not_flipped, n_little_effect,
                              attack_stats['ece'] - clean_stats['ece']))

    all_results["attacks"][path] = attack_stats

print("\n" + "#" * 90)
print("SUMMARY TABLE (for the paper -- directly answers the supervisor's Finding 2 request)")
print("#" * 90)
print(f"{'Attack':<45}{'Mean dp':<10}{'Max dp':<10}{'Flipped':<10}{'Displaced':<12}{'Little effect':<15}{'dECE':<10}")
for row in summary_rows:
    path, mean_dp, max_dp, n_flip, n_disp, n_little, d_ece = row
    print(f"{path:<45}{mean_dp:<10.4f}{max_dp:<10.4f}{n_flip:<10}{n_disp:<12}{n_little:<15}{d_ece:<+10.4f}")

with open("calibration_full_analysis_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved calibration_full_analysis_results.json")
print("\nFor reliability diagrams: the ece_bins field in the saved JSON has everything needed")
print("(bin range, n, avg_confidence, accuracy per bin) to plot confidence-vs-accuracy for")
print("both clean and each attack condition -- ask for the plotting script once you know")
print("which specific attacks you want diagrams for (not all 7 -- pick the 2-3 most relevant).")