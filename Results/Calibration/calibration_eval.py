
import json
import sys
import statistics


def brier_score(y_true, y_prob):
    return sum((p - t) ** 2 for p, t in zip(y_prob, y_true)) / len(y_true)


def analyze_file(path):
    with open(path) as f:
        d = json.load(f)

    clean_key = 'clean_phish_only' if 'clean_phish_only' in d else 'clean'
    if clean_key not in d:
        print(f"[{path}] No clean baseline key found (looked for 'clean_phish_only' or 'clean') -- skipping.")
        return

    clean = d[clean_key]
    y_true = clean['y_true']
    clean_probs = clean['y_prob']

    dists = [abs(p - 0.5) for p in clean_probs]
    print("=" * 90)
    print(f"FILE: {path}")
    print("=" * 90)
    print(f"Clean baseline (n={len(y_true)}):")
    print(f"  Brier score: {brier_score(y_true, clean_probs):.4f}")
    print(f"  Min distance from boundary: {min(dists):.4f}")
    print(f"  Median distance from boundary: {statistics.median(dists):.4f}")
    print(f"  # within 0.05 of boundary: {sum(1 for x in dists if x < 0.05)} / {len(dists)}")
    print(f"  # within 0.10 of boundary: {sum(1 for x in dists if x < 0.10)} / {len(dists)}")

    attack_keys = [k for k in d.keys() if k not in
                   ('clean', 'clean_full', 'clean_phish_only', 'n_zero_eligible_urls',
                    'fallback_count', 'evasion_success_rate', 'n_pre_correct', 'n_evaded')]

    prev_flip_set = None
    for key in attack_keys:
        entry = d[key]
        if not isinstance(entry, dict) or 'y_prob' not in entry:
            continue
        adv_probs = entry['y_prob']
        if len(adv_probs) != len(clean_probs):
            print(f"\n[{key}] length mismatch vs clean baseline ({len(adv_probs)} vs {len(clean_probs)}) -- skipping.")
            continue

        diffs = [abs(a - b) for a, b in zip(clean_probs, adv_probs)]
        max_diff = max(diffs)
        mean_diff = sum(diffs) / len(diffs)
        flips = [i for i, (a, b) in enumerate(zip(clean_probs, adv_probs)) if (a > 0.5) != (b > 0.5)]
        flip_set = set(flips)

        print(f"\n[{key}]")
        print(f"  Brier score: {brier_score(y_true, adv_probs):.4f}")
        print(f"  Max |delta P|: {max_diff:.4f} | Mean |delta P|: {mean_diff:.4f}")
        print(f"  Hard-label flips: {len(flips)} / {len(clean_probs)} -> indices: {flips}")
        if 'evasion_success_rate' in entry:
            print(f"  Evasion success rate: {entry['evasion_success_rate']:.4f} "
                  f"({entry.get('n_evaded','?')}/{entry.get('n_pre_correct','?')})")

        # Sanity flags -- catch the two bug classes already found in this project.
        if max_diff < 1e-9:
            print("  [FLAG] y_prob is BIT-IDENTICAL to clean baseline -- almost certainly a pipeline bug "
                  "(perturbed data not reaching the model), not a real robustness result.")
        if prev_flip_set is not None and flip_set == prev_flip_set and len(flip_set) > 0:
            print("  [NOTE] Same flip set as previous strength level -- effect may be saturating "
                  "(expected/interesting if distance-from-boundary is large; worth cross-checking).")
        prev_flip_set = flip_set


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibration_eval.py <results.json> [<results2.json> ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        analyze_file(path)
