"""
deployment_study.py

Addresses the review's #3 highest-value experiment: "realistic prevalence
and deployment study demonstrating operational benefit."

Three independent parts:
  1. Prevalence-adjusted precision/recall -- your test set is ~43-57%
     phishing (balanced for evaluation), but real inboxes see maybe 1-5%
     phishing. Precision at the SAME classifier operating point looks very
     different at realistic prevalence -- this is the actual number that
     matters for "would this be usable in production."
  2. Latency benchmark -- inference time per email, canonical model, your
     hardware.
  3. Resource usage summary -- parameter counts.

Usage:
    python deployment_study.py <predictions.json>
(predictions.json = any file with y_true/y_prob at the canonical model's
clean operating point, e.g. one of the seed prediction files)
"""
import sys
import json
import time
import numpy as np
import torch

# ====================== PART 1: Prevalence-adjusted precision/recall ======================

def prevalence_adjusted_metrics(y_true, y_prob, threshold=0.5, prevalences=(0.01, 0.02, 0.05, 0.10, 0.20)):
    y_true = np.array(y_true)
    y_pred = (np.array(y_prob) > threshold).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')

    print(f"Test-set operating point (threshold={threshold}):")
    print(f"  TPR (recall/sensitivity): {tpr:.4f}")
    print(f"  FPR (false positive rate): {fpr:.4f}")
    print(f"  Test-set prevalence: {(y_true==1).mean():.4f}")
    print()
    print("Precision at REALISTIC deployment prevalence (using this TPR/FPR, Bayes' rule):")
    print(f"{'Prevalence':<15}{'Precision':<15}{'Expected FPs per 10k emails':<30}")
    for prev in prevalences:
        denom = tpr * prev + fpr * (1 - prev)
        precision = (tpr * prev) / denom if denom > 0 else float('nan')
        expected_fp_per_10k = fpr * (1 - prev) * 10000
        print(f"{prev*100:>6.1f}%       {precision:<15.4f}{expected_fp_per_10k:<30.1f}")
    print()
    print("Interpretation: at low real-world prevalence (1-2%, typical for a corporate inbox),")
    print("precision often drops sharply even for a model with excellent test-set metrics --")
    print("this is the actual number that determines whether flagged emails are trustworthy")
    print("enough for automated action vs. just a warning banner.")
    return tpr, fpr


# ====================== PART 2 & 3: Latency + resource usage ======================

def latency_and_resource_benchmark():
    from model_and_data_utils import load_model, build_dataset
    email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
    _, _, test_df = build_dataset()

    n_params = sum(p.numel() for p in email_model.parameters()) + \
               sum(p.numel() for p in url_model.parameters()) + \
               sum(p.numel() for p in comm_model.parameters())
    print(f"\nTotal model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    device = next(email_model.parameters()).device
    sample = test_df.iloc[:100]

    for _ in range(3):
        e_in = email_tokenizer(sample['email_text'].iloc[:1].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(sample['url'].iloc[:1].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            comm_model(e_feat, u_feat)

    times = []
    for i in range(len(sample)):
        row = sample.iloc[i:i+1]
        start = time.perf_counter()
        e_in = email_tokenizer(row['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(row['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            comm_model(e_feat, u_feat)
        times.append(time.perf_counter() - start)

    times = np.array(times) * 1000
    print(f"\nSingle-email inference latency ({device}), n={len(times)}:")
    print(f"  Mean: {times.mean():.2f} ms | Median: {np.median(times):.2f} ms | "
          f"p95: {np.percentile(times, 95):.2f} ms | p99: {np.percentile(times, 99):.2f} ms")
    print(f"  Implied throughput: {1000/times.mean():.1f} emails/sec (single-request, unbatched)")
    print("\nNote: batched throughput would be substantially higher -- this measures the")
    print("realistic inline-scanning latency (one email at a time), which is what matters")
    print("for user-facing delivery delay, not batch processing throughput.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            data = json.load(f)
        y_true = data.get('y_true')
        y_prob = data.get('y_prob')
        if y_true is None or y_prob is None:
            print("Could not find y_true/y_prob in this file -- point this at a raw predictions")
            print("JSON (e.g. combined_corpus_test_predictions_seed123.json)")
        else:
            prevalence_adjusted_metrics(y_true, y_prob)
    else:
        print("No predictions file given -- skipping Part 1 (prevalence-adjusted metrics).")
        print("Usage: python deployment_study.py <predictions.json>")

    print("\n" + "=" * 90)
    print("Latency & resource benchmark")
    print("=" * 90)
    latency_and_resource_benchmark()
