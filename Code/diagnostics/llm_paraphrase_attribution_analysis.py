
import sys
import json
import numpy as np
import torch
from tqdm import tqdm

from model_and_data_utils import load_model, DEVICE

if len(sys.argv) < 2:
    print("Usage: python llm_paraphrase_attribution_analysis.py <samples.json>")
    sys.exit(1)

SAMPLES_PATH = sys.argv[1]
TOP_K = 5

with open(SAMPLES_PATH) as f:
    samples = json.load(f)
print(f"Loaded {len(samples)} original/paraphrased pairs from {SAMPLES_PATH}")
if len(samples) < 50:
    print(f"[NOTE] This is a small-scale analysis ({len(samples)} examples) -- report it as such, "
          f"not as a full-corpus result. For a full 639-example analysis, re-run paraphrase "
          f"generation saving ALL pairs, not just the Appendix sample subset.")

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
device = DEVICE


def true_class_prob(text, url, label):
    with torch.no_grad():
        e_in = email_tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer([url], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        prob_phish = torch.sigmoid(comm_model(e_feat, u_feat)).item()
    return prob_phish if label == 1 else (1 - prob_phish)


def top_salient_words(text, url, label, top_k=TOP_K):
    words = text.split()
    if len(words) < top_k + 1:
        return []
    original_prob = true_class_prob(text, url, label)
    saliency_scores = []
    for i, word in enumerate(words):
        if not word.isalpha():
            saliency_scores.append(0.0)
            continue
        deleted = words[:i] + words[i+1:]
        p = true_class_prob(" ".join(deleted) if deleted else word, url, label)
        saliency_scores.append(abs(original_prob - p))
    ranked_idx = np.argsort(saliency_scores)[::-1][:top_k]
    return [(words[i], saliency_scores[i]) for i in ranked_idx if saliency_scores[i] > 0]


results = []
survival_rates = []

for sample in tqdm(samples, desc="Computing attribution"):
    if sample.get("label") != 1:
        continue
    original = str(sample.get("original", ""))
    paraphrased = str(sample.get("paraphrased", sample.get("adv_email", "")))
    placeholder_url = sample.get("url", "http://example.com")

    top_words = top_salient_words(original, placeholder_url, TOP_K)
    if not top_words:
        continue

    paraphrased_lower = paraphrased.lower()
    survived = [w for w, score in top_words if w.lower() in paraphrased_lower]
    survival_rate = len(survived) / len(top_words)
    survival_rates.append(survival_rate)

    results.append({
        "original_snippet": original[:150],
        "top_salient_words": [{"word": w, "saliency": float(s)} for w, s in top_words],
        "survived_in_paraphrase": survived,
        "survival_rate": survival_rate,
    })

print("\n" + "=" * 90)
print("ATTRIBUTION ANALYSIS SUMMARY")
print("=" * 90)
print(f"Examples analyzed: {len(results)}")
if survival_rates:
    print(f"Mean top-{TOP_K}-word survival rate in paraphrase: {np.mean(survival_rates):.4f}")
    print(f"Median: {np.median(survival_rates):.4f} | Min: {min(survival_rates):.4f} | Max: {max(survival_rates):.4f}")
    high_survival = sum(1 for r in survival_rates if r >= 0.6)
    print(f"Examples with >=60% of top salient words surviving: {high_survival}/{len(survival_rates)}")
    print("\nInterpretation: a high survival rate supports the hypothesis that meaning-preserving")
    print("paraphrasing retains the specific words the classifier relies on, unlike attacks that")
    print("explicitly target and replace exactly those words. A low survival rate combined with")
    print("still-near-zero evasion would argue AGAINST this explanation and warrant further")
    print("investigation into what else might be driving the near-zero evasion result.")

with open("llm_paraphrase_attribution_results.json", "w") as f:
    json.dump({"n_examples": len(results),
               "mean_survival_rate": float(np.mean(survival_rates)) if survival_rates else None,
               "per_example": results}, f, indent=2)
print("\nSaved llm_paraphrase_attribution_results.json")
