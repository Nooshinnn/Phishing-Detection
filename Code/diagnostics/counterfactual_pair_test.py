
import numpy as np
import torch
import random
from tqdm import tqdm
import json

from model_and_data_utils import load_model, build_dataset_combined_corpus, DEVICE

print("=" * 90)
print("DIAGNOSTIC: Counterfactual-pairing test (COMBINED CORPUS, canonical seed=31415)")
print("=" * 90)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model("combined_corpus_best_model_seed31415.pth")
train_df, val_df, test_df = build_dataset_combined_corpus(near_dup_threshold=0.90)

random.seed(42)

phish_urls = test_df[test_df['label'] == 1]['url'].tolist()
legit_urls = test_df[test_df['label'] == 0]['url'].tolist()


def predict(email_text, url):
    with torch.no_grad():
        e_in = email_tokenizer([email_text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        u_in = url_tokenizer([url], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        return torch.sigmoid(comm_model(e_feat, u_feat)).item()


records = []
for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Counterfactual pairs"):
    email_text, url, label = row['email_text'], row['url'], int(row['label'])

    original_prob = predict(email_text, url)

    if label == 1:
        same_label_url = random.choice([u for u in phish_urls if u != url]) if len(phish_urls) > 1 else url
        cross_label_url = random.choice(legit_urls) if legit_urls else url
    else:
        same_label_url = random.choice([u for u in legit_urls if u != url]) if len(legit_urls) > 1 else url
        cross_label_url = random.choice(phish_urls) if phish_urls else url

    same_label_prob = predict(email_text, same_label_url)
    cross_label_prob = predict(email_text, cross_label_url)

    records.append({
        "label": label,
        "original_prob": original_prob,
        "same_label_swap_prob": same_label_prob,
        "cross_label_swap_prob": cross_label_prob,
        "same_label_shift": abs(original_prob - same_label_prob),
        "cross_label_shift": abs(original_prob - cross_label_prob),
    })

same_shifts = [r['same_label_shift'] for r in records]
cross_shifts = [r['cross_label_shift'] for r in records]

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"Mean |shift| from same-label URL swap:  {np.mean(same_shifts):.4f}")
print(f"Mean |shift| from cross-label URL swap: {np.mean(cross_shifts):.4f}")
print(f"Max |shift| from same-label URL swap:   {np.max(same_shifts):.4f}")
print(f"Max |shift| from cross-label URL swap:  {np.max(cross_shifts):.4f}")

ratio = np.mean(cross_shifts) / np.mean(same_shifts) if np.mean(same_shifts) > 0 else float('inf')
print(f"\nCross-label / same-label shift ratio: {ratio:.2f}x")
print("\nInterpretation guide:")
print("  - Ratio >> 1: model IS sensitive to label-mismatched pairs -- evidence of")
print("    genuine cross-modal reasoning, not just per-modality independent scoring.")
print("  - Ratio close to 1: swapping to a mismatched URL barely matters more than")
print("    swapping to a same-label URL -- suggests the model may be scoring each")
print("    modality largely independently, which would support a label-consistency")
print("    shortcut concern for the synthetically-paired portion of the dataset.")

with open("counterfactual_pair_results.json", "w") as f:
    json.dump(records, f, indent=2)
print("\nSaved counterfactual_pair_results.json")
