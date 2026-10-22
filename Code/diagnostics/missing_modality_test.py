
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import json

from model_and_data_utils import load_model, build_dataset_combined_corpus, DEVICE

print("=" * 90)
print("DIAGNOSTIC: Missing-modality test (COMBINED CORPUS, canonical seed=31415)")
print("=" * 90)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model("combined_corpus_best_model_seed31415.pth")
train_df, val_df, test_df = build_dataset_combined_corpus(near_dup_threshold=0.90)


def evaluate(df_eval, mode):

    y_true, y_prob = [], []
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i + 8]

            email_texts = batch['email_text'].tolist() if mode != 'url_only' else [""] * len(batch)
            urls = batch['url'].tolist() if mode != 'email_only' else [""] * len(batch)

            e_in = email_tokenizer(
                email_texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(DEVICE)
            u_in = url_tokenizer(
                urls, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(DEVICE)
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            prob = logits.sigmoid().cpu().numpy()
            y_true.extend(batch['label'].values)
            y_prob.extend(prob)
    y_pred = (np.array(y_prob) > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "y_true": [int(x) for x in y_true],
        "y_prob": [float(x) for x in y_prob],
    }


results = {}
for mode in ['both', 'email_only', 'url_only']:
    print(f"\nEvaluating mode: {mode}")
    r = evaluate(test_df, mode)
    print(f"  F1: {r['f1']:.4f} | Acc: {r['accuracy']:.4f} | MCC: {r['mcc']:.4f}")
    results[mode] = r

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"{'Mode':<15}{'F1':<10}{'Accuracy':<12}{'MCC':<10}")
for mode in ['both', 'email_only', 'url_only']:
    r = results[mode]
    print(f"{mode:<15}{r['f1']:<10.4f}{r['accuracy']:<12.4f}{r['mcc']:<10.4f}")

drop_email_only = results['both']['f1'] - results['email_only']['f1']
drop_url_only = results['both']['f1'] - results['url_only']['f1']
print(f"\nF1 drop when URL is missing (email-only):  {drop_email_only:.4f}")
print(f"F1 drop when email is missing (url-only):   {drop_url_only:.4f}")
print("\nInterpretation guide:")
print("  - If BOTH drops are small: the model may have learned a shortcut where either")
print("    modality alone is nearly sufficient -- worth cross-checking against the")
print("    counterfactual-pairing test (does swapping URLs also barely matter?).")
print("  - If one drop is much larger than the other: model relies heavily on that")
print("    modality -- relevant context for interpreting PWWS/BAE/TextFooler (email-only")
print("    attacks) vs URLGAN (url-only attack) effect sizes reported elsewhere.")

with open("missing_modality_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved missing_modality_results.json")
