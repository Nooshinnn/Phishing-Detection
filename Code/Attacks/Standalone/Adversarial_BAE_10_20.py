import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForMaskedLM
import warnings
from tqdm import tqdm
import random
import json

from model_and_data_utils import load_model, build_dataset, DEVICE

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = DEVICE

print("=" * 90)
print("ADVERSARIAL ATTACK: BAE -- FIXED candidate selection, SPLIT RUN (10% + 20% only)")
print("Using shared module -- canonical checkpoint & combined corpus, auto-selected")
print("=" * 90)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
train_df, val_df, test_df = build_dataset()
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


test_df_phish = test_df[test_df['label'] == 1].reset_index(drop=True)
print(f"Evaluating evasion on phishing-labeled test examples only: "
      f"{len(test_df_phish)} / {len(test_df)}")

mlm_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
mlm_model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device)
mlm_model.eval()


def bae_attack(text, url, change_percent=0.30):
    words = text.split()
    if len(words) < 5:
        return text

    num_changes = max(1, int(len(words) * change_percent))

    u_in = url_tokenizer(
        [url], padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        e_in = email_tokenizer(
            [text], padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(device)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        original_prob = torch.sigmoid(comm_model(e_feat, u_feat)).item()

    saliency_scores = []
    for i, word in enumerate(words):
        if not word.isalpha():
            saliency_scores.append(0.0)
            continue
        masked = words.copy()
        masked[i] = mlm_tokenizer.mask_token
        masked_text = " ".join(masked)

        with torch.no_grad():
            e_in = email_tokenizer(
                [masked_text], padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(device)
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            new_prob = torch.sigmoid(comm_model(e_feat, u_feat)).item()

        saliency_scores.append(abs(original_prob - new_prob))

    indices = np.argsort(saliency_scores)[::-1][:num_changes]

    for idx in indices:
        if not words[idx].isalpha():
            continue
        masked = words.copy()
        masked[idx] = mlm_tokenizer.mask_token
        masked_text = " ".join(masked)

        inputs = mlm_tokenizer(
            masked_text, return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            logits = mlm_model(**inputs).logits

        mask_positions = (inputs.input_ids == mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        if mask_positions.numel() == 0:
            continue
        mask_pos = mask_positions[0]
        top_tokens = logits[0, mask_pos].topk(40).indices

        candidates = []
        for token_id in top_tokens:
            candidate = mlm_tokenizer.decode(token_id).strip()
            if candidate.isalpha() and candidate.lower() != words[idx].lower():
                candidates.append(candidate)
            if len(candidates) >= 10:  # cap trials per word to bound runtime
                break

        best_word = None
        best_prob = None
        for cand in candidates:
            trial = words.copy()
            trial[idx] = cand
            trial_text = " ".join(trial)
            with torch.no_grad():
                e_in = email_tokenizer(
                    [trial_text], padding=True, truncation=True, max_length=256, return_tensors="pt"
                ).to(device)
                e_feat = email_model(e_in.input_ids, e_in.attention_mask)
                u_feat = url_model(u_in.input_ids, u_in.attention_mask)
                trial_prob = torch.sigmoid(comm_model(e_feat, u_feat)).item()
            if best_prob is None or trial_prob < best_prob:
                best_prob = trial_prob
                best_word = cand

        if best_word is not None:
            words[idx] = best_word

    return " ".join(words)

def evaluate(df_eval, use_adv=False):
    y_true, y_prob = [], []
    col = 'adv_email' if use_adv else 'email_text'
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i + 8]
            e_in = email_tokenizer(
                batch[col].tolist(), padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            ).to(device)
            u_in = url_tokenizer(
                batch['url'].tolist(), padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(device)
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

percentages = [0.1, 0.2]
all_results = {}

print("\nRunning BAE attacks (10%, 20%)...\n")
clean_full = evaluate(test_df, use_adv=False)
clean_phish = evaluate(test_df_phish, use_adv=False)
print(f"Full-set clean baseline (reference) -> F1: {clean_full['f1']:.4f} | Acc: {clean_full['accuracy']:.4f}")
print(f"Phishing-only clean accuracy         -> {clean_phish['accuracy']:.4f}")
all_results['clean_full'] = clean_full
all_results['clean_phish_only'] = clean_phish

for percent in percentages:
    print(f"\n-> Attacking with {percent * 100:.0f}% changes using BAE (phishing-only)...")
    test_adv = test_df_phish.copy()
    random.seed(42)
    test_adv['adv_email'] = [
        bae_attack(str(row['email_text']), str(row['url']), change_percent=percent)
        for _, row in tqdm(test_adv.iterrows(), total=len(test_adv), desc=f"BAE {percent * 100:.0f}%")
    ]
    adv = evaluate(test_adv, use_adv=True)

    pre_correct = [i for i, p in enumerate(clean_phish['y_prob']) if p > 0.5]
    n_evaded = sum(1 for i in pre_correct if adv['y_prob'][i] <= 0.5)
    evasion_rate = n_evaded / len(pre_correct) if pre_correct else float('nan')
    adv['evasion_success_rate'] = evasion_rate
    adv['n_pre_correct'] = len(pre_correct)
    adv['n_evaded'] = n_evaded

    print(
        f"   Phishing-only Acc: {adv['accuracy']:.4f} | "
        f"Evasion success rate: {evasion_rate:.4f} ({n_evaded}/{len(pre_correct)})"
    )
    all_results[f"bae_{int(percent*100)}"] = adv

with open("bae_raw_results_10_20.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved bae_raw_results_10_20.json")
