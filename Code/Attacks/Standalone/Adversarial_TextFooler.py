import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import json
import gensim.downloader as gensim_api

from model_and_data_utils import load_model, build_dataset, is_english_word, DEVICE

print("=" * 90)
print("ADVERSARIAL ATTACK: TextFooler (word-embedding nearest-neighbor substitution)")
print("Using canonical checkpoint: original_best_model_fixed.pth")
print("=" * 90)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
train_df, val_df, test_df = build_dataset()

# FIX (consistency with PDGAN/PWWS/BAE fix): restrict the attack itself to
# phishing-labeled examples, matching this paper's evasion threat model.
test_df_phish = test_df[test_df['label'] == 1].reset_index(drop=True)
print(f"Evaluating evasion on phishing-labeled test examples only: "
      f"{len(test_df_phish)} / {len(test_df)}")

print("\nLoading GloVe word vectors (first run downloads ~66MB via gensim)...")
word_vectors = gensim_api.load("glove-wiki-gigaword-100")
print("Loaded.")


def true_class_prob(text, url, label):
    with torch.no_grad():
        e_in = email_tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        u_in = url_tokenizer([url], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        prob_phish = torch.sigmoid(comm_model(e_feat, u_feat)).item()
    return prob_phish if label == 1 else (1 - prob_phish)


def get_embedding_neighbors(word, topn=15):

    word_lower = word.lower()
    if word_lower not in word_vectors:
        return []
    try:
        neighbors = word_vectors.most_similar(word_lower, topn=topn)
    except KeyError:
        return []
    candidates = []
    for cand_word, similarity in neighbors:
        if not cand_word.isalpha():
            continue
        if cand_word.lower() == word_lower:
            continue
        if not is_english_word(cand_word):
            continue
        candidates.append(cand_word)
    return candidates


def textfooler_attack(text, url, label, change_percent=0.30):
    words = text.split()
    if len(words) < 5:
        return text, 0

    num_changes_budget = max(1, int(len(words) * change_percent))
    original_true_prob = true_class_prob(text, url, label)
    original_decision = original_true_prob > 0.5 if label == 1 else original_true_prob < 0.5

    importance = []
    for i, word in enumerate(words):
        if not word.isalpha():
            importance.append(0.0)
            continue
        deleted = words[:i] + words[i+1:]
        deleted_text = " ".join(deleted) if deleted else word
        p = true_class_prob(deleted_text, url, label)
        importance.append(original_true_prob - p)  

    ranked_indices = np.argsort(importance)[::-1]

    attacked_words = words.copy()
    n_changed = 0
    for idx in ranked_indices:
        if n_changed >= num_changes_budget:
            break
        if importance[idx] <= 0:
            continue
        candidates = get_embedding_neighbors(words[idx])
        if not candidates:
            continue

        best_drop = -1.0
        best_word = None
        for cand in candidates:
            trial = attacked_words.copy()
            trial[idx] = cand
            trial_text = " ".join(trial)
            p = true_class_prob(trial_text, url, label)
            drop = original_true_prob - p
            if drop > best_drop:
                best_drop = drop
                best_word = cand

        if best_word is not None and best_drop > 0:
            attacked_words[idx] = best_word
            n_changed += 1
            # early stop if we've flipped the raw phishing-probability decision
            current_full_prob = true_class_prob(" ".join(attacked_words), url, label)
            flipped = (current_full_prob > 0.5) != (original_true_prob > 0.5)
            if flipped:
                break

    return " ".join(attacked_words), n_changed


def evaluate(df_eval, use_adv=False):
    y_true, y_prob = [], []
    col = 'adv_email' if use_adv else 'email_text'
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i + 8]
            e_in = email_tokenizer(
                batch[col].tolist(), padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            ).to(DEVICE)
            u_in = url_tokenizer(
                batch['url'].tolist(), padding=True, truncation=True,
                max_length=128, return_tensors="pt"
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


percentages = [0.10, 0.20, 0.30, 0.40, 0.50]
all_results = {}

print("\nRunning TextFooler attacks with different change-percent budgets...\n")
clean_full = evaluate(test_df, use_adv=False)
clean_phish = evaluate(test_df_phish, use_adv=False)
print(f"Full-set clean baseline (reference) -> F1: {clean_full['f1']:.4f} | Acc: {clean_full['accuracy']:.4f}")
print(f"Phishing-only clean accuracy         -> {clean_phish['accuracy']:.4f}")
all_results['clean_full'] = clean_full
all_results['clean_phish_only'] = clean_phish

for percent in percentages:
    print(f"\n-> Attacking with up to {percent * 100:.0f}% word budget (TextFooler, phishing-only)...")
    test_adv = test_df_phish.copy()
    adv_texts, n_changed_list = [], []
    for _, row in tqdm(test_adv.iterrows(), total=len(test_adv), desc=f"TextFooler {percent*100:.0f}%"):
        adv_text, n_changed = textfooler_attack(
            str(row['email_text']), str(row['url']), int(row['label']), change_percent=percent
        )
        adv_texts.append(adv_text)
        n_changed_list.append(n_changed)
    test_adv['adv_email'] = adv_texts
    adv = evaluate(test_adv, use_adv=True)
    adv['mean_words_changed'] = float(np.mean(n_changed_list))

    pre_correct = [i for i, p in enumerate(clean_phish['y_prob']) if p > 0.5]
    n_evaded = sum(1 for i in pre_correct if adv['y_prob'][i] <= 0.5)
    evasion_rate = n_evaded / len(pre_correct) if pre_correct else float('nan')
    adv['evasion_success_rate'] = evasion_rate
    adv['n_pre_correct'] = len(pre_correct)
    adv['n_evaded'] = n_evaded

    print(
        f"   Phishing-only Acc: {adv['accuracy']:.4f} | "
        f"Evasion success rate: {evasion_rate:.4f} ({n_evaded}/{len(pre_correct)}) | "
        f"mean words changed: {adv['mean_words_changed']:.2f}"
    )
    all_results[f"textfooler_{int(percent*100)}"] = adv

with open("textfooler_raw_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved raw per-example predictions to textfooler_raw_results.json")
