import numpy as np
import torch
import random
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import json
import gensim.downloader as gensim_api

from model_and_data_utils import load_model, build_dataset, is_english_word, DEVICE

print("=" * 90)
print("ADVERSARIAL ATTACK: TextBugger (character + word level)")
print("Using canonical checkpoint: original_best_model_fixed.pth")
print("=" * 90)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
train_df, val_df, test_df = build_dataset()

test_df_phish = test_df[test_df['label'] == 1].reset_index(drop=True)
print(f"Evaluating evasion on phishing-labeled test examples only: "
      f"{len(test_df_phish)} / {len(test_df)}")

print("\nLoading GloVe word vectors for word-level substitution candidates...")
word_vectors = gensim_api.load("glove-wiki-gigaword-100")
print("Loaded.")


HOMOGLYPHS = {
    'o': '0', 'l': '1', 'i': '1', 'e': '3', 'a': '@',
    's': '5', 'g': '9', 'b': '6', 't': '7',
}


def true_class_prob(text, url, label):
    with torch.no_grad():
        e_in = email_tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        u_in = url_tokenizer([url], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        prob_phish = torch.sigmoid(comm_model(e_feat, u_feat)).item()
    return prob_phish if label == 1 else (1 - prob_phish)


import zlib


def _word_rng(word):

    seed = zlib.crc32(word.encode('utf-8'))
    return random.Random(seed)


def char_swap_adjacent(word):
    if len(word) < 4:
        return None
    rng = _word_rng(word)
    i = rng.randint(1, len(word) - 3)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def char_delete(word):
    if len(word) < 4:
        return None
    rng = _word_rng(word)
    i = rng.randint(1, len(word) - 2)
    return word[:i] + word[i + 1:]


def char_insert_space(word):
    if len(word) < 4:
        return None
    rng = _word_rng(word)
    i = rng.randint(1, len(word) - 2)
    return word[:i] + " " + word[i:]


def char_homoglyph(word):
    candidates = [c for c in word.lower() if c in HOMOGLYPHS]
    if not candidates:
        return None
    chars = list(word)
    for i, c in enumerate(word.lower()):
        if c in HOMOGLYPHS:
            chars[i] = HOMOGLYPHS[c]
            break
    return "".join(chars)


def word_embedding_substitute(word):

    word_lower = word.lower()
    if word_lower not in word_vectors:
        return None
    try:
        neighbors = word_vectors.most_similar(word_lower, topn=5)
    except KeyError:
        return None
    for cand_word, _ in neighbors:
        if cand_word.isalpha() and cand_word.lower() != word_lower and is_english_word(cand_word):
            return cand_word
    return None


PERTURBATION_FNS = [char_swap_adjacent, char_delete, char_insert_space, char_homoglyph, word_embedding_substitute]


def textbugger_attack(text, url, label, change_percent=0.30):
    words = text.split()
    if len(words) < 5:
        return text, 0

    num_changes_budget = max(1, int(len(words) * change_percent))
    original_true_prob = true_class_prob(text, url, label)


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

  
        best_drop = -1.0
        best_variant = None
        for fn in PERTURBATION_FNS:
            variant = fn(words[idx])
            if variant is None:
                continue
            trial = attacked_words.copy()
            trial[idx] = variant
            trial_text = " ".join(trial)
            p = true_class_prob(trial_text, url, label)
            drop = original_true_prob - p
            if drop > best_drop:
                best_drop = drop
                best_variant = variant

        if best_variant is not None and best_drop > 0:
            attacked_words[idx] = best_variant
            n_changed += 1
            current_full_prob = true_class_prob(" ".join(attacked_words), url, label)
            if (current_full_prob > 0.5) != (original_true_prob > 0.5):
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

print("\nRunning TextBugger attacks with different change-percent budgets...\n")
clean_full = evaluate(test_df, use_adv=False)
clean_phish = evaluate(test_df_phish, use_adv=False)
print(f"Full-set clean baseline (reference) -> F1: {clean_full['f1']:.4f} | Acc: {clean_full['accuracy']:.4f}")
print(f"Phishing-only clean accuracy         -> {clean_phish['accuracy']:.4f}")
all_results['clean_full'] = clean_full
all_results['clean_phish_only'] = clean_phish

samples_out = []

for percent in percentages:
    print(f"\n-> Attacking with up to {percent * 100:.0f}% word budget (TextBugger, phishing-only)...")
    test_adv = test_df_phish.copy()
    adv_texts, n_changed_list = [], []
    for _, row in tqdm(test_adv.iterrows(), total=len(test_adv), desc=f"TextBugger {percent*100:.0f}%"):
        adv_text, n_changed = textbugger_attack(
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
    all_results[f"textbugger_{int(percent*100)}"] = adv

    if percent == 0.30: 
        for i in range(min(10, len(test_adv))):
            samples_out.append({
                "label": int(test_adv.iloc[i]['label']),
                "original": str(test_df_phish.iloc[i]['email_text'])[:500],
                "adversarial": str(test_adv.iloc[i]['adv_email'])[:500],
            })

with open("textbugger_raw_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
with open("textbugger_samples.json", "w") as f:
    json.dump(samples_out, f, indent=2)
print("\nSaved textbugger_raw_results.json and textbugger_samples.json")
