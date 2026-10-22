import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import json

from model_and_data_utils import load_model, build_dataset, DEVICE

print("=" * 90)
print("ADVERSARIAL ATTACK: URLGAN (structural URL perturbation, visual-similarity preserving)")
print("Using canonical checkpoint: original_best_model_fixed.pth")
print("NOTE: despite the name, this is a homoglyph/typosquat-style character")
print("substitution attack, not a generative adversarial network -- matches")
print("the mechanism actually described in the dissertation's Section 3.6.2.")
print("=" * 90)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
train_df, val_df, test_df = build_dataset()

test_df_phish = test_df[test_df['label'] == 1].reset_index(drop=True)
print(f"Evaluating evasion on phishing-labeled test examples only: "
      f"{len(test_df_phish)} / {len(test_df)}")

HOMOGLYPHS = {
    'o': '0', 'l': '1', 'i': '1', 'e': '3', 'a': '4',
    's': '5', 'g': '9', 'b': '6', 't': '7', 'z': '2',
}


def true_class_prob(text, url, label):
    with torch.no_grad():
        e_in = email_tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        u_in = url_tokenizer([url], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        prob_phish = torch.sigmoid(comm_model(e_feat, u_feat)).item()
    return prob_phish if label == 1 else (1 - prob_phish)


def urlgan_attack(text, url, label, change_percent=0.30):

    eligible_positions = [i for i, c in enumerate(url) if c.lower() in HOMOGLYPHS]
    if not eligible_positions:
        return url, 0

    num_changes_budget = max(1, int(len(eligible_positions) * change_percent))
    original_true_prob = true_class_prob(text, url, label)
    orig_true = original_true_prob if label == 1 else (1 - original_true_prob)

    url_chars = list(url)
    remaining = set(eligible_positions)
    n_changed = 0

    while n_changed < num_changes_budget and remaining:
        current_url = "".join(url_chars)
        current_prob = true_class_prob(text, current_url, label)
        current_true = current_prob if label == 1 else (1 - current_prob)

        best_i, best_effect = None, -1.0
        for i in remaining:
            trial_chars = url_chars.copy()
            trial_chars[i] = HOMOGLYPHS[trial_chars[i].lower()]
            trial_url = "".join(trial_chars)
            p = true_class_prob(text, trial_url, label)
            p_true = p if label == 1 else (1 - p)
            effect = current_true - p_true  
            if effect > best_effect:
                best_effect = effect
                best_i = i

        if best_i is None or best_effect <= 0:
            break  

        url_chars[best_i] = HOMOGLYPHS[url_chars[best_i].lower()]
        remaining.discard(best_i)
        n_changed += 1

        new_prob = true_class_prob(text, "".join(url_chars), label)
        if (new_prob > 0.5) != (original_true_prob > 0.5):
            break

    return "".join(url_chars), n_changed


def evaluate(df_eval, use_adv=False):
    y_true, y_prob = [], []
    col = 'adv_url' if use_adv else 'url'
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i + 8]
            e_in = email_tokenizer(
                batch['email_text'].tolist(), padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            ).to(DEVICE)
            u_in = url_tokenizer(
                batch[col].tolist(), padding=True, truncation=True,
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

print("\nRunning URLGAN attacks with different change-percent budgets...\n")
clean_full = evaluate(test_df, use_adv=False)
clean_phish = evaluate(test_df_phish, use_adv=False)
print(f"Full-set clean baseline (reference) -> F1: {clean_full['f1']:.4f} | Acc: {clean_full['accuracy']:.4f}")
print(f"Phishing-only clean accuracy         -> {clean_phish['accuracy']:.4f}")
all_results['clean_full'] = clean_full
all_results['clean_phish_only'] = clean_phish

samples_out = []
n_zero_eligible = 0  
                 
                   

for percent in percentages:
    print(f"\n-> Attacking with up to {percent * 100:.0f}% char budget (URLGAN, phishing-only)...")
    test_adv = test_df_phish.copy()
    adv_urls, n_changed_list = [], []
    for _, row in tqdm(test_adv.iterrows(), total=len(test_adv), desc=f"URLGAN {percent*100:.0f}%"):
        adv_url, n_changed = urlgan_attack(
            str(row['email_text']), str(row['url']), int(row['label']), change_percent=percent
        )
        adv_urls.append(adv_url)
        n_changed_list.append(n_changed)
        if percent == percentages[0] and n_changed == 0:
            n_zero_eligible += 1
    test_adv['adv_url'] = adv_urls
    adv = evaluate(test_adv, use_adv=True)
    adv['mean_chars_changed'] = float(np.mean(n_changed_list))

    pre_correct = [i for i, p in enumerate(clean_phish['y_prob']) if p > 0.5]
    n_evaded = sum(1 for i in pre_correct if adv['y_prob'][i] <= 0.5)
    evasion_rate = n_evaded / len(pre_correct) if pre_correct else float('nan')
    adv['evasion_success_rate'] = evasion_rate
    adv['n_pre_correct'] = len(pre_correct)
    adv['n_evaded'] = n_evaded

    print(
        f"   Phishing-only Acc: {adv['accuracy']:.4f} | "
        f"Evasion success rate: {evasion_rate:.4f} ({n_evaded}/{len(pre_correct)}) | "
        f"mean chars changed: {adv['mean_chars_changed']:.2f}"
    )
    all_results[f"urlgan_{int(percent*100)}"] = adv

    if percent == 0.30:
        for i in range(min(10, len(test_adv))):
            samples_out.append({
                "label": int(test_adv.iloc[i]['label']),
                "original_url": str(test_df_phish.iloc[i]['url']),
                "adversarial_url": str(test_adv.iloc[i]['adv_url']),
            })

print(f"\nURLs with ZERO homoglyph-eligible characters at 10% budget: "
      f"{n_zero_eligible}/{len(test_df_phish)} "
      f"-- if this is large, it partly explains any near-zero attack effect "
      f"WITHOUT that being a bug (nothing to perturb, not a broken perturbation).")
all_results['n_zero_eligible_urls'] = n_zero_eligible

with open("urlgan_raw_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
with open("urlgan_samples.json", "w") as f:
    json.dump(samples_out, f, indent=2)
print("\nSaved urlgan_raw_results.json and urlgan_samples.json")
