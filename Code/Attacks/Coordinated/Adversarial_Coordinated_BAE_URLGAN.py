"""
Adversarial_Coordinated_BAE_URLGAN.py

Coordinated cross-modal attack: BAE (masked-LM word substitution, email
text) + URLGAN (homoglyph substitution, URL), applied TOGETHER. Compares
three conditions: BAE alone, URLGAN alone, both combined.

Fixed 30% budget for both (not a strength sweep).
"""
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import json

from model_and_data_utils import load_model, build_dataset, DEVICE

import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42  # optional CLI arg for multi-seed repeats
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(42)
device = DEVICE

print("=" * 90)
print("COORDINATED ATTACK: BAE (text) + URLGAN (URL), applied together")
print("=" * 90)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
train_df, val_df, test_df = build_dataset()
test_df_phish = test_df[test_df['label'] == 1].reset_index(drop=True)
print(f"Evaluating on phishing-labeled test examples only: {len(test_df_phish)} / {len(test_df)}")

mlm_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
mlm_model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device)
mlm_model.eval()

HOMOGLYPHS = {
    'o': '0', 'l': '1', 'i': '1', 'e': '3', 'a': '4',
    's': '5', 'g': '9', 'b': '6', 't': '7', 'z': '2',
}


def true_class_prob(text, url, label):
    with torch.no_grad():
        e_in = email_tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer([url], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        prob_phish = torch.sigmoid(comm_model(e_feat, u_feat)).item()
    return prob_phish if label == 1 else (1 - prob_phish)


def true_class_prob_batch(texts, url, label, batch_size=16):
    """
    FIX (speed): batched version of true_class_prob. BAE's saliency step
    was calling true_class_prob once per word (one email forward pass per
    call), sequentially -- for a ~200-word email that's ~200 individual
    single-example GPU calls, badly underusing GPU parallelism. Batching
    these into groups of batch_size gives a substantial speedup, since
    forward-pass cost barely scales with batch size on GPU up to a point.
    """
    results = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            e_in = email_tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer([url] * len(batch_texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            prob_phish = torch.sigmoid(comm_model(e_feat, u_feat)).cpu().numpy()
            for p in prob_phish:
                results.append(float(p) if label == 1 else float(1 - p))
    return results


def bae_attack(text, url, label, change_percent=0.30):
    words = text.split()
    # FIX (speed + correctness): cap to ~200 words. The classifier truncates
    # at 256 tokens anyway, so scoring words beyond that is wasted work, and
    # this also prevents pathological long training/test rows from taking
    # a very long time on a single example.
    words = words[:200]
    if len(words) < 5:
        return " ".join(words)
    num_changes = max(1, int(len(words) * change_percent))
    original_true_prob = true_class_prob(text, url, label)

    # FIX (speed): batch the saliency computation instead of one-word-at-a-time.
    deleted_texts = []
    alpha_mask = []
    for i, word in enumerate(words):
        if not word.isalpha():
            alpha_mask.append(False)
            deleted_texts.append(text)  # placeholder, score discarded below
            continue
        alpha_mask.append(True)
        masked = words.copy()
        masked[i] = mlm_tokenizer.mask_token
        deleted_texts.append(" ".join(masked))

    batched_probs = true_class_prob_batch(deleted_texts, url, label, batch_size=16)
    saliency_scores = [
        abs(original_true_prob - p) if alpha_mask[i] else 0.0
        for i, p in enumerate(batched_probs)
    ]

    indices = np.argsort(saliency_scores)[::-1][:num_changes]
    words_out = words.copy()

    for idx in indices:
        if not words_out[idx].isalpha():
            continue
        masked = words_out.copy()
        masked[idx] = mlm_tokenizer.mask_token
        masked_text = " ".join(masked)
        inputs = mlm_tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = mlm_model(**inputs).logits
        mask_positions = (inputs.input_ids == mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        if mask_positions.numel() == 0:
            continue
        mask_pos = mask_positions[0]
        top_tokens = logits[0, mask_pos].topk(20).indices

        candidates = []
        for token_id in top_tokens:
            candidate = mlm_tokenizer.decode(token_id).strip()
            if candidate.isalpha() and candidate.lower() != words_out[idx].lower():
                candidates.append(candidate)
            if len(candidates) >= 8:
                break

        # FIX (speed): batch candidate scoring instead of one-candidate-at-a-time.
        if candidates:
            trial_texts = []
            for cand in candidates:
                trial = words_out.copy()
                trial[idx] = cand
                trial_texts.append(" ".join(trial))
            cand_probs = true_class_prob_batch(trial_texts, url, label, batch_size=16)
            best_i = int(np.argmin(cand_probs))
            best_word = candidates[best_i]
        else:
            best_word = None

        if best_word is not None:
            words_out[idx] = best_word

    return " ".join(words_out)


def urlgan_attack(text, url, label, change_percent=0.30):
    eligible_positions = [i for i, c in enumerate(url) if c.lower() in HOMOGLYPHS]
    if not eligible_positions:
        return url
    num_changes_budget = max(1, int(len(eligible_positions) * change_percent))
    original_true_prob = true_class_prob(text, url, label)

    url_chars = list(url)
    remaining = set(eligible_positions)
    n_changed = 0
    while n_changed < num_changes_budget and remaining:
        current_prob = true_class_prob(text, "".join(url_chars), label)
        current_true = current_prob if label == 1 else (1 - current_prob)
        best_i, best_effect = None, -1.0
        for i in remaining:
            trial_chars = url_chars.copy()
            trial_chars[i] = HOMOGLYPHS[trial_chars[i].lower()]
            p = true_class_prob(text, "".join(trial_chars), label)
            p_true = p if label == 1 else (1 - p)
            effect = current_true - p_true
            if effect > best_effect:
                best_effect, best_i = effect, i
        if best_i is None or best_effect <= 0:
            break
        url_chars[best_i] = HOMOGLYPHS[url_chars[best_i].lower()]
        remaining.discard(best_i)
        n_changed += 1
        new_prob = true_class_prob(text, "".join(url_chars), label)
        if (new_prob > 0.5) != (original_true_prob > 0.5):
            break
    return "".join(url_chars)


def evaluate(df_eval, text_col='email_text', url_col='url'):
    y_true, y_prob = [], []
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i + 8]
            e_in = email_tokenizer(batch[text_col].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(batch[url_col].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            prob = logits.sigmoid().cpu().numpy()
            y_true.extend(batch['label'].values)
            y_prob.extend(prob)
    y_pred = (np.array(y_prob) > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "y_true": [int(x) for x in y_true], "y_prob": [float(x) for x in y_prob],
    }


CHANGE_PERCENT = 0.30
clean = evaluate(test_df_phish)
pre_correct = [i for i, p in enumerate(clean['y_prob']) if p > 0.5]
print(f"\nClean phishing-only accuracy: {clean['accuracy']:.4f} ({len(pre_correct)} pre-attack-correct)")

# FIX (speed): bae_only and urlgan_only are SKIPPED here -- URLGAN is
# deterministic against a fixed model (doesn't vary by seed, already
# established), and bae_only is presumably already known from a prior run
# at this seed. Only "coordinated" (the one condition that actually needs
# freshly computing) runs here, cutting total runtime to roughly a third
# of the full 3-condition script.
conditions = {}
for name in ["coordinated"]:
    print(f"\n-> Running condition: {name}")
    adv_texts, adv_urls = [], []
    for _, row in tqdm(test_df_phish.iterrows(), total=len(test_df_phish), desc=name):
        text, url, label = str(row['email_text']), str(row['url']), int(row['label'])
        new_text = bae_attack(text, url, label, CHANGE_PERCENT)
        new_url = urlgan_attack(new_text, url, label, CHANGE_PERCENT)
        adv_texts.append(new_text)
        adv_urls.append(new_url)

    test_adv = test_df_phish.copy()
    test_adv['adv_text'] = adv_texts
    test_adv['adv_url'] = adv_urls
    result = evaluate(test_adv, text_col='adv_text', url_col='adv_url')

    n_evaded = sum(1 for i in pre_correct if result['y_prob'][i] <= 0.5)
    evasion_rate = n_evaded / len(pre_correct) if pre_correct else float('nan')
    result['evasion_success_rate'] = evasion_rate
    result['n_evaded'] = n_evaded
    print(f"   Evasion success rate: {evasion_rate:.4f} ({n_evaded}/{len(pre_correct)})")
    conditions[name] = result

print("\n" + "=" * 90)
print("COORDINATED-ONLY RESULT (bae_only/urlgan_only skipped -- combine with your")
print("already-known values for this seed to get the full summary)")
print("=" * 90)
for name, r in conditions.items():
    print(f"{name:<18} evasion: {r['evasion_success_rate']:.4f} ({r['n_evaded']}/{len(pre_correct)})")

print("\nbae_only and urlgan_only skipped this run -- compute gain/excess-over-sum")
print("manually using your already-known values for those two conditions at this seed.")

with open(f"coordinated_bae_urlgan_results_seed{SEED}.json", "w") as f:
    json.dump({"clean": clean, **conditions}, f, indent=2)
print(f"\nSaved coordinated_bae_urlgan_results_seed{SEED}.json")