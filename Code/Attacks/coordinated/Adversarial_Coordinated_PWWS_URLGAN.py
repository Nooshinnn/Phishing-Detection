
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import json
import nltk
from nltk.corpus import wordnet

from model_and_data_utils import load_model, build_dataset, is_english_word, DEVICE

for pkg in ["wordnet", "omw-1.4", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42  # optional CLI arg for multi-seed repeats
torch.manual_seed(SEED)
np.random.seed(SEED)
device = DEVICE

print("=" * 90)
print("COORDINATED ATTACK: PWWS (text) + URLGAN (URL), applied together")
print("=" * 90)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
train_df, val_df, test_df = build_dataset()
test_df_phish = test_df[test_df['label'] == 1].reset_index(drop=True)
print(f"Evaluating on phishing-labeled test examples only: {len(test_df_phish)} / {len(test_df)}")

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


def get_wordnet_synonyms(word, max_candidates=8):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace('_', ' ')
            if candidate.isalpha() and candidate.lower() != word.lower():
                synonyms.add(candidate)
            if len(synonyms) >= max_candidates:
                break
        if len(synonyms) >= max_candidates:
            break
    return list(synonyms)


def pwws_attack(text, url, label, change_percent=0.30):
    words = text.split()
    if len(words) < 5:
        return text
    num_changes_budget = max(1, int(len(words) * change_percent))
    original_true_prob = true_class_prob(text, url, label)
    true_prob_of_label = original_true_prob if label == 1 else (1 - original_true_prob)

    saliency = []
    for i, word in enumerate(words):
        if not word.isalpha():
            saliency.append(0.0)
            continue
        deleted = words[:i] + words[i+1:]
        deleted_text = " ".join(deleted) if deleted else word
        p = true_class_prob(deleted_text, url, label)
        p_true = p if label == 1 else (1 - p)
        saliency.append(true_prob_of_label - p_true)

    best_subs = {}
    delta = []
    for i, word in enumerate(words):
        if not word.isalpha() or saliency[i] == 0.0:
            delta.append(0.0)
            continue
        candidates = get_wordnet_synonyms(word)
        if not candidates:
            delta.append(0.0)
            continue
        best_drop, best_word = -1.0, None
        for cand in candidates:
            trial = words.copy()
            trial[i] = cand
            p = true_class_prob(" ".join(trial), url, label)
            p_true = p if label == 1 else (1 - p)
            drop = true_prob_of_label - p_true
            if drop > best_drop:
                best_drop, best_word = drop, cand
        delta.append(max(best_drop, 0.0))
        if best_word is not None:
            best_subs[i] = best_word

    saliency_arr = np.array(saliency)
    exp_s = np.exp(saliency_arr - saliency_arr.max())
    softmax_s = exp_s / exp_s.sum() if exp_s.sum() > 0 else exp_s
    combined_score = softmax_s * np.array(delta)
    ranked_indices = np.argsort(combined_score)[::-1]

    attacked_words = words.copy()
    n_changed = 0
    original_decision = original_true_prob > 0.5
    for idx in ranked_indices:
        if n_changed >= num_changes_budget:
            break
        if idx not in best_subs or combined_score[idx] <= 0:
            continue
        attacked_words[idx] = best_subs[idx]
        n_changed += 1
        current_prob = true_class_prob(" ".join(attacked_words), url, label)
        if (current_prob > 0.5) != original_decision:
            break
    return " ".join(attacked_words)


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

conditions = {}
for name in ["pwws_only", "urlgan_only", "coordinated"]:
    print(f"\n-> Running condition: {name}")
    adv_texts, adv_urls = [], []
    for _, row in tqdm(test_df_phish.iterrows(), total=len(test_df_phish), desc=name):
        text, url, label = str(row['email_text']), str(row['url']), int(row['label'])
        if name == "pwws_only":
            adv_texts.append(pwws_attack(text, url, label, CHANGE_PERCENT))
            adv_urls.append(url)
        elif name == "urlgan_only":
            adv_texts.append(text)
            adv_urls.append(urlgan_attack(text, url, label, CHANGE_PERCENT))
        else: 
            new_text = pwws_attack(text, url, label, CHANGE_PERCENT)
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
print("SUMMARY")
print("=" * 90)
for name, r in conditions.items():
    print(f"{name:<15} evasion: {r['evasion_success_rate']:.4f} ({r['n_evaded']}/{len(pre_correct)})")

coord_gain_over_best_single = conditions['coordinated']['evasion_success_rate'] - max(
    conditions['pwws_only']['evasion_success_rate'], conditions['urlgan_only']['evasion_success_rate']
)
print(f"\nCoordinated attack gain over best single-modality attack: {coord_gain_over_best_single:+.4f}")
print("If this gain is large: coordinated cross-modal attacks are a meaningfully bigger threat")
print("than single-modality attacks studied in isolation -- worth highlighting in the paper.")
print("If small/negative: single-modality results are not substantially understating risk.")

with open(f"coordinated_attack_results_seed{SEED}.json", "w") as f:
    json.dump({"clean": clean, **conditions}, f, indent=2)
print("\nSaved coordinated_attack_results.json")
