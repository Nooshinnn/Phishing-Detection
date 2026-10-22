"""
Adversarial_Coordinated_LLMParaphrase_URLGAN.py

Coordinated cross-modal attack: LLM paraphrasing (FLAN-T5-large, classifier-
guided, email text) + URLGAN (homoglyph substitution, URL), applied
TOGETHER. Compares three conditions: LLM-paraphrase alone, URLGAN alone,
both combined.

Heavier than the other coordinated scripts (FLAN-T5-large generation is
the slowest single step in this whole project) -- budget accordingly.
URLGAN uses a fixed 30% budget (not a strength sweep).
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import json

from model_and_data_utils import load_model, build_dataset, DEVICE

torch.manual_seed(42)
np.random.seed(42)
device = DEVICE

print("=" * 90)
print("COORDINATED ATTACK: LLM Paraphrase (text) + URLGAN (URL), applied together")
print("=" * 90)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
train_df, val_df, test_df = build_dataset()
test_df_phish = test_df[test_df['label'] == 1].reset_index(drop=True)
print(f"Evaluating on phishing-labeled test examples only: {len(test_df_phish)} / {len(test_df)}")

print("\nLoading FLAN-T5-large (slow)...")
paraphraser = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
para_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
paraphraser.eval()

HOMOGLYPHS = {
    'o': '0', 'l': '1', 'i': '1', 'e': '3', 'a': '4',
    's': '5', 'g': '9', 'b': '6', 't': '7', 'z': '2',
}

PROMPT = (
    "You are a highly skilled professional writer. Rewrite the following email to be "
    "extremely convincing, natural, formal, and trustworthy. Make it sound like an "
    "important legitimate business or personal message. Use polite, urgent, and "
    "professional English. Keep the exact same meaning and intent but make it "
    "very difficult to detect as suspicious or phishing:\n\n{text}\n\n"
    "Professional Rewritten Email (in English):"
)


def true_class_prob(text, url, label):
    with torch.no_grad():
        e_in = email_tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer([url], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        prob_phish = torch.sigmoid(comm_model(e_feat, u_feat)).item()
    return prob_phish if label == 1 else (1 - prob_phish)


def is_english_text(text, min_chars=15):
    from langdetect import detect, LangDetectException
    if len(str(text).strip()) < min_chars:
        return True
    try:
        return detect(str(text)) == "en"
    except LangDetectException:
        return False


def llm_paraphrase_attack(text, url, label, n_candidates=4, max_retries=2):
    prompt = PROMPT.format(text=text)
    inputs = para_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_len = inputs.input_ids.shape[1]
    min_new = min(int(input_len * 0.7), 500)
    original_true_prob = true_class_prob(text, url, label)

    candidates = []
    for attempt in range(max_retries + n_candidates):
        if len(candidates) >= n_candidates:
            break
        with torch.no_grad():
            outputs = paraphraser.generate(
                **inputs, max_length=512, min_new_tokens=min_new, num_beams=10,
                temperature=0.95, top_p=0.98, do_sample=True, early_stopping=True,
                repetition_penalty=1.25,
            )
        result = para_tokenizer.decode(outputs[0], skip_special_tokens=True)
        result_len_ok = len(result.split()) >= 0.4 * len(text.split()[:int(input_len * 0.75)])
        if is_english_text(result) and result_len_ok:
            candidates.append(result)

    if not candidates:
        return text

    best_text, best_prob = text, original_true_prob
    for cand in candidates:
        p = true_class_prob(cand, url, label)
        if p < best_prob:
            best_prob, best_text = p, cand
    return best_text


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
            emails = [str(t) for t in batch[text_col].tolist()]
            urls = [str(u) for u in batch[url_col].tolist()]
            e_in = email_tokenizer(emails, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(urls, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            prob = torch.sigmoid(logits).cpu().numpy()
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
for name in ["llmparaphrase_only", "urlgan_only", "coordinated"]:
    print(f"\n-> Running condition: {name}")
    adv_texts, adv_urls = [], []
    for _, row in tqdm(test_df_phish.iterrows(), total=len(test_df_phish), desc=name):
        text, url, label = str(row['email_text']), str(row['url']), int(row['label'])
        if name == "llmparaphrase_only":
            adv_texts.append(llm_paraphrase_attack(text, url, label))
            adv_urls.append(url)
        elif name == "urlgan_only":
            adv_texts.append(text)
            adv_urls.append(urlgan_attack(text, url, label, CHANGE_PERCENT))
        else:
            new_text = llm_paraphrase_attack(text, url, label)
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
    print(f"{name:<20} evasion: {r['evasion_success_rate']:.4f} ({r['n_evaded']}/{len(pre_correct)})")

coord_gain = conditions['coordinated']['evasion_success_rate'] - max(
    conditions['llmparaphrase_only']['evasion_success_rate'], conditions['urlgan_only']['evasion_success_rate']
)
print(f"\nCoordinated attack gain over best single-modality attack: {coord_gain:+.4f}")

with open("coordinated_llmparaphrase_urlgan_results.json", "w") as f:
    json.dump({"clean": clean, **conditions}, f, indent=2)
print("\nSaved coordinated_llmparaphrase_urlgan_results.json")
