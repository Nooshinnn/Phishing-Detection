import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import json
import random

from model_and_data_utils import load_model, build_dataset, DEVICE

torch.manual_seed(42)
random.seed(42)

print("=" * 90)
print("ADVERSARIAL ATTACK: LLM Paraphrasing (FLAN-T5-large) -- REBUILT")
print("Using canonical checkpoint: original_best_model_fixed.pth")
print("=" * 90)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
train_df, val_df, test_df = build_dataset()

# FIX: matches the phishing-only evasion methodology used everywhere else in
# the attack suite (PWWS/BAE/TextFooler/PDGAN/URLGAN) -- this script was the
# one remaining attack still evaluating the mixed test set, which obscures
# the actual evasion rate on phishing examples specifically.
test_df_phish = test_df[test_df['label'] == 1].reset_index(drop=True)
print(f"Evaluating evasion on phishing-labeled test examples only: "
      f"{len(test_df_phish)} / {len(test_df)}")

print("\nLoading FLAN-T5-large...")
paraphraser = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(DEVICE)
para_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")


def is_english_text(text, min_chars=15):
    """Whole-output language check (different from the per-word check used in
    TextFooler/PWWS -- paraphrasing produces full sentences, not word swaps)."""
    from langdetect import detect, LangDetectException
    if len(str(text).strip()) < min_chars:
        return True
    try:
        return detect(str(text)) == "en"
    except LangDetectException:
        return False


PROMPTS = {
    "strong": (
        "You are a highly skilled professional writer. Rewrite the following email to be "
        "extremely convincing, natural, formal, and trustworthy. Make it sound like an "
        "important legitimate business or personal message. Use polite, urgent, and "
        "professional English. Keep the exact same meaning and intent but make it "
        "very difficult to detect as suspicious or phishing:\n\n{text}\n\n"
        "Professional Rewritten Email (in English):"
    ),
}


def true_class_prob(text, url, label):
    """P(model assigns the TRUE label) -- consistent with how every other
    attack in this suite scores candidates."""
    with torch.no_grad():
        e_in = email_tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        u_in = url_tokenizer([url], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        prob_phish = torch.sigmoid(comm_model(e_feat, u_feat)).item()
    return prob_phish if label == 1 else (1 - prob_phish)


def generate_paraphrase(text, url, label, prompt_style, n_candidates=4, max_retries=2):
    """
    FIX: this attack was the only one in the suite that never queried the
    victim classifier -- it generated one blind sample and hoped the prompt
    wording alone did the work. Now generates n_candidates via sampling and
    keeps whichever ACTUALLY reduces true-label confidence most, matching
    the query-based, classifier-guided pattern used by PWWS/TextFooler/BAE/
    PDGAN. This is a genuine methodological strengthening (uses real
    feedback to pick the best candidate), not just a more aggressive prompt.
    """
    prompt = PROMPTS[prompt_style].format(text=text)
    inputs = para_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    input_len = inputs.input_ids.shape[1]
    min_new = min(int(input_len * 0.7), 500)
    original_true_prob = true_class_prob(text, url, label)

    candidates = []
    for attempt in range(max_retries + n_candidates):
        if len(candidates) >= n_candidates:
            break
        with torch.no_grad():
            outputs = paraphraser.generate(
                **inputs,
                max_length=512,
                min_new_tokens=min_new,
                num_beams=10,
                temperature=0.95,
                top_p=0.98,
                do_sample=True,
                early_stopping=True,
                repetition_penalty=1.25,
            )
        result = para_tokenizer.decode(outputs[0], skip_special_tokens=True)
        result_len_ok = len(result.split()) >= 0.4 * len(text.split()[:int(input_len * 0.75)])
        if is_english_text(result) and result_len_ok:
            candidates.append(result)

    if not candidates:
        return text, -1, 0 

    best_text, best_prob = text, original_true_prob
    for cand in candidates:
        p = true_class_prob(cand, url, label)
        if p < best_prob:
            best_prob = p
            best_text = cand

    return best_text, 0, len(candidates)


def evaluate(df_eval, use_adv=False, batch_size=8):
    y_true, y_prob = [], []
    col = 'adv_email' if use_adv else 'email_text'
    with torch.no_grad():
        for i in range(0, len(df_eval), batch_size):
            batch = df_eval.iloc[i:i + batch_size]
            emails = [str(t) for t in batch[col].tolist()]
            urls = [str(u) for u in batch['url'].tolist()]
            e_in = email_tokenizer(emails, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
            u_in = url_tokenizer(urls, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            prob = torch.sigmoid(logits).cpu().numpy()
            y_true.extend(batch['label'].values)
            y_prob.extend(prob)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "y_true": [int(x) for x in y_true],
        "y_prob": [float(x) for x in y_prob],
    }


print("\nEvaluating clean baseline...")
clean_full = evaluate(test_df, use_adv=False)
clean_phish = evaluate(test_df_phish, use_adv=False)
print(f"Full-set clean baseline (reference) -> F1: {clean_full['f1']:.4f} | Acc: {clean_full['accuracy']:.4f}")
print(f"Phishing-only clean accuracy         -> {clean_phish['accuracy']:.4f}")

all_results = {"clean_full": clean_full, "clean_phish_only": clean_phish}
SAMPLE_N = 10 

style = "strong"  
print(f"\n-> Generating paraphrases with '{style}' prompt (classifier-guided, "
      f"best-of-4 candidates per email)...")
test_adv = test_df_phish.copy()
adv_texts = []
fallback_count = 0
n_candidates_list = []
for _, row in tqdm(test_adv.iterrows(), total=len(test_adv), desc=f"LLM paraphrase ({style})"):
    result, attempt, n_cand = generate_paraphrase(
        str(row['email_text']), str(row['url']), int(row['label']), style
    )
    if attempt == -1:
        fallback_count += 1
    adv_texts.append(result)
    n_candidates_list.append(n_cand)
test_adv['adv_email'] = adv_texts
print(f"   ({fallback_count}/{len(test_adv)} examples fell back to original text "
      f"after failing the English/length check on every attempt)")
print(f"   Mean valid candidates generated per email: {np.mean(n_candidates_list):.2f}")

adv = evaluate(test_adv, use_adv=True)

pre_correct = [i for i, p in enumerate(clean_phish['y_prob']) if p > 0.5]
n_evaded = sum(1 for i in pre_correct if adv['y_prob'][i] <= 0.5)
evasion_rate = n_evaded / len(pre_correct) if pre_correct else float('nan')
adv['evasion_success_rate'] = evasion_rate
adv['n_pre_correct'] = len(pre_correct)
adv['n_evaded'] = n_evaded
adv['fallback_count'] = fallback_count

print(f"   Phishing-only Acc: {adv['accuracy']:.4f} | "
      f"Evasion success rate: {evasion_rate:.4f} ({n_evaded}/{len(pre_correct)})")
all_results[f"llm_paraphrase_{style}"] = adv

sample_df = test_adv.sample(n=min(SAMPLE_N, len(test_adv)), random_state=42)
samples = [
    {"label": int(r['label']), "original": r['email_text'], "paraphrased": r['adv_email']}
    for _, r in sample_df.iterrows()
]
with open(f"llm_paraphrase_{style}_samples.json", "w") as f:
    json.dump(samples, f, indent=2)

with open("llm_paraphrase_raw_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved raw results to llm_paraphrase_raw_results.json")
print(f"Saved sample before/after pairs to llm_paraphrase_{style}_samples.json")
