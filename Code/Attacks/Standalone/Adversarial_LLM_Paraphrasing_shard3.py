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

SHARD_INDEX = 2   
TOTAL_SHARDS = 3

print("=" * 90)
print(f"ADVERSARIAL ATTACK: LLM Paraphrasing (FLAN-T5-large) -- SHARD {SHARD_INDEX+1}/{TOTAL_SHARDS}")
print("Using shared module -- canonical checkpoint & combined corpus, auto-selected")
print("=" * 90)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
train_df, val_df, test_df = build_dataset()

test_df_phish = test_df[test_df['label'] == 1].reset_index(drop=True)
print(f"Full phishing-labeled test set: {len(test_df_phish)} / {len(test_df)}")

shard_size = len(test_df_phish) // TOTAL_SHARDS
start = SHARD_INDEX * shard_size
end = (SHARD_INDEX + 1) * shard_size if SHARD_INDEX < TOTAL_SHARDS - 1 else len(test_df_phish)
shard_df = test_df_phish.iloc[start:end].reset_index(drop=True)
print(f"This shard covers rows [{start}:{end}] -> {len(shard_df)} examples")

print("\nLoading FLAN-T5-large...")
paraphraser = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(DEVICE)
para_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")


def is_english_text(text, min_chars=15):
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
    with torch.no_grad():
        e_in = email_tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        u_in = url_tokenizer([url], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        prob_phish = torch.sigmoid(comm_model(e_feat, u_feat)).item()
    return prob_phish if label == 1 else (1 - prob_phish)


def generate_paraphrase(text, url, label, prompt_style, n_candidates=4, max_retries=2):
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

    
    if best_text == text:
        return best_text, 1, len(candidates)  
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


print("\nEvaluating clean baseline (full test set, for consistent evasion-rate denominator)...")
clean_phish_full = evaluate(test_df_phish, use_adv=False)
print(f"Full phishing-only clean accuracy -> {clean_phish_full['accuracy']:.4f}")

style = "strong"
print(f"\n-> Generating paraphrases with '{style}' prompt for THIS SHARD only "
      f"({len(shard_df)} examples, rows [{start}:{end}])...")
shard_adv = shard_df.copy()
adv_texts = []
fallback_count = 0
no_improvement_count = 0
n_candidates_list = []
for _, row in tqdm(shard_adv.iterrows(), total=len(shard_adv), desc=f"LLM paraphrase shard {SHARD_INDEX+1}"):
    result, attempt, n_cand = generate_paraphrase(
        str(row['email_text']), str(row['url']), int(row['label']), style
    )
    if attempt == -1:
        fallback_count += 1
    elif attempt == 1:
        no_improvement_count += 1
    adv_texts.append(result)
    n_candidates_list.append(n_cand)
shard_adv['adv_email'] = adv_texts
print(f"   ({fallback_count}/{len(shard_adv)} generation failures (fallback), "
      f"{no_improvement_count}/{len(shard_adv)} generated but no candidate beat original -- "
      f"TRUE no-op rate: {fallback_count + no_improvement_count}/{len(shard_adv)})")
print(f"   Mean valid candidates generated per email (this shard): {np.mean(n_candidates_list):.2f}")

shard_result = evaluate(shard_adv, use_adv=True)
print(f"\nShard {SHARD_INDEX+1}/{TOTAL_SHARDS} phishing-only Acc (this shard's rows only): {shard_result['accuracy']:.4f}")

sample_df = shard_adv.sample(n=min(4, len(shard_adv)), random_state=42)
samples = [
    {"label": int(r['label']), "original": r['email_text'], "paraphrased": r['adv_email']}
    for _, r in sample_df.iterrows()
]

output = {
    "shard_index": SHARD_INDEX,
    "total_shards": TOTAL_SHARDS,
    "row_range": [start, end],
    "clean_phish_full": clean_phish_full,
    "shard_adv_result": shard_result,
    "fallback_count": fallback_count,
    "no_improvement_count": no_improvement_count,
}
with open(f"llm_paraphrase_shard{SHARD_INDEX+1}of{TOTAL_SHARDS}.json", "w") as f:
    json.dump(output, f, indent=2)
with open(f"llm_paraphrase_shard{SHARD_INDEX+1}of{TOTAL_SHARDS}_samples.json", "w") as f:
    json.dump(samples, f, indent=2)

print(f"\nSaved llm_paraphrase_shard{SHARD_INDEX+1}of{TOTAL_SHARDS}.json")
print(f"Once all {TOTAL_SHARDS} shards finish, run combine_llm_shards.py to merge into the final result.")