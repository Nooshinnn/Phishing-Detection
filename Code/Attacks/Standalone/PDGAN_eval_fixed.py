import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import AutoModelForMaskedLM
from tqdm import tqdm
import json

from model_and_data_utils import load_model, build_dataset, DEVICE

print("=" * 90)
print("PDGAN ATTACK EVALUATION -- FIXED: generator never trained on this test_df")
print("Using canonical checkpoint: original_best_model_fixed.pth")
print("=" * 90)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
train_df, val_df, test_df = build_dataset()

# Usage: python PDGAN_eval_fixed.py [seed] [fluency_weight]
#   no args -> default generator pdgan_generator_fixed_seed42.pth, default output names
#   with a seed (and optional lambda) -> loads pdgan_generator_fixed_<tag>.pth
#   and writes pdgan_raw_results_<tag>.json / pdgan_samples_<tag>.json
import sys
if len(sys.argv) > 1:
    EVAL_SEED = int(sys.argv[1])
    _lam = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    TAG = f"seed{EVAL_SEED}" + ("" if _lam == 5.0 else f"_lam{_lam}")
    GENERATOR_CHECKPOINT = f"pdgan_generator_fixed_{TAG}.pth"
    RAW_OUT, SAMPLES_OUT = f"pdgan_raw_results_{TAG}.json", f"pdgan_samples_{TAG}.json"
else:
    EVAL_SEED, TAG = 42, None
    GENERATOR_CHECKPOINT = "pdgan_generator_fixed_seed42.pth"
    RAW_OUT, SAMPLES_OUT = "pdgan_raw_results.json", "pdgan_samples.json"
torch.manual_seed(EVAL_SEED)   # the masking at attack time is random: fix it
np.random.seed(EVAL_SEED)


class PDGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlm = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
    def forward(self, input_ids, attention_mask=None):
        return self.mlm(input_ids=input_ids, attention_mask=attention_mask).logits


generator = PDGAN_Generator().to(DEVICE)
generator.load_state_dict(torch.load(GENERATOR_CHECKPOINT, map_location=DEVICE))
generator.eval()
print("PDGAN generator loaded.")


def is_english_text(text, min_chars=15):
    from langdetect import detect, LangDetectException
    if len(str(text).strip()) < min_chars:
        return True
    try:
        return detect(str(text)) == "en"
    except LangDetectException:
        return False


def generate_pdgan_adv(text, mask_prob=0.15):
    # FIX: was 0.28, nearly double the training mask_prob (0.15 in
    # PDGAN_train_fixed.py). The generator's fluency regularization was
    # only ever trained/validated at ~15% simultaneous masking -- testing
    # at a much higher masking density asks it to handle a regime it never
    # saw, which could independently degrade coherence regardless of the
    # FLUENCY_WEIGHT retuning. Aligned to match training exactly.
    text = str(text)
    if len(text) < 20:
        return text, True  # too short to attack meaningfully; not a fallback

    inputs = email_tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    input_ids = inputs.input_ids

    mask = torch.rand_like(input_ids, dtype=torch.float32, device=DEVICE) < mask_prob
    mask = mask & (input_ids != email_tokenizer.pad_token_id)
    input_ids_masked = input_ids.clone()
    input_ids_masked[mask] = email_tokenizer.mask_token_id

    with torch.no_grad():
        logits = generator(input_ids_masked, inputs.attention_mask)
        fake_ids = logits.argmax(dim=-1)
        fake_ids = torch.where(mask, fake_ids, input_ids)

    adv_text = email_tokenizer.decode(fake_ids[0], skip_special_tokens=True)

    # English-only check on the full decoded output, with fallback to original
    # (same pattern as the LLM paraphrasing script -- consistent across attacks).
    if is_english_text(adv_text):
        return adv_text, True
    return text, False


def evaluate(df_eval, use_adv=False):
    y_true, y_prob = [], []
    col = 'adv_email' if use_adv else 'email_text'
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i + 8]
            emails = batch[col].tolist()
            urls = batch['url'].tolist()
            e_in = email_tokenizer(emails, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
            u_in = url_tokenizer(urls, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
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


print("\nGenerating adversarial emails with PDGAN (on held-out test_df only)...")

# FIX: matches the retrained generator's objective -- evaluate evasion success
# on PHISHING-labeled test examples only. The original version ran the attack
# over the whole test set (both classes), which is why the previous run's
# headline F1 drop was almost entirely driven by legitimate emails being
# pushed toward false-positive phishing predictions (a different, explicitly
# out-of-scope attack per Limitations) rather than genuine evasion.
test_df_phish = test_df[test_df['label'] == 1].reset_index(drop=True)
print(f"Evaluating evasion on phishing-labeled test examples only: "
      f"{len(test_df_phish)} / {len(test_df)}")

test_adv = test_df_phish.copy()
adv_texts, kept_flags = [], []
for t in tqdm(test_adv['email_text']):
    adv_text, kept = generate_pdgan_adv(t)
    adv_texts.append(adv_text)
    kept_flags.append(kept)
test_adv['adv_email'] = adv_texts
n_fallback = sum(1 for k in kept_flags if not k)
print(f"({n_fallback}/{len(test_adv)} examples fell back to original text after failing the English check)")

clean_full = evaluate(test_df, use_adv=False)          # full-set reference, both classes
clean_phish = evaluate(test_df_phish, use_adv=False)    # phishing-only baseline
adv_phish = evaluate(test_adv, use_adv=True)             # phishing-only, post-attack

# Evasion success rate: of the phishing examples correctly detected pre-attack,
# what fraction flip to "legitimate" post-attack? This is the number that
# actually answers "does PDGAN evade this detector," unlike aggregate F1/Acc
# over a mixed-class set.
pre_correct = [i for i, p in enumerate(clean_phish['y_prob']) if p > 0.5]
n_pre_correct = len(pre_correct)
n_evaded = sum(1 for i in pre_correct if adv_phish['y_prob'][i] <= 0.5)
evasion_rate = n_evaded / n_pre_correct if n_pre_correct else float('nan')

print("\n" + "=" * 80)
print("PDGAN ATTACK RESULTS (leakage-free, evasion-only: phishing-labeled examples)")
print("=" * 80)
print(f"Full-set clean baseline (reference)     -> F1: {clean_full['f1']:.4f} | Acc: {clean_full['accuracy']:.4f}")
print(f"Phishing-only clean accuracy             -> {clean_phish['accuracy']:.4f}")
print(f"Phishing-only accuracy AFTER PDGAN        -> {adv_phish['accuracy']:.4f}")
print(f"Evasion success rate (of pre-attack correct detections) -> {evasion_rate:.4f} "
      f"({n_evaded}/{n_pre_correct})")

results = {
    "clean_full": clean_full,
    "clean_phish_only": clean_phish,
    "pdgan_phish_only": adv_phish,
    "evasion_success_rate": evasion_rate,
    "n_pre_correct": n_pre_correct,
    "n_evaded": n_evaded,
    "fallback_count": n_fallback,
}
with open(RAW_OUT, "w") as f:
    json.dump(results, f, indent=2)

# Save representative before/after samples for Appendix F
sample_df = test_adv.sample(n=min(10, len(test_adv)), random_state=42)
samples = [
    {"label": int(r['label']), "original": r['email_text'], "adversarial": r['adv_email']}
    for _, r in sample_df.iterrows()
]
with open(SAMPLES_OUT, "w") as f:
    json.dump(samples, f, indent=2)

print(f"\nSaved {RAW_OUT} and {SAMPLES_OUT}")