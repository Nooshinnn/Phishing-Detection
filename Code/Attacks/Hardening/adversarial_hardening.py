
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import warnings
import random

from model_and_data_utils import load_model, build_dataset, DEVICE

warnings.filterwarnings("ignore")
SEED = 31415
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = DEVICE

HOMOGLYPHS = {
    'o': '0', 'l': '1', 'i': '1', 'e': '3', 'a': '4',
    's': '5', 'g': '9', 'b': '6', 't': '7', 'z': '2',
}

print("=" * 90)
print("Adversarial hardening: augmenting training data with URLGAN perturbations")
print("=" * 90)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
train_df, val_df, test_df = build_dataset()
train_df_phish = train_df[train_df['label'] == 1].reset_index(drop=True)
print(f"Train: {len(train_df)} (phishing subset for hardening: {len(train_df_phish)})")


def true_class_prob(text, url, label):
    with torch.no_grad():
        e_in = email_tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer([url], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        prob_phish = torch.sigmoid(comm_model(e_feat, u_feat)).item()
    return prob_phish if label == 1 else (1 - prob_phish)


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


AUGMENT_FRACTION = 0.30
augment_source = train_df_phish.sample(frac=AUGMENT_FRACTION, random_state=SEED).reset_index(drop=True)
print(f"\nGenerating URLGAN-perturbed augmentation examples for {len(augment_source)} training rows...")

adv_urls = []
for _, row in tqdm(augment_source.iterrows(), total=len(augment_source), desc="Generating hardening data"):
    adv_url = urlgan_attack(str(row['email_text']), str(row['url']), int(row['label']), change_percent=0.30)
    adv_urls.append(adv_url)

augmented_rows = augment_source.copy()
augmented_rows['url'] = adv_urls
hardened_train_df = pd.concat([train_df, augmented_rows]).sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"Hardened training set: {len(hardened_train_df)} (original {len(train_df)} + {len(augmented_rows)} adversarial)")


def run_eval(df_eval, batch_size=16):
    email_model.eval(); url_model.eval(); comm_model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for i in range(0, len(df_eval), batch_size):
            batch = df_eval.iloc[i:i+batch_size]
            e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            prob = logits.sigmoid().cpu().numpy()
            y_true.extend(batch['label'].values)
            y_prob.extend(prob)
    y_pred = (np.array(y_prob) > 0.5).astype(int)
    return {"accuracy": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred)}


optimizer = optim.AdamW(
    list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()),
    lr=8e-6, weight_decay=0.01
)
criterion = nn.BCEWithLogitsLoss()
batch_size = 16
best_val_f1 = -1.0
patience, patience_counter = 4, 0
best_state = None

print("\nFine-tuning canonical model on hardened (clean + URLGAN-augmented) training set...")
for epoch in range(10):
    email_model.train(); url_model.train(); comm_model.train()
    for i in range(0, len(hardened_train_df), batch_size):
        batch = hardened_train_df.iloc[i:i+batch_size]
        labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)
        e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        logits = comm_model(e_feat, u_feat)
        loss = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    val_metrics = run_eval(val_df)
    print(f"Epoch {epoch+1} | Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        patience_counter = 0
        best_state = {
            'email_model': {k: v.cpu().clone() for k, v in email_model.state_dict().items()},
            'url_model': {k: v.cpu().clone() for k, v in url_model.state_dict().items()},
            'comm_model': {k: v.cpu().clone() for k, v in comm_model.state_dict().items()},
        }
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

email_model.load_state_dict(best_state['email_model'])
url_model.load_state_dict(best_state['url_model'])
comm_model.load_state_dict(best_state['comm_model'])

torch.save(best_state, "hardened_model_urlgan.pth")
clean_test_result = run_eval(test_df)
print(f"\nHardened model clean test performance: {clean_test_result}")
print("(Compare against canonical model's clean F1 ~0.9938 -- hardening should not")
print("meaningfully hurt clean performance; a big drop here would be a real cost to flag.)")

print("\n" + "=" * 90)
print("NEXT STEPS -- cross-attack matrix")
print("=" * 90)
print("Saved hardened_model_urlgan.pth. To build the cross-attack matrix:")
print("1. For EACH attack script (PWWS/TextFooler/BAE/TextBugger/URLGAN/PDGAN),")
print("   run it TWICE: once with load_model() (canonical, unhardened -- already done")
print("   for most), once with load_model('hardened_model_urlgan.pth').")
print("2. To point a script at the hardened model, change its load_model() call to:")
print("     load_model('hardened_model_urlgan.pth')")
print("3. Compare evasion success rate: hardened vs unhardened, PER ATTACK.")
print("   - If hardening reduces URLGAN's evasion a lot but does little for")
print("     PWWS/TextFooler/BAE/TextBugger: attack-specific memorization (the")
print("     failure mode the review warns about).")
print("   - If hardening reduces evasion broadly across attacks: genuine,")
print("     defensible general robustness claim.")
