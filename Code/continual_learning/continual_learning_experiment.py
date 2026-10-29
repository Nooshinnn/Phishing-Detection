
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import json
import warnings
import random
import sys

from model_and_data_utils import DistilBertWrapper, DomURLBERT, MessagePassing, dedup_pool, DEVICE
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

# Now CLI-configurable for multi-seed repeats:
#   python continual_learning_experiment.py <held_out_source> <seed>
#   e.g. python continual_learning_experiment.py nigerian_fraud 100
HELD_OUT_SOURCE = sys.argv[1] if len(sys.argv) > 1 else "nigerian_fraud"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 31415

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = DEVICE

LR = 8e-6  # matches canonical model training -- see module docstring

print("=" * 90)
print(f"Continual learning under campaign drift: '{HELD_OUT_SOURCE}' as new campaign")
print("=" * 90)

df = pd.read_csv("combined_phishing_dataset.csv")
df = df.rename(columns={"body": "email_text"})
df['url'] = df['urls'].apply(lambda u: str(u).split(" | ")[0] if pd.notna(u) and str(u).strip() else "")
df = df[df['url'] != ""].reset_index(drop=True)
df = dedup_pool(df, near_dup_threshold=0.90, verbose=False)
print(f"After dedup: {len(df)} rows")

base_sources_df = df[df['source'] != HELD_OUT_SOURCE].reset_index(drop=True)
new_campaign_df = df[df['source'] == HELD_OUT_SOURCE].reset_index(drop=True)
print(f"Base (5 sources): {len(base_sources_df)} | New campaign ({HELD_OUT_SOURCE}): {len(new_campaign_df)}")

base_sources_df = base_sources_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
n_base_test = int(len(base_sources_df) * 0.15)
n_base_val = int(len(base_sources_df) * 0.15)
base_test_df = base_sources_df.iloc[:n_base_test]
base_val_df = base_sources_df.iloc[n_base_test:n_base_test+n_base_val]
base_train_df = base_sources_df.iloc[n_base_test+n_base_val:]

new_campaign_df = new_campaign_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
n_new_test = int(len(new_campaign_df) * 0.30)
new_test_df = new_campaign_df.iloc[:n_new_test]
new_update_df = new_campaign_df.iloc[n_new_test:]

print(f"Base train/val/test: {len(base_train_df)}/{len(base_val_df)}/{len(base_test_df)}")
print(f"New campaign update/test: {len(new_update_df)}/{len(new_test_df)}")

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")


def make_model():
    return (DistilBertWrapper().to(device), DomURLBERT().to(device), MessagePassing(rounds=2).to(device))


def run_eval(df_eval, email_model, url_model, comm_model, batch_size=16):
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
    if len(set(y_true)) > 1:
        return {"accuracy": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred, zero_division=0),
                "mcc": matthews_corrcoef(y_true, y_pred)}
    return {"accuracy": accuracy_score(y_true, y_pred), "f1": None, "mcc": None}


def train_loop(email_model, url_model, comm_model, train_df, val_df, n_epochs=15, patience=4, batch_size=16, lr=LR):
    optimizer = optim.AdamW(
        list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()),
        lr=lr, weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(n_epochs):
        email_model.train(); url_model.train(); comm_model.train()
        for i in range(0, len(train_df), batch_size):
            batch = train_df.iloc[i:i+batch_size]
            labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)
            e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        val_result = run_eval(val_df, email_model, url_model, comm_model, batch_size)
        with torch.no_grad():
            probs = []
            for i in range(0, len(val_df), batch_size):
                batch = val_df.iloc[i:i+batch_size]
                e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                e_feat = email_model(e_in.input_ids, e_in.attention_mask)
                u_feat = url_model(u_in.input_ids, u_in.attention_mask)
                logits = comm_model(e_feat, u_feat)
                probs.extend(logits.sigmoid().cpu().numpy().tolist())
            val_labels = torch.tensor([float(x) for x in val_df['label'].values])
            val_loss = criterion(torch.logit(torch.tensor(probs).clamp(1e-6, 1-1e-6)), val_labels).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                'email_model': {k: v.cpu().clone() for k, v in email_model.state_dict().items()},
                'url_model': {k: v.cpu().clone() for k, v in url_model.state_dict().items()},
                'comm_model': {k: v.cpu().clone() for k, v in comm_model.state_dict().items()},
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    email_model.load_state_dict(best_state['email_model'])
    url_model.load_state_dict(best_state['url_model'])
    comm_model.load_state_dict(best_state['comm_model'])
    return best_state


# ====================== PHASE 1: base model on 5 sources ======================
print("\n" + "=" * 90)
print("PHASE 1: Training base model on 5 sources (excludes new campaign)")
print("=" * 90)
email_model, url_model, comm_model = make_model()
base_state = train_loop(email_model, url_model, comm_model, base_train_df, base_val_df)
torch.save(base_state, f"continual_base_model_{HELD_OUT_SOURCE}_seed{SEED}.pth")

base_test_result = run_eval(base_test_df, email_model, url_model, comm_model)
new_test_result_before = run_eval(new_test_df, email_model, url_model, comm_model)
print(f"Base model on base_test: {base_test_result}")
print(f"Base model on NEW campaign test (before any update): {new_test_result_before}")

results = {
    "held_out_source": HELD_OUT_SOURCE,
    "base_model_on_base_test": base_test_result,
    "base_model_on_new_campaign_before_update": new_test_result_before,
    "methods": {},
}


def load_base_copy():
    e, u, c = make_model()
    e.load_state_dict(base_state['email_model'])
    u.load_state_dict(base_state['url_model'])
    c.load_state_dict(base_state['comm_model'])
    return e, u, c

# --- Method 1: Naive fine-tuning (train ONLY on new campaign data) ---
print("\n" + "=" * 90)
print("METHOD 1: Naive fine-tuning (new-campaign data only)")
print("=" * 90)
e1, u1, c1 = load_base_copy()
n_val_split = max(1, int(len(new_update_df) * 0.15))
naive_val = new_update_df.iloc[:n_val_split]
naive_train = new_update_df.iloc[n_val_split:]
train_loop(e1, u1, c1, naive_train, naive_val, n_epochs=10, patience=3)
naive_on_base = run_eval(base_test_df, e1, u1, c1)
naive_on_new = run_eval(new_test_df, e1, u1, c1)
print(f"Naive fine-tune -> base_test (forgetting check): {naive_on_base}")
print(f"Naive fine-tune -> new_campaign_test (adaptation check): {naive_on_new}")
results["methods"]["naive_finetune"] = {"on_base_test": naive_on_base, "on_new_campaign_test": naive_on_new}

# --- Method 2: Replay/rehearsal ---
print("\n" + "=" * 90)
print("METHOD 2: Replay/rehearsal (new-campaign data + 20% replay buffer from base sources)")
print("=" * 90)
e2, u2, c2 = load_base_copy()
replay_size = int(len(naive_train) * 0.20)
replay_buffer = base_train_df.sample(n=min(replay_size, len(base_train_df)), random_state=SEED)
replay_train = pd.concat([naive_train, replay_buffer]).sample(frac=1, random_state=SEED).reset_index(drop=True)
train_loop(e2, u2, c2, replay_train, naive_val, n_epochs=10, patience=3)
replay_on_base = run_eval(base_test_df, e2, u2, c2)
replay_on_new = run_eval(new_test_df, e2, u2, c2)
print(f"Replay -> base_test (forgetting check): {replay_on_base}")
print(f"Replay -> new_campaign_test (adaptation check): {replay_on_new}")
results["methods"]["replay"] = {"on_base_test": replay_on_base, "on_new_campaign_test": replay_on_new}

# --- Method 3: Frozen-encoder update ---
print("\n" + "=" * 90)
print("METHOD 3: Frozen-encoder update (encoders frozen, only comm_model updated)")
print("=" * 90)
e3, u3, c3 = load_base_copy()
for p in e3.parameters():
    p.requires_grad = False
for p in u3.parameters():
    p.requires_grad = False
optimizer3 = optim.AdamW(c3.parameters(), lr=LR, weight_decay=0.01)
criterion = nn.BCEWithLogitsLoss()
batch_size = 16
for epoch in range(10):
    e3.eval(); u3.eval(); c3.train()
    for i in range(0, len(naive_train), batch_size):
        batch = naive_train.iloc[i:i+batch_size]
        labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)
        e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            e_feat = e3(e_in.input_ids, e_in.attention_mask)
            u_feat = u3(u_in.input_ids, u_in.attention_mask)
        logits = c3(e_feat, u_feat)
        loss = criterion(logits, labels)
        optimizer3.zero_grad(); loss.backward(); optimizer3.step()
frozen_on_base = run_eval(base_test_df, e3, u3, c3)
frozen_on_new = run_eval(new_test_df, e3, u3, c3)
print(f"Frozen-encoder -> base_test (forgetting check): {frozen_on_base}")
print(f"Frozen-encoder -> new_campaign_test (adaptation check): {frozen_on_new}")
results["methods"]["frozen_encoder"] = {"on_base_test": frozen_on_base, "on_new_campaign_test": frozen_on_new}

print("\n" + "=" * 90)
print("METHOD 4 (reference): Full retrain -- use the already-trained canonical model's")
print("known results (trained on all 6 sources) as the upper-bound comparison.")
print("No new training needed -- see results_log.md for canonical model numbers.")
print("=" * 90)

with open(f"continual_learning_results_{HELD_OUT_SOURCE}_seed{SEED}.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 90)
print("SUMMARY -- Forgetting comparison (accuracy on base_test, higher = less forgetting)")
print("=" * 90)
print(f"Before any update:  {base_test_result['accuracy']:.4f}")
print(f"Naive fine-tune:     {naive_on_base['accuracy']:.4f}")
print(f"Replay:              {replay_on_base['accuracy']:.4f}")
print(f"Frozen-encoder:      {frozen_on_base['accuracy']:.4f}")
print("\nSUMMARY -- Adaptation comparison (accuracy on new_campaign_test, higher = better adaptation)")
print(f"Before any update:  {new_test_result_before['accuracy']:.4f}")
print(f"Naive fine-tune:     {naive_on_new['accuracy']:.4f}")
print(f"Replay:              {replay_on_new['accuracy']:.4f}")
print(f"Frozen-encoder:      {frozen_on_new['accuracy']:.4f}")
print(f"\nSaved continual_learning_results_{HELD_OUT_SOURCE}_seed{SEED}.json")
