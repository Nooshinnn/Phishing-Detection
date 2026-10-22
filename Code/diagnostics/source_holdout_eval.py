
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import json
import warnings
import random

from model_and_data_utils import (
    DistilBertWrapper, DomURLBERT, MessagePassing, dedup_pool, DEVICE
)
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
SEED = 31415  
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = DEVICE

print("=" * 90)
print("Leave-one-source-out holdout evaluation")
print("=" * 90)

df_full = pd.read_csv("combined_phishing_dataset.csv")
df_full = df_full.rename(columns={"body": "email_text"})
df_full['url'] = df_full['urls'].apply(lambda u: str(u).split(" | ")[0] if pd.notna(u) and str(u).strip() else "")
df_full = df_full[df_full['url'] != ""].reset_index(drop=True)
df_full = dedup_pool(df_full, near_dup_threshold=0.90, verbose=False)
print(f"After dedup: {len(df_full)} rows across sources: {df_full['source'].value_counts().to_dict()}")

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")


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
    result = {"y_true": [int(x) for x in y_true], "y_prob": [float(x) for x in y_prob]}
    if len(set(y_true)) > 1:
        result["accuracy"] = accuracy_score(y_true, y_pred)
        result["f1"] = f1_score(y_true, y_pred, zero_division=0)
        result["mcc"] = matthews_corrcoef(y_true, y_pred)
    else:
       
        result["accuracy"] = accuracy_score(y_true, y_pred)  
        result["f1"] = None
        result["mcc"] = None
        result["note"] = f"mono-class held-out set (label={y_true[0]}) -- accuracy above IS the recall on this class"
    return result


def train_one_fold(train_df, val_df, n_epochs=15, patience=4, batch_size=16):
    email_model = DistilBertWrapper().to(device)
    url_model = DomURLBERT().to(device)
    comm_model = MessagePassing(rounds=2).to(device)
    optimizer = optim.AdamW(
        list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()),
        lr=8e-6, weight_decay=0.01
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
            val_probs = torch.tensor(val_result['y_prob'])
            val_labels = torch.tensor([float(x) for x in val_result['y_true']])
            val_loss = criterion(torch.logit(val_probs.clamp(1e-6, 1-1e-6)), val_labels).item()
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
    return email_model, url_model, comm_model


sources = sorted(df_full['source'].unique())
all_results = {}

for held_out in sources:
    print(f"\n{'='*90}\nHolding out: {held_out}\n{'='*90}")
    remaining = df_full[df_full['source'] != held_out].reset_index(drop=True)
    held_out_df = df_full[df_full['source'] == held_out].reset_index(drop=True)

    remaining = remaining.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_val = int(len(remaining) * 0.15)
    val_df = remaining.iloc[:n_val]
    train_df = remaining.iloc[n_val:]

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Held-out test ({held_out}): {len(held_out_df)}")

    email_model, url_model, comm_model = train_one_fold(train_df, val_df)
    result = run_eval(held_out_df, email_model, url_model, comm_model)

    if result.get("f1") is not None:
        print(f"Result on held-out {held_out}: F1={result['f1']:.4f} Acc={result['accuracy']:.4f} MCC={result['mcc']:.4f}")
    else:
        print(f"Result on held-out {held_out}: {result['note']} -> {result['accuracy']:.4f}")

    all_results[held_out] = result

with open("source_holdout_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 90)
print("SUMMARY -- Leave-one-source-out")
print("=" * 90)
print(f"{'Held-out source':<18}{'N':<8}{'Metric':<20}{'Value':<10}")
for source, r in all_results.items():
    n = len(r['y_true'])
    if r.get('f1') is not None:
        print(f"{source:<18}{n:<8}{'F1':<20}{r['f1']:<10.4f}")
    else:
        label = r['y_true'][0]
        metric_name = f"Recall (label={label})"
        print(f"{source:<18}{n:<8}{metric_name:<20}{r['accuracy']:<10.4f}")

print("\nCompare these numbers against the random-split result (F1 ~0.98-0.99).")
print("A meaningfully lower F1/recall here indicates the model relies on")
print("source-specific patterns rather than generalizing across campaigns --")
print("the failure mode a random split cannot detect.")
