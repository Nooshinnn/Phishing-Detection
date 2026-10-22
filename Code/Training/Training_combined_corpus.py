# Trains on the alternate multi-source corpus (combined_phishing_dataset.csv)
# 1): ran this multiple times with different seeds:
#   python Training_combined_corpus.py 42
#   python Training_combined_corpus.py 123
#   python Training_combined_corpus.py 2024
#   python Training_combined_corpus.py 777
#   python Training_combined_corpus.py 31415
# 2) ran ensemble_eval.py
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import warnings
from tqdm import tqdm
import random
import json

from model_and_data_utils import (
    DistilBertWrapper, DomURLBERT, MessagePassing, build_dataset_combined_corpus, DEVICE
)
from transformers import AutoTokenizer

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
NEAR_DUP_THRESHOLD = 0.90  # set to whichever threshold you've settled on

warnings.filterwarnings("ignore")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = DEVICE

CHECKPOINT_OUT = f"combined_corpus_best_model_seed{SEED}.pth"
PREDICTIONS_OUT = f"combined_corpus_test_predictions_seed{SEED}.json"

print("=" * 80)
print(f"Message Passing (2 rounds) Training -- COMBINED CORPUS (seed={SEED})")
print(f"Will save to: {CHECKPOINT_OUT}")
print("=" * 80)


#the train/val/test SPLIT itself is controlled by random_state=42
train_df, val_df, test_df = build_dataset_combined_corpus(near_dup_threshold=NEAR_DUP_THRESHOLD)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} (held out, untouched until final eval)")

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")


def run_eval(df_eval, email_model, url_model, comm_model, batch_size=16):
    email_model.eval(); url_model.eval(); comm_model.eval()
    total_loss = 0.0
    y_true, y_prob = [], []
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for i in range(0, len(df_eval), batch_size):
            batch = df_eval.iloc[i:i+batch_size]
            labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)
            e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            prob = logits.sigmoid().cpu().numpy()
            y_true.extend(batch['label'].values)
            y_prob.extend(prob)
    avg_loss = total_loss / (len(df_eval) // batch_size + 1)
    y_pred = (np.array(y_prob) > 0.5).astype(int)
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "y_true": [int(x) for x in y_true],
        "y_prob": [float(x) for x in y_prob],
    }
    return metrics

# early stopping
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
patience = 7
patience_counter = 0
batch_size = 16

print("\nStarting training -- early stopping monitors VAL loss only...\n")

for epoch in range(40):
    email_model.train(); url_model.train(); comm_model.train()
    train_loss = 0.0
    for i in tqdm(range(0, len(train_df), batch_size), desc=f"Epoch {epoch+1}"):
        batch = train_df.iloc[i:i+batch_size]
        labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)
        e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        logits = comm_model(e_feat, u_feat)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_metrics = run_eval(val_df, email_model, url_model, comm_model, batch_size)
    scheduler.step(val_metrics['loss'])
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_df):.4f} | "
          f"Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f} | LR: {current_lr:.2e}")

    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        patience_counter = 0
        torch.save({
            'email_model': email_model.state_dict(),
            'url_model': url_model.state_dict(),
            'comm_model': comm_model.state_dict()
        }, CHECKPOINT_OUT)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

best_checkpoint = torch.load(CHECKPOINT_OUT, map_location=device)
email_model.load_state_dict(best_checkpoint['email_model'])
url_model.load_state_dict(best_checkpoint['url_model'])
comm_model.load_state_dict(best_checkpoint['comm_model'])

test_metrics = run_eval(test_df, email_model, url_model, comm_model, batch_size)

print("\n" + "="*70)
print("TRAINING FINISHED -- COMBINED CORPUS")
print("="*70)
print(f"Held-out TEST results -> Accuracy: {test_metrics['accuracy']:.4f} | "
      f"F1: {test_metrics['f1']:.4f} | MCC: {test_metrics['mcc']:.4f}")
print(f"Best model (by val loss) saved as {CHECKPOINT_OUT}")

with open(PREDICTIONS_OUT, "w") as f:
    json.dump({
        "seed": SEED,
        "accuracy": test_metrics['accuracy'],
        "f1": test_metrics['f1'],
        "mcc": test_metrics['mcc'],
        "y_true": test_metrics['y_true'],
        "y_prob": test_metrics['y_prob'],
    }, f, indent=2)
print(f"Test predictions saved to {PREDICTIONS_OUT}")
print(f"\nRun this for 4-5 different seeds, then use ensemble_eval.py to combine them.")