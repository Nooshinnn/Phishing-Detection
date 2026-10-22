
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
    DistilBertWrapper, DomURLBERT, build_dataset_combined_corpus, DEVICE
)
from transformers import AutoTokenizer

VALID_TYPES = ["message_passing_r1", "message_passing_r2", "message_passing_r3",
               "concatenation", "late_fusion", "cross_attention"]

if len(sys.argv) < 3:
    print(f"Usage: python Training_architecture_ablation.py <{'|'.join(VALID_TYPES)}> <seed>")
    sys.exit(1)

FUSION_TYPE = sys.argv[1]
assert FUSION_TYPE in VALID_TYPES, f"fusion type must be one of {VALID_TYPES}"
SEED = int(sys.argv[2])

warnings.filterwarnings("ignore")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = DEVICE

CHECKPOINT_OUT = f"arch_{FUSION_TYPE}_seed{SEED}.pth"
PREDICTIONS_OUT = f"arch_{FUSION_TYPE}_seed{SEED}_predictions.json"

print("=" * 80)
print(f"Architecture ablation: {FUSION_TYPE}, seed={SEED}")
print(f"Will save to: {CHECKPOINT_OUT}")
print("=" * 80)

train_df, val_df, test_df = build_dataset_combined_corpus(near_dup_threshold=0.90)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")


class MessagePassingFusion(nn.Module):
    def __init__(self, rounds=2):
        super().__init__()
        self.rounds = rounds
        self.update_e = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))
        self.update_u = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))
        self.fc = nn.Linear(256, 1)
    def forward(self, e, u):
        for _ in range(self.rounds):
            e = self.update_e(torch.cat([e, u], dim=1))
            u = self.update_u(torch.cat([u, e], dim=1))
        return self.fc(torch.cat([e, u], dim=1)).squeeze(-1)


class ConcatenationFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    def forward(self, e, u):
        return self.head(torch.cat([e, u], dim=1)).squeeze(-1)


class LateFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.email_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
        self.url_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
        self.combine = nn.Linear(2, 1)
    def forward(self, e, u):
        e_logit = self.email_head(e)
        u_logit = self.url_head(u)
        combined = torch.cat([e_logit, u_logit], dim=1)
        return self.combine(combined).squeeze(-1)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=128, n_heads=4):
        super().__init__()
        self.u_to_e_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True, dropout=0.3)
        self.e_to_u_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(dim * 2, 1)
    def forward(self, e, u):
        e_seq = e.unsqueeze(1)  # [B, 1, dim] -- treat each vector as a length-1 sequence
        u_seq = u.unsqueeze(1)
        e_attended, _ = self.u_to_e_attn(e_seq, u_seq, u_seq)  # email representation attends to URL
        u_attended, _ = self.e_to_u_attn(u_seq, e_seq, e_seq)  # URL representation attends to email
        e_attended = e_attended.squeeze(1)
        u_attended = u_attended.squeeze(1)
        return self.fc(torch.cat([e_attended, u_attended], dim=1)).squeeze(-1)


def make_comm_model(fusion_type):
    if fusion_type == "message_passing_r1":
        return MessagePassingFusion(rounds=1)
    elif fusion_type == "message_passing_r2":
        return MessagePassingFusion(rounds=2)
    elif fusion_type == "message_passing_r3":
        return MessagePassingFusion(rounds=3)
    elif fusion_type == "concatenation":
        return ConcatenationFusion()
    elif fusion_type == "late_fusion":
        return LateFusion()
    elif fusion_type == "cross_attention":
        return CrossAttentionFusion()
    else:
        raise ValueError(fusion_type)


comm_model = make_comm_model(FUSION_TYPE).to(device)
email_model = DistilBertWrapper().to(device)
url_model = DomURLBERT().to(device)

n_comm_params = sum(p.numel() for p in comm_model.parameters())
print(f"Fusion module parameter count ({FUSION_TYPE}): {n_comm_params:,}")

optimizer = optim.AdamW(
    list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()),
    lr=8e-6, weight_decay=0.01
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
criterion = nn.BCEWithLogitsLoss()


def run_eval(df_eval, batch_size=16):
    email_model.eval(); url_model.eval(); comm_model.eval()
    total_loss = 0.0
    y_true, y_prob = [], []
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
    return {
        "loss": avg_loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "y_true": [int(x) for x in y_true],
        "y_prob": [float(x) for x in y_prob],
    }


best_val_loss = float('inf')
patience, patience_counter = 7, 0
batch_size = 16
best_state = None

for epoch in range(40):
    email_model.train(); url_model.train(); comm_model.train()
    for i in tqdm(range(0, len(train_df), batch_size), desc=f"Epoch {epoch+1}"):
        batch = train_df.iloc[i:i+batch_size]
        labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)
        e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        logits = comm_model(e_feat, u_feat)
        loss = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    val_metrics = run_eval(val_df, batch_size)
    scheduler.step(val_metrics['loss'])
    print(f"Epoch {epoch+1} | Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f}")

    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
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
torch.save(best_state, CHECKPOINT_OUT)

test_metrics = run_eval(test_df, batch_size)
print("\n" + "=" * 70)
print(f"ARCHITECTURE ABLATION RESULT: {FUSION_TYPE}, seed={SEED}")
print(f"Test F1: {test_metrics['f1']:.4f} | Acc: {test_metrics['accuracy']:.4f} | MCC: {test_metrics['mcc']:.4f}")
print(f"Fusion module params: {n_comm_params:,}")

with open(PREDICTIONS_OUT, "w") as f:
    json.dump({
        "fusion_type": FUSION_TYPE, "seed": SEED, "n_comm_params": n_comm_params,
        "f1": test_metrics['f1'], "accuracy": test_metrics['accuracy'], "mcc": test_metrics['mcc'],
        "y_true": test_metrics['y_true'], "y_prob": test_metrics['y_prob'],
    }, f, indent=2)
print(f"Saved predictions to {PREDICTIONS_OUT}")
