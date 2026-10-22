
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
import random

from model_and_data_utils import DistilBertWrapper, dedup_pool, DEVICE
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = DEVICE

print("=" * 90)
print("Source-prediction diagnostic (does NOT use the phishing/legit label)")
print("=" * 90)

df = pd.read_csv("combined_phishing_dataset.csv")
df = df.rename(columns={"body": "email_text"})
df['url'] = df['urls'].apply(lambda u: str(u).split(" | ")[0] if pd.notna(u) and str(u).strip() else "")
df = df[df['url'] != ""].reset_index(drop=True)
df = dedup_pool(df, near_dup_threshold=0.90, verbose=False)
print(f"After dedup: {len(df)} rows")

le = LabelEncoder()
df['source_id'] = le.fit_transform(df['source'])
n_classes = len(le.classes_)
print(f"Sources: {list(le.classes_)} ({n_classes} classes)")

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
n_test = int(len(df) * 0.15)
n_val = int(len(df) * 0.15)
test_df = df.iloc[:n_test]
val_df = df.iloc[n_test:n_test+n_val]
train_df = df.iloc[n_test+n_val:]
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class SourceClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = DistilBertWrapper()  
        self.head = nn.Linear(128, n_classes)
    def forward(self, input_ids, attention_mask):
        feat = self.encoder(input_ids, attention_mask)
        return self.head(feat)

model = SourceClassifier(n_classes).to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
batch_size = 16

def run_eval(df_eval):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(0, len(df_eval), batch_size):
            batch = df_eval.iloc[i:i+batch_size]
            e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            logits = model(e_in.input_ids, e_in.attention_mask)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(batch['source_id'].values)
            y_pred.extend(pred)
    return y_true, y_pred

best_val_acc = 0.0
patience, patience_counter = 4, 0
best_state = None

for epoch in range(15):
    model.train()
    for i in range(0, len(train_df), batch_size):
        batch = train_df.iloc[i:i+batch_size]
        labels = torch.tensor(batch['source_id'].values, dtype=torch.long).to(device)
        e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        logits = model(e_in.input_ids, e_in.attention_mask)
        loss = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    y_true_val, y_pred_val = run_eval(val_df)
    val_acc = accuracy_score(y_true_val, y_pred_val)
    print(f"Epoch {epoch+1} | Val source-prediction accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

model.load_state_dict(best_state)
y_true_test, y_pred_test = run_eval(test_df)
test_acc = accuracy_score(y_true_test, y_pred_test)
test_f1 = f1_score(y_true_test, y_pred_test, average='macro', zero_division=0)

print("\n" + "=" * 90)
print("RESULTS")
print("=" * 90)
print(f"Source-prediction TEST accuracy: {test_acc:.4f} (macro-F1: {test_f1:.4f})")
print(f"Random-guess baseline (1/{n_classes} classes): {1/n_classes:.4f}")

print("\nConfusion matrix (rows=true, cols=predicted):")
cm = confusion_matrix(y_true_test, y_pred_test)
print(f"{'':<15}" + "".join(f"{c[:8]:<10}" for c in le.classes_))
for i, row in enumerate(cm):
    print(f"{le.classes_[i]:<15}" + "".join(f"{v:<10}" for v in row))

print("\nInterpretation guide:")
print("- If source-prediction accuracy is high (e.g. >70-80%): sources are")
print("  strongly stylistically distinct. Combined with source/label correlation")
print("  (4 of 6 sources are mono-class), this means the main phishing classifier")
print("  COULD be partially learning source style as a proxy for label, even")
print("  without seeing source directly -- worth flagging as a limitation.")
print("- If accuracy is low/near-random: sources are not easily distinguishable")
print("  by a text classifier, reducing (not eliminating) this concern.")
