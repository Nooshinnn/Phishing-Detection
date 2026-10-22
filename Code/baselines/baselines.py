
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import warnings
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from model_and_data_utils import DistilBertWrapper, build_dataset_combined_corpus, DEVICE
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
SEED = 31415  
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = DEVICE

print("=" * 90)
print("Baseline models (email-text-only, same split as canonical model)")
print("=" * 90)

train_df, val_df, test_df = build_dataset_combined_corpus(near_dup_threshold=0.90)
results = {}

print("\n" + "=" * 90)
print("Baseline 1: TF-IDF + Logistic Regression")
print("=" * 90)
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df['email_text'])
X_test = vectorizer.transform(test_df['email_text'])
y_train = train_df['label'].values
y_test = test_df['label'].values

lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_f1 = f1_score(y_test, y_pred_lr, zero_division=0)
lr_acc = accuracy_score(y_test, y_pred_lr)
lr_mcc = matthews_corrcoef(y_test, y_pred_lr)
print(f"TF-IDF + LogReg -> F1: {lr_f1:.4f} | Acc: {lr_acc:.4f} | MCC: {lr_mcc:.4f}")
results['tfidf_logreg'] = {"f1": lr_f1, "accuracy": lr_acc, "mcc": lr_mcc}

print("\n" + "=" * 90)
print("Baseline 2: TF-IDF + Random Forest")
print("=" * 90)
rf = RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_mcc = matthews_corrcoef(y_test, y_pred_rf)
print(f"TF-IDF + RandomForest -> F1: {rf_f1:.4f} | Acc: {rf_acc:.4f} | MCC: {rf_mcc:.4f}")
results['tfidf_randomforest'] = {"f1": rf_f1, "accuracy": rf_acc, "mcc": rf_mcc}

print("\n" + "=" * 90)
print("Baseline 3: Standalone DistilBERT (email-only, no URL branch at all)")
print("=" * 90)

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class EmailOnlyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DistilBertWrapper()
        self.head = nn.Linear(128, 1)
    def forward(self, input_ids, attention_mask):
        feat = self.encoder(input_ids, attention_mask)
        return self.head(feat).squeeze(-1)

model = EmailOnlyClassifier().to(device)
optimizer = optim.AdamW(model.parameters(), lr=8e-6, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
criterion = nn.BCEWithLogitsLoss()
batch_size = 16

def run_eval(df_eval):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for i in range(0, len(df_eval), batch_size):
            batch = df_eval.iloc[i:i+batch_size]
            e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            logits = model(e_in.input_ids, e_in.attention_mask)
            prob = logits.sigmoid().cpu().numpy()
            y_true.extend(batch['label'].values)
            y_prob.extend(prob)
    y_pred = (np.array(y_prob) > 0.5).astype(int)
    return y_true, y_prob, y_pred

best_val_loss = float('inf')
patience, patience_counter = 5, 0
best_state = None

for epoch in range(20):
    model.train()
    for i in range(0, len(train_df), batch_size):
        batch = train_df.iloc[i:i+batch_size]
        labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)
        e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        logits = model(e_in.input_ids, e_in.attention_mask)
        loss = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    y_true_val, y_prob_val, y_pred_val = run_eval(val_df)
    val_loss = criterion(
        torch.logit(torch.tensor(y_prob_val).clamp(1e-6, 1-1e-6)),
        torch.tensor([float(x) for x in y_true_val])
    ).item()
    scheduler.step(val_loss)
    val_f1 = f1_score(y_true_val, y_pred_val, zero_division=0)
    print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

model.load_state_dict(best_state)
y_true_test, y_prob_test, y_pred_test = run_eval(test_df)
bert_f1 = f1_score(y_true_test, y_pred_test, zero_division=0)
bert_acc = accuracy_score(y_true_test, y_pred_test)
bert_mcc = matthews_corrcoef(y_true_test, y_pred_test)
print(f"\nStandalone DistilBERT (email-only) -> F1: {bert_f1:.4f} | Acc: {bert_acc:.4f} | MCC: {bert_mcc:.4f}")
results['standalone_distilbert_email_only'] = {"f1": bert_f1, "accuracy": bert_acc, "mcc": bert_mcc}

# ====================== Summary ======================
print("\n" + "=" * 90)
print("BASELINE COMPARISON SUMMARY")
print("=" * 90)
print(f"{'Model':<35}{'F1':<10}{'Accuracy':<12}{'MCC':<10}")
for name, r in results.items():
    print(f"{name:<35}{r['f1']:<10.4f}{r['accuracy']:<12.4f}{r['mcc']:<10.4f}")
print(f"{'canonical (multimodal, seed31415)':<35}{'0.9938':<10}{'0.9929':<12}{'0.9855':<10}  (from prior run)")

with open("baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved baseline_results.json")
