
import re
import sys
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from model_and_data_utils import build_dataset_combined_corpus

if len(sys.argv) < 2:
    print("Usage: python check_leak_impact.py <predictions.json>")
    sys.exit(1)

with open(sys.argv[1]) as f:
    preds = json.load(f)

train_df, val_df, test_df = build_dataset_combined_corpus(near_dup_threshold=0.90)

if len(test_df) != len(preds['y_true']):
    print(f"[ERROR] test_df ({len(test_df)}) and predictions file "
          f"({len(preds['y_true'])}) sizes don't match -- can't align positions.")
    sys.exit(1)

def normalize(t):
    return re.sub(r'\s+', ' ', str(t).lower().strip())

train_norm = train_df['email_text'].apply(normalize)
test_norm = test_df['email_text'].apply(normalize)

combined_text = pd.concat([train_norm, test_norm], ignore_index=True)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf = vectorizer.fit_transform(combined_text)
n_train = len(train_df)
train_tfidf = tfidf[:n_train]
test_tfidf = tfidf[n_train:]

max_sims = []
chunk = 200
for i in range(0, test_tfidf.shape[0], chunk):
    block = test_tfidf[i:i+chunk]
    sims = cosine_similarity(block, train_tfidf)
    max_sims.extend(sims.max(axis=1).tolist())
max_sims = np.array(max_sims)

leaked_positions = np.where(max_sims > 0.90)[0]
print(f"Leaked test row positions: {leaked_positions.tolist()}")

y_true = np.array(preds['y_true'])
y_prob = np.array(preds['y_prob'])
y_pred = (y_prob > 0.5).astype(int)

full_f1 = f1_score(y_true, y_pred, zero_division=0)
full_acc = accuracy_score(y_true, y_pred)
full_mcc = matthews_corrcoef(y_true, y_pred)

mask = np.ones(len(y_true), dtype=bool)
mask[leaked_positions] = False
clean_f1 = f1_score(y_true[mask], y_pred[mask], zero_division=0)
clean_acc = accuracy_score(y_true[mask], y_pred[mask])
clean_mcc = matthews_corrcoef(y_true[mask], y_pred[mask])

if len(leaked_positions) > 0:
    leaked_acc = accuracy_score(y_true[leaked_positions], y_pred[leaked_positions])
else:
    leaked_acc = float('nan')

print("\n" + "=" * 90)
print("IMPACT OF THE LEAKED ROWS")
print("=" * 90)
print(f"{'':<20}{'F1':<10}{'Accuracy':<12}{'MCC':<10}")
print(f"{'Full test set':<20}{full_f1:<10.4f}{full_acc:<12.4f}{full_mcc:<10.4f}")
print(f"{'Excl. leaked rows':<20}{clean_f1:<10.4f}{clean_acc:<12.4f}{clean_mcc:<10.4f}")
print(f"\nAccuracy on JUST the {len(leaked_positions)} leaked rows: {leaked_acc:.4f}")
print(f"\nF1 change from removing leaked rows: {clean_f1 - full_f1:+.4f}")
print("\nIf this change is small (<0.005 or so), the leak isn't the main story --")
print("the high F1 is more likely genuine (or driven by something else, like the")
print("split landing on an easier partition by chance). If it's large, the leak")
print("matters more than its 1% row-count share suggests.")
