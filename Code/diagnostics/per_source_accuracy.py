
import sys
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from model_and_data_utils import build_dataset_combined_corpus

if len(sys.argv) < 2:
    print("Usage: python per_source_accuracy.py <predictions.json>")
    sys.exit(1)

with open(sys.argv[1]) as f:
    preds = json.load(f)

train_df, val_df, test_df = build_dataset_combined_corpus(near_dup_threshold=0.90)

if len(test_df) != len(preds['y_true']):
    print(f"[ERROR] size mismatch: test_df={len(test_df)}, predictions={len(preds['y_true'])}")
    sys.exit(1)

y_true = np.array(preds['y_true'])
y_prob = np.array(preds['y_prob'])
y_pred = (y_prob > 0.5).astype(int)

test_df = test_df.reset_index(drop=True)
test_df['pred'] = y_pred
test_df['true'] = y_true
test_df['correct'] = (test_df['pred'] == test_df['true'])

print("=" * 90)
print("Per-source accuracy breakdown")
print("=" * 90)
print(f"{'Source':<18}{'N':<8}{'Classes present':<20}{'Accuracy':<12}{'F1':<10}")
for source, group in test_df.groupby('source'):
    classes_present = sorted(group['true'].unique().tolist())
    acc = accuracy_score(group['true'], group['pred'])
    if len(classes_present) > 1:
        f1_val = f1_score(group['true'], group['pred'], zero_division=0)
        f1_str = f"{f1_val:.4f}"
    else:
        f1_str = "n/a (mono-class)"
    print(f"{source:<18}{len(group):<8}{str(classes_present):<20}{acc:<12.4f}{f1_str:<10}")

mono_class_sources = ['nigerian_fraud', 'phishing_pot', 'darkknight', 'azario']
mixed_class_sources = ['chakraborty', 'phishnchips']

mono_mask = test_df['source'].isin(mono_class_sources)
mixed_mask = test_df['source'].isin(mixed_class_sources)

print(f"\n--- Aggregated ---")
if mono_mask.sum() > 0:
    print(f"Mono-class bulk sources (n={mono_mask.sum()}): "
          f"accuracy {accuracy_score(test_df[mono_mask]['true'], test_df[mono_mask]['pred']):.4f}")
if mixed_mask.sum() > 0:
    mixed_acc = accuracy_score(test_df[mixed_mask]['true'], test_df[mixed_mask]['pred'])
    mixed_f1 = f1_score(test_df[mixed_mask]['true'], test_df[mixed_mask]['pred'], zero_division=0)
    print(f"Mixed-class sources (n={mixed_mask.sum()}): accuracy {mixed_acc:.4f}, F1 {mixed_f1:.4f}")
    print(f"\nTHIS is the real test of the classifier -- compare this F1 against the")
    print(f"headline 0.9938 to see how much of the aggregate number is inflated by")
    print(f"the easy mono-class sources.")
