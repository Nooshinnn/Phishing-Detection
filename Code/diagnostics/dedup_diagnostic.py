
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from model_and_data_utils import build_dataset

print("=" * 90)
print("DIAGNOSTIC: Deduplication check (report-only, does not modify the pipeline)")
print("=" * 90)

train_df, val_df, test_df = build_dataset()

# Combine all three splits to check for duplication ACROSS splits too --
# this is the more serious case (a near-duplicate in both train and test
# would be actual leakage, not just redundancy).
train_df = train_df.copy(); train_df['split'] = 'train'
val_df = val_df.copy(); val_df['split'] = 'val'
test_df = test_df.copy(); test_df['split'] = 'test'
combined = train_df._append(val_df)._append(test_df).reset_index(drop=True) if hasattr(train_df, '_append') \
    else train_df.append(val_df).append(test_df).reset_index(drop=True)

print(f"\nTotal examples across train+val+test: {len(combined)}")


def normalize(text):
    return re.sub(r'\s+', ' ', str(text).lower().strip())


# --- Exact duplicates ---
combined['norm_text'] = combined['email_text'].apply(normalize)
exact_dup_mask = combined.duplicated(subset='norm_text', keep=False)
n_exact_dup_rows = exact_dup_mask.sum()
n_exact_dup_groups = combined[exact_dup_mask]['norm_text'].nunique()

cross_split_exact = 0
for text, group in combined[exact_dup_mask].groupby('norm_text'):
    if group['split'].nunique() > 1:
        cross_split_exact += len(group)

print(f"\nEXACT duplicates:")
print(f"  Rows involved in an exact-duplicate group: {n_exact_dup_rows} / {len(combined)}")
print(f"  Number of distinct duplicate groups: {n_exact_dup_groups}")
print(f"  Rows in groups that SPAN MULTIPLE SPLITS (train/val/test leakage): {cross_split_exact}")


print(f"\nComputing near-duplicate similarity matrix (n={len(combined)})... this may take a moment.")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf = vectorizer.fit_transform(combined['norm_text'])
sim_matrix = cosine_similarity(tfidf)
np.fill_diagonal(sim_matrix, 0)

THRESHOLD = 0.90
near_dup_pairs = np.argwhere(sim_matrix > THRESHOLD)
near_dup_pairs = [(i, j) for i, j in near_dup_pairs if i < j] 

cross_split_near = 0
for i, j in near_dup_pairs:
    if combined.iloc[i]['split'] != combined.iloc[j]['split']:
        cross_split_near += 1

print(f"\nNEAR duplicates (TF-IDF cosine similarity > {THRESHOLD}):")
print(f"  Number of near-duplicate pairs: {len(near_dup_pairs)}")
print(f"  Rows involved in at least one near-duplicate pair: "
      f"{len(set([i for pair in near_dup_pairs for i in pair]))}")
print(f"  Pairs that SPAN MULTIPLE SPLITS (train/val/test leakage): {cross_split_near}")

print("\n" + "=" * 90)
print("DECISION GUIDE")
print("=" * 90)
print("- If cross-split exact/near duplicates are non-trivial (e.g. >1% of test set):")
print("  this is real leakage risk -- worth rebuilding the split with dedup applied,")
print("  even though it means re-running completed attacks.")
print("- If duplication exists but is entirely WITHIN a single split (e.g. train has")
print("  repeated template spam): less urgent -- doesn't leak train info into test,")
print("  though it can still bias what the model overfits to. Lower priority fix.")
print("- If duplication is minimal overall: document the check as done (with the")
print("  numbers) in the Open Science / methodology section and move on.")
