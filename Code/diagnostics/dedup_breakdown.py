
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

from model_and_data_utils import build_dataset

print("=" * 90)
print("Dedup breakdown by split pair")
print("=" * 90)

train_df, val_df, test_df = build_dataset()

train_df = train_df.copy(); train_df['split'] = 'train'
val_df = val_df.copy(); val_df['split'] = 'val'
test_df = test_df.copy(); test_df['split'] = 'test'
combined = train_df._append(val_df)._append(test_df).reset_index(drop=True) if hasattr(train_df, '_append') \
    else train_df.append(val_df).append(test_df).reset_index(drop=True)


def normalize(text):
    return re.sub(r'\s+', ' ', str(text).lower().strip())


combined['norm_text'] = combined['email_text'].apply(normalize)

exact_dup_mask = combined.duplicated(subset='norm_text', keep=False)
exact_split_pairs = Counter()
test_rows_with_train_exact_dup = set()

for text, group in combined[exact_dup_mask].groupby('norm_text'):
    splits_present = set(group['split'])
    if len(splits_present) > 1:
        for pair in [('train', 'test'), ('train', 'val'), ('val', 'test')]:
            if pair[0] in splits_present and pair[1] in splits_present:
                exact_split_pairs[pair] += 1
        if 'train' in splits_present and 'test' in splits_present:
            test_rows_with_train_exact_dup.update(group[group['split'] == 'test'].index)

print("\nEXACT duplicate groups by split pair involved:")
for pair, count in exact_split_pairs.items():
    print(f"  {pair[0]} <-> {pair[1]}: {count} groups")
print(f"\nTEST rows that have an EXACT duplicate in TRAIN: "
      f"{len(test_rows_with_train_exact_dup)} / {len(test_df)} "
      f"({len(test_rows_with_train_exact_dup)/len(test_df)*100:.1f}% of test set)")

# --- Near duplicates, broken down by split pair ---
print(f"\nComputing near-duplicate similarity matrix (n={len(combined)})...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf = vectorizer.fit_transform(combined['norm_text'])
sim_matrix = cosine_similarity(tfidf)
np.fill_diagonal(sim_matrix, 0)

THRESHOLD = 0.90
pairs = np.argwhere(sim_matrix > THRESHOLD)
pairs = [(i, j) for i, j in pairs if i < j]

near_split_pairs = Counter()
test_rows_with_train_near_dup = set()

for i, j in pairs:
    si, sj = combined.iloc[i]['split'], combined.iloc[j]['split']
    if si != sj:
        key = tuple(sorted([si, sj]))
        near_split_pairs[key] += 1
        if key == ('test', 'train'):
            test_idx = i if si == 'test' else j
            test_rows_with_train_near_dup.add(test_idx)

print("\nNEAR duplicate pairs by split pair involved:")
for pair, count in near_split_pairs.items():
    print(f"  {pair[0]} <-> {pair[1]}: {count} pairs")
print(f"\nTEST rows that have a NEAR duplicate (cosine>{THRESHOLD}) in TRAIN: "
      f"{len(test_rows_with_train_near_dup)} / {len(test_df)} "
      f"({len(test_rows_with_train_near_dup)/len(test_df)*100:.1f}% of test set)")

print("\n" + "=" * 90)
print("BOTTOM LINE")
print("=" * 90)
total_test_contaminated = len(test_rows_with_train_exact_dup | test_rows_with_train_near_dup)
print(f"TEST rows with an exact OR near duplicate in TRAIN: "
      f"{total_test_contaminated} / {len(test_df)} "
      f"({total_test_contaminated/len(test_df)*100:.1f}% of test set)")
print("\nThis fraction of your reported clean/attack baseline is potentially")
print("evaluated on examples the model has effectively already seen (as a")
print("near-copy) during training -- inflating F1/Acc by an unknown amount.")
