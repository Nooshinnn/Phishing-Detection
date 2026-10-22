
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from model_and_data_utils import build_dataset_combined_corpus

print("=" * 90)
print("Verifying the group-split combined corpus result")
print("=" * 90)

train_df, val_df, test_df = build_dataset_combined_corpus(near_dup_threshold=0.90)


def normalize(t):
    return re.sub(r'\s+', ' ', str(t).lower().strip())

train_df = train_df.copy()
test_df = test_df.copy()
train_df['norm'] = train_df['email_text'].apply(normalize)
test_df['norm'] = test_df['email_text'].apply(normalize)

print(f"\nChecking near-duplicate text leakage: {len(test_df)} test rows vs {len(train_df)} train rows...")

combined_text = pd.concat([train_df['norm'], test_df['norm']], ignore_index=True)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf = vectorizer.fit_transform(combined_text)

n_train = len(train_df)
train_tfidf = tfidf[:n_train]
test_tfidf = tfidf[n_train:]

THRESHOLD = 0.90
max_sims = []
chunk = 200
for i in range(0, test_tfidf.shape[0], chunk):
    block = test_tfidf[i:i+chunk]
    sims = cosine_similarity(block, train_tfidf)
    max_sims.extend(sims.max(axis=1).tolist())

max_sims = np.array(max_sims)
n_leaked = (max_sims > THRESHOLD).sum()
print(f"TEST rows with a near-duplicate (cosine>{THRESHOLD}) TRAIN row: "
      f"{n_leaked} / {len(test_df)} ({n_leaked/len(test_df)*100:.1f}%)")
print(f"Max similarity to any train row -- mean: {max_sims.mean():.4f}, max: {max_sims.max():.4f}, "
      f"95th percentile: {np.percentile(max_sims, 95):.4f}")

if n_leaked > 0:
    print(f"\n[FOUND IT] {n_leaked} test rows leak near-duplicate text from train despite the URL-group")
    print("check passing -- these rows have DIFFERENT urls but near-identical body text, so the")
    print("URL-based grouping never caught them. This likely explains the inflated F1.")
else:
    print("\nNo direct text-leakage found via this check -- the high F1 may be genuine, or from")
    print("a different cause (see composition check below).")


print("\n" + "=" * 90)
print("Test set composition check")
print("=" * 90)
full_source_dist = pd.concat([train_df, val_df, test_df])['source'].value_counts(normalize=True)
test_source_dist = test_df['source'].value_counts(normalize=True)
print("\nSource distribution -- full corpus vs test set:")
comparison = pd.DataFrame({'full_corpus': full_source_dist, 'test_set': test_source_dist}).fillna(0)
print(comparison.round(3))

print(f"\nTest set class balance: {dict(test_df['label'].value_counts(normalize=True).round(3))}")
