"""
model_and_data_utils.py

Single source of truth for: model architecture, checkpoint loading, and dataset
construction (including the identical train/val/test split and the new
English-only filter). Every attack script should import from here instead of
redefining classes/split logic locally -- that duplication is exactly what
caused the PWWS/BAE architecture mismatch and dataset mismatch bugs.

Usage in an attack script:
    from model_and_data_utils import load_model, build_dataset, DEVICE

    email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
    train_df, val_df, test_df = build_dataset()
"""
import pandas as pd
import re
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import warnings
import random

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "combined_corpus_best_model_seed31415.pth" 


class DistilBertWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(768, 128)
        self.dropout = nn.Dropout(0.3)
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.fc(pooled)

class DomURLBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("amahdaouy/DomURLs_BERT")
        self.fc = nn.Linear(768, 128)
        self.dropout = nn.Dropout(0.3)
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.fc(pooled)

class MessagePassing(nn.Module):
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


def load_model(checkpoint_path=CHECKPOINT_PATH):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    email_model = DistilBertWrapper().to(DEVICE)
    url_model = DomURLBERT().to(DEVICE)
    comm_model = MessagePassing(rounds=2).to(DEVICE)

    email_model.load_state_dict(checkpoint['email_model'])
    url_model.load_state_dict(checkpoint['url_model'])
    comm_model.load_state_dict(checkpoint['comm_model'])

    email_model.eval()
    url_model.eval()
    comm_model.eval()

    email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

    return email_model, url_model, comm_model, email_tokenizer, url_tokenizer


def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else ""


def is_english_word(word, _wordset={}):
    """
    Per-candidate English validity check, for use INSIDE an attack's substitution
    loop -- i.e. "is this proposed replacement word actually English", not
    "is this email in English" (the dataset is already all-English).

    Uses NLTK's words corpus (~236k English words).
    """
    if not _wordset:
        import nltk
        try:
            nltk.data.find("corpora/words")
        except LookupError:
            nltk.download("words")
        from nltk.corpus import words as nltk_words
        _wordset["set"] = set(w.lower() for w in nltk_words.words())
    return word.lower() in _wordset["set"]


def dedup_pool(df, near_dup_threshold=0.90, verbose=True):
    """
    Deduplicates BEFORE splitting
    Removes exact duplicates (normalized text match) and collapses
    near-duplicate clusters (TF-IDF cosine similarity > near_dup_threshold)
    down to one representative each

    Found via dedup_diagnostic.py / dedup_breakdown.py: 22.1% of the test
    set had an exact-or-near duplicate in train under the old (no-dedup)
    split -- non-trivial leakage risk, not just redundancy.
    """
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    def normalize(text):
        return re.sub(r'\s+', ' ', str(text).lower().strip())

    df = df.copy().reset_index(drop=True)
    df['norm_text'] = df['email_text'].apply(normalize)

    n_before = len(df)
    df = df.drop_duplicates(subset='norm_text', keep='first').reset_index(drop=True)
    n_after_exact = len(df)
    if verbose:
        print(f"Exact dedup: {n_before} -> {n_after_exact} ({n_before - n_after_exact} removed)")

    # Near-duplicate clustering via connected components on the similarity graph.
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf = vectorizer.fit_transform(df['norm_text'])
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0)
    THRESHOLD = near_dup_threshold

    n = len(df)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    pairs = np.argwhere(sim > THRESHOLD)
    for i, j in pairs:
        if i < j:
            union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    keep_indices = sorted([min(members) for members in clusters.values()])
    n_before_near = len(df)
    df = df.iloc[keep_indices].reset_index(drop=True)
    n_after_near = len(df)
    if verbose:
        print(f"Near-dup dedup (cosine>{THRESHOLD}): {n_before_near} -> {n_after_near} "
              f"({n_before_near - n_after_near} removed)")

    return df.drop(columns=['norm_text'])


def build_dataset_combined_corpus(csv_path="combined_phishing_dataset.csv", near_dup_threshold=0.90, verbose=True):
    """

    CSV columns expected: source, body, label, urls

    Splits by URL GROUP, not by row: rows that share the same URL are always
    kept together in the same split. Without this, two different emails
    pointing at the same URL could land on opposite sides of the split which is a 
    a subtler leakage path than duplicate email text, but the same class of
    problem. Groups are also approximately stratified by label (each label's
    groups are split independently, then recombined) so class balance stays
    close to 70/15/15 despite the grouping constraint.
    """
    from sklearn.model_selection import GroupShuffleSplit

    df = pd.read_csv(csv_path)
    df = df.rename(columns={"body": "email_text"})
    df['url'] = df['urls'].apply(lambda u: str(u).split(" | ")[0] if pd.notna(u) and str(u).strip() else "")
    df = df[df['url'] != ""].reset_index(drop=True)
    if verbose:
        print(f"Loaded combined corpus: {len(df)} rows with a usable URL")

    df = dedup_pool(df, near_dup_threshold=near_dup_threshold, verbose=verbose)
    if verbose:
        print(f"After deduplication (pre-split, near-dup threshold={near_dup_threshold}): {len(df)}")

    n_groups = df['url'].nunique()
    n_multi = int((df['url'].value_counts() > 1).sum())
    if verbose:
        print(f"Split groups: {len(df)} rows -> {n_groups} groups "
              f"({n_multi} groups contain >1 row, i.e. shared a URL)")

    group_label = df.groupby('url')['label'].agg(lambda x: x.value_counts().idxmax())
    df['_group_label'] = df['url'].map(group_label)

    train_parts, val_parts, test_parts = [], [], []
    for lbl in sorted(df['_group_label'].unique()):
        sub = df[df['_group_label'] == lbl]
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        train_idx, temp_idx = next(gss1.split(sub, groups=sub['url']))
        train_parts.append(sub.iloc[train_idx])
        temp = sub.iloc[temp_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
        val_idx, test_idx = next(gss2.split(temp, groups=temp['url']))
        val_parts.append(temp.iloc[val_idx])
        test_parts.append(temp.iloc[test_idx])

    train_df = pd.concat(train_parts).drop(columns=['_group_label']).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat(val_parts).drop(columns=['_group_label']).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test_parts).drop(columns=['_group_label']).sample(frac=1, random_state=42).reset_index(drop=True)


    train_urls, val_urls, test_urls = set(train_df['url']), set(val_df['url']), set(test_df['url'])
    overlap = (train_urls & val_urls) | (train_urls & test_urls) | (val_urls & test_urls)

    if verbose:
        print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            print(f"  {name} label balance: {dict(d['label'].value_counts(normalize=True).round(3))}")
        if overlap:
            print(f"[WARNING] {len(overlap)} URL(s) span multiple splits -- group-integrity check FAILED")
        else:
            print("Group-integrity check passed: no group spans multiple splits.")
        print(f"Source distribution in test set:\n{test_df['source'].value_counts()}")

    return train_df, val_df, test_df


def build_dataset_cybersectony(verbose=True):
    """
    kept for
    reference/footnote purposes only (this is what every PDGAN/URLGAN/BAE/etc.
    result from earlier in the project was built on). It is not use for new
    experiments and build_dataset() is used instead.

    A direct cross-corpus check found 18.3% exact + 10.6%+ near-duplicate text
    overlap between this dataset and the combined corpus -- the two are not
    independent, which is part of why the combined corpus was chosen as sole
    primary going forward rather than reporting both as separate evidence.
    """
    dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
    df = pd.DataFrame(dataset)
    df = df[df['labels'].isin([0, 1])].reset_index(drop=True)
    df = df.rename(columns={"labels": "label", "content": "email_text"})

    df['url'] = df['email_text'].apply(extract_first_url)
    df = df[df['url'] != ""].reset_index(drop=True)
    if verbose:
        print(f"After dropping rows with no embedded URL: {len(df)}")

    df = dedup_pool(df, verbose=verbose)
    if verbose:
        print(f"After deduplication (pre-split): {len(df)}")

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if verbose:
        print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    return train_df, val_df, test_df


def build_dataset(verbose=True):
    """
    Every
    existing attack script that calls build_dataset() picks up the combined
    corpus automatically, with zero per-script edits needed.
    """
    return build_dataset_combined_corpus(near_dup_threshold=0.90, verbose=verbose)
