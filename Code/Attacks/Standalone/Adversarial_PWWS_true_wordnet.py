import pandas as pd
import re
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import warnings
from tqdm import tqdm
import random
import json
import nltk
from nltk.corpus import wordnet

warnings.filterwarnings("ignore")
for pkg in ["wordnet", "omw-1.4", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 90)
print("ADVERSARIAL ATTACK: TRUE PWWS (Probability Weighted Word Saliency, WordNet synonyms)")
print("Using canonical checkpoint: original_best_model_fixed.pth")
print("=" * 90)

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


checkpoint = torch.load("original_best_model_fixed.pth", map_location=device)
email_model = DistilBertWrapper().to(device)
url_model = DomURLBERT().to(device)
comm_model = MessagePassing(rounds=2).to(device)

email_model.load_state_dict(checkpoint['email_model'])
url_model.load_state_dict(checkpoint['url_model'])
comm_model.load_state_dict(checkpoint['comm_model'])

email_model.eval()
url_model.eval()
comm_model.eval()

email_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
url_tokenizer = AutoTokenizer.from_pretrained("amahdaouy/DomURLs_BERT")

dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
df = pd.DataFrame(dataset)
df = df[df['labels'].isin([0, 1])].reset_index(drop=True)
df = df.rename(columns={"labels": "label", "content": "email_text"})

def extract_first_url(text):
    urls = re.findall(r'https?://\S+', str(text))
    return urls[0] if urls else ""

df['url'] = df['email_text'].apply(extract_first_url)
df = df[df['url'] != ""].reset_index(drop=True)  # matches BAE's dataset construction

train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)
test_df = test_df.reset_index(drop=True)
print(f"Test set size: {len(test_df)}")

test_df_phish = test_df[test_df['label'] == 1].reset_index(drop=True)
print(f"Evaluating evasion on phishing-labeled test examples only: "
      f"{len(test_df_phish)} / {len(test_df)}")


def true_class_prob(text, url):
    """P(model assigns the TRUE label), used for both saliency and substitution scoring --
    this is what PWWS actually optimizes against (not just P(phishing) regardless of label)."""
    with torch.no_grad():
        e_in = email_tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer([url], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        prob_phish = torch.sigmoid(comm_model(e_feat, u_feat)).item()
    return prob_phish


def get_wordnet_synonyms(word, max_candidates=8):
    """Real PWWS candidate generation: WordNet synonym sets, not MLM completions."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace('_', ' ')
            if candidate.isalpha() and candidate.lower() != word.lower():
                synonyms.add(candidate)
            if len(synonyms) >= max_candidates:
                break
        if len(synonyms) >= max_candidates:
            break
    return list(synonyms)


def pwws_attack(text, url, label, change_percent=0.30):
    """
    Genuine PWWS (Ren et al. 2019):
      1. Word saliency S(wi) = true-class-prob(x) - true-class-prob(x with wi deleted)
      2. Substitution effect delta(wi) = max over WordNet synonyms of the true-class-prob decrease
      3. Rank words by softmax(S) * delta(wi); greedily substitute in that order
      4. Stop early if the decision flips, or once the word budget is used up
    """
    words = text.split()
    if len(words) < 5:
        return text, 0

    num_changes_budget = max(1, int(len(words) * change_percent))
    original_prob = true_class_prob(text, url)
    true_prob_of_label = original_prob if label == 1 else (1 - original_prob)

    # word saliency via leave-one-out deletion
    saliency = []
    for i, word in enumerate(words):
        if not word.isalpha():
            saliency.append(0.0)
            continue
        deleted = words[:i] + words[i+1:]
        deleted_text = " ".join(deleted) if deleted else word
        p = true_class_prob(deleted_text, url)
        p_true = p if label == 1 else (1 - p)
        saliency.append(true_prob_of_label - p_true)  # positive = word supports the true label

    # best substitution + its effect, per word
    best_subs = {}
    delta = []
    for i, word in enumerate(words):
        if not word.isalpha() or saliency[i] == 0.0:
            delta.append(0.0)
            continue
        candidates = get_wordnet_synonyms(word)
        if not candidates:
            delta.append(0.0)
            continue
        best_drop = -1.0
        best_word = None
        for cand in candidates:
            trial = words.copy()
            trial[i] = cand
            trial_text = " ".join(trial)
            p = true_class_prob(trial_text, url)
            p_true = p if label == 1 else (1 - p)
            drop = true_prob_of_label - p_true
            if drop > best_drop:
                best_drop = drop
                best_word = cand
        delta.append(max(best_drop, 0.0))
        if best_word is not None:
            best_subs[i] = best_word

    # combined ranking score = softmax(saliency) * delta
    saliency_arr = np.array(saliency)
    exp_s = np.exp(saliency_arr - saliency_arr.max())
    softmax_s = exp_s / exp_s.sum() if exp_s.sum() > 0 else exp_s
    combined_score = softmax_s * np.array(delta)

    ranked_indices = np.argsort(combined_score)[::-1]

    # greedy substitution, stop early if decision flips
    attacked_words = words.copy()
    n_changed = 0
    original_decision = original_prob > 0.5
    for idx in ranked_indices:
        if n_changed >= num_changes_budget:
            break
        if idx not in best_subs or combined_score[idx] <= 0:
            continue
        attacked_words[idx] = best_subs[idx]
        n_changed += 1
        current_prob = true_class_prob(" ".join(attacked_words), url)
        if (current_prob > 0.5) != original_decision:
            break  

    return " ".join(attacked_words), n_changed


def evaluate(df_eval, use_adv=False):
    y_true, y_prob = [], []
    col = 'adv_email' if use_adv else 'email_text'
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i + 8]
            e_in = email_tokenizer(
                batch[col].tolist(), padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            ).to(device)
            u_in = url_tokenizer(
                batch['url'].tolist(), padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(device)
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            prob = logits.sigmoid().cpu().numpy()
            y_true.extend(batch['label'].values)
            y_prob.extend(prob)
    y_pred = (np.array(y_prob) > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "y_true": [int(x) for x in y_true],
        "y_prob": [float(x) for x in y_prob],
    }


percentages = [0.10, 0.20, 0.30, 0.40, 0.50]
all_results = {}

print("\nRunning TRUE PWWS attacks with different change-percent budgets...\n")
clean_full = evaluate(test_df, use_adv=False)             
clean_phish = evaluate(test_df_phish, use_adv=False)        # phishing-only baseline
print(f"Full-set clean baseline (reference) -> F1: {clean_full['f1']:.4f} | Acc: {clean_full['accuracy']:.4f}")
print(f"Phishing-only clean accuracy         -> {clean_phish['accuracy']:.4f}")
all_results['clean_full'] = clean_full
all_results['clean_phish_only'] = clean_phish

for percent in percentages:
    print(f"\n-> Attacking with up to {percent * 100:.0f}% word budget (true PWWS, phishing-only)...")
    test_adv = test_df_phish.copy()
    adv_texts = []
    n_changed_list = []
    for _, row in tqdm(test_adv.iterrows(), total=len(test_adv), desc=f"PWWS {percent*100:.0f}%"):
        adv_text, n_changed = pwws_attack(
            str(row['email_text']), str(row['url']), int(row['label']), change_percent=percent
        )
        adv_texts.append(adv_text)
        n_changed_list.append(n_changed)
    test_adv['adv_email'] = adv_texts
    adv = evaluate(test_adv, use_adv=True)
    adv['mean_words_changed'] = float(np.mean(n_changed_list))

    # Evasion success rate: of phishing examples correctly detected pre-attack,
    # what fraction flip to "legitimate" post-attack.
    pre_correct = [i for i, p in enumerate(clean_phish['y_prob']) if p > 0.5]
    n_evaded = sum(1 for i in pre_correct if adv['y_prob'][i] <= 0.5)
    evasion_rate = n_evaded / len(pre_correct) if pre_correct else float('nan')
    adv['evasion_success_rate'] = evasion_rate
    adv['n_pre_correct'] = len(pre_correct)
    adv['n_evaded'] = n_evaded

    print(
        f"   Phishing-only Acc: {adv['accuracy']:.4f} | "
        f"Evasion success rate: {evasion_rate:.4f} ({n_evaded}/{len(pre_correct)}) | "
        f"mean words changed: {adv['mean_words_changed']:.2f}"
    )
    all_results[f"pwws_{int(percent*100)}"] = adv

with open("pwws_true_raw_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved raw per-example predictions to pwws_true_raw_results.json")
