
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import zlib
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import warnings

from model_and_data_utils import load_model, build_dataset, is_english_word, DEVICE

warnings.filterwarnings("ignore")
ATTACK_NAME = sys.argv[1] if len(sys.argv) > 1 else None
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 31415  # optional 2nd CLI arg for multi-seed repeats
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = DEVICE

VALID_ATTACKS = ["urlgan", "pwws", "textfooler", "bae", "textbugger", "pdgan", "llmparaphrase"]
if ATTACK_NAME not in VALID_ATTACKS:
    print(f"Usage: python adversarial_hardening_generic.py <{'|'.join(VALID_ATTACKS)}> [seed]")
    sys.exit(1)

CHECKPOINT_OUT = f"hardened_model_{ATTACK_NAME}_seed{SEED}.pth"
print("=" * 90)
print(f"Adversarial hardening: augmenting training data with {ATTACK_NAME.upper()} perturbations")
print(f"Will save to: {CHECKPOINT_OUT}")
print("=" * 90)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
train_df, val_df, test_df = build_dataset()
train_df_phish = train_df[train_df['label'] == 1].reset_index(drop=True)
print(f"Train: {len(train_df)} (phishing subset for hardening: {len(train_df_phish)})")

AUGMENT_FRACTION = 0.30
augment_source = train_df_phish.sample(frac=AUGMENT_FRACTION, random_state=SEED).reset_index(drop=True)
print(f"Generating {len(augment_source)} augmentation examples with {ATTACK_NAME}...")


def true_class_prob(text, url, label):
    with torch.no_grad():
        e_in = email_tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer([url], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        prob_phish = torch.sigmoid(comm_model(e_feat, u_feat)).item()
    return prob_phish if label == 1 else (1 - prob_phish)


def gen_urlgan():
    HOMOGLYPHS = {'o': '0', 'l': '1', 'i': '1', 'e': '3', 'a': '4', 's': '5', 'g': '9', 'b': '6', 't': '7', 'z': '2'}
    def attack(text, url, label, change_percent=0.30):
        eligible = [i for i, c in enumerate(url) if c.lower() in HOMOGLYPHS]
        if not eligible:
            return url
        budget = max(1, int(len(eligible) * change_percent))
        orig_prob = true_class_prob(text, url, label)
        chars = list(url)
        remaining = set(eligible)
        n = 0
        while n < budget and remaining:
            cur_prob = true_class_prob(text, "".join(chars), label)
            cur_true = cur_prob if label == 1 else (1 - cur_prob)
            best_i, best_e = None, -1.0
            for i in remaining:
                trial = chars.copy(); trial[i] = HOMOGLYPHS[trial[i].lower()]
                p = true_class_prob(text, "".join(trial), label)
                p_true = p if label == 1 else (1 - p)
                e = cur_true - p_true
                if e > best_e:
                    best_e, best_i = e, i
            if best_i is None or best_e <= 0:
                break
            chars[best_i] = HOMOGLYPHS[chars[best_i].lower()]; remaining.discard(best_i); n += 1
            if (true_class_prob(text, "".join(chars), label) > 0.5) != (orig_prob > 0.5):
                break
        return "".join(chars)

    rows = []
    for _, row in tqdm(augment_source.iterrows(), total=len(augment_source), desc="urlgan augment"):
        new_url = attack(str(row['email_text']), str(row['url']), int(row['label']))
        rows.append((row['email_text'], new_url))
    return rows


def gen_pwws():
    import nltk
    from nltk.corpus import wordnet
    for pkg in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)

    def synonyms(word, max_c=8):
        out = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                c = lemma.name().replace('_', ' ')
                if c.isalpha() and c.lower() != word.lower():
                    out.add(c)
                if len(out) >= max_c:
                    break
            if len(out) >= max_c:
                break
        return list(out)

    def attack(text, url, label, change_percent=0.30):
        words = text.split()
        words = words[:200]
        if len(words) < 5:
            return text
        budget = max(1, int(len(words) * change_percent))
        orig = true_class_prob(text, url, label)
        true_p = orig if label == 1 else (1 - orig)
        sal = []
        for i, w in enumerate(words):
            if not w.isalpha():
                sal.append(0.0); continue
            d = words[:i] + words[i+1:]
            p = true_class_prob(" ".join(d) if d else w, url, label)
            p_t = p if label == 1 else (1 - p)
            sal.append(true_p - p_t)
        subs, delta = {}, []
        for i, w in enumerate(words):
            if not w.isalpha() or sal[i] == 0.0:
                delta.append(0.0); continue
            cands = synonyms(w)
            if not cands:
                delta.append(0.0); continue
            best_d, best_w = -1.0, None
            for c in cands:
                trial = words.copy(); trial[i] = c
                p = true_class_prob(" ".join(trial), url, label)
                p_t = p if label == 1 else (1 - p)
                d = true_p - p_t
                if d > best_d:
                    best_d, best_w = d, c
            delta.append(max(best_d, 0.0))
            if best_w:
                subs[i] = best_w
        sal_arr = np.array(sal)
        exp_s = np.exp(sal_arr - sal_arr.max())
        soft = exp_s / exp_s.sum() if exp_s.sum() > 0 else exp_s
        score = soft * np.array(delta)
        ranked = np.argsort(score)[::-1]
        out = words.copy(); n = 0
        for idx in ranked:
            if n >= budget:
                break
            if idx not in subs or score[idx] <= 0:
                continue
            out[idx] = subs[idx]; n += 1
            if (true_class_prob(" ".join(out), url, label) > 0.5) != (orig > 0.5):
                break
        return " ".join(out)

    rows = []
    for _, row in tqdm(augment_source.iterrows(), total=len(augment_source), desc="pwws augment"):
        new_text = attack(str(row['email_text']), str(row['url']), int(row['label']))
        rows.append((new_text, row['url']))
    return rows


def gen_textfooler():
    import gensim.downloader as gensim_api
    wv = gensim_api.load("glove-wiki-gigaword-100")

    def neighbors(word, topn=15):
        wl = word.lower()
        if wl not in wv:
            return []
        try:
            n = wv.most_similar(wl, topn=topn)
        except KeyError:
            return []
        return [c for c, _ in n if c.isalpha() and c.lower() != wl and is_english_word(c)]

    def attack(text, url, label, change_percent=0.30):
        words = text.split()
        words = words[:200]
        if len(words) < 5:
            return text
        budget = max(1, int(len(words) * change_percent))
        orig = true_class_prob(text, url, label)
        imp = []
        for i, w in enumerate(words):
            if not w.isalpha():
                imp.append(0.0); continue
            d = words[:i] + words[i+1:]
            p = true_class_prob(" ".join(d) if d else w, url, label)
            imp.append(orig - p)
        ranked = np.argsort(imp)[::-1]
        out = words.copy(); n = 0
        for idx in ranked:
            if n >= budget:
                break
            if imp[idx] <= 0:
                continue
            cands = neighbors(words[idx])
            if not cands:
                continue
            best_d, best_w = -1.0, None
            for c in cands:
                trial = out.copy(); trial[idx] = c
                p = true_class_prob(" ".join(trial), url, label)
                d = orig - p
                if d > best_d:
                    best_d, best_w = d, c
            if best_w and best_d > 0:
                out[idx] = best_w; n += 1
                if (true_class_prob(" ".join(out), url, label) > 0.5) != (orig > 0.5):
                    break
        return " ".join(out)

    rows = []
    for _, row in tqdm(augment_source.iterrows(), total=len(augment_source), desc="textfooler augment"):
        new_text = attack(str(row['email_text']), str(row['url']), int(row['label']))
        rows.append((new_text, row['url']))
    return rows


def gen_bae():
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    mlm_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    mlm = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device)
    mlm.eval()

    def attack(text, url, label, change_percent=0.30):
        words = text.split()
        words = words[:200]
        if len(words) < 5:
            return text
        n_changes = max(1, int(len(words) * change_percent))
        orig = true_class_prob(text, url, label)
        sal = []
        for i, w in enumerate(words):
            if not w.isalpha():
                sal.append(0.0); continue
            m = words.copy(); m[i] = mlm_tok.mask_token
            p = true_class_prob(" ".join(m), url, label)
            sal.append(abs(orig - p))
        idxs = np.argsort(sal)[::-1][:n_changes]
        out = words.copy()
        for idx in idxs:
            if not out[idx].isalpha():
                continue
            m = out.copy(); m[idx] = mlm_tok.mask_token
            inputs = mlm_tok(" ".join(m), return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                logits = mlm(**inputs).logits
            mp = (inputs.input_ids == mlm_tok.mask_token_id).nonzero(as_tuple=True)[1]
            if mp.numel() == 0:
                continue
            top = logits[0, mp[0]].topk(20).indices
            cands = []
            for t in top:
                c = mlm_tok.decode(t).strip()
                if c.isalpha() and c.lower() != out[idx].lower():
                    cands.append(c)
                if len(cands) >= 8:
                    break
            best_w, best_p = None, None
            for c in cands:
                trial = out.copy(); trial[idx] = c
                p = true_class_prob(" ".join(trial), url, label)
                if best_p is None or p < best_p:
                    best_p, best_w = p, c
            if best_w:
                out[idx] = best_w
        return " ".join(out)

    rows = []
    for _, row in tqdm(augment_source.iterrows(), total=len(augment_source), desc="bae augment"):
        new_text = attack(str(row['email_text']), str(row['url']), int(row['label']))
        rows.append((new_text, row['url']))
    return rows


def gen_textbugger():
    import gensim.downloader as gensim_api
    wv = gensim_api.load("glove-wiki-gigaword-100")
    HOMOGLYPHS = {'o': '0', 'l': '1', 'i': '1', 'e': '3', 'a': '@', 's': '5', 'g': '9', 'b': '6', 't': '7'}

    def word_rng(w):
        return random.Random(zlib.crc32(w.encode('utf-8')))

    def swap(w):
        if len(w) < 4: return None
        r = word_rng(w); i = r.randint(1, len(w)-3)
        c = list(w); c[i], c[i+1] = c[i+1], c[i]; return "".join(c)
    def delete(w):
        if len(w) < 4: return None
        r = word_rng(w); i = r.randint(1, len(w)-2)
        return w[:i] + w[i+1:]
    def space(w):
        if len(w) < 4: return None
        r = word_rng(w); i = r.randint(1, len(w)-2)
        return w[:i] + " " + w[i:]
    def homo(w):
        if not any(c in HOMOGLYPHS for c in w.lower()): return None
        chars = list(w)
        for i, c in enumerate(w.lower()):
            if c in HOMOGLYPHS:
                chars[i] = HOMOGLYPHS[c]; break
        return "".join(chars)
    def embed(w):
        wl = w.lower()
        if wl not in wv: return None
        try:
            n = wv.most_similar(wl, topn=5)
        except KeyError:
            return None
        for c, _ in n:
            if c.isalpha() and c.lower() != wl and is_english_word(c):
                return c
        return None

    fns = [swap, delete, space, homo, embed]

    def attack(text, url, label, change_percent=0.30):
        words = text.split()
        words = words[:200]
        if len(words) < 5:
            return text
        budget = max(1, int(len(words) * change_percent))
        orig = true_class_prob(text, url, label)
        imp = []
        for i, w in enumerate(words):
            if not w.isalpha():
                imp.append(0.0); continue
            d = words[:i] + words[i+1:]
            p = true_class_prob(" ".join(d) if d else w, url, label)
            imp.append(orig - p)
        ranked = np.argsort(imp)[::-1]
        out = words.copy(); n = 0
        for idx in ranked:
            if n >= budget:
                break
            if imp[idx] <= 0:
                continue
            best_d, best_v = -1.0, None
            for fn in fns:
                v = fn(words[idx])
                if v is None:
                    continue
                trial = out.copy(); trial[idx] = v
                p = true_class_prob(" ".join(trial), url, label)
                d = orig - p
                if d > best_d:
                    best_d, best_v = d, v
            if best_v and best_d > 0:
                out[idx] = best_v; n += 1
                if (true_class_prob(" ".join(out), url, label) > 0.5) != (orig > 0.5):
                    break
        return " ".join(out)

    rows = []
    for _, row in tqdm(augment_source.iterrows(), total=len(augment_source), desc="textbugger augment"):
        new_text = attack(str(row['email_text']), str(row['url']), int(row['label']))
        rows.append((new_text, row['url']))
    return rows


def gen_pdgan():
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    print("Loading pdgan_generator_fixed.pth -- must already exist (your finalized generator).")

    class PDGAN_Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlm = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
        def forward(self, input_ids, attention_mask=None):
            return self.mlm(input_ids=input_ids, attention_mask=attention_mask).logits

    gen = PDGAN_Generator().to(device)
    gen.load_state_dict(torch.load("pdgan_generator_fixed.pth", map_location=device))
    gen.eval()

    def is_english_text(text, min_chars=15):
        from langdetect import detect, LangDetectException
        if len(str(text).strip()) < min_chars:
            return True
        try:
            return detect(str(text)) == "en"
        except LangDetectException:
            return False

    def attack(text, mask_prob=0.15):
        text = str(text)
        if len(text) < 20:
            return text
        inputs = email_tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        input_ids = inputs.input_ids
        mask = torch.rand_like(input_ids, dtype=torch.float32, device=device) < mask_prob
        mask = mask & (input_ids != email_tokenizer.pad_token_id)
        masked = input_ids.clone(); masked[mask] = email_tokenizer.mask_token_id
        with torch.no_grad():
            logits = gen(masked, inputs.attention_mask)
            fake_ids = logits.argmax(dim=-1)
            fake_ids = torch.where(mask, fake_ids, input_ids)
        adv_text = email_tokenizer.decode(fake_ids[0], skip_special_tokens=True)
        return adv_text if is_english_text(adv_text) else text

    rows = []
    for _, row in tqdm(augment_source.iterrows(), total=len(augment_source), desc="pdgan augment"):
        rows.append((attack(str(row['email_text'])), row['url']))
    return rows


def gen_llmparaphrase():
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    print("Loading FLAN-T5-large (slow)...")
    para_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    paraphraser = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)
    paraphraser.eval()

    prompt_template = (
        "You are a highly skilled professional writer. Rewrite the following email to be "
        "extremely convincing, natural, formal, and trustworthy. Keep the exact same meaning "
        "and intent but make it very difficult to detect as suspicious or phishing:\n\n{text}\n\n"
        "Professional Rewritten Email (in English):"
    )

    def attack(text, url, label):
        prompt = prompt_template.format(text=text)
        inputs = para_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        input_len = inputs.input_ids.shape[1]
        min_new = min(int(input_len * 0.7), 500)
        with torch.no_grad():
            outputs = paraphraser.generate(
                **inputs, max_length=512, min_new_tokens=min_new, num_beams=4,
                do_sample=True, temperature=0.95, top_p=0.98, repetition_penalty=1.25,
            )
        return para_tokenizer.decode(outputs[0], skip_special_tokens=True)

    rows = []
    for _, row in tqdm(augment_source.iterrows(), total=len(augment_source), desc="llmparaphrase augment"):
        new_text = attack(str(row['email_text']), str(row['url']), int(row['label']))
        rows.append((new_text, row['url']))
    return rows


GENERATORS = {
    "urlgan": gen_urlgan, "pwws": gen_pwws, "textfooler": gen_textfooler,
    "bae": gen_bae, "textbugger": gen_textbugger, "pdgan": gen_pdgan,
    "llmparaphrase": gen_llmparaphrase,
}

augmented_rows = GENERATORS[ATTACK_NAME]()
augmented_df = augment_source.copy()
augmented_df['email_text'] = [r[0] for r in augmented_rows]
augmented_df['url'] = [r[1] for r in augmented_rows]

hardened_train_df = pd.concat([train_df, augmented_df]).sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"Hardened training set: {len(hardened_train_df)} (original {len(train_df)} + {len(augmented_df)} adversarial)")


def run_eval(df_eval, batch_size=16):
    email_model.eval(); url_model.eval(); comm_model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for i in range(0, len(df_eval), batch_size):
            batch = df_eval.iloc[i:i+batch_size]
            e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            prob = logits.sigmoid().cpu().numpy()
            y_true.extend(batch['label'].values)
            y_prob.extend(prob)
    y_pred = (np.array(y_prob) > 0.5).astype(int)
    return {"accuracy": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred)}


optimizer = optim.AdamW(
    list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()),
    lr=8e-6, weight_decay=0.01
)
criterion = nn.BCEWithLogitsLoss()
batch_size = 16
best_val_f1 = -1.0
patience, patience_counter = 4, 0
best_state = None

print(f"\nFine-tuning canonical model on hardened (clean + {ATTACK_NAME}-augmented) training set...")
for epoch in range(10):
    email_model.train(); url_model.train(); comm_model.train()
    for i in range(0, len(hardened_train_df), batch_size):
        batch = hardened_train_df.iloc[i:i+batch_size]
        labels = torch.tensor(batch['label'].values, dtype=torch.float32).to(device)
        e_in = email_tokenizer(batch['email_text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer(batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        logits = comm_model(e_feat, u_feat)
        loss = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    val_metrics = run_eval(val_df)
    print(f"Epoch {epoch+1} | Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
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

clean_test_result = run_eval(test_df)
print(f"\nHardened model ({ATTACK_NAME}) clean test performance: {clean_test_result}")
print(f"Saved {CHECKPOINT_OUT}")
print("\nNext: run cross_attack_eval_generic.py with this checkpoint against each of the 7 attacks.")
