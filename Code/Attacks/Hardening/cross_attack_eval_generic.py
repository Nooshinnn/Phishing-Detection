
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import random
import zlib
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import json

from model_and_data_utils import load_model, build_dataset, is_english_word, DEVICE

if len(sys.argv) < 3:
    print("Usage: python cross_attack_eval_generic.py <checkpoint.pth> <attack_name>")
    print("attack_name in: urlgan, pwws, textfooler, bae, textbugger, pdgan, llmparaphrase")
    sys.exit(1)

CHECKPOINT = sys.argv[1]
ATTACK_NAME = sys.argv[2]
VALID_ATTACKS = ["urlgan", "pwws", "textfooler", "bae", "textbugger", "pdgan", "llmparaphrase"]
if ATTACK_NAME not in VALID_ATTACKS:
    print(f"attack_name must be one of: {VALID_ATTACKS}")
    sys.exit(1)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = DEVICE

checkpoint_stem = os.path.splitext(os.path.basename(CHECKPOINT))[0]
OUT_FILE = f"matrix_{ATTACK_NAME}_on_{checkpoint_stem}.json"

print("=" * 90)
print(f"CROSS-ATTACK MATRIX: {ATTACK_NAME.upper()} vs {CHECKPOINT}")
print("=" * 90)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model(CHECKPOINT)
train_df, val_df, test_df = build_dataset()
test_df_phish = test_df[test_df['label'] == 1].reset_index(drop=True)
print(f"Evaluating on phishing-labeled test examples only: {len(test_df_phish)} / {len(test_df)}")


def true_class_prob(text, url, label):
    with torch.no_grad():
        e_in = email_tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        u_in = url_tokenizer([url], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        e_feat = email_model(e_in.input_ids, e_in.attention_mask)
        u_feat = url_model(u_in.input_ids, u_in.attention_mask)
        prob_phish = torch.sigmoid(comm_model(e_feat, u_feat)).item()
    return prob_phish if label == 1 else (1 - prob_phish)


def attack_urlgan(text, url, label):
    HOMOGLYPHS = {'o': '0', 'l': '1', 'i': '1', 'e': '3', 'a': '4', 's': '5', 'g': '9', 'b': '6', 't': '7', 'z': '2'}
    eligible = [i for i, c in enumerate(url) if c.lower() in HOMOGLYPHS]
    if not eligible:
        return text, url
    budget = max(1, int(len(eligible) * 0.50))
    orig = true_class_prob(text, url, label)
    chars = list(url)
    remaining = set(eligible)
    n = 0
    while n < budget and remaining:
        cur = true_class_prob(text, "".join(chars), label)
        cur_true = cur if label == 1 else (1 - cur)
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
        if (true_class_prob(text, "".join(chars), label) > 0.5) != (orig > 0.5):
            break
    return text, "".join(chars)


def attack_pwws(text, url, label):
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

    words = text.split()
    words = words[:200]
    if len(words) < 5:
        return text, url
    budget = max(1, int(len(words) * 0.50))
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
    return " ".join(out), url


_TF_WV = None
def attack_textfooler(text, url, label):
    global _TF_WV
    if _TF_WV is None:
        import gensim.downloader as gensim_api
        _TF_WV = gensim_api.load("glove-wiki-gigaword-100")
    wv = _TF_WV

    def neighbors(word, topn=15):
        wl = word.lower()
        if wl not in wv:
            return []
        try:
            n = wv.most_similar(wl, topn=topn)
        except KeyError:
            return []
        return [c for c, _ in n if c.isalpha() and c.lower() != wl and is_english_word(c)]

    words = text.split()
    words = words[:200]
    if len(words) < 5:
        return text, url
    budget = max(1, int(len(words) * 0.50))
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
    return " ".join(out), url


_BAE_MLM = None
_BAE_TOK = None
def attack_bae(text, url, label):
    global _BAE_MLM, _BAE_TOK
    if _BAE_MLM is None:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        _BAE_TOK = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        _BAE_MLM = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device)
        _BAE_MLM.eval()
    mlm_tok, mlm = _BAE_TOK, _BAE_MLM

    words = text.split()
    words = words[:200]
    if len(words) < 5:
        return text, url
    n_changes = max(1, int(len(words) * 0.50))
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
    return " ".join(out), url


_TB_WV = None
def attack_textbugger(text, url, label):
    global _TB_WV
    if _TB_WV is None:
        import gensim.downloader as gensim_api
        _TB_WV = gensim_api.load("glove-wiki-gigaword-100")
    wv = _TB_WV
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
    words = text.split()
    words = words[:200]
    if len(words) < 5:
        return text, url
    budget = max(1, int(len(words) * 0.50))
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
    return " ".join(out), url


_PDGAN_GEN = None
def attack_pdgan(text, url, label):
    global _PDGAN_GEN
    if _PDGAN_GEN is None:
        from transformers import AutoModelForMaskedLM
        class PDGAN_Generator(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlm = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
            def forward(self, input_ids, attention_mask=None):
                return self.mlm(input_ids=input_ids, attention_mask=attention_mask).logits
        _PDGAN_GEN = PDGAN_Generator().to(device)
        _PDGAN_GEN.load_state_dict(torch.load("pdgan_generator_fixed.pth", map_location=device))
        _PDGAN_GEN.eval()

    def is_english_text(t, min_chars=15):
        from langdetect import detect, LangDetectException
        if len(str(t).strip()) < min_chars:
            return True
        try:
            return detect(str(t)) == "en"
        except LangDetectException:
            return False

    text = str(text)
    if len(text) < 20:
        return text, url
    inputs = email_tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    input_ids = inputs.input_ids
    mask = torch.rand_like(input_ids, dtype=torch.float32, device=device) < 0.15
    mask = mask & (input_ids != email_tokenizer.pad_token_id)
    masked = input_ids.clone(); masked[mask] = email_tokenizer.mask_token_id
    with torch.no_grad():
        logits = _PDGAN_GEN(masked, inputs.attention_mask)
        fake_ids = logits.argmax(dim=-1)
        fake_ids = torch.where(mask, fake_ids, input_ids)
    adv_text = email_tokenizer.decode(fake_ids[0], skip_special_tokens=True)
    return (adv_text if is_english_text(adv_text) else text), url


_LLM_MODEL = None
_LLM_TOK = None
def attack_llmparaphrase(text, url, label):
    global _LLM_MODEL, _LLM_TOK
    if _LLM_MODEL is None:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        _LLM_TOK = T5Tokenizer.from_pretrained("google/flan-t5-large")
        _LLM_MODEL = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)
        _LLM_MODEL.eval()

    prompt = (
        "You are a highly skilled professional writer. Rewrite the following email to be "
        "extremely convincing, natural, formal, and trustworthy. Keep the exact same meaning "
        f"and intent but make it very difficult to detect as suspicious or phishing:\n\n{text}\n\n"
        "Professional Rewritten Email (in English):"
    )
    inputs = _LLM_TOK(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_len = inputs.input_ids.shape[1]
    min_new = min(int(input_len * 0.7), 500)
    with torch.no_grad():
        outputs = _LLM_MODEL.generate(
            **inputs, max_length=512, min_new_tokens=min_new, num_beams=4,
            do_sample=True, temperature=0.95, top_p=0.98, repetition_penalty=1.25,
        )
    return _LLM_TOK.decode(outputs[0], skip_special_tokens=True), url


ATTACKS = {
    "urlgan": attack_urlgan, "pwws": attack_pwws, "textfooler": attack_textfooler,
    "bae": attack_bae, "textbugger": attack_textbugger, "pdgan": attack_pdgan,
    "llmparaphrase": attack_llmparaphrase,
}


def evaluate(df_eval, text_col='email_text', url_col='url'):
    y_true, y_prob = [], []
    with torch.no_grad():
        for i in range(0, len(df_eval), 8):
            batch = df_eval.iloc[i:i + 8]
            e_in = email_tokenizer(batch[text_col].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            u_in = url_tokenizer(batch[url_col].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            e_feat = email_model(e_in.input_ids, e_in.attention_mask)
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)
            logits = comm_model(e_feat, u_feat)
            prob = logits.sigmoid().cpu().numpy()
            y_true.extend(batch['label'].values)
            y_prob.extend(prob)
    y_pred = (np.array(y_prob) > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "y_true": [int(x) for x in y_true], "y_prob": [float(x) for x in y_prob],
    }


clean = evaluate(test_df_phish)
pre_correct = [i for i, p in enumerate(clean['y_prob']) if p > 0.5]
print(f"Clean phishing-only accuracy ({checkpoint_stem}): {clean['accuracy']:.4f} ({len(pre_correct)} pre-attack-correct)")

attack_fn = ATTACKS[ATTACK_NAME]
adv_texts, adv_urls = [], []
for _, row in tqdm(test_df_phish.iterrows(), total=len(test_df_phish), desc=f"{ATTACK_NAME} @ 50%"):
    text, url = attack_fn(str(row['email_text']), str(row['url']), int(row['label']))
    adv_texts.append(text)
    adv_urls.append(url)

test_adv = test_df_phish.copy()
test_adv['adv_text'] = adv_texts
test_adv['adv_url'] = adv_urls
adv = evaluate(test_adv, text_col='adv_text', url_col='adv_url')

n_evaded = sum(1 for i in pre_correct if adv['y_prob'][i] <= 0.5)
evasion_rate = n_evaded / len(pre_correct) if pre_correct else float('nan')
adv['evasion_success_rate'] = evasion_rate
adv['n_evaded'] = n_evaded

print(f"\n{ATTACK_NAME.upper()} @ 50% vs {checkpoint_stem}: Evasion success rate: {evasion_rate:.4f} ({n_evaded}/{len(pre_correct)})")

with open(OUT_FILE, "w") as f:
    json.dump({"clean": clean, "attack_result": adv, "checkpoint": CHECKPOINT, "attack": ATTACK_NAME}, f, indent=2)
print(f"Saved {OUT_FILE}")
