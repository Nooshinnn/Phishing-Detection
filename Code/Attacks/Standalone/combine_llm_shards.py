
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

shard_files = ["llm_paraphrase_shard1of3.json", "llm_paraphrase_shard2of3.json", "llm_paraphrase_shard3of3.json"]
shards = []
for fname in shard_files:
    with open(fname) as f:
        shards.append(json.load(f))

shards.sort(key=lambda s: s['shard_index'])

total_len = len(shards[0]['clean_phish_full']['y_prob'])
covered = [False] * total_len
for s in shards:
    start, end = s['row_range']
    for i in range(start, end):
        if covered[i]:
            print(f"[ERROR] row {i} covered by multiple shards -- check row_range values")
        covered[i] = True
if not all(covered):
    missing = [i for i, c in enumerate(covered) if not c]
    print(f"[ERROR] {len(missing)} rows not covered by any shard: {missing[:20]}...")
    print("Do not trust the merged result until this is resolved.")
else:
    print(f"All {total_len} rows covered exactly once across {len(shards)} shards -- merge is valid.")

clean_phish_full = shards[0]['clean_phish_full']
merged_y_true = [None] * total_len
merged_y_prob = [None] * total_len
total_fallback = 0

for s in shards:
    start, end = s['row_range']
    shard_result = s['shard_adv_result']
    for local_i, global_i in enumerate(range(start, end)):
        merged_y_true[global_i] = shard_result['y_true'][local_i]
        merged_y_prob[global_i] = shard_result['y_prob'][local_i]
    total_fallback += s['fallback_count']

y_true_arr = np.array(merged_y_true)
y_pred_arr = (np.array(merged_y_prob) > 0.5).astype(int)

merged_adv_result = {
    "accuracy": accuracy_score(y_true_arr, y_pred_arr),
    "f1": f1_score(y_true_arr, y_pred_arr, zero_division=0),
    "mcc": matthews_corrcoef(y_true_arr, y_pred_arr),
    "y_true": merged_y_true,
    "y_prob": merged_y_prob,
    "fallback_count": total_fallback,
}

clean_probs = clean_phish_full['y_prob']
pre_correct = [i for i, p in enumerate(clean_probs) if p > 0.5]
n_evaded = sum(1 for i in pre_correct if merged_y_prob[i] <= 0.5)
evasion_rate = n_evaded / len(pre_correct) if pre_correct else float('nan')
merged_adv_result['evasion_success_rate'] = evasion_rate
merged_adv_result['n_pre_correct'] = len(pre_correct)
merged_adv_result['n_evaded'] = n_evaded

all_results = {
    "clean_full": clean_phish_full,
    "clean_phish_only": clean_phish_full,
    "llm_paraphrase_strong": merged_adv_result,
}

with open("llm_paraphrase_raw_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

all_samples = []
for i in range(1, 4):
    try:
        with open(f"llm_paraphrase_shard{i}of3_samples.json") as f:
            all_samples.extend(json.load(f))
    except FileNotFoundError:
        pass
with open("llm_paraphrase_strong_samples.json", "w") as f:
    json.dump(all_samples, f, indent=2)

print("\n" + "=" * 90)
print("MERGED RESULT")
print("=" * 90)
print(f"Phishing-only clean accuracy: {clean_phish_full['accuracy']:.4f}")
print(f"Phishing-only accuracy AFTER LLM paraphrase: {merged_adv_result['accuracy']:.4f}")
print(f"Evasion success rate: {evasion_rate:.4f} ({n_evaded}/{len(pre_correct)})")
print(f"Total fallback count across all shards: {total_fallback}/{total_len}")
print("\nSaved llm_paraphrase_raw_results.json and llm_paraphrase_strong_samples.json")
print("(same filenames the single-file script would have produced -- drop-in compatible)")
