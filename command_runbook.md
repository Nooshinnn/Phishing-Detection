# Complete Command Runbook

*Every command needed to finish the experimental program. Organized by category. Check items off as they complete. Spread across your GPUs -- items within each category are independent of each other unless noted.*

---

```

---

## 1. Multi-seed: Source-Holdout (3 seeds -- matches "three is much better than one")

Seed 31415 already done (original run). Adding 3 more:

```
python source_holdout_eval.py 100
python source_holdout_eval.py 200
python source_holdout_eval.py 300
```

Each run does all 6 source-holdout folds internally -- 3 separate invocations, one per seed, fully independent, put each on its own GPU. This is his "strongest result in the paper" -- highest priority of the multi-seed asks if you have to triage.

---

## 2. Multi-seed: Coordinated Attacks (3 seeds per pair)

Seed 42 already done for all 6 pairs (the original run). Adding seeds 100, 200, 300 to each -- **18 commands**:

```
python Adversarial_Coordinated_PWWS_URLGAN.py 100
python Adversarial_Coordinated_PWWS_URLGAN.py 200
python Adversarial_Coordinated_PWWS_URLGAN.py 300

python Adversarial_Coordinated_TextFooler_URLGAN.py 100
python Adversarial_Coordinated_TextFooler_URLGAN.py 200
python Adversarial_Coordinated_TextFooler_URLGAN.py 300

python Adversarial_Coordinated_TextBugger_URLGAN.py 100
python Adversarial_Coordinated_TextBugger_URLGAN.py 200
python Adversarial_Coordinated_TextBugger_URLGAN.py 300

python Adversarial_Coordinated_BAE_URLGAN.py 100
python Adversarial_Coordinated_BAE_URLGAN.py 200
python Adversarial_Coordinated_BAE_URLGAN.py 300

python Adversarial_Coordinated_PDGAN_URLGAN.py 100
python Adversarial_Coordinated_PDGAN_URLGAN.py 200
python Adversarial_Coordinated_PDGAN_URLGAN.py 300

python Adversarial_Coordinated_LLMParaphrase_URLGAN.py 100
python Adversarial_Coordinated_LLMParaphrase_URLGAN.py 200
python Adversarial_Coordinated_LLMParaphrase_URLGAN.py 300
```

**This is what lets you compute a real confidence interval on S = A_combined - (A_text + A_url)**, the formal super-additivity quantity your supervisor asked for. BAE and TextFooler pairs are the slowest (~1.5-2hrs each based on your earlier logs) -- prioritize those if time gets tight, since they're also the strongest standalone attacks.

---

## 3. Multi-seed: Continual Learning / Replay (3 seeds x 2 sources)

Seed 31415 already done for both sources. Adding seeds 100, 200, 300 -- **6 commands**:

```
python continual_learning_experiment.py nigerian_fraud 100
python continual_learning_experiment.py nigerian_fraud 200
python continual_learning_experiment.py nigerian_fraud 300

python continual_learning_experiment.py phishing_pot 100
python continual_learning_experiment.py phishing_pot 200
python continual_learning_experiment.py phishing_pot 300
```

Each run trains and compares all 3 update strategies (naive/replay/frozen-encoder) -- this gives you multi-seed data on all three, not just replay, as a side benefit.

---

## 4. Multi-seed: Adversarial Hardening -- SCOPED DOWN, read this first

**A full 3x replication of the entire 7x7 cross-attack matrix is 21 training runs + up to 147 eval runs -- well over 100 GPU-hours, not realistic in a week.** Recommended scope instead: repeat only the **two hardening bases that anchor the headline finding** (BAE -- strongest "flexible" attack; PWWS -- clearest "constrained" attack), tested only against **PDGAN** (the attack the whole finding is about) plus **self** (sanity check). This gives a real confidence interval on the two most important matrix cells without the combinatorial blowup.

```
# Hardening training (4 new runs -- seed 31415 already done for both)
python adversarial_hardening_generic.py bae 100
python adversarial_hardening_generic.py bae 200
python adversarial_hardening_generic.py pwws 100
python adversarial_hardening_generic.py pwws 200

# Cross-attack eval against PDGAN + self (8 new runs)
python cross_attack_eval_generic.py hardened_model_bae_seed100.pth pdgan
python cross_attack_eval_generic.py hardened_model_bae_seed100.pth bae
python cross_attack_eval_generic.py hardened_model_bae_seed200.pth pdgan
python cross_attack_eval_generic.py hardened_model_bae_seed200.pth bae

python cross_attack_eval_generic.py hardened_model_pwws_seed100.pth pdgan
python cross_attack_eval_generic.py hardened_model_pwws_seed100.pth pwws
python cross_attack_eval_generic.py hardened_model_pwws_seed200.pth pdgan
python cross_attack_eval_generic.py hardened_model_pwws_seed200.pth pwws
```

**If you want the full 7x7 replicated instead**, tell me and I'll give you the exhaustive list -- just flagging that it's a very different time commitment (roughly 5-10x more compute than the scoped version above) before you commit to it.

---

## 5. One-time analysis scripts (no multi-seed needed, run once each)

```
# Calibration / Delta-p / ECE / Brier / NLL -- run against every attack's raw results
python calibration_full_analysis.py combined_corpus_test_predictions_seed31415.json bae_raw_results.json textfooler_raw_results.json pwws_true_raw_results.json urlgan_raw_results.json pdgan_raw_results.json llm_paraphrase_raw_results.json

# Attribution analysis for the LLM-paraphrase null result
python llm_paraphrase_attribution_analysis.py llm_paraphrase_strong_samples.json

# Reliability diagrams -- pick 2-3 attacks for the main-paper figure, e.g. BAE (strongest) and LLM-paraphrase (the null result) for contrast
python plot_reliability_diagrams.py calibration_full_analysis_results.json bae_raw_results.json llm_paraphrase_raw_results.json
```

---

## Realistic total cost estimate

| Category | Runs | Rough GPU-hours |
|---|---|---|
| Source-holdout (3 seeds) | 3 | 5-6 |
| Coordinated attacks (3 seeds x 6 pairs) | 18 | 20-35 |
| Continual learning (3 seeds x 2 sources) | 6 | 6 |
| Hardening, scoped | 12 | 4-6 |
| Analysis scripts | 3 | <1 |
| **Total new work** | **42** | **~35-50** |

Spread across multiple GPUs this is very doable in a week, but it's the coordinated-attack repeats (Category 2) that dominate the time budget -- worth starting those first and letting them run continuously.
