#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
export NLTK_DATA="$PWD/.cache/nltk"
export GENSIM_DATA_DIR="$PWD/.cache/gensim"
export TOKENIZERS_PARALLELISM=false
# Hardened model: pdgan. Trains it, then tests: pdgan urlgan pwws textfooler bae textbugger llmparaphrase. Resumable.
H=pdgan
mkdir -p matrix_logs
ck="hardened_model_${H}_seed31415.pth"
[ -f "$ck" ] || python adversarial_hardening_generic.py $H 31415 > matrix_logs/train_$H.txt 2>&1
for t in pdgan urlgan pwws textfooler bae textbugger llmparaphrase; do
  out="matrix_${t}_on_hardened_model_${H}_seed31415.json"
  [ -f "$out" ] && { echo "skip $out"; continue; }
  echo "eval $H x $t"
  python cross_attack_eval_generic.py $ck $t > matrix_logs/eval_${H}_${t}.txt 2>&1
done
echo "DONE $H"
