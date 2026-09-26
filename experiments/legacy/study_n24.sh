#!/bin/bash
# n24 training-investment study: 3 configs on the shared tokens_n24 data
set -e
cd /home/fischeri/Projects/Compiler/gr
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad

for f in train.jsonl val.jsonl vocab.json meta.json; do
  colab upload -s study "data/tokens_n24/$f" "/content/$f"
done
colab upload -s study training/train_ar.py /content/train_ar.py
colab upload -s study training/grammar_mask.py /content/grammar_mask.py

run_cfg () {
  local NAME=$1 WRAPPER=$2
  echo "=== $NAME start ==="
  colab exec -s study -f "$S/$WRAPPER" --timeout 3600 2>&1 | tail -6
  mkdir -p "runs/$NAME"
  for f in samples.json history.json model.pt; do
    colab download -s study "/content/$NAME/$f" "runs/$NAME/$f"
  done
  uv run python -m training.eval_samples \
    --samples "runs/$NAME/samples.json" \
    --vocab data/tokens_n24/vocab.json \
    --train-tokens data/tokens_n24/train.jsonl \
    --out "runs/$NAME/eval.json" | grep -E "well_formed_rate|novelty_rate"
  echo "=== $NAME done ==="
}

run_cfg n24_e120 wE120.py
run_cfg n24_c256 wC256.py
run_cfg n24_c256e120 wC256E120.py

colab stop -s study
echo "STUDY-DONE"
