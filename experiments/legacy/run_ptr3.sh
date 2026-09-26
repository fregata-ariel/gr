#!/bin/bash
# 案 2 (--pointer) on sweep_n24 / sweep_n32 data
set -e
cd /home/fischeri/Projects/Compiler/gr
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad
colab upload -s ptr3 training/train_ar.py /content/train_ar.py
colab upload -s ptr3 training/grammar_mask.py /content/grammar_mask.py
for n in 24 32; do
  NAME=ptr3_n$n; DATA=sweep_n$n
  echo "=== $NAME start ==="
  for f in train.jsonl val.jsonl vocab.json meta.json; do
    colab upload -s ptr3 "data/tokens_$DATA/$f" "/content/$f"
  done
  colab exec -s ptr3 -f "$S/w_$NAME.py" --timeout 5400 2>&1 | grep -E "epoch   1 |early stop|wrote|Error|error|Traceback" | tail -6
  mkdir -p "runs/$NAME"
  for f in samples.json history.json model.pt val_scores.jsonl; do
    colab download -s ptr3 "/content/$NAME/$f" "runs/$NAME/$f"
  done
  uv run python -m training.eval_samples --samples "runs/$NAME/samples.json" \
    --vocab "data/tokens_$DATA/vocab.json" --train-tokens "data/tokens_$DATA/train.jsonl" \
    --out "runs/$NAME/eval.json" | grep -E "well_formed_rate|novelty_rate"
  uv run python -m training.eval_axes report --run "runs/$NAME" \
    --dataset "data/ds_$DATA" --tokens "data/tokens_$DATA" > /dev/null
  echo "=== $NAME done ==="
done
colab stop -s ptr
echo "PTR3-DONE"
