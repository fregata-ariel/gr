#!/bin/bash
# B: remaining n32 runs on a fresh GPU session "b2"
cd /home/fischeri/Projects/Compiler/gr
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad
SES=b2
colab new --gpu T4 -s $SES 2>&1 | tail -2
colab upload -s $SES training/train_ar.py /content/train_ar.py 2>&1 | tail -1
colab upload -s $SES training/grammar_mask.py /content/grammar_mask.py 2>&1 | tail -1
n=32
for f in train.jsonl val.jsonl test.jsonl vocab.json meta.json; do
  colab upload -s $SES data/tokens_b_n$n/$f /content/$f 2>&1 | tail -1
done
for NAME in b_mask_n32_s1 b_mask_n32_s2 b_ptr_n32_s0 b_ptr_n32_s1 b_ptr_n32_s2; do
  if [ -f runs/$NAME/test_scores.jsonl ]; then echo "skip $NAME"; continue; fi
  echo "=== $NAME start $(date +%T) ==="
  colab exec -s $SES -f $S/w_$NAME.py --timeout 5400 2>&1 | grep -E "epoch   1 |early stop|test NLL|wrote|Error|error|Traceback" | tail -8
  mkdir -p runs/$NAME
  for f in samples.json history.json model.pt val_scores.jsonl test_scores.jsonl samples_constrained.json; do
    colab download -s $SES /content/$NAME/$f runs/$NAME/$f 2>&1 | grep -iE "error|not found"
  done
  uv run python -m training.eval_samples --samples runs/$NAME/samples.json \
    --vocab data/tokens_b_n$n/vocab.json --train-tokens data/tokens_b_n$n/train.jsonl \
    --out runs/$NAME/eval.json 2>&1 | grep -E "well_formed_rate"
  echo "=== $NAME done $(date +%T) ==="
done
echo RUN-B-REST-DONE
