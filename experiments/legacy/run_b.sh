#!/bin/bash
# B: 3 configs x 3 seeds x n24/n32 = 18 runs on Colab session "b"
cd /home/fischeri/Projects/Compiler/gr
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad
for n in 24 32; do
  for f in train.jsonl val.jsonl test.jsonl vocab.json meta.json; do
    colab upload -s b data/tokens_b_n$n/$f /content/$f 2>&1 | tail -1
  done
  for cfg in base mask ptr; do
    for seed in 0 1 2; do
      NAME=b_${cfg}_n${n}_s${seed}
      if [ -f runs/$NAME/test_scores.jsonl ]; then echo "skip $NAME"; continue; fi
      echo "=== $NAME start $(date +%T) ==="
      colab exec -s b -f $S/w_$NAME.py --timeout 5400 2>&1 | grep -E "epoch   1 |early stop|test NLL|wrote|Error|error|Traceback" | tail -8
      mkdir -p runs/$NAME
      for f in samples.json history.json model.pt val_scores.jsonl test_scores.jsonl samples_constrained.json; do
        colab download -s b /content/$NAME/$f runs/$NAME/$f 2>&1 | grep -iE "error|not found" 
      done
      uv run python -m training.eval_samples --samples runs/$NAME/samples.json \
        --vocab data/tokens_b_n$n/vocab.json --train-tokens data/tokens_b_n$n/train.jsonl \
        --out runs/$NAME/eval.json 2>&1 | grep -E "well_formed_rate"
      echo "=== $NAME done $(date +%T) ==="
    done
  done
done
echo RUN-B-DONE
