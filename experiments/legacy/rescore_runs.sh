#!/bin/bash
# Re-score given runs on CPU session "samp": rescore_runs.sh <N> run1 run2 ...
cd /home/fischeri/Projects/Compiler/gr
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad
N=$1; shift
colab upload -s samp training/train_ar.py /content/train_ar.py 2>&1 | tail -1
for f in vocab.json meta.json test.jsonl; do colab upload -s samp data/tokens_b_n$N/$f /content/$f 2>&1 | tail -1; done
runs=""
for r in "$@"; do
  runs="$runs \"$r\","
  colab upload -s samp runs/$r/model.pt /content/${r}_model.pt 2>&1 | tail -1
  colab upload -s samp runs/$r/samples.json /content/${r}_samples.json 2>&1 | tail -1
done
echo "[${runs%,}]" > $S/rescore_list.json
colab upload -s samp $S/rescore_list.json /content/rescore_list.json 2>&1 | tail -1
colab exec -s samp -f $S/rescore_b.py --timeout 1800 2>&1 | grep -E "RESCORED|RESCORE-DONE|Error|error|Traceback|line [0-9]+"
for r in "$@"; do
  colab download -s samp /content/${r}_test_scores.jsonl runs/$r/test_scores.jsonl 2>&1 | grep -iE "error|not found"
done
echo "RESCORE-RUNS-DONE"
