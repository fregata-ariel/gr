#!/bin/bash
# Re-score the nine n24 checkpoints on CPU session "samp" (adds ref_type_nll)
cd /home/fischeri/Projects/Compiler/gr
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad
N=${1:-24}
for f in vocab.json meta.json test.jsonl; do colab upload -s samp data/tokens_b_n$N/$f /content/$f 2>&1 | tail -1; done
runs=""
for cfg in base mask ptr; do for s in 0 1 2; do
  r=b_${cfg}_n${N}_s$s; runs="$runs \"$r\","
  colab upload -s samp runs/$r/model.pt /content/${r}_model.pt 2>&1 | tail -1
  colab upload -s samp runs/$r/samples.json /content/${r}_samples.json 2>&1 | tail -1
done; done
echo "[${runs%,}]" > $S/rescore_list.json
colab upload -s samp $S/rescore_list.json /content/rescore_list.json 2>&1 | tail -1
echo "--- rescore n$N ---"
colab exec -s samp -f $S/rescore_b.py --timeout 1800 2>&1 | grep -E "RESCORED|RESCORE-DONE|Error|error|Traceback|line [0-9]+"
for cfg in base mask ptr; do for s in 0 1 2; do
  r=b_${cfg}_n${N}_s$s
  colab download -s samp /content/${r}_test_scores.jsonl runs/$r/test_scores.jsonl 2>&1 | grep -iE "error|not found"
done; done
echo "RESCORE-N$N-DONE"
