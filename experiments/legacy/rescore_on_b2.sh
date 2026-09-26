#!/bin/bash
# After the remaining runs: re-score the 4 earlier n32 runs on session b2 (adds ref_type_nll)
cd /home/fischeri/Projects/Compiler/gr
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad
SES=b2
runs=""
for r in b_base_n32_s0 b_base_n32_s1 b_base_n32_s2 b_mask_n32_s0; do
  runs="$runs \"$r\","
  colab upload -s $SES runs/$r/model.pt /content/${r}_model.pt 2>&1 | tail -1
  colab upload -s $SES runs/$r/samples.json /content/${r}_samples.json 2>&1 | tail -1
done
echo "[${runs%,}]" > $S/rescore_list.json
colab upload -s $SES $S/rescore_list.json /content/rescore_list.json 2>&1 | tail -1
colab exec -s $SES -f $S/rescore_b.py --timeout 1800 2>&1 | grep -E "RESCORED|RESCORE-DONE|Error|error|Traceback|line [0-9]+"
for r in b_base_n32_s0 b_base_n32_s1 b_base_n32_s2 b_mask_n32_s0; do
  colab download -s $SES /content/${r}_test_scores.jsonl runs/$r/test_scores.jsonl 2>&1 | grep -iE "error|not found"
done
echo "RESCORE-B2-DONE"
