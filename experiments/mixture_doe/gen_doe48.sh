#!/bin/bash
cd /home/fischeri/Projects/Compiler/gr
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad/doe
EX="--exclude-dataset data/s48_layered --exclude-dataset data/s48_structured --exclude-dataset data/s48_spaghetti --allow-version-mismatch --allow-incomplete"
for p in 1 2 3 7 8; do
  [ -f data/d48_p$p/manifest.json ] || { rm -rf data/d48_p$p; echo "=== d48_p$p $(date +%T)"; uv run python -m cfg_reducer.dataset_v2 --spec $S/specs/spec48_p$p.json --out data/d48_p$p --split train=810000:812150 --split val=812150:812370 --split test=812370:812400 $EX 2>&1 | tail -1; python3 -c "import json;m=json.load(open('data/d48_p$p/manifest.json'));print('   ', {s:(v['accepted'],v['attempts']) for s,v in m['splits'].items()})"; }
  for t in lay str spa; do case $t in lay) T=s48_layered;; str) T=s48_structured;; spa) T=s48_spaghetti;; esac; out=data/tok_d48_p${p}_$t; [ -f $out/meta.json ] && continue; rm -rf $out
    uv run python -m training.prepare_tokens --dataset data/d48_p$p --test-dataset data/$T --max-offset 37 --out $out 2>&1 | grep -E "vocab window|excluded|Error" | tr '\n' ' '; echo " <- $out"; done
done
echo GEN-DOE48-DONE $(date +%T)
