#!/bin/bash
cd /home/fischeri/Projects/Compiler/gr
for p in 1 2 3 4 5 6 7 8 9 10; do for t in lay str spa; do
  case $t in lay) T=s24_layered;; str) T=s24_structured;; spa) T=s24_spaghetti;; esac
  out=data/tok_d24_p${p}_$t; [ -f $out/meta.json ] && { echo "skip $out"; continue; }; rm -rf $out
  uv run python -m training.prepare_tokens --dataset data/d24_p$p --test-dataset data/$T --max-offset 20 --out $out 2>&1 | grep -E "vocab window|excluded|Error" | tr '\n' ' '; echo " <- $out"
done; done
echo TOK-DOE-DONE $(date +%T)
