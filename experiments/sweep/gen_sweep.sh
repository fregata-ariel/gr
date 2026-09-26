#!/bin/bash
# Milestone sweep data: per size, layered (train/val/test), structured test, spaghetti test, mixed (train/val/test); then bundles.
cd /home/fischeri/Projects/Compiler/gr
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad/sw
i=0
for n in 12 16 24 32 48; do
  B=$((700000 + i*10000)); i=$((i+1))
  echo "=== n$n (seed base $B) ==="
  rm -rf data/s${n}_layered data/s${n}_structured data/s${n}_spaghetti data/s${n}_mixed
  uv run python -m cfg_reducer.dataset_v2 --spec $S/spec_layered_n$n.json --out data/s${n}_layered --allow-incomplete --split train=$B:$((B+2000)) --split val=$((B+2000)):$((B+2200)) --split test=$((B+2200)):$((B+2400)) 2>&1 | tail -1
  uv run python -m cfg_reducer.dataset_v2 --spec $S/spec_structured_n$n.json --out data/s${n}_structured --allow-incomplete --split test=$((B+2400)):$((B+2630)) --exclude-dataset data/s${n}_layered 2>&1 | tail -1
  uv run python -m cfg_reducer.dataset_v2 --spec $S/spec_spaghetti_n$n.json --out data/s${n}_spaghetti --allow-incomplete --split test=$((B+2630)):$((B+2850)) --exclude-dataset data/s${n}_layered --exclude-dataset data/s${n}_structured 2>&1 | tail -1
  uv run python -m cfg_reducer.dataset_v2 --spec $S/spec_mixed_n$n.json --out data/s${n}_mixed --allow-incomplete --split train=$B:$((B+2150)) --split val=$((B+2150)):$((B+2370)) --split test=$((B+2370)):$((B+2600)) --exclude-dataset data/s${n}_layered --exclude-dataset data/s${n}_structured --exclude-dataset data/s${n}_spaghetti 2>&1 | tail -1
  python3 -c "
import json
for d in ('layered','structured','spaghetti','mixed'):
    m=json.load(open('data/s${n}_'+d+'/manifest.json')); print('   ', d, {s: (v['accepted'], v['attempts']) for s,v in m['splits'].items()})"
  tok() { out=$1; src=$2; tgt=$3; rm -rf data/$out; if [ -n "$tgt" ]; then extra="--test-dataset data/$tgt"; else extra=""; fi; uv run python -m training.prepare_tokens --dataset data/$src $extra --window-from train --out data/$out 2>&1 | grep -E "vocab window|excluded" | tr '\n' ' '; echo " <- $out"; }
  tok tok_s${n}_lay s${n}_layered
  tok tok_s${n}_lay2str s${n}_layered s${n}_structured
  tok tok_s${n}_lay2spa s${n}_layered s${n}_spaghetti
  tok tok_s${n}_mix s${n}_mixed
  tok tok_s${n}_mix2str s${n}_mixed s${n}_structured
  tok tok_s${n}_mix2spa s${n}_mixed s${n}_spaghetti
  tok tok_s${n}_mix2lay s${n}_mixed s${n}_layered
done
echo GEN-SWEEP-DONE
