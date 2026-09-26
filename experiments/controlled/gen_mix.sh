#!/bin/bash
# Mixed-family training data (layered + structured + spaghetti, equal weights) and OOD bundles.
cd /home/fischeri/Projects/Compiler/gr
C=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad/c
rm -rf data/c_mixed
uv run python -m cfg_reducer.dataset_v2 --spec $C/spec_mixed.json --out data/c_mixed --allow-incomplete \
  --split train=500000:502100 --split val=502100:502310 --split test=502310:502520 \
  --exclude-dataset data/c_layered --exclude-dataset data/c_structured --exclude-dataset data/c_spaghetti \
  --exclude-dataset data/c_structured_depth3 --exclude-dataset data/c_structured_merge4 --allow-version-mismatch 2>&1 | tail -4
python3 -c "
import json; m=json.load(open('data/c_mixed/manifest.json')); print({s: (v['accepted'], v['attempts'], v.get('rejected_by_reason')) for s,v in m['splits'].items()})"
tok() { out=$1; src=$2; tgt=$3; rm -rf data/$out; if [ -n "$tgt" ]; then extra="--test-dataset data/$tgt"; else extra=""; fi; echo "=== tok $out ==="; uv run python -m training.prepare_tokens --dataset data/$src $extra --window-from train --out data/$out 2>&1 | tail -2; }
tok tok_c_mixed c_mixed
tok tok_c_mix2lay c_mixed c_layered
tok tok_c_mix2str c_mixed c_structured
tok tok_c_mix2spa c_mixed c_spaghetti
tok tok_c_mix2depth c_mixed c_structured_depth3
tok tok_c_mix2merge c_mixed c_structured_merge4
tok tok_c_lay2depth c_layered c_structured_depth3
tok tok_c_lay2merge c_layered c_structured_merge4
tok tok_c_str2depth c_structured c_structured_depth3
tok tok_c_str2merge c_structured c_structured_merge4
echo GEN-MIX-DONE
