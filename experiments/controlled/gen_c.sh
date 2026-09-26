#!/bin/bash
# C production data (n24). Datasets then token bundles.
cd /home/fischeri/Projects/Compiler/gr
C=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad/c
gen() { # name spec plan splits... [--exclude-dataset ...]
  name=$1; spec=$2; plan=$3; shift 3
  rm -rf data/$name
  if [ "$plan" = "-" ]; then planarg=""; else planarg="--plan $C/$plan"; fi
  echo "=== gen $name $(date +%T) ==="
  uv run python -m cfg_reducer.dataset_v2 --spec $C/$spec $planarg --out data/$name --allow-incomplete "$@" 2>&1 | tail -4
  python3 -c "
import json; m=json.load(open('data/$name/manifest.json')); print('   ', {s: (v['accepted'], v['attempts']) for s,v in m['splits'].items()}, 'excluded', m['selection']['excluded_datasets'][:2])"
}
gen c_layered spec_layered.json - --split train=100000:102000 --split val=102000:102200 --split test=102200:102400
gen c_structured spec_structured.json - --split train=100000:102100 --split val=102100:102310 --split test=102310:102520 --exclude-dataset data/c_layered
gen c_spaghetti spec_spaghetti.json - --split test=100000:100210 --exclude-dataset data/c_layered --exclude-dataset data/c_structured
gen c_structured_depth1 spec_depth1.json plan_depth1.json --split train=200000:202200 --split val=202200:202420 --split test=202420:202640
gen c_structured_depth3 spec_depth3.json plan_depth3.json --split test=200000:200220 --exclude-dataset data/c_structured_depth1
gen c_structured_merge2 spec_merge2.json plan_merge2.json --split train=300000:302200 --split val=302200:302420 --split test=302420:302640
gen c_structured_merge4 spec_merge4.json plan_merge4.json --split test=300000:301000 --exclude-dataset data/c_structured_merge2
gen c_layered_balanced spec_layered.json plan_balanced.json --split train=400000:406000 --split val=406000:406600 --split test=406600:407200
tok() { # out source [target]
  out=$1; src=$2; tgt=$3
  rm -rf data/$out
  if [ -n "$tgt" ]; then extra="--test-dataset data/$tgt"; else extra=""; fi
  echo "=== tok $out $(date +%T) ==="
  uv run python -m training.prepare_tokens --dataset data/$src $extra --window-from train --out data/$out 2>&1 | tail -2
}
tok tok_c_layered c_layered
tok tok_c_lay2str c_layered c_structured
tok tok_c_lay2spa c_layered c_spaghetti
tok tok_c_structured c_structured
tok tok_c_str2lay c_structured c_layered
tok tok_c_str2spa c_structured c_spaghetti
tok tok_c_depth_id c_structured_depth1
tok tok_c_depth c_structured_depth1 c_structured_depth3
tok tok_c_merge_id c_structured_merge2
tok tok_c_merge c_structured_merge2 c_structured_merge4
tok tok_c_balanced c_layered_balanced
echo GEN-C-DONE
