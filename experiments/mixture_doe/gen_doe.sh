#!/bin/bash
# DoE datasets: 10 design points at n24, common seed range, 3 fixed test sets excluded.
cd /home/fischeri/Projects/Compiler/gr
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad/doe
EX="--exclude-dataset data/s24_layered --exclude-dataset data/s24_structured --exclude-dataset data/s24_spaghetti --allow-version-mismatch --allow-incomplete"
for p in 1 2 3 4 5 6 7 8 9 10; do
  [ -f data/d24_p$p/manifest.json ] && { echo "skip p$p"; continue; }
  rm -rf data/d24_p$p
  echo "=== p$p $(date +%T)"
  uv run python -m cfg_reducer.dataset_v2 --spec $S/specs/spec_p$p.json --out data/d24_p$p --split train=800000:802150 --split val=802150:802370 --split test=802370:802400 $EX 2>&1 | tail -1
  python3 -c "import json;m=json.load(open('data/d24_p$p/manifest.json'));print('   ', {s:(v['accepted'],v['attempts']) for s,v in m['splits'].items()})"
done
echo GEN-DOE-DONE $(date +%T)
