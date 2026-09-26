#!/usr/bin/env bash
# experiments/pretrain/gen_data.sh — P1 data: per-n test datasets (200 each, reserved first), the mixed
# train/val dataset (20k / 1k, excluding every test dataset) and the token bundles (REF window 128,
# unlimited evaluation length). See docs/design/pretrain_longseq_detailed.md §2.
set -euo pipefail
cd "$(dirname "$0")/../.."
VERSION=$(git rev-parse HEAD)
NS="8 12 16 24 32 48 64 96 128 192 256"
echo "=== version $VERSION  start $(date +%T)"
for n in $NS; do
  [ -f data/pretrain_test_n${n}/manifest.json ] && { echo "skip test n$n"; continue; }
  uv run python -m cfg_reducer.dataset_v2 --spec experiments/pretrain/specs/n${n}.json \
    --out data/pretrain_test_n${n} --version "$VERSION" \
    --split test=$((2000000+n*10000)):$((2010000+n*10000)) --target-count test=200 --allow-incomplete 2>&1 | tail -1
  python3 -c "import json;m=json.load(open('data/pretrain_test_n${n}/manifest.json'))['splits']['test'];print('   test n$n accepted',m['accepted'],'attempts',m['attempts'],'complete',m['complete'])"
done
EXCLUDE=()
for n in $NS; do EXCLUDE+=(--exclude-dataset data/pretrain_test_n${n}); done
if [ ! -f data/pretrain_mix/manifest.json ]; then
  echo "=== mixed train/val $(date +%T)"
  uv run python -m cfg_reducer.dataset_v2 --spec experiments/pretrain/specs/mixed.json \
    --out data/pretrain_mix --version "$VERSION" --split train=1000000:1200000 \
    --split val=1200000:1250000 --target-count train=20000 --target-count val=1000 \
    "${EXCLUDE[@]}" 2>&1 | tail -1
fi
python3 -c "
import json;m=json.load(open('data/pretrain_mix/manifest.json'))['splits']
for s in ('train','val'): print('   ',s,'accepted',m[s]['accepted'],'attempts',m[s]['attempts'],'complete',m[s]['complete'], {n:v['accepted'] for n,v in m[s]['per_num_nodes'].items()})"
echo "=== tokenize $(date +%T)"
[ -f data/tok_pretrain_mix/meta.json ] || uv run python -m training.prepare_tokens --dataset data/pretrain_mix --out data/tok_pretrain_mix --max-offset 128 2>&1 | tail -1
for n in $NS; do
  [ -f data/tok_pretrain_n${n}/meta.json ] && continue
  uv run python -m training.prepare_tokens --dataset data/pretrain_mix --test-dataset data/pretrain_test_n${n} \
    --out data/tok_pretrain_n${n} --max-offset 128 --eval-length-policy unlimited 2>&1 | tail -1
done
echo "GEN-PRETRAIN-DONE $(date +%T)"
