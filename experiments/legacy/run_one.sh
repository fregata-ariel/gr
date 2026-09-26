#!/bin/bash
# One scale-experiment run: generate -> tokenize -> Colab train -> eval
set -e
NAME=$1; NODES=$2; TRAIN=$3; VAL=$4
cd /home/fischeri/Projects/Compiler/gr

uv run python -m cfg_reducer.dataset --out "data/ds_$NAME" \
  --split "train=$TRAIN" --split "val=$VAL" \
  --num-nodes "$NODES" --edge-prob 0.18
uv run python -m training.prepare_tokens \
  --dataset "data/ds_$NAME" --out "data/tokens_$NAME"

for f in train.jsonl val.jsonl vocab.json meta.json; do
  colab upload -s scale "data/tokens_$NAME/$f" "/content/$f"
done

colab exec -s scale -f training/train_ar.py --timeout 2400 2>&1 | tail -4

mkdir -p "runs/$NAME"
for f in samples.json history.json model.pt; do
  colab download -s scale "/content/run1/$f" "runs/$NAME/$f"
done

uv run python -m training.eval_samples \
  --samples "runs/$NAME/samples.json" \
  --vocab "data/tokens_$NAME/vocab.json" \
  --train-tokens "data/tokens_$NAME/train.jsonl" \
  --out "runs/$NAME/eval.json" | tail -12
