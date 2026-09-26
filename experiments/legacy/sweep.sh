#!/bin/bash
# Node sweep: separate seed ranges per node count, train to saturation
set -e
cd /home/fischeri/Projects/Compiler/gr
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad
NODES=(12 16 20 24 28 32 40 48)

colab upload -s sweep training/train_ar.py /content/train_ar.py
colab upload -s sweep training/grammar_mask.py /content/grammar_mask.py

i=0
for n in "${NODES[@]}"; do
  base=$((i * 100000)); i=$((i + 1))
  NAME="sweep_n$n"
  echo "=== $NAME start (seeds from $base) ==="
  uv run python -m cfg_reducer.dataset --out "data/ds_$NAME" \
    --split "train=$base:$((base + 2000))" --split "val=$((base + 2000)):$((base + 2200))" \
    --num-nodes "$n" --edge-prob 0.18
  uv run python -m training.prepare_tokens --dataset "data/ds_$NAME" --out "data/tokens_$NAME"
  for f in train.jsonl val.jsonl vocab.json meta.json; do
    colab upload -s sweep "data/tokens_$NAME/$f" "/content/$f"
  done
  cat > "$S/w_$NAME.py" <<EOF
import sys
sys.path.insert(0, "/content")
import train_ar
train_ar.main(["--out", "/content/$NAME", "--epochs", "300",
               "--patience", "20", "--num-samples", "400"])
EOF
  colab exec -s sweep -f "$S/w_$NAME.py" --timeout 5400 2>&1 \
    | grep -E "early stop|wrote|Error|error" | tail -4
  mkdir -p "runs/$NAME"
  for f in samples.json history.json model.pt val_scores.jsonl; do
    colab download -s sweep "/content/$NAME/$f" "runs/$NAME/$f"
  done
  uv run python -m training.eval_samples --samples "runs/$NAME/samples.json" \
    --vocab "data/tokens_$NAME/vocab.json" --train-tokens "data/tokens_$NAME/train.jsonl" \
    --out "runs/$NAME/eval.json" | grep -E "well_formed_rate|novelty_rate"
  echo "=== $NAME done ==="
done

colab stop -s sweep
echo "SWEEP-DONE"
