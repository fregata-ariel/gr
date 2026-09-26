#!/bin/bash
# 案 1' (sinusoidal level position) on sweep_n24 data
set -e
cd /home/fischeri/Projects/Compiler/gr
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad
NAME=sp2_n24; DATA=sweep_n24
colab upload -s sp2 training/train_ar.py /content/train_ar.py
colab upload -s sp2 training/grammar_mask.py /content/grammar_mask.py
for f in train.jsonl val.jsonl vocab.json meta.json; do
  colab upload -s sp2 "data/tokens_$DATA/$f" "/content/$f"
done
echo "=== $NAME start ==="
colab exec -s sp2 -f "$S/w_$NAME.py" --timeout 5400 2>&1 | grep -E "early stop|wrote|Error|error" | tail -4
mkdir -p "runs/$NAME"
for f in samples.json history.json model.pt val_scores.jsonl; do
  colab download -s sp2 "/content/$NAME/$f" "runs/$NAME/$f"
done
uv run python -m training.eval_samples --samples "runs/$NAME/samples.json" \
  --vocab "data/tokens_$DATA/vocab.json" --train-tokens "data/tokens_$DATA/train.jsonl" \
  --out "runs/$NAME/eval.json" | grep -E "well_formed_rate|novelty_rate"
uv run python -m training.eval_axes report --run "runs/$NAME" \
  --dataset "data/ds_$DATA" --tokens "data/tokens_$DATA" > /dev/null
echo "=== $NAME done ==="
colab stop -s sp2
echo "SP2-DONE"
