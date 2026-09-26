#!/bin/bash
# B aggregation: dev canaries (eval_axes), constrained-sample fidelity (sketch_stats), test summary (controlled_eval)
cd /home/fischeri/Projects/Compiler/gr
for n in 24 32; do
  for cfg in base mask ptr; do for seed in 0 1 2; do
    R=runs/b_${cfg}_n${n}_s${seed}
    [ -f $R/val_scores.jsonl ] || continue
    [ -f $R/axes.json ] || uv run python -m training.eval_axes report --run $R --dataset data/ds_b_n$n --tokens data/tokens_b_n$n > /dev/null
    [ -f $R/sketch_stats.txt ] || [ ! -f $R/samples_constrained.json ] || uv run python -m training.sketch_stats --reference data/tokens_b_n$n/test.jsonl --samples $R/samples_constrained.json --vocab data/tokens_b_n$n/vocab.json > $R/sketch_stats.txt
  done; done
  uv run python -m training.controlled_eval --size $n --tokens data/tokens_b_n$n --out runs/b_summary_n$n.json
done
