#!/bin/bash
cd /home/fischeri/Projects/Compiler/gr
declare -A TK=( [layered]=tok_c_layered [structured]=tok_c_structured [depth1]=tok_c_depth_id )
echo "=== canary flags (base sX -> mask sX) ==="
for src in layered structured depth1; do for s in 0 1 2; do
  out=$(uv run python -m training.eval_axes compare runs/c_${src}_base_n24_s$s/axes.json runs/c_${src}_mask_n24_s$s/axes.json 2>/dev/null | grep -E "flags|要検証" | tr '\n' ' ')
  echo "$src s$s: $out"
done; done
for s in 0 1 2; do
  out=$(uv run python -m training.eval_axes compare runs/c_layered_base_n24_s$s/axes.json runs/c_layered_ptr_n24_s$s/axes.json 2>/dev/null | grep -E "flags|要検証" | tr '\n' ' ')
  echo "layered ptr s$s: $out"
done
echo "=== token-class NLL (dev, seed mean) ==="
python3 - <<'PY'
import json
from statistics import mean
for src in ("layered","structured","depth1"):
    for cfg in ("base","mask") + (("ptr",) if src=="layered" else ()):
        vals={k:[] for k in ("nll_kind_mean","nll_loop_mean","nll_eos_mean")}
        for s in range(3):
            a=json.load(open(f"runs/c_{src}_{cfg}_n24_s{s}/axes.json"))["canary_token_class"]
            for k in vals: vals[k].append(a[k])
        print(src, cfg, {k[4:-5]: round(mean(v),3) for k,v in vals.items()})
PY
echo "=== KS fidelity (constrained samples vs own test) ==="
for src in layered structured depth1; do
  args=""; for cfg in base mask; do for s in 0 1 2; do args="$args --samples ${cfg}_s$s=runs/c_${src}_${cfg}_n24_s$s/samples_constrained.json"; done; done
  echo "-- $src"; uv run python -m training.sketch_stats --vocab data/${TK[$src]}/vocab.json --reference data/${TK[$src]}/test.jsonl $args 2>&1 | grep "mean KS" | sed 's/n_reference=.*mean/mean/' | tr '\n' ' '; echo
done
echo CANARY-KS-DONE
