#!/bin/bash
cd /home/fischeri/Projects/Compiler/gr
for cfg in base mask; do for s in 0 1 2; do d=runs/c_mixed_${cfg}_n24_s$s; [ -f $d/axes.json ] || uv run python -m training.eval_axes report --run $d --dataset data/c_mixed --tokens data/tok_c_mixed > /dev/null 2>&1; done; done
echo "=== canary flags (base -> mask) ==="
for s in 0 1 2; do out=$(uv run python -m training.eval_axes compare runs/c_mixed_base_n24_s$s/axes.json runs/c_mixed_mask_n24_s$s/axes.json 2>/dev/null | grep -E "flags|要検証" | tr '\n' ' '); echo "mixed s$s: $out"; done
echo "=== token-class NLL (dev, seed mean) ==="
python3 - <<'PY'
import json
from statistics import mean
for cfg in ("base","mask"):
    vals={k:[] for k in ("nll_kind_mean","nll_loop_mean","nll_eos_mean")}
    for s in range(3):
        a=json.load(open(f"runs/c_mixed_{cfg}_n24_s{s}/axes.json"))["canary_token_class"]
        for k in vals: vals[k].append(a[k])
    print("mixed", cfg, {k[4:-5]: round(mean(v),3) for k,v in vals.items()})
PY
echo "=== KS fidelity (constrained samples vs mixed test) ==="
args=""; for cfg in base mask; do for s in 0 1 2; do args="$args --samples ${cfg}_s$s=runs/c_mixed_${cfg}_n24_s$s/samples_constrained.json"; done; done
uv run python -m training.sketch_stats --vocab data/tok_c_mixed/vocab.json --reference data/tok_c_mixed/test.jsonl $args 2>&1 | grep "mean KS" | sed 's/n_reference=.*mean/mean/' | tr '\n' ' '; echo
echo "=== violations (diagnostic samples, seed sum) ==="
python3 - <<'PY'
import json
for cfg in ("base","mask"):
    tot={}
    for s in range(3):
        for k,v in json.load(open(f"runs/c_mixed_{cfg}_n24_s{s}/eval.json"))["violations"].items(): tot[k]=tot.get(k,0)+v
    print(cfg, dict(sorted(tot.items(), key=lambda kv:-kv[1])))
PY
echo CANARY-MIX-DONE
