#!/usr/bin/env bash
# experiments/pretrain/run_final_parallel.sh <pos> — final 3 seeds on three concurrent Colab sessions.
#   make_plan --phase final --pos <pos> -> split_plan -> 3 runners (sessions final0/1/2, --gpu L4) -> wait
#   -> eval_buckets. Compute-unit balance is recorded before and after with the isolated 0.7.2 CLI.
set -euo pipefail
cd "$(dirname "$0")/../.."
POS=${1:?usage: run_final_parallel.sh <sinusoidal|alibi|none>}
GPU=${GR_COLAB_GPU:-L4}
L=$HOME/.cache/gr-runner/colab.lock; H=$HOME/.cache/gr-runner/colab072-home
usage() { HOME=$H flock $L timeout 120 uvx --from google-colab-cli==0.7.2 colab usage 2>&1 | grep -E 'balance|rate|assignments'; }
echo "=== $(date +%T) usage before"; usage || true
uv run python experiments/pretrain/make_plan.py --phase final --pos "$POS" --out-dir experiments/pretrain
uv run python experiments/pretrain/split_plan.py --plan experiments/pretrain/plan_final.json --out-dir experiments/pretrain
pids=()
for s in 0 1 2; do
  plan=experiments/pretrain/plan_final_pretrain_final_${POS}_s${s}.json
  log=experiments/pretrain/final_${POS}_s${s}_colab.log
  GR_BACKEND=colab nohup uv run python -m training.runner run --plan "$plan" --gpu "$GPU" --session "final${s}" > "$log" 2>&1 &
  pids+=($!); echo "=== $(date +%T) launched seed $s pid $! session final${s} -> $log"
  sleep 20   # stagger the three 'colab new' calls (global lock) so the runners do not all wait on it
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "=== $(date +%T) runners finished rc=$rc"
for s in 0 1 2; do grep -E 'RUN-DONE|GIVE-UP|FAILED' experiments/pretrain/final_${POS}_s${s}_colab.log | tail -1; done
echo "=== $(date +%T) usage after"; usage || true
uv run python -m training.eval_buckets --index experiments/pretrain/eval_index_final.json --out experiments/pretrain/final_eval.json && echo "EVAL-FINAL-DONE $(date +%T)"
echo "FINAL-PARALLEL-DONE $(date +%T) rc=$rc"
