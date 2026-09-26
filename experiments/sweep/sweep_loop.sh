#!/bin/bash
# Keep (re)creating a T4 session and running the idempotent sweep runner until it reports RUN-SWEEP-DONE.
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad
cd /home/fischeri/Projects/Compiler/gr
export SWEEP_SESSION=sw
for cycle in $(seq 1 12); do
  for i in $(seq 1 36); do
    out=$(timeout 120 colab new --gpu T4 -s sw 2>&1 | tail -1); echo "cycle $cycle attempt $i $(date +%T): $out"
    colab sessions 2>&1 | grep -q "^\[sw\]" && { echo "SESSION-OK $(date +%T)"; break; }
    sleep 600
  done
  colab sessions 2>&1 | grep -q "^\[sw\]" || { echo "GIVE-UP $(date +%T)"; exit 1; }
  bash $S/sw/run_sweep.sh >> $S/sw/run_sweep.log 2>&1
  rc=$?
  timeout 90 colab stop -s sw > /dev/null 2>&1
  if grep -q RUN-SWEEP-DONE $S/sw/run_sweep.log; then echo "SWEEP-COMPLETE $(date +%T)"; exit 0; fi
  echo "cycle $cycle ended rc=$rc $(date +%T); retrying"; sleep 60
done
echo "GIVE-UP $(date +%T)"; exit 1
