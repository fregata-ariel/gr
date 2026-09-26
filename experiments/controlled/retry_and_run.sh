#!/bin/bash
# Retry GPU session creation (Colab capacity), then start the C runner.
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad
cd /home/fischeri/Projects/Compiler/gr
for i in $(seq 1 60); do
  out=$(timeout 120 colab new --gpu T4 -s c2 2>&1 | tail -1)
  echo "attempt $i $(date +%T): $out"
  if colab sessions 2>&1 | grep -q "^\[c2\]"; then echo "SESSION-OK $(date +%T)"; break; fi
  sleep 120
done
if colab sessions 2>&1 | grep -q "^\[c2\]"; then
  bash $S/c/run_c.sh > $S/c/run_c.log 2>&1
else
  echo "GIVE-UP $(date +%T)"
fi
