#!/usr/bin/env bash
# Empirical check of Colab Pro concurrency and compute-unit rates. Every colab call takes the runner's
# lock (~/.cache/gr-runner/colab.lock) so it interleaves safely with a running Plan. `usage` needs
# CLI >= 0.7 and is run from an isolated copy (uvx, separate HOME with a copy of the token) so the
# pinned 0.6.0 install and its token file are untouched.
set -uo pipefail
L=$HOME/.cache/gr-runner/colab.lock; H=$HOME/.cache/gr-runner/colab072-home
clean() { grep -v 'new version\|colab update\|uv tool install\|silence this' | grep -v '^$'; }
usage() { HOME=$H flock $L timeout 120 uvx --from google-colab-cli==0.7.2 colab usage 2>&1 | clean; }
c() { flock $L timeout 240 colab "$@" 2>&1 | clean; }
echo "=== $(date +%T) usage before"; usage
echo "=== $(date +%T) sessions before"; c sessions
for s in par1 par2 par3; do echo "=== $(date +%T) new T4 -s $s"; c new --gpu T4 -s $s; echo "rc=${PIPESTATUS[0]}"; done
echo "=== $(date +%T) sessions with extras"; c sessions
echo "=== $(date +%T) usage with extras"; usage
for s in par1 par2 par3; do echo "=== $(date +%T) stop $s"; c stop -s $s; done
echo "=== $(date +%T) sessions after"; c sessions
echo "=== $(date +%T) usage after"; usage
echo "PARALLEL-TEST-DONE $(date +%T)"
