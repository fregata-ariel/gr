#!/usr/bin/env bash
set -euo pipefail
# repository root から実行する。
uv run python -m training.mixture_doe collect --prefix a24 --size 24 --runs runs --data data --points 103 --seeds 0,1,2,3,4,5 --baseline-dir experiments/mixture_doe/base_target --baseline-mode target --out experiments/mixture_doe/adaptive/r2/obs_p103.json
