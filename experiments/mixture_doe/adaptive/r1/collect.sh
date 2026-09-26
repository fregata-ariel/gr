#!/usr/bin/env bash
set -euo pipefail
# repository root から実行する。
uv run python -m training.mixture_doe collect --prefix a24 --size 24 --runs runs --data data --points 100 --seeds 0,1 --baseline-dir experiments/mixture_doe/base_target --baseline-mode target --out experiments/mixture_doe/adaptive/r1/obs_p100.json
uv run python -m training.mixture_doe collect --prefix a24 --size 24 --runs runs --data data --points 101 --seeds 0,1 --baseline-dir experiments/mixture_doe/base_target --baseline-mode target --out experiments/mixture_doe/adaptive/r1/obs_p101.json
uv run python -m training.mixture_doe collect --prefix a24 --size 24 --runs runs --data data --points 102 --seeds 0,1 --baseline-dir experiments/mixture_doe/base_target --baseline-mode target --out experiments/mixture_doe/adaptive/r1/obs_p102.json
