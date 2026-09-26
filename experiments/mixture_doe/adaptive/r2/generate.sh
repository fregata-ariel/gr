#!/usr/bin/env bash
set -euo pipefail
# repository root から実行する。
uv run python -c 'import subprocess,sys; actual=subprocess.check_output(['"'"'git'"'"','"'"'rev-parse'"'"','"'"'HEAD'"'"'],text=True).strip(); sys.exit(0 if actual == sys.argv[1] else '"'"'dataset_version differs from HEAD'"'"')' 697c816084b7bd0686c5cd21fbc3a6236b407a2d
uv run python -c 'import hashlib,pathlib,sys; actual=hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(); sys.exit(0 if actual == sys.argv[2] else '"'"'fixed test manifest hash mismatch'"'"')' data/s24_layered/manifest.json fce01258ac8c39cbedd923467bb49c814471fc722f5388ef7bcd4abb32bc6e65
uv run python -c 'import hashlib,pathlib,sys; actual=hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(); sys.exit(0 if actual == sys.argv[2] else '"'"'fixed test manifest hash mismatch'"'"')' data/s24_spaghetti/manifest.json 1eb8e0135c7615e6bb8291f18748e4aff2b1a686797010e1d9dab50f0acb2fc9
uv run python -c 'import hashlib,pathlib,sys; actual=hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(); sys.exit(0 if actual == sys.argv[2] else '"'"'fixed test manifest hash mismatch'"'"')' data/s24_structured/manifest.json f1414916dd69dcc5ad5a30ef5008f7fde60750b5cac4bb262150c1916401c892
if [ -e data/d24_p103 ]; then printf '%s\n' 'existing dataset/bundle requires verification: data/d24_p103' >&2; exit 2; fi
uv run python -m cfg_reducer.dataset_v2 --spec experiments/mixture_doe/adaptive/r2/specs/spec_p103.json --out data/d24_p103 --split train=800000:802150 --split val=802150:802370 --split test=802370:802400 --exclude-dataset data/s24_layered --exclude-dataset data/s24_structured --exclude-dataset data/s24_spaghetti --allow-version-mismatch --allow-incomplete
