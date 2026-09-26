#!/usr/bin/env bash
set -euo pipefail
# repository root から実行する。
uv run python -c 'import hashlib,pathlib,sys; actual=hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(); sys.exit(0 if actual == sys.argv[2] else '"'"'fixed test manifest hash mismatch'"'"')' data/s24_layered/manifest.json fce01258ac8c39cbedd923467bb49c814471fc722f5388ef7bcd4abb32bc6e65
uv run python -c 'import hashlib,pathlib,sys; actual=hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(); sys.exit(0 if actual == sys.argv[2] else '"'"'fixed test manifest hash mismatch'"'"')' data/s24_spaghetti/manifest.json 1eb8e0135c7615e6bb8291f18748e4aff2b1a686797010e1d9dab50f0acb2fc9
uv run python -c 'import hashlib,pathlib,sys; actual=hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(); sys.exit(0 if actual == sys.argv[2] else '"'"'fixed test manifest hash mismatch'"'"')' data/s24_structured/manifest.json f1414916dd69dcc5ad5a30ef5008f7fde60750b5cac4bb262150c1916401c892
if [ -e data/tok_d24_p100_lay ]; then printf '%s\n' 'existing dataset/bundle requires verification: data/tok_d24_p100_lay' >&2; exit 2; fi
uv run python -m training.prepare_tokens --dataset data/d24_p100 --test-dataset data/s24_layered --max-offset 20 --out data/tok_d24_p100_lay
if [ -e data/tok_d24_p100_str ]; then printf '%s\n' 'existing dataset/bundle requires verification: data/tok_d24_p100_str' >&2; exit 2; fi
uv run python -m training.prepare_tokens --dataset data/d24_p100 --test-dataset data/s24_structured --max-offset 20 --out data/tok_d24_p100_str
if [ -e data/tok_d24_p100_spa ]; then printf '%s\n' 'existing dataset/bundle requires verification: data/tok_d24_p100_spa' >&2; exit 2; fi
uv run python -m training.prepare_tokens --dataset data/d24_p100 --test-dataset data/s24_spaghetti --max-offset 20 --out data/tok_d24_p100_spa
if [ -e data/tok_d24_p101_lay ]; then printf '%s\n' 'existing dataset/bundle requires verification: data/tok_d24_p101_lay' >&2; exit 2; fi
uv run python -m training.prepare_tokens --dataset data/d24_p101 --test-dataset data/s24_layered --max-offset 20 --out data/tok_d24_p101_lay
if [ -e data/tok_d24_p101_str ]; then printf '%s\n' 'existing dataset/bundle requires verification: data/tok_d24_p101_str' >&2; exit 2; fi
uv run python -m training.prepare_tokens --dataset data/d24_p101 --test-dataset data/s24_structured --max-offset 20 --out data/tok_d24_p101_str
if [ -e data/tok_d24_p101_spa ]; then printf '%s\n' 'existing dataset/bundle requires verification: data/tok_d24_p101_spa' >&2; exit 2; fi
uv run python -m training.prepare_tokens --dataset data/d24_p101 --test-dataset data/s24_spaghetti --max-offset 20 --out data/tok_d24_p101_spa
if [ -e data/tok_d24_p102_lay ]; then printf '%s\n' 'existing dataset/bundle requires verification: data/tok_d24_p102_lay' >&2; exit 2; fi
uv run python -m training.prepare_tokens --dataset data/d24_p102 --test-dataset data/s24_layered --max-offset 20 --out data/tok_d24_p102_lay
if [ -e data/tok_d24_p102_str ]; then printf '%s\n' 'existing dataset/bundle requires verification: data/tok_d24_p102_str' >&2; exit 2; fi
uv run python -m training.prepare_tokens --dataset data/d24_p102 --test-dataset data/s24_structured --max-offset 20 --out data/tok_d24_p102_str
if [ -e data/tok_d24_p102_spa ]; then printf '%s\n' 'existing dataset/bundle requires verification: data/tok_d24_p102_spa' >&2; exit 2; fi
uv run python -m training.prepare_tokens --dataset data/d24_p102 --test-dataset data/s24_spaghetti --max-offset 20 --out data/tok_d24_p102_spa
