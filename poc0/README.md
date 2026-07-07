# PoC-0 Colab Workflow

This is the staged Colab flow from `docs/plans/poc0_training_impl.md`.

Stage 2 delivers Cell 1-3. Cell 4 (`poc0.train`) lands in Stage 3. Cell 5-6 (`poc0.eval` and result inspection) land in Stage 4.

Cell 1, install pinned training dependencies:

```sh
%cd /content/gr
!python -m pip install --upgrade pip
!python -m pip install \
  "jax[cuda12]==0.10.2" \
  "flax==0.12.7" \
  "optax==0.2.8" \
  "orbax-checkpoint==0.12.1" \
  "grain==0.2.18" \
  "array-record==0.8.3"
```

Cell 2, generate locally, then upload the JSONL:

Local command before upload:

```sh
./.venv/bin/python scripts/gen_corpus.py \
  --out /tmp/poc0_corpus.jsonl \
  --min-nodes 6 --max-nodes 20 \
  --edge-probs 0.10,0.15,0.20,0.25,0.30 \
  --seeds-per-config 1000
```

Upload in Colab:

```python
from google.colab import files
uploaded = files.upload()  # upload poc0_corpus.jsonl
```

```sh
!mkdir -p /content/poc0_data
!mv poc0_corpus.jsonl /content/poc0_data/poc0_corpus.jsonl
!python - <<'PY'
import json
path = "/content/poc0_data/poc0_corpus.jsonl"
count = sum(1 for _ in open(path, encoding="utf-8"))
print({"jsonl_records": count})
PY
```

Cell 3, convert JSONL to ArrayRecord on Colab Linux:

```sh
!python -m poc0.array_record_data convert \
  --jsonl /content/poc0_data/poc0_corpus.jsonl \
  --out-dir /content/poc0_data/arrayrecord \
  --max-seq-len 80 \
  --split-seed 20260707
```

Cell 4, train. This command becomes available in Stage 3:

```sh
!python -m poc0.train \
  --data-dir /content/poc0_data/arrayrecord \
  --workdir /content/poc0_runs/poc0_t4 \
  --batch-size 256 \
  --steps 3000 \
  --warmup-steps 200 \
  --peak-lr 3e-4 \
  --end-lr 3e-5 \
  --weight-decay 0.01 \
  --eval-every 100 \
  --ckpt-every 500 \
  --seed 20260709
```

Cell 5, eval. This command becomes available in Stage 4:

```sh
!python -m poc0.eval \
  --checkpoint /content/poc0_runs/poc0_t4/checkpoints/final \
  --jsonl /content/poc0_data/poc0_corpus.jsonl \
  --manifest /content/poc0_data/arrayrecord/manifest.json \
  --out-dir /content/poc0_runs/poc0_t4/eval \
  --n-samples 1000 \
  --temperature 1.0 \
  --seed 20260710
```

Cell 6, inspect summary. This becomes useful once Stage 4 eval outputs exist:

```sh
!cat /content/poc0_runs/poc0_t4/eval/metrics.json
!ls -lh /content/poc0_runs/poc0_t4/eval
```
