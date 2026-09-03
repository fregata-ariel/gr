"""Training-pipeline tests (torch-free): prepare_tokens end-to-end,
eval metrics, and self-containment guards for the Colab-only trainer."""

import ast
import json
from pathlib import Path

from cfg_reducer import dataset, model_input, store
from training import data_utils, eval_samples, prepare_tokens

CONFIG = {"num_nodes": 8, "edge_prob": 0.3}
TRAIN_AR = Path(__file__).parent.parent / "training" / "train_ar.py"


def _prepare(tmp_path):
    ds = tmp_path / "ds"
    dataset.build_dataset(
        ds, {"train": (0, 5), "val": (5, 8)}, CONFIG, version="test"
    )
    out = tmp_path / "tokens"
    meta = prepare_tokens.prepare(ds, out)
    return ds, out, meta


def test_prepare_tokens_end_to_end(tmp_path):
    ds, out, meta = _prepare(tmp_path)
    vocab = json.loads((out / "vocab.json").read_text())
    manifest = json.loads((ds / "manifest.json").read_text())

    assert vocab["PAD"] == 0
    assert f"REF_{meta['max_offset']}" in vocab

    seen_max_len = 0
    for split in ("train", "val"):
        rows = data_utils.read_jsonl(out / f"{split}.jsonl")
        assert len(rows) == manifest["splits"][split]["kept"]
        assert len(rows) == meta["splits"][split]

        for row in rows:
            mg = store.load_sample(ds / split / f"{row['sample_id']}.json")
            assert model_input.detokenize(row["tokens"], vocab) == \
                model_input.sketch_of(mg)
            seen_max_len = max(seen_max_len, len(row["tokens"]))

    assert meta["max_len"] == seen_max_len


def test_eval_metrics_on_valid_and_corrupted_streams(tmp_path):
    _, out, _ = _prepare(tmp_path)
    vocab = json.loads((out / "vocab.json").read_text())
    train = [r["tokens"] for r in data_utils.read_jsonl(out / "train.jsonl")]
    val = [r["tokens"] for r in data_utils.read_jsonl(out / "val.jsonl")]

    report = eval_samples.evaluate(val, vocab, train)
    assert report["total"] == len(val)
    assert report["well_formed"] == len(val)
    assert report["well_formed_rate"] == 1.0
    assert 0.0 <= report["novelty_rate"] <= 1.0

    # A replay of a training stream must not count as novel
    replay = eval_samples.evaluate([train[0]], vocab, train)
    assert replay["well_formed"] == 1
    assert replay["novel_sketches"] == 0

    # Dropping EOS breaks the grammar
    corrupted = [s[:-1] for s in val]
    broken = eval_samples.evaluate(corrupted, vocab, train)
    assert broken["well_formed"] == 0
    assert broken["well_formed_rate"] == 0.0


def test_jsonl_roundtrip(tmp_path):
    rows = [{"sample_id": "a", "tokens": [1, 2]}, {"sample_id": "b", "tokens": []}]
    path = tmp_path / "rows.jsonl"
    data_utils.write_jsonl(path, rows)
    assert data_utils.read_jsonl(path) == rows


def test_train_ar_is_valid_and_self_contained():
    source = TRAIN_AR.read_text(encoding="utf-8")
    tree = ast.parse(source)     # syntax guard (torch not installed locally)

    imported = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    # Single-file Colab upload: only stdlib + torch allowed
    assert "cfg_reducer" not in imported
    assert "training" not in imported
