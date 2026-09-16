"""Fast, torch-free tests for training/info_baseline.py."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cfg_reducer.model_input import build_vocab
from training import info_baseline, mixture_doe
from training.controlled_eval import ref_records

VOCAB = build_vocab(3)

BASE = ["BOS", "KIND_ENTRY", "KIND_LINEAR", "KIND_MERGE", "REF_1", "EOS"]
LOOPED = ["BOS", "KIND_ENTRY", "KIND_LINEAR", "KIND_LOOP", "LOOP_START",
          "KIND_ENTRY", "KIND_LINEAR", "REF_1", "LOOP_END",
          "KIND_MERGE", "REF_1", "REF_2", "EOS"]


def _stream(names: list[str]) -> list[int]:
    return [VOCAB[name] for name in names]


def _fit(names_list: list[list[str]], alpha: float = 0.5) -> info_baseline.InfoBaseline:
    rows = [{"sample_id": f"s{i}", "seed": i % 3, "tokens": _stream(names)}
            for i, names in enumerate(names_list)]
    return info_baseline.InfoBaseline.fit(rows, VOCAB, 3, alpha)


def test_streams_are_well_formed():
    for names in (BASE, LOOPED):
        assert names[0] == "BOS" and names[-1] == "EOS"


def test_type_table_and_structural_distribution_sum_to_one():
    base = _fit([BASE])
    key = ("KIND_MERGE", 0)
    p_ref = base.type_probability(key)
    assert 0.0 < p_ref < 1.0
    assert p_ref + (1.0 - p_ref) == pytest.approx(1.0)

    names = base.structural_names
    seen = sum(base.structural_probability("BOS", "KIND_ENTRY", c)
               for c in names)
    assert seen == pytest.approx(1.0)
    # EOS is never a context token, so this falls back to the unigram.
    unseen = sum(base.structural_probability("KIND_ENTRY", "EOS", c)
                 for c in names)
    assert unseen == pytest.approx(1.0)


def test_score_shapes_and_class_partition():
    base = _fit([BASE, LOOPED])
    for names in (BASE, LOOPED):
        result = base.score(_stream(names))
        n = len(names) - 1
        assert result["n_tokens"] == n
        assert len(result["token_nll"]) == n
        assert all(math.isfinite(v) for v in result["token_nll"])
        assert result["nll"] == pytest.approx(sum(result["token_nll"]))
        assert result["nll_per_token"] == pytest.approx(result["nll"] / n)
        classes = result["class_nll"]
        assert set(classes) == {"KIND", "REF", "LOOP", "EOS", "TYPE"}
        total = sum(classes[k] for k in ("KIND", "REF", "LOOP", "EOS"))
        assert total == pytest.approx(result["nll"], abs=1e-9)


def test_model_prefers_training_statistics():
    base = _fit([BASE] * 20)
    base_nll = base.score(_stream(BASE))["nll"]
    # The largest legal alternative to the trained REF_1 is REF_2.
    ref_alt = ["BOS", "KIND_ENTRY", "KIND_LINEAR", "KIND_MERGE", "REF_2", "EOS"]
    assert base.score(_stream(ref_alt))["nll"] > base_nll
    kind_reversed = ["BOS", "KIND_MERGE", "KIND_LINEAR", "KIND_ENTRY",
                     "REF_1", "EOS"]
    assert base.score(_stream(kind_reversed))["nll"] > base_nll


def test_ref_values_agree_with_ref_records():
    base = _fit([BASE, LOOPED])
    assert base.max_k == 3
    for names in (BASE, LOOPED):
        tokens = _stream(names)
        ours = info_baseline.ref_values(tokens, VOCAB, 3)
        score = {"sample_id": "s", "token_nll": [0.0] * (len(tokens) - 1)}
        records = ref_records({"s": tokens}, [score], VOCAB, 3)
        assert [(r["k"], r["rel"], r["n_legal"]) for r in ours] == [
            (r["k"], r["rel"], r["n_legal"]) for r in records]
        assert [r["pos"] for r in ours] == [
            i for i in range(1, len(tokens)) if names[i].startswith("REF_")]


def test_score_rows_carry_sample_seed_and_baseline_name():
    base = _fit([BASE])
    rows = info_baseline.score_rows(
        base, [{"sample_id": "z", "seed": 2, "tokens": _stream(BASE)}])
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "z"
    assert rows[0]["seed"] == 2
    assert rows[0]["baseline"] == "freq_trigram"
    assert set(rows[0]) == {
        "sample_id", "seed", "n_tokens", "nll", "nll_per_token",
        "token_nll", "class_nll", "baseline"}


def test_cli_writes_rows_readable_by_mixture_doe(tmp_path, capsys):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "vocab.json").write_text(json.dumps(VOCAB))
    (bundle / "meta.json").write_text(json.dumps({"max_offset": 3}))
    train = [{"sample_id": "t", "seed": 0, "tokens": _stream(BASE)}]
    test = [{"sample_id": "a", "seed": 0, "tokens": _stream(BASE)},
            {"sample_id": "b", "seed": 1, "tokens": _stream(LOOPED)}]
    (bundle / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in train))
    (bundle / "test.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in test))
    out = tmp_path / "nested" / "base_lay.jsonl"
    info_baseline.main([
        "score", "--train", str(bundle / "train.jsonl"),
        "--vocab", str(bundle / "vocab.json"),
        "--meta", str(bundle / "meta.json"),
        "--test", str(bundle / "test.jsonl"), "--out", str(out),
    ])
    printed = capsys.readouterr().out
    assert "nll/token" in printed and "TYPE" in printed
    assert out.exists()

    baseline = info_baseline.InfoBaseline.fit(train, VOCAB, 3)
    results = info_baseline.score_rows(baseline, test)
    expected = (sum(r["nll"] for r in results)
                / sum(r["n_tokens"] for r in results))
    assert mixture_doe.nll_per_token(out) == pytest.approx(expected)
