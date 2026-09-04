"""controlled_eval: interval statistics, paired deltas, REF records and
frequency baselines on synthetic token streams (torch-free)."""

import math

import pytest

from cfg_reducer import model_input
from training import controlled_eval as ce


def test_wilson_interval_brackets_the_rate_and_shrinks_with_n():
    lo, hi = ce.wilson(90, 100)
    assert lo < 0.9 < hi
    lo2, hi2 = ce.wilson(900, 1000)
    assert hi2 - lo2 < hi - lo
    assert ce.wilson(0, 0) == (0.0, 0.0)
    assert ce.wilson(0, 10)[0] == 0.0 and ce.wilson(10, 10)[1] == 1.0


def test_bootstrap_ci_contains_the_mean_and_is_deterministic():
    values = [float(i) for i in range(50)]
    lo, hi = ce.bootstrap_ci(values, n_boot=500, seed=1)
    assert lo < sum(values) / len(values) < hi
    assert ce.bootstrap_ci(values, n_boot=500, seed=1) == (lo, hi)


def test_paired_delta_joins_on_sample_id():
    base = [{"sample_id": "a", "nll_per_token": 1.0},
            {"sample_id": "b", "nll_per_token": 2.0},
            {"sample_id": "zzz", "nll_per_token": 9.0}]
    conf = [{"sample_id": "b", "nll_per_token": 1.5},
            {"sample_id": "a", "nll_per_token": 1.2}]
    out = ce.paired_delta(conf, base)
    assert out["n"] == 2
    assert out["mean_delta"] == pytest.approx((0.2 - 0.5) / 2)
    assert out["frac_improved"] == 0.5
    with pytest.raises(ValueError):
        ce.paired_delta([{"sample_id": "q", "nll_per_token": 0.0}], base)


def _stream(vocab, names):
    return [vocab[n] for n in names]


def test_ref_records_capture_k_n_legal_rel_and_edge_correctness():
    vocab = model_input.build_vocab(3)
    names = ["BOS", "KIND_ENTRY", "KIND_LINEAR", "KIND_LINEAR", "KIND_MERGE",
             "REF_1", "REF_3", "EOS"]
    tokens = _stream(vocab, names)
    # targets are tokens[1:]; REF_1 is target index 4, REF_3 is index 5
    score = {"sample_id": "s", "token_nll": [0.1] * 4 + [0.5, 1.5, 0.2],
             "ref_pos": [4, 5], "ref_k": [1, 3], "ref_correct": [1, 0]}
    recs = ce.ref_records({"s": tokens}, [score], vocab, max_k=3)
    assert [r["k"] for r in recs] == [1, 3]
    # after KIND_MERGE (4th motif): 3 earlier motifs, window 3 -> 3 legal
    assert recs[0]["n_legal"] == 3 and recs[0]["rel"] == 1
    # after REF_1: legal k in {2, 3} -> 2 legal, rel = 3 - 1
    assert recs[1]["n_legal"] == 2 and recs[1]["rel"] == 2
    assert [r["correct"] for r in recs] == [1, 0]
    assert [r["nll"] for r in recs] == [0.5, 1.5]

    acc = ce.edge_accuracy(recs)
    assert acc["overall"] == 0.5 and acc["n"] == 2
    assert acc["by_k"] == {1: {"n": 1, "acc": 1.0}, 3: {"n": 1, "acc": 0.0}}


def test_by_k_macro_micro_and_strata():
    recs = ([{"k": 1, "nll": 0.2, "n_legal": 2, "rel": 1, "correct": None}] * 30
            + [{"k": 2, "nll": 1.0, "n_legal": 2, "rel": 2, "correct": None}] * 10
            + [{"k": 2, "nll": 3.0, "n_legal": 5, "rel": 2, "correct": None}] * 5)
    table = ce.by_k(recs)
    assert table[1]["n"] == 30 and "ci95" in table[1]
    assert table[2]["n"] == 15 and table[2]["mean"] == pytest.approx(5 / 3)
    mm = ce.macro_micro(table)
    assert mm["macro"] == pytest.approx((0.2 + 5 / 3) / 2)
    assert mm["micro"] == pytest.approx((30 * 0.2 + 15 * 5 / 3) / 45)
    strata = ce.within_n_legal(recs)
    assert set(strata[2]) == {1, 2}          # both k have >= 10 records
    assert 5 not in strata or strata[5] == {}  # 5 records fall under min_count


def test_frequency_baselines_fit_on_train_refs():
    vocab = model_input.build_vocab(3)
    # every stream: three motifs then a merge with REF_1 REF_2 -> k=1 (rel 1,
    # n_legal 3) and k=2 (rel 1, n_legal 2)
    names = ["BOS", "KIND_ENTRY", "KIND_LINEAR", "KIND_LINEAR", "KIND_MERGE",
             "REF_1", "REF_2", "EOS"]
    tokens = _stream(vocab, names)
    rows = [{"sample_id": f"t{i}", "tokens": tokens} for i in range(4)]
    fb = ce.FrequencyBaselines(rows, vocab, max_k=3, alpha=0.0)
    assert fb.total == 8
    assert fb.unigram_nll(1) == pytest.approx(-math.log(0.5))
    assert fb.unigram_nll(2) == pytest.approx(-math.log(0.5))
    # conditional: given n_legal 3 the only observed rel is 1 -> cost 0
    assert fb.conditional_nll(1, 3) == pytest.approx(0.0)
    smoothed = ce.FrequencyBaselines(rows, vocab, max_k=3, alpha=1.0)
    assert smoothed.conditional_nll(2, 3) > smoothed.conditional_nll(1, 3)
    recs = ce.ref_records({"t0": tokens},
                          [{"sample_id": "t0", "token_nll": [0.0] * 7}], vocab, 3)
    tables = smoothed.tables(recs, min_count=1)
    assert set(tables["unigram_nll_by_k"]) == {1, 2}
    assert set(tables["conditional_nll_by_k"]) == {1, 2}
