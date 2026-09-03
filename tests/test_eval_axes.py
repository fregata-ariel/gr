"""eval_axes: REF-by-k aggregation and the record-then-verify flags."""

from cfg_reducer import model_input
from training import eval_axes


def _base_report(**over):
    rep = {
        "run": "a", "n_val": 10,
        "primary": {"ref_nll_by_k": {1: 0.2}, "nll_by_offset_bin": [],
                    "feature_rho": {}, "val_nll_per_token": 0.7,
                    "unconstrained": None},
        "canary_flat": {"n_tokens": 0.0, "max_in_degree": 0.1},
        "canary_sign": {"n_loops": -0.3},
        "canary_token_class": {"nll_kind_mean": 0.7, "nll_loop_mean": 0.7,
                               "nll_eos_mean": 0.1},
        "invariants": {"n_entry_all_one": True},
    }
    for k, v in over.items():
        rep[k] = {**rep[k], **v} if isinstance(v, dict) else v
    return rep


def test_ref_nll_by_k_buckets_by_offset():
    vocab = model_input.build_vocab(3)
    # stream: BOS KIND_ENTRY KIND_LINEAR REF_1 KIND_MERGE REF_1 REF_2 EOS
    names = ["BOS", "KIND_ENTRY", "KIND_LINEAR", "REF_1", "KIND_MERGE",
             "REF_1", "REF_2", "EOS"]
    tokens = [vocab[n] for n in names]
    token_nll = [0.5, 0.5, 1.0, 0.5, 3.0, 5.0, 0.1]   # aligns with tokens[1:]
    out = eval_axes.ref_nll_by_k({"s": tokens}, [{"sample_id": "s", "token_nll": token_nll}],
                                 vocab, min_count=1)
    assert out == {1: 2.0, 2: 5.0}


def test_compare_flags_only_meaningful_deltas():
    before = _base_report()
    same = _base_report(canary_flat={"n_tokens": 0.05})       # within noise
    assert eval_axes.compare(before, same) == []

    drift = _base_report(canary_flat={"max_in_degree": 0.45})  # |Δ| 0.35
    flip = _base_report(canary_sign={"n_loops": 0.25})          # sign flip
    worse = _base_report(canary_token_class={"nll_kind_mean": 0.95})
    broken = _base_report(invariants={"n_entry_all_one": False})

    for rep, axis in [(drift, "canary_flat.max_in_degree"),
                      (flip, "canary_sign.n_loops"),
                      (worse, "canary_token_class.nll_kind_mean"),
                      (broken, "invariants.n_entry_all_one")]:
        flags = eval_axes.compare(before, rep)
        assert [f["axis"] for f in flags] == [axis]
        assert "要検証" in flags[0]["note"]     # never a verdict


def test_nll_by_offset_bin_is_monotone_in_offset_ranges():
    rows = [{"mean_offset": i / 10, "nll_per_token": i} for i in range(9)]
    bins = eval_axes.nll_by_offset_bin(rows, bins=3)
    assert [b["n"] for b in bins] == [3, 3, 3]
    assert bins[0]["mean_offset_range"][1] <= bins[1]["mean_offset_range"][0]


# ── sketch_stats (distribution fidelity) ─────

def test_sketch_features_and_ks():
    from cfg_reducer import GraphEngine, dataset
    from training import sketch_stats
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "C"), ("D", "B"), ("B", "E")]
    engine = GraphEngine()
    for n in sorted({n for e in edges for n in e}):
        engine.add_node(n)
    for s, d in edges:
        engine.add_edge(s, d)
    mg = dataset.reduce_to_metagraph(engine)
    vocab = model_input.build_vocab(2)
    tokens = model_input.tokenize(mg, vocab)
    f = sketch_stats.sketch_features(model_input.detokenize(tokens, vocab), len(tokens))
    assert f["n_motifs"] == 7 and f["n_loops"] == 2 and f["max_depth"] == 2
    assert f["max_width"] == 3 and f["mean_offset"] == 1.0

    assert sketch_stats.ks_distance([1, 2, 3], [1, 2, 3]) == 0.0
    assert sketch_stats.ks_distance([1, 2, 3], [4, 5, 6]) == 1.0
    # malformed streams are dropped, not counted
    rows = sketch_stats.features_of_streams([tokens, tokens[:-1]], vocab)
    assert len(rows) == 1
