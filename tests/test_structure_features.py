"""Structure-feature extraction and the rank statistics used to
explore which features separate hard from easy samples."""

from cfg_reducer import GraphEngine, dataset, model_input
from training import analyze_features, structure_features

NESTED_LOOP = [
    ("A", "B"), ("B", "C"), ("C", "D"),
    ("D", "C"), ("D", "B"), ("B", "E"),
]
DIAMOND = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]


def _mg(edges):
    engine = GraphEngine()
    for n in sorted({n for e in edges for n in e}):
        engine.add_node(n)
    for s, d in edges:
        engine.add_edge(s, d)
    return dataset.reduce_to_metagraph(engine)


def test_features_nested_loop():
    f = structure_features.features_for(_mg(NESTED_LOOP))
    assert f["n_motifs"] == 7
    assert f["n_loops"] == 2
    assert f["n_nested_loops"] == 1
    assert f["max_depth"] == 2
    assert f["n_levels"] == 3
    assert f["top_width"] == 3
    assert f["max_width"] == 3
    assert f["n_edges"] == 4          # (0,5),(5,6) + (1,4) + (2,3)
    assert f["max_offset"] == 1
    assert f["max_scc_size"] == 3
    assert f["n_back_edges"] == 2


def test_features_diamond():
    f = structure_features.features_for(_mg(DIAMOND))
    assert f["n_loops"] == 0 and f["max_depth"] == 0
    assert f["n_merge"] == 1 and f["max_in_degree"] == 2
    assert f["max_out_degree"] == 2   # entry A -> B, C
    assert f["max_offset"] == 2


def test_token_class_nll_buckets():
    vocab = model_input.build_vocab(2)
    mg = _mg(NESTED_LOOP)
    tokens = model_input.tokenize(mg, vocab)
    token_nll = [1.0] * (len(tokens) - 1)
    out = structure_features.token_class_nll(tokens, token_nll, vocab)
    # 7 KIND, 4 REF, 4 LOOP_*, 1 EOS  (see docs/design/model_input.md)
    assert out["n_kind_tokens"] == 7
    assert out["n_ref_tokens"] == 4
    assert out["n_loop_tokens"] == 4
    assert out["n_eos_tokens"] == 1
    assert out["nll_ref_mean"] == 1.0


def test_spearman_and_auc():
    up = analyze_features.spearman([1, 2, 3, 4], [10, 20, 30, 40])
    down = analyze_features.spearman([1, 2, 3, 4], [40, 30, 20, 10])
    assert up is not None and abs(up - 1.0) < 1e-9
    assert down is not None and abs(down + 1.0) < 1e-9
    # perfectly separating feature -> AUC 1.0; anti-separating -> 0.0
    assert analyze_features.auc([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert analyze_features.auc([4, 3, 2, 1], [False, False, True, True]) == 0.0
    assert analyze_features.auc([1, 1, 1, 1], [False, False, True, True]) == 0.5


def test_analyze_ranks_separating_feature_first():
    rows = [
        {"sample_id": str(i), "seed": i, "width": i, "noise": (i * 7) % 5,
         "nll_per_token": i * 0.1, "nll": 1.0, "acc": 0.5, "n_tokens": 10}
        for i in range(20)
    ]
    results = analyze_features.analyze(rows, "nll_per_token")
    assert results[0]["feature"] == "width"
    assert results[0]["auc_hard"] == 1.0
