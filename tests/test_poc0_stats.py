from __future__ import annotations

from cfg_reducer.linearize import decode

from poc0.stats import normalized_histogram, stats_for_skeleton, total_variation_distance


def test_stats_for_skeleton_matches_corpus_stats_shape():
    tokens = [
        "ADD_ENTRY",
        "ADD_LINEAR",
        "ptr_0",
        "ADD_LOOP",
        "ptr_1",
        "OPEN",
        "ADD_ENTRY",
        "STOP",
        "CLOSE",
        "ADD_MERGE",
        "ptr_0",
        "ptr_1",
        "STOP",
    ]

    stats = stats_for_skeleton(decode(tokens), tokens)

    assert stats == {
        "n_motifs": 5,
        "kinds": {
            "entry": 2,
            "linear": 1,
            "merge": 1,
            "loop": 1,
        },
        "n_tokens": len(tokens),
        "depth": 3,
        "max_width": 2,
        "loop_nest": 1,
    }


def test_total_variation_distance_uses_union_of_bins():
    train_hist = normalized_histogram([1, 1, 3])
    sample_hist = normalized_histogram([2, 3, 3])

    assert train_hist == {1: 2.0 / 3.0, 3: 1.0 / 3.0}
    assert sample_hist == {2: 1.0 / 3.0, 3: 2.0 / 3.0}
    assert total_variation_distance(train_hist, sample_hist) == 2.0 / 3.0
