"""Training-pipeline tests (torch-free): prepare_tokens end-to-end,
eval metrics, and self-containment guards for the Colab-only trainer."""

import ast
import json
import random
from pathlib import Path

from cfg_reducer import GraphEngine, dataset, model_input, store
from cfg_reducer.generate import generate_cfg
from training import data_utils, eval_samples, grammar_mask, prepare_tokens

CONFIG = {"num_nodes": 8, "edge_prob": 0.3}
TRAIN_AR = Path(__file__).parent.parent / "training" / "train_ar.py"
GRAMMAR_MASK = Path(__file__).parent.parent / "training" / "grammar_mask.py"


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

    assert report["violations"] == {}

    # Dropping EOS breaks the grammar — and is classified as a clean
    # truncation, not structural damage
    corrupted = [s[:-1] for s in val]
    broken = eval_samples.evaluate(corrupted, vocab, train)
    assert broken["well_formed"] == 0
    assert broken["well_formed_rate"] == 0.0
    assert broken["violations"] == {
        "no_eos:would_close_cleanly": len(corrupted),
    }


def test_violation_classification():
    vocab = model_input.build_vocab(2)

    def ids(names):
        return [vocab[n] for n in names]

    cases = {
        "ref_out_of_range":
            ids(["BOS", "KIND_ENTRY", "KIND_LINEAR", "REF_2", "EOS"]),
        "loop_missing_start":
            ids(["BOS", "KIND_LOOP", "EOS"]),
        "unclosed_loop":
            ids(["BOS", "KIND_LOOP", "LOOP_START", "EOS"]),
        "no_eos:would_close_cleanly":
            ids(["BOS", "KIND_ENTRY", "KIND_LINEAR", "REF_1"]),
        "no_eos:unclosed_loop":
            ids(["BOS", "KIND_LOOP", "LOOP_START", "KIND_ENTRY"]),
        "ok":
            ids(["BOS", "KIND_ENTRY", "KIND_LINEAR", "REF_1", "EOS"]),
    }
    for expected, stream in cases.items():
        assert eval_samples.classify_stream(stream, vocab) == expected


def test_jsonl_roundtrip(tmp_path):
    rows = [{"sample_id": "a", "tokens": [1, 2]}, {"sample_id": "b", "tokens": []}]
    path = tmp_path / "rows.jsonl"
    data_utils.write_jsonl(path, rows)
    assert data_utils.read_jsonl(path) == rows


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))   # syntax guard
    return {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }


def test_colab_files_do_not_depend_on_cfg_reducer():
    # train_ar.py + grammar_mask.py are the only files uploaded to the VM
    assert "cfg_reducer" not in _imports_of(TRAIN_AR)
    assert "cfg_reducer" not in _imports_of(GRAMMAR_MASK)
    # grammar_mask must stay torch-free (local tests import it)
    assert "torch" not in _imports_of(GRAMMAR_MASK)


# ── grammar-constrained decoding ─────────────

def _mg_from_seed(seed: int):
    engine = GraphEngine()
    generate_cfg(engine, num_nodes=10, edge_prob=0.3, seed=seed)
    return dataset.reduce_to_metagraph(engine)


def test_grammar_state_accepts_every_real_stream():
    for seed in range(4):
        mg = _mg_from_seed(seed)
        vocab = model_input.build_vocab(
            max(1, model_input.max_offset_needed(mg))
        )
        tokens = model_input.tokenize(mg, vocab)

        state = grammar_mask.GrammarState(vocab)
        for token in tokens[1:]:
            assert token in state.allowed_ids()
            state.push(token)
        assert state.done


def _random_walk(vocab, rng, max_len):
    state = grammar_mask.GrammarState(vocab)
    ids = [vocab["BOS"]]
    while not state.done:
        # +2 guard band, same as train_ar.sample_stream
        if (max_len - len(ids)) <= state.min_close_cost() + 2:
            next_id = state.forced_close_id()
        else:
            next_id = rng.choice(state.allowed_ids())
        ids.append(next_id)
        state.push(next_id)
    return ids


def test_random_walks_over_allowed_ids_are_well_formed():
    vocab = model_input.build_vocab(6)
    rng = random.Random(0)
    for max_len in (12, 40):     # short budget exercises forced closing
        for _ in range(40):
            ids = _random_walk(vocab, rng, max_len)
            assert len(ids) <= max_len
            model_input.detokenize(ids, vocab)   # must not raise


# ── 案 1: structural positions ───────────────

def test_structural_positions_nested_loop():
    vocab = model_input.build_vocab(1)
    names = ["BOS", "KIND_ENTRY", "KIND_LOOP", "REF_1", "LOOP_START",
             "KIND_LINEAR", "KIND_LOOP", "REF_1", "LOOP_START",
             "KIND_LINEAR", "KIND_LINEAR", "REF_1", "LOOP_END", "LOOP_END",
             "KIND_LINEAR", "REF_1", "EOS"]
    ids = [vocab[n] for n in names]
    depth, count = grammar_mask.structural_positions(ids, vocab)
    assert depth == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 0, 0, 0, 0]
    assert count == [0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 2, 2, 3, 3, 3]


def test_structural_positions_match_grammar_state_on_real_streams():
    for seed in range(3):
        mg = _mg_from_seed(seed)
        vocab = model_input.build_vocab(
            max(1, model_input.max_offset_needed(mg)))
        tokens = model_input.tokenize(mg, vocab)
        depth, count = grammar_mask.structural_positions(tokens, vocab)

        state = grammar_mask.GrammarState(vocab)
        assert (depth[0], count[0]) == (state.depth, state.level_count)
        for i, tok in enumerate(tokens[1:], start=1):
            state.push(tok)
            assert (depth[i], count[i]) == (state.depth, state.level_count)


def test_structural_positions_tolerate_malformed_prefixes():
    vocab = model_input.build_vocab(1)
    # stray LOOP_END at depth 0, REF before any motif, PAD inside
    names = ["BOS", "LOOP_END", "REF_1", "PAD", "KIND_ENTRY", "LOOP_START"]
    ids = [vocab[n] for n in names]
    depth, count = grammar_mask.structural_positions(ids, vocab)
    assert depth == [0, 0, 0, 0, 0, 1]
    assert count == [0, 0, 0, 0, 1, 0]


# ── 案 2: pointer context ────────────────────

def test_pointer_context_nested_loop():
    vocab = model_input.build_vocab(1)
    names = ["BOS", "KIND_ENTRY", "KIND_LOOP", "REF_1", "LOOP_START",
             "KIND_LINEAR", "KIND_LOOP", "REF_1", "LOOP_START",
             "KIND_LINEAR", "KIND_LINEAR", "REF_1", "LOOP_END", "LOOP_END",
             "KIND_LINEAR", "REF_1", "EOS"]
    ids = [vocab[n] for n in names]
    ctx = grammar_mask.pointer_context(ids, vocab)
    assert ctx["level_id"] == [-1, -1, -1, -1, 4, 4, 4, 4, 8, 8, 8, 8, 4, -1, -1, -1, -1]
    # REF_1 at index 3 (after KIND_LOOP at 2) points to KIND_ENTRY at 1
    assert ctx["ref_target"][2] == 1
    # inner: REF_1 at 7 -> KIND_LINEAR at 5;  REF_1 at 11 -> KIND_LINEAR at 9
    assert ctx["ref_target"][6] == 5 and ctx["ref_target"][10] == 9
    # top-level REF_1 at 15 (after KIND_LINEAR at 14) -> the loop motif at 2
    assert ctx["ref_target"][14] == 2
    assert ctx["klast"][3] == 1 and ctx["klast"][4] == 0
    assert [i for i, k in enumerate(ctx["is_kind"]) if k] == [1, 2, 5, 6, 9, 10, 14]


def test_pointer_targets_consistent_on_real_streams():
    vocab = model_input.build_vocab(12)
    for seed in range(4):
        mg = _mg_from_seed(seed)
        tokens = model_input.tokenize(mg, vocab)
        ctx = grammar_mask.pointer_context(tokens, vocab)
        names = {i: t for t, i in vocab.items()}
        for t in range(len(tokens) - 1):
            name = names[tokens[t + 1]]
            if not name.startswith("REF_"):
                assert ctx["ref_target"][t] == -1
                continue
            j = ctx["ref_target"][t]
            assert j >= 0 and j < t and ctx["is_kind"][j]
            assert ctx["level_id"][j] == ctx["level_id"][t]
            k = int(name[4:])
            assert (ctx["lpos"][j] - 1) == (ctx["lpos"][t] - 1) - k
            assert k > ctx["klast"][t]          # strictly increasing refs


# ── B: legal offsets for the masked vocabulary baseline ──

def test_legal_offsets_contain_every_real_ref_and_match_grammar_in_ref_state():
    for seed in range(4):
        mg = _mg_from_seed(seed)
        max_k = max(1, model_input.max_offset_needed(mg))
        vocab = model_input.build_vocab(max_k)
        tokens = model_input.tokenize(mg, vocab)
        legal = grammar_mask.legal_offsets(tokens, vocab, max_k)
        names = {i: t for t, i in vocab.items()}

        state = grammar_mask.GrammarState(vocab)
        for t in range(len(tokens) - 1):
            nxt = names[tokens[t + 1]]
            if nxt.startswith("REF_"):
                assert int(nxt[4:]) in legal[t]
            # the grammar's own REF set (only non-empty right after KIND/REF)
            grammar_ks = sorted(int(names[i][4:]) for i in state.allowed_ids()
                                if names[i].startswith("REF_"))
            prev = names[tokens[t]]
            if prev.startswith(("KIND_", "REF_")):
                assert legal[t] == grammar_ks
            else:
                assert grammar_ks == []           # grammar forbids REF here
            state.push(tokens[t + 1])


def test_legal_offsets_respect_window_and_monotone_refs():
    vocab = model_input.build_vocab(3)
    names = ["BOS", "KIND_ENTRY", "KIND_LINEAR", "KIND_LINEAR", "KIND_LINEAR",
             "KIND_MERGE", "REF_1", "REF_2"]
    ids = [vocab[n] for n in names]
    legal = grammar_mask.legal_offsets(ids, vocab, max_k=3)
    assert legal[0] == [] and legal[1] == []          # BOS, first motif
    assert legal[2] == [1]
    assert legal[4] == [1, 2, 3]                      # 4 earlier motifs, window 3
    assert legal[5] == [1, 2, 3]                      # after KIND_MERGE (5th motif)
    assert legal[6] == [2, 3]                         # after REF_1
    assert legal[7] == [3]                            # after REF_2


def test_prepare_tokens_window_from_train_only(tmp_path):
    ds = tmp_path / "ds"
    dataset.build_dataset(
        ds, {"train": (0, 4), "val": (4, 14), "test": (14, 24)},
        {"num_nodes": 10, "edge_prob": 0.3}, version="test",
    )
    manifest = json.loads((ds / "manifest.json").read_text())

    def needed(split, sid):
        return model_input.max_offset_needed(
            store.load_sample(ds / split / f"{sid}.json"))

    train_window = max(needed("train", e["sample_id"])
                       for e in manifest["splits"]["train"]["samples"])

    out = tmp_path / "tokens"
    meta = prepare_tokens.prepare(ds, out, window_from="train")
    assert meta["max_offset"] == max(1, train_window)
    assert meta["window_from"] == "train"

    for split in ("val", "test"):
        entries = manifest["splits"][split]["samples"]
        expect_excluded = sorted(e["seed"] for e in entries
                                 if needed(split, e["sample_id"]) > meta["max_offset"])
        got = meta["excluded_over_window"].get(split, {"count": 0, "seeds": []})
        assert sorted(got["seeds"]) == expect_excluded
        assert got["count"] == len(expect_excluded)
        rows = data_utils.read_jsonl(out / f"{split}.jsonl")
        assert len(rows) == len(entries) - len(expect_excluded)
        assert meta["splits"][split] == len(rows)

    # default (no window_from) keeps the global window and excludes nothing
    meta_all = prepare_tokens.prepare(ds, tmp_path / "tokens_all")
    assert "excluded_over_window" not in meta_all
    assert meta_all["max_offset"] >= meta["max_offset"]
    assert sum(meta_all["splits"].values()) == sum(
        len(v["samples"]) for v in manifest["splits"].values())
