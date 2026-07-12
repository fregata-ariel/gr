from __future__ import annotations

import numpy as np

from cfg_reducer.linearize import decode
from poc0.constants import TOKEN_TO_ID, VOCAB, VOCAB_SIZE
from poc0.grammar import GrammarTracker


def _advance_all(tracker: GrammarTracker, tokens: list[str]) -> GrammarTracker:
    for token in tokens:
        tracker = tracker.advance_token(token)
    return tracker


def _legal_tokens(tracker: GrammarTracker) -> set[str]:
    return {VOCAB[token_id] for token_id in tracker.legal_token_ids()}


def test_initial_item_boundary_legality():
    tracker = GrammarTracker.initial()

    assert _legal_tokens(tracker) == {
        "ADD_ENTRY",
        "ADD_LINEAR",
        "ADD_MERGE",
        "ADD_LOOP",
        "STOP",
    }
    assert tracker.is_legal_token("OPEN") is False
    assert tracker.is_legal_token("CLOSE") is False
    assert tracker.is_legal_token("PAD") is False
    assert tracker.is_legal_token("BOS") is False

    for token in VOCAB:
        if token.startswith("ptr_"):
            assert tracker.is_legal_token(token) is False


def test_nonloop_parent_delimiters_match_parser():
    tracker = GrammarTracker.initial().advance_token("ADD_ENTRY")

    assert _legal_tokens(tracker) == {
        "ADD_ENTRY",
        "ADD_LINEAR",
        "ADD_MERGE",
        "ADD_LOOP",
        "STOP",
    }
    assert tracker.is_legal_token("OPEN") is False
    assert tracker.is_legal_token("CLOSE") is False
    assert tracker.is_legal_token("ptr_0") is False

    advanced = tracker.advance_token("ADD_LINEAR")
    assert advanced.is_legal_token("ptr_0") is True


def test_loop_parent_list_requires_open():
    tracker = _advance_all(
        GrammarTracker.initial(),
        ["ADD_ENTRY", "ADD_LINEAR", "ADD_MERGE", "ADD_LOOP"],
    )

    assert tracker.is_legal_token("OPEN") is True
    assert tracker.is_legal_token("ptr_0") is True
    assert tracker.is_legal_token("ptr_1") is True
    assert tracker.is_legal_token("ptr_2") is True
    assert tracker.is_legal_token("ADD_ENTRY") is False
    assert tracker.is_legal_token("ADD_LOOP") is False
    assert tracker.is_legal_token("STOP") is False
    assert tracker.is_legal_token("CLOSE") is False

    tracker = tracker.advance_token("ptr_0")
    assert tracker.is_legal_token("OPEN") is True
    assert tracker.is_legal_token("ptr_1") is True
    assert tracker.is_legal_token("ptr_2") is True
    assert tracker.is_legal_token("ptr_0") is False
    assert tracker.is_legal_token("ADD_MERGE") is False
    assert tracker.is_legal_token("STOP") is False


def test_pointer_bounds_are_current_level_ascending_unique():
    tracker = _advance_all(
        GrammarTracker.initial(),
        ["ADD_ENTRY", "ADD_LINEAR", "ADD_MERGE"],
    )
    assert tracker.is_legal_token("ptr_0") is True
    assert tracker.is_legal_token("ptr_1") is True
    assert tracker.is_legal_token("ptr_2") is False
    assert tracker.is_legal_token("ptr_31") is False

    tracker = tracker.advance_token("ptr_0")
    assert tracker.is_legal_token("ptr_0") is False
    assert tracker.is_legal_token("ptr_1") is True
    assert tracker.is_legal_token("ptr_2") is False

    tracker = tracker.advance_token("ptr_1")
    assert tracker.is_legal_token("ptr_0") is False
    assert tracker.is_legal_token("ptr_1") is False
    assert tracker.is_legal_token("ptr_2") is False

    wide_tracker = GrammarTracker.initial()
    for _ in range(33):
        wide_tracker = wide_tracker.advance_token("ADD_ENTRY")

    pointer_tokens = {
        token for token in _legal_tokens(wide_tracker) if token.startswith("ptr_")
    }
    assert pointer_tokens == {f"ptr_{index}" for index in range(32)}


def test_child_stop_requires_close_only():
    tracker = _advance_all(
        GrammarTracker.initial(),
        ["ADD_LOOP", "OPEN", "STOP"],
    )

    assert _legal_tokens(tracker) == {"CLOSE"}

    tracker = tracker.advance_token("CLOSE")
    assert _legal_tokens(tracker) == {
        "ADD_ENTRY",
        "ADD_LINEAR",
        "ADD_MERGE",
        "ADD_LOOP",
        "STOP",
    }
    assert tracker.is_legal_token("ptr_0") is False
    assert tracker.is_legal_token("CLOSE") is False


def test_child_level_pointer_counts_reset_and_parent_counts_restore():
    tracker = _advance_all(
        GrammarTracker.initial(),
        ["ADD_ENTRY", "ADD_LOOP", "OPEN"],
    )
    assert tracker.is_legal_token("ptr_0") is False

    tracker = tracker.advance_token("ADD_ENTRY")
    assert tracker.is_legal_token("ptr_0") is False

    tracker = tracker.advance_token("ADD_LINEAR")
    assert tracker.is_legal_token("ptr_0") is True
    assert tracker.is_legal_token("ptr_1") is False

    tracker = _advance_all(tracker, ["STOP", "CLOSE", "ADD_ENTRY"])
    assert tracker.is_legal_token("ptr_0") is True
    assert tracker.is_legal_token("ptr_1") is True
    assert tracker.is_legal_token("ptr_2") is False


def test_terminal_state_has_no_legal_tokens():
    tracker = GrammarTracker.initial().advance_token("STOP")

    assert tracker.is_terminal is True
    assert tracker.legal_token_ids() == ()
    assert tracker.legal_mask().sum() == 0

    for token_id in range(VOCAB_SIZE):
        assert tracker.is_legal_token_id(token_id) is False


def test_no_kind_arity_constraints():
    tracker = _advance_all(
        GrammarTracker.initial(),
        ["ADD_ENTRY", "ADD_ENTRY"],
    )

    assert tracker.is_legal_token("ptr_0") is True

    tokens = ["ADD_ENTRY", "ADD_ENTRY", "ptr_0", "STOP"]
    decode(tokens)
    assert _advance_all(GrammarTracker.initial(), tokens).is_terminal is True


def test_legal_mask_shape_and_ids():
    tracker = _advance_all(
        GrammarTracker.initial(),
        ["ADD_ENTRY", "ADD_LOOP"],
    )

    mask = tracker.legal_mask()

    assert mask.shape == (VOCAB_SIZE,)
    assert mask.dtype == np.bool_
    assert set(np.flatnonzero(mask)) == set(tracker.legal_token_ids())
