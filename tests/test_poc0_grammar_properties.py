from __future__ import annotations

import random
import re

import jax
import jax.numpy as jnp

from cfg_reducer.linearize import decode
from poc0.constants import TOKEN_TO_ID, VOCAB, VOCAB_SIZE
from poc0.grammar import GrammarTracker
from poc0.sample import sample_tokens
from poc0.tokenizer import id_to_token

A1_SAMPLE_COUNT = 64
A2_ROLLOUT_COUNT = 48
A2_MIN_PREFIX_STATES = 300
_ERROR_POS_RE = re.compile(r" at position (\d+)$")


def _parse_error_position(exc: ValueError) -> int:
    match = _ERROR_POS_RE.search(str(exc))
    if match is None:
        raise AssertionError(f"missing parser position in error: {exc}") from exc
    return int(match.group(1))


def _oracle_legal_token_ids(prefix_tokens: list[str]) -> tuple[int, ...]:
    legal_ids: list[int] = []
    candidate_pos = len(prefix_tokens)

    for token_id, candidate in enumerate(VOCAB):
        try:
            decode([*prefix_tokens, candidate])
        except ValueError as exc:
            error_pos = _parse_error_position(exc)
            if error_pos > candidate_pos:
                legal_ids.append(token_id)
            elif error_pos < candidate_pos:
                raise AssertionError(
                    f"prefix became invalid before candidate: prefix={prefix_tokens}, "
                    f"candidate={candidate}, error={exc}"
                ) from exc
        else:
            legal_ids.append(token_id)

    return tuple(legal_ids)


def _depth(tokens: list[str]) -> int:
    depth = 0
    for token in tokens:
        if token == "OPEN":
            depth += 1
        elif token == "CLOSE":
            depth -= 1
    return depth


def _adversarial_logits_fn(salt: int):
    def logits_fn(prefix_ids: jax.Array) -> jax.Array:
        prefix_tokens = [id_to_token(int(token_id)) for token_id in prefix_ids[1:]]
        legal_ids = set(_oracle_legal_token_ids(prefix_tokens))
        depth = _depth(prefix_tokens)
        rng = random.Random(f"{salt}:{','.join(prefix_tokens)}")
        logits: list[float] = []

        for token_id, token in enumerate(VOCAB):
            noise = rng.random()
            if token_id in legal_ids:
                score = noise
                if token == "STOP":
                    score += 1.0 if depth == 0 else 0.25
                elif token == "CLOSE":
                    score += 0.75
                elif token.startswith("ptr_"):
                    score += 0.5
            else:
                score = 10.0 + noise
            logits.append(score)

        return jnp.asarray(logits, dtype=jnp.float32)

    return logits_fn


def test_mask_implies_decode_accepts():
    total_prefix_states = 0

    for seed in range(A1_SAMPLE_COUNT):
        result = sample_tokens(
            logits_fn=_adversarial_logits_fn(seed),
            temperature=1.0,
            rng=jax.random.key(seed),
            constrained=True,
        )
        total_prefix_states += len(result.tokens)

        if result.success:
            decode(result.tokens)
        else:
            assert result.invalid_reason == "length_cap"

    assert total_prefix_states >= A1_SAMPLE_COUNT


def _weighted_choice(rng: random.Random, legal_ids: list[int]) -> int:
    weights: list[int] = []
    for token_id in legal_ids:
        token = VOCAB[token_id]
        if token == "ADD_LOOP":
            weights.append(6)
        elif token.startswith("ADD_"):
            weights.append(5)
        elif token == "OPEN":
            weights.append(4)
        elif token == "CLOSE":
            weights.append(3)
        elif token.startswith("ptr_"):
            weights.append(3)
        else:
            weights.append(1)
    return rng.choices(legal_ids, weights=weights, k=1)[0]


def test_mask_never_overblocks_decode_oracle():
    rng = random.Random(20260713)
    prefix_states_checked = 0

    for _ in range(A2_ROLLOUT_COUNT):
        tracker = GrammarTracker.initial()
        prefix_tokens: list[str] = []
        depth = 0

        for step in range(16):
            oracle_ids = set(_oracle_legal_token_ids(prefix_tokens))
            tracker_ids = set(tracker.legal_token_ids())

            assert tracker_ids == oracle_ids, prefix_tokens
            prefix_states_checked += 1

            legal_ids = sorted(oracle_ids)
            if (
                depth == 0
                and TOKEN_TO_ID["STOP"] in legal_ids
                and len(legal_ids) > 1
                and step < 5
            ):
                legal_ids = [
                    token_id for token_id in legal_ids if token_id != TOKEN_TO_ID["STOP"]
                ]

            next_token_id = _weighted_choice(rng, legal_ids)
            next_token = VOCAB[next_token_id]
            tracker = tracker.advance_token_id(next_token_id)
            prefix_tokens.append(next_token)

            if next_token == "OPEN":
                depth += 1
            elif next_token == "CLOSE":
                depth -= 1
            elif next_token == "STOP" and depth == 0:
                assert tracker.is_terminal is True
                break

    assert prefix_states_checked >= A2_MIN_PREFIX_STATES
