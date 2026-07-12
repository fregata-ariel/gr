from __future__ import annotations

import jax
import jax.numpy as jnp

from cfg_reducer.linearize import decode
from poc0.constants import MAX_SEQ_LEN, TOKEN_TO_ID, VOCAB_SIZE
from poc0.sample import InvalidSampleReason, SampleResult, sample_tokens


def _scripted_logits_fn(token_ids: list[int]):
    def logits_fn(prefix_ids: jax.Array) -> jax.Array:
        emitted = prefix_ids[1:]
        index = int(emitted.shape[0])
        next_token_id = token_ids[min(index, len(token_ids) - 1)]
        logits = jnp.full((VOCAB_SIZE,), -1.0e9, dtype=jnp.float32)
        return logits.at[next_token_id].set(0.0)

    return logits_fn


def test_depth_zero_stop_terminates_successfully():
    result = sample_tokens(
        logits_fn=_scripted_logits_fn([TOKEN_TO_ID["STOP"]]),
        temperature=1.0,
        rng=jax.random.key(0),
    )

    assert result.success is True
    assert result.invalid_reason is None
    assert result.tokens == ["STOP"]
    assert result.depth == 0


def test_stop_inside_open_level_does_not_terminate():
    result = sample_tokens(
        logits_fn=_scripted_logits_fn(
            [
                TOKEN_TO_ID["ADD_LOOP"],
                TOKEN_TO_ID["OPEN"],
                TOKEN_TO_ID["STOP"],
                TOKEN_TO_ID["CLOSE"],
                TOKEN_TO_ID["STOP"],
            ]
        ),
        temperature=1.0,
        rng=jax.random.key(1),
    )

    assert result.success is True
    assert result.invalid_reason is None
    assert result.tokens == ["ADD_LOOP", "OPEN", "STOP", "CLOSE", "STOP"]
    assert result.depth == 0


def test_length_cap_without_top_level_stop_is_invalid():
    result = sample_tokens(
        logits_fn=_scripted_logits_fn([TOKEN_TO_ID["OPEN"]] * MAX_SEQ_LEN),
        temperature=1.0,
        rng=jax.random.key(2),
    )

    assert result.success is False
    assert result.invalid_reason == InvalidSampleReason.LENGTH_CAP
    assert len(result.tokens) == MAX_SEQ_LEN - 1
    assert result.tokens == ["OPEN"] * (MAX_SEQ_LEN - 1)


def test_negative_depth_returns_invalid_reason():
    result = sample_tokens(
        logits_fn=_scripted_logits_fn([TOKEN_TO_ID["CLOSE"]]),
        temperature=1.0,
        rng=jax.random.key(3),
    )

    assert result.success is False
    assert result.invalid_reason == InvalidSampleReason.NEGATIVE_DEPTH
    assert result.tokens == ["CLOSE"]
    assert result.depth == -1


def test_unconstrained_sampling_regression_matches_poc0_snapshot():
    cases = [
        (
            "top_level_stop",
            [TOKEN_TO_ID["STOP"]],
            0,
            SampleResult(
                tokens=["STOP"],
                success=True,
                invalid_reason=None,
                depth=0,
            ),
        ),
        (
            "child_stop_then_close",
            [
                TOKEN_TO_ID["ADD_LOOP"],
                TOKEN_TO_ID["OPEN"],
                TOKEN_TO_ID["STOP"],
                TOKEN_TO_ID["CLOSE"],
                TOKEN_TO_ID["STOP"],
            ],
            1,
            SampleResult(
                tokens=["ADD_LOOP", "OPEN", "STOP", "CLOSE", "STOP"],
                success=True,
                invalid_reason=None,
                depth=0,
            ),
        ),
        (
            "length_cap_opens",
            [TOKEN_TO_ID["OPEN"]] * MAX_SEQ_LEN,
            2,
            SampleResult(
                tokens=["OPEN"] * (MAX_SEQ_LEN - 1),
                success=False,
                invalid_reason=InvalidSampleReason.LENGTH_CAP.value,
                depth=79,
            ),
        ),
        (
            "negative_depth_close",
            [TOKEN_TO_ID["CLOSE"]],
            3,
            SampleResult(
                tokens=["CLOSE"],
                success=False,
                invalid_reason=InvalidSampleReason.NEGATIVE_DEPTH.value,
                depth=-1,
            ),
        ),
    ]

    for _, token_ids, seed, expected in cases:
        result = sample_tokens(
            logits_fn=_scripted_logits_fn(token_ids),
            temperature=1.0,
            rng=jax.random.key(seed),
        )

        assert result == expected


def test_constrained_mask_blocks_negative_depth_and_decode_errors():
    def logits_fn(prefix_ids: jax.Array) -> jax.Array:
        emitted = [int(token_id) for token_id in prefix_ids[1:]]
        logits = jnp.full((VOCAB_SIZE,), -1.0e9, dtype=jnp.float32)

        if emitted == []:
            preferred = TOKEN_TO_ID["CLOSE"]
            fallback = TOKEN_TO_ID["ADD_LOOP"]
        elif emitted == [TOKEN_TO_ID["ADD_LOOP"]]:
            preferred = TOKEN_TO_ID["STOP"]
            fallback = TOKEN_TO_ID["OPEN"]
        elif emitted == [TOKEN_TO_ID["ADD_LOOP"], TOKEN_TO_ID["OPEN"]]:
            preferred = TOKEN_TO_ID["ptr_0"]
            fallback = TOKEN_TO_ID["STOP"]
        elif emitted == [
            TOKEN_TO_ID["ADD_LOOP"],
            TOKEN_TO_ID["OPEN"],
            TOKEN_TO_ID["STOP"],
        ]:
            preferred = TOKEN_TO_ID["ADD_ENTRY"]
            fallback = TOKEN_TO_ID["CLOSE"]
        else:
            preferred = TOKEN_TO_ID["ptr_0"]
            fallback = TOKEN_TO_ID["STOP"]

        logits = logits.at[preferred].set(2.0)
        return logits.at[fallback].set(1.0)

    result = sample_tokens(
        logits_fn=logits_fn,
        temperature=1.0,
        rng=jax.random.key(7),
        constrained=True,
    )

    assert result == SampleResult(
        tokens=["ADD_LOOP", "OPEN", "STOP", "CLOSE", "STOP"],
        success=True,
        invalid_reason=None,
        depth=0,
    )
    decode(result.tokens)
