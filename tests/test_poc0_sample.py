from __future__ import annotations

import jax
import jax.numpy as jnp

from poc0.constants import MAX_SEQ_LEN, TOKEN_TO_ID, VOCAB_SIZE
from poc0.sample import InvalidSampleReason, sample_tokens


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
