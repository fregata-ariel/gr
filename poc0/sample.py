from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp

from poc0.checkpoints import restore_checkpoint
from poc0.constants import BOS_ID, MAX_SEQ_LEN, PAD_ID, VOCAB_SIZE
from poc0.tokenizer import id_to_token
from poc0.train import create_train_state

LogitsFn = Callable[[jax.Array], jax.Array]


class InvalidSampleReason(str, Enum):
    LENGTH_CAP = "length_cap"
    NEGATIVE_DEPTH = "negative_depth"
    DECODE_ERROR = "decode_error"


@dataclass(frozen=True)
class SampleResult:
    tokens: list[str]
    success: bool
    invalid_reason: str | None
    depth: int


def _masked_logits(logits: jax.Array) -> jax.Array:
    masked = jnp.asarray(logits, dtype=jnp.float32)
    if masked.shape != (VOCAB_SIZE,):
        raise ValueError(f"logits_fn must return shape ({VOCAB_SIZE},), got {masked.shape}")
    masked = masked.at[PAD_ID].set(-jnp.inf)
    masked = masked.at[BOS_ID].set(-jnp.inf)
    return masked


def sample_tokens(
    *,
    logits_fn: LogitsFn,
    temperature: float,
    rng: jax.Array,
) -> SampleResult:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    prefix_token_ids = [BOS_ID]
    emitted_tokens: list[str] = []
    depth = 0

    for _ in range(MAX_SEQ_LEN - 1):
        logits = _masked_logits(
            logits_fn(jnp.asarray(prefix_token_ids, dtype=jnp.int32))
        )
        rng, sample_rng = jax.random.split(rng)
        next_token_id = int(
            jax.random.categorical(sample_rng, logits / temperature).item()
        )
        next_token = id_to_token(next_token_id)
        prefix_token_ids.append(next_token_id)
        emitted_tokens.append(next_token)

        if next_token == "OPEN":
            depth += 1
        elif next_token == "CLOSE":
            depth -= 1
            if depth < 0:
                return SampleResult(
                    tokens=emitted_tokens,
                    success=False,
                    invalid_reason=InvalidSampleReason.NEGATIVE_DEPTH.value,
                    depth=depth,
                )
        elif next_token == "STOP" and depth == 0:
            return SampleResult(
                tokens=emitted_tokens,
                success=True,
                invalid_reason=None,
                depth=depth,
            )

    return SampleResult(
        tokens=emitted_tokens,
        success=False,
        invalid_reason=InvalidSampleReason.LENGTH_CAP.value,
        depth=depth,
    )


def make_model_logits_fn(*, apply_fn: Callable[..., jax.Array], params: object) -> LogitsFn:
    @jax.jit
    def _forward(input_ids: jax.Array) -> jax.Array:
        return apply_fn({"params": params}, input_ids, train=False)

    def logits_fn(prefix_token_ids: jax.Array) -> jax.Array:
        prefix_length = int(prefix_token_ids.shape[0])
        if prefix_length <= 0 or prefix_length > MAX_SEQ_LEN:
            raise ValueError(
                f"prefix length must be in [1, {MAX_SEQ_LEN}], got {prefix_length}"
            )

        padded = jnp.full((1, MAX_SEQ_LEN), PAD_ID, dtype=jnp.int32)
        padded = padded.at[0, :prefix_length].set(prefix_token_ids)
        logits = _forward(padded)
        return logits[0, prefix_length - 1, :]

    return logits_fn


def restore_logits_fn_from_checkpoint(
    checkpoint_dir: Path,
) -> tuple[LogitsFn, dict[str, object]]:
    state = create_train_state()
    restored_state, manifest = restore_checkpoint(Path(checkpoint_dir), state)
    return (
        make_model_logits_fn(
            apply_fn=restored_state.apply_fn,
            params=restored_state.params,
        ),
        manifest,
    )
