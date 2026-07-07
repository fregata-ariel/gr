from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from poc0.constants import MAX_SEQ_LEN, VOCAB_SIZE
from poc0.model import CausalTransformerLM, TransformerConfig, count_parameters


def test_model_init_logits_shape_and_param_count():
    config = TransformerConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )
    model = CausalTransformerLM(config)
    input_ids = jnp.zeros((2, MAX_SEQ_LEN), dtype=jnp.int32)

    variables = model.init(jax.random.key(0), input_ids, train=False)
    logits = cast(jax.Array, model.apply(variables, input_ids, train=False))
    param_count = count_parameters(variables["params"])

    assert logits.shape == (2, MAX_SEQ_LEN, VOCAB_SIZE)
    assert param_count == 3_190_528
    assert 2_000_000 <= param_count <= 4_000_000


def test_causal_mask_blocks_future_positions():
    config = TransformerConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )
    model = CausalTransformerLM(config)

    input_ids_a = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
    input_ids_b = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
    input_ids_a[0, 20] = 3
    input_ids_b[0, 20] = 4

    variables = model.init(jax.random.key(1), jnp.asarray(input_ids_a), train=False)
    logits_a = cast(
        jax.Array,
        model.apply(variables, jnp.asarray(input_ids_a), train=False),
    )
    logits_b = cast(
        jax.Array,
        model.apply(variables, jnp.asarray(input_ids_b), train=False),
    )

    np.testing.assert_allclose(
        np.asarray(logits_a[:, :20, :]),
        np.asarray(logits_b[:, :20, :]),
        rtol=0.0,
        atol=1e-6,
    )
