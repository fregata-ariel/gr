from __future__ import annotations

import jax
import numpy as np

from poc0.constants import (
    END_LEARNING_RATE,
    INIT_SEED,
    MAX_SEQ_LEN,
    PEAK_LEARNING_RATE,
    TRAIN_STEPS,
    VOCAB_SIZE,
    WARMUP_STEPS,
)
from poc0.data import Batch
from poc0.tokenizer import record_to_example
from poc0.train import (
    create_learning_rate_schedule,
    create_train_state,
    masked_cross_entropy_loss,
    train_step,
)


def _tiny_batch() -> Batch:
    example = record_to_example(["ADD_ENTRY", "ptr_0", "STOP"])
    input_ids = np.stack([example.input_ids, example.input_ids], axis=0).astype(
        np.int32, copy=False
    )
    target_ids = np.stack([example.target_ids, example.target_ids], axis=0).astype(
        np.int32, copy=False
    )
    loss_mask = np.stack([example.loss_mask, example.loss_mask], axis=0).astype(
        np.bool_, copy=False
    )
    return Batch(
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=loss_mask,
    )


def test_masked_loss_ignores_padding():
    batch = _tiny_batch()
    logits = np.zeros((2, MAX_SEQ_LEN, VOCAB_SIZE), dtype=np.float32)

    changed_targets = batch.target_ids.copy()
    changed_targets[:, -1] = (changed_targets[:, -1] + 7) % VOCAB_SIZE

    baseline_loss = masked_cross_entropy_loss(
        logits=logits,
        target_ids=batch.target_ids,
        loss_mask=batch.loss_mask,
    )
    changed_loss = masked_cross_entropy_loss(
        logits=logits,
        target_ids=changed_targets,
        loss_mask=batch.loss_mask,
    )

    np.testing.assert_allclose(
        np.asarray(baseline_loss),
        np.asarray(changed_loss),
        rtol=0.0,
        atol=0.0,
    )


def test_train_step_is_deterministic_for_fixed_seed():
    batch = _tiny_batch()
    schedule = create_learning_rate_schedule(
        peak_lr=PEAK_LEARNING_RATE,
        end_lr=END_LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        total_steps=TRAIN_STEPS,
    )

    state_a = create_train_state(
        seed=INIT_SEED,
        learning_rate_schedule=schedule,
    )
    state_b = create_train_state(
        seed=INIT_SEED,
        learning_rate_schedule=schedule,
    )

    next_state_a, metrics_a = train_step(state_a, batch)
    next_state_b, metrics_b = train_step(state_b, batch)

    np.testing.assert_allclose(
        np.asarray(metrics_a["loss"]),
        np.asarray(metrics_b["loss"]),
        rtol=0.0,
        atol=0.0,
    )

    leaves_a = jax.tree_util.tree_leaves(next_state_a.params)
    leaves_b = jax.tree_util.tree_leaves(next_state_b.params)
    assert len(leaves_a) == len(leaves_b)
    for leaf_a, leaf_b in zip(leaves_a, leaves_b, strict=True):
        np.testing.assert_allclose(
            np.asarray(leaf_a),
            np.asarray(leaf_b),
            rtol=0.0,
            atol=0.0,
        )
