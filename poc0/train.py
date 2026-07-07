from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from poc0.checkpoints import restore_checkpoint, save_checkpoint
from poc0.constants import (
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPS,
    BATCH_SIZE,
    CHECKPOINT_EVERY_STEPS,
    CHECKPOINT_KEEP_LAST,
    END_LEARNING_RATE,
    EVAL_EVERY_STEPS,
    GLOBAL_GRAD_CLIP_NORM,
    INIT_SEED,
    MAX_SEQ_LEN,
    PEAK_LEARNING_RATE,
    TRAIN_SHUFFLE_SEED,
    TRAIN_STEPS,
    VOCAB_SIZE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from poc0.data import Batch, BatchSource, InMemoryTokenDataset
from poc0.model import CausalTransformerLM, TransformerConfig


Array = jax.Array
ScheduleValue = (
    jax.Array
    | np.ndarray
    | np.bool_
    | np.number
    | bool
    | int
    | float
    | complex
)
LearningRateSchedule = Callable[[ScheduleValue], ScheduleValue]


def masked_cross_entropy_loss(
    *,
    logits: Array | np.ndarray,
    target_ids: Array | np.ndarray,
    loss_mask: Array | np.ndarray,
) -> Array:
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        jnp.asarray(logits, dtype=jnp.float32),
        jnp.asarray(target_ids, dtype=jnp.int32),
    )
    weights = jnp.asarray(loss_mask, dtype=jnp.float32)
    total_weight = jnp.maximum(jnp.sum(weights), 1.0)
    return jnp.sum(per_token_loss * weights) / total_weight


def _masked_loss_totals(
    *,
    logits: Array,
    target_ids: Array,
    loss_mask: Array,
) -> tuple[Array, Array]:
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits,
        target_ids,
    )
    weights = loss_mask.astype(jnp.float32)
    return jnp.sum(per_token_loss * weights), jnp.sum(weights)


def create_learning_rate_schedule(
    *,
    peak_lr: float,
    end_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> LearningRateSchedule:
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if warmup_steps > total_steps:
        raise ValueError(
            f"warmup_steps must be <= total_steps, got {warmup_steps} > {total_steps}"
        )

    decay_span = max(total_steps - warmup_steps, 1)

    def schedule(step: ScheduleValue) -> Array:
        step_array = jnp.asarray(step, dtype=jnp.float32)
        if warmup_steps == 0:
            warmup_value = jnp.asarray(peak_lr, dtype=jnp.float32)
        else:
            warmup_progress = jnp.clip(step_array / warmup_steps, 0.0, 1.0)
            warmup_value = jnp.asarray(peak_lr, dtype=jnp.float32) * warmup_progress

        decay_progress = jnp.clip(
            (step_array - warmup_steps) / decay_span,
            0.0,
            1.0,
        )
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_progress))
        decay_value = jnp.asarray(end_lr, dtype=jnp.float32) + (
            jnp.asarray(peak_lr, dtype=jnp.float32) - jnp.asarray(end_lr, dtype=jnp.float32)
        ) * cosine
        return jnp.where(step_array < warmup_steps, warmup_value, decay_value)

    return schedule


def create_optimizer(
    *,
    learning_rate_schedule: LearningRateSchedule,
    weight_decay: float = WEIGHT_DECAY,
) -> optax.GradientTransformationExtraArgs:
    return optax.chain(
        optax.clip_by_global_norm(GLOBAL_GRAD_CLIP_NORM),
        optax.adamw(
            learning_rate=learning_rate_schedule,
            b1=ADAM_BETA1,
            b2=ADAM_BETA2,
            eps=ADAM_EPS,
            weight_decay=weight_decay,
        ),
    )


def create_train_state(
    *,
    seed: int = INIT_SEED,
    learning_rate_schedule: LearningRateSchedule | None = None,
    config: TransformerConfig | None = None,
    weight_decay: float = WEIGHT_DECAY,
) -> train_state.TrainState:
    config = config or TransformerConfig()
    learning_rate_schedule = learning_rate_schedule or create_learning_rate_schedule(
        peak_lr=PEAK_LEARNING_RATE,
        end_lr=END_LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        total_steps=TRAIN_STEPS,
    )

    model = CausalTransformerLM(config=config)
    dummy_input_ids = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)
    variables = model.init(jax.random.key(seed), dummy_input_ids, train=False)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=create_optimizer(
            learning_rate_schedule=learning_rate_schedule,
            weight_decay=weight_decay,
        ),
    )


def _batch_to_device(batch: Batch) -> dict[str, Array]:
    return {
        "input_ids": jnp.asarray(batch.input_ids, dtype=jnp.int32),
        "target_ids": jnp.asarray(batch.target_ids, dtype=jnp.int32),
        "loss_mask": jnp.asarray(batch.loss_mask, dtype=jnp.bool_),
    }


@jax.jit
def _train_step_jit(
    state: train_state.TrainState,
    batch: Mapping[str, Array],
) -> tuple[train_state.TrainState, dict[str, Array]]:
    def loss_fn(params: Any) -> Array:
        logits = state.apply_fn({"params": params}, batch["input_ids"], train=True)
        return masked_cross_entropy_loss(
            logits=logits,
            target_ids=batch["target_ids"],
            loss_mask=batch["loss_mask"],
        )

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grad_norm = optax.tree.norm(grads)
    next_state = state.apply_gradients(grads=grads)
    metrics = {
        "loss": loss,
        "grad_norm": grad_norm,
    }
    return next_state, metrics


def train_step(
    state: train_state.TrainState,
    batch: Batch,
) -> tuple[train_state.TrainState, dict[str, Array]]:
    return _train_step_jit(state, _batch_to_device(batch))


@jax.jit
def _eval_batch_loss_jit(
    state: train_state.TrainState,
    batch: Mapping[str, Array],
) -> tuple[Array, Array]:
    logits = state.apply_fn({"params": state.params}, batch["input_ids"], train=False)
    return _masked_loss_totals(
        logits=logits,
        target_ids=batch["target_ids"],
        loss_mask=batch["loss_mask"],
    )


def evaluate_loss(
    state: train_state.TrainState,
    batch_iter: Iterator[Batch],
) -> float:
    total_loss = 0.0
    total_weight = 0.0
    for batch in batch_iter:
        batch_loss, batch_weight = _eval_batch_loss_jit(
            state,
            _batch_to_device(batch),
        )
        total_loss += float(batch_loss)
        total_weight += float(batch_weight)

    if total_weight == 0.0:
        raise ValueError("Validation dataset produced zero active targets")
    return total_loss / total_weight


def _load_batch_sources(
    *,
    data_dir: Path | None,
    jsonl_path: Path | None,
) -> tuple[BatchSource, BatchSource, str]:
    if data_dir is not None:
        manifest_path = data_dir / "manifest.json"
        if manifest_path.exists():
            from poc0.array_record_data import GrainArrayRecordDataset

            train_source = GrainArrayRecordDataset(
                manifest_path=manifest_path,
                split="train",
            )
            val_source = GrainArrayRecordDataset(
                manifest_path=manifest_path,
                split="val",
            )
            return train_source, val_source, "array_record"

    if jsonl_path is None:
        raise ValueError(
            "Provide --jsonl for local fallback, or point --data-dir at a directory containing manifest.json"
        )

    train_source = InMemoryTokenDataset(jsonl_path=jsonl_path, split="train")
    val_source = InMemoryTokenDataset(jsonl_path=jsonl_path, split="val")
    return train_source, val_source, "jsonl"


def _checkpoint_hyperparams(args: argparse.Namespace) -> dict[str, object]:
    return {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "peak_lr": args.peak_lr,
        "end_lr": args.end_lr,
        "weight_decay": args.weight_decay,
        "eval_every": args.eval_every,
        "ckpt_every": args.ckpt_every,
        "seed": args.seed,
        "train_shuffle_seed": TRAIN_SHUFFLE_SEED,
        "max_seq_len": MAX_SEQ_LEN,
    }


def _prune_step_checkpoints(checkpoints_dir: Path, *, keep_last: int) -> None:
    step_dirs = sorted(
        path
        for path in checkpoints_dir.iterdir()
        if path.is_dir() and path.name.startswith("step_")
    )
    to_delete = step_dirs[:-keep_last]
    for path in to_delete:
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                child.rmdir()
        path.rmdir()


def _write_run_config(workdir: Path, config: Mapping[str, object]) -> None:
    (workdir / "train_config.json").write_text(
        json.dumps(dict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _to_python_float(value: object) -> float:
    return float(np.asarray(value, dtype=np.float32).item())


def run_training(args: argparse.Namespace) -> Path:
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = workdir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_source, val_source, dataset_kind = _load_batch_sources(
        data_dir=args.data_dir,
        jsonl_path=args.jsonl,
    )
    if train_source.info.n_records < args.batch_size:
        raise ValueError(
            f"Training split has {train_source.info.n_records} records, which is smaller than batch_size={args.batch_size} with drop_remainder=True"
        )

    learning_rate_schedule = create_learning_rate_schedule(
        peak_lr=args.peak_lr,
        end_lr=args.end_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
    )
    state = create_train_state(
        seed=args.seed,
        learning_rate_schedule=learning_rate_schedule,
        weight_decay=args.weight_decay,
    )

    run_config = {
        **_checkpoint_hyperparams(args),
        "dataset_kind": dataset_kind,
        "train_records": train_source.info.n_records,
        "val_records": val_source.info.n_records,
        "vocab_size": VOCAB_SIZE,
    }
    _write_run_config(workdir, run_config)

    train_iter = train_source.iter_batches(
        batch_size=args.batch_size,
        shuffle=True,
        seed=TRAIN_SHUFFLE_SEED,
        drop_remainder=True,
        repeat=True,
    )

    best_val_loss: float | None = None
    first_train_loss: float | None = None
    last_train_loss: float | None = None

    for step in range(1, args.steps + 1):
        state, metrics = train_step(state, next(train_iter))
        train_loss = float(metrics["loss"])
        first_train_loss = train_loss if first_train_loss is None else first_train_loss
        last_train_loss = train_loss

        if step == 1 or step == args.steps:
            print(
                json.dumps(
                    {
                        "event": "train_step",
                        "step": step,
                        "loss": train_loss,
                        "lr": _to_python_float(learning_rate_schedule(int(state.step))),
                    },
                    sort_keys=True,
                )
            )

        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate_loss(
                state,
                val_source.iter_batches(
                    batch_size=args.batch_size,
                    shuffle=False,
                    seed=0,
                    drop_remainder=False,
                    repeat=False,
                ),
            )
            print(
                json.dumps(
                    {
                        "event": "eval",
                        "step": step,
                        "val_loss": val_loss,
                    },
                    sort_keys=True,
                )
            )
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    checkpoints_dir / "best",
                    state,
                    hyperparams=run_config,
                    vocab_size=VOCAB_SIZE,
                )

        if step % args.ckpt_every == 0:
            save_checkpoint(
                checkpoints_dir / f"step_{step:06d}",
                state,
                hyperparams=run_config,
                vocab_size=VOCAB_SIZE,
            )
            _prune_step_checkpoints(
                checkpoints_dir,
                keep_last=CHECKPOINT_KEEP_LAST,
            )

    final_checkpoint = save_checkpoint(
        checkpoints_dir / "final",
        state,
        hyperparams=run_config,
        vocab_size=VOCAB_SIZE,
    )

    restored_state, manifest = restore_checkpoint(final_checkpoint, state)
    if not jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda a, b: jnp.array_equal(a, b),
            state.params,
            restored_state.params,
        )
    ):
        raise AssertionError("Final checkpoint restore did not reproduce params")
    if int(restored_state.step) != int(state.step):
        raise AssertionError("Final checkpoint restore did not reproduce step")

    print(
        json.dumps(
            {
                "event": "train_complete",
                "steps": args.steps,
                "first_train_loss": first_train_loss,
                "last_train_loss": last_train_loss,
                "best_val_loss": best_val_loss,
                "final_checkpoint": str(final_checkpoint),
                "restored_step": manifest["step"],
            },
            sort_keys=True,
        )
    )
    return final_checkpoint


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m poc0.train")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--peak-lr", type=float, default=PEAK_LEARNING_RATE)
    parser.add_argument("--end-lr", type=float, default=END_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY_STEPS)
    parser.add_argument("--ckpt-every", type=int, default=CHECKPOINT_EVERY_STEPS)
    parser.add_argument("--seed", type=int, default=INIT_SEED)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
