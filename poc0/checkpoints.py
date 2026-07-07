from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp
from flax.training import orbax_utils, train_state


def _checkpoint_payload(state: train_state.TrainState) -> dict[str, Any]:
    return {
        "step": int(state.step),
        "params": state.params,
        "opt_state": state.opt_state,
    }


def _manifest_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "manifest.json"


def save_checkpoint(
    checkpoint_dir: Path,
    state: train_state.TrainState,
    *,
    hyperparams: Mapping[str, object],
    vocab_size: int,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)

    payload = _checkpoint_payload(state)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(
        checkpoint_dir,
        payload,
        save_args=orbax_utils.save_args_from_target(payload),
        force=True,
    )

    manifest = {
        "step": int(state.step),
        "hyperparams": dict(hyperparams),
        "vocab_size": vocab_size,
    }
    _manifest_path(checkpoint_dir).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint_dir


def restore_checkpoint(
    checkpoint_dir: Path,
    state: train_state.TrainState,
) -> tuple[train_state.TrainState, dict[str, object]]:
    checkpoint_dir = Path(checkpoint_dir)

    checkpointer = ocp.PyTreeCheckpointer()
    restored_payload = checkpointer.restore(
        checkpoint_dir,
        item=_checkpoint_payload(state),
    )
    manifest = json.loads(_manifest_path(checkpoint_dir).read_text(encoding="utf-8"))

    restored_state = state.replace(
        step=restored_payload["step"],
        params=restored_payload["params"],
        opt_state=restored_payload["opt_state"],
    )
    return restored_state, manifest
