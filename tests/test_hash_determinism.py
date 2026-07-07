from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

_PIPELINE = """
import json
from cfg_reducer.gen import build_cfg
from cfg_reducer import ReductionAlgorithm, metagraph, motif
from cfg_reducer.linearize import encode

engine = build_cfg(num_nodes={num_nodes}, edge_prob={edge_prob}, seed={seed})
algorithm = ReductionAlgorithm(engine, entry="N00")
while algorithm.step() is not None:
    pass
mg = metagraph.build(motif.extract(engine.history))
print(json.dumps(encode(mg)))
"""


def _encode_in_subprocess(
    num_nodes: int,
    edge_prob: float,
    seed: int,
    hash_seed: int,
) -> list[str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(hash_seed)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _PIPELINE.format(
                num_nodes=num_nodes,
                edge_prob=edge_prob,
                seed=seed,
            ),
        ],
        check=True,
        capture_output=True,
        cwd=ROOT,
        env=env,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    ("num_nodes", "edge_prob", "seed"),
    [
        (12, 0.20, 7),
        (15, 0.10, 580),
        (15, 0.10, 627),
    ],
)
def test_build_cfg_encoding_is_invariant_across_python_hash_seeds(
    num_nodes: int,
    edge_prob: float,
    seed: int,
) -> None:
    token_streams = {
        hash_seed: _encode_in_subprocess(
            num_nodes=num_nodes,
            edge_prob=edge_prob,
            seed=seed,
            hash_seed=hash_seed,
        )
        for hash_seed in (0, 1, 2)
    }

    assert len({tuple(tokens) for tokens in token_streams.values()}) == 1, (
        f"cross-process token stream changed for "
        f"num_nodes={num_nodes}, edge_prob={edge_prob}, seed={seed}: "
        f"{token_streams}"
    )
