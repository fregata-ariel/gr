"""Write concrete mixture specs for variable-length pretraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CHOICES = (8, 12, 16, 24, 32, 48, 64, 96, 128)
WEIGHTS = (3, 3, 3, 3, 2, 2, 2, 1, 1)


def params_for_n(n: int) -> dict:
    if type(n) is not int or n < 3:
        raise ValueError("n must be an integer >= 3")
    loops, gotos = 3 * n // 20, n // 10
    capped_loops = min(loops, (11 * n // 10 - 3) // 3)
    capped_gotos = min(gotos, (11 * n // 10 - 3 - 3 * capped_loops) // 2)
    structured = dict(loop_count=capped_loops, goto_count=capped_gotos,
                      max_layer_width=3, branch_degree=2, merge_degree=3,
                      span_mode="uniform", target_depth=None)
    return {"components": [
        {"family": "layered", "weight": 1, "params": dict(
            loop_count=loops, goto_count=gotos, max_layer_width=3,
            edge_prob=0.18, span_mode="uniform")},
        {"family": "structured", "weight": 1, "params": structured},
        {"family": "spaghetti", "weight": 1,
         "params": structured | {"spaghetti_rate": 0.1}},
    ]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    specs = {f"n{n}": {"family": "mixture", "num_nodes": n, "params": params_for_n(n)}
             for n in (*CHOICES, 192, 256)}
    specs["mixed"] = {"family": "mixture", "num_nodes": {
        "choices": list(CHOICES), "weights": list(WEIGHTS)},
        "params_by_num_nodes": {str(n): params_for_n(n) for n in CHOICES}}
    for name, spec in specs.items():
        (args.out / f"{name}.json").write_text(
            json.dumps(spec, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
