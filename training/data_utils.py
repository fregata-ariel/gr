"""Torch-free I/O helpers shared by the local training-pipeline scripts.

Not used by train_ar.py, which must stay single-file self-contained for
the Colab upload.
"""

from __future__ import annotations
import json
from pathlib import Path


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
