"""
Serialise / deserialise Op history and MetaGraph samples to JSON.

Op history uses a deliberately flat schema so that a Rust/C++ reader
can consume the same files without a Python dependency.
MetaGraph samples follow the canonical schema in
docs/design/metagraph_schema.md.
"""

from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import Any

from .types import Op, Motif, MetaGraph


# ──────────────────────────────────────────────
#  Codec helpers
# ──────────────────────────────────────────────

def _encode_value(v: Any) -> Any:
    """Convert sets to sorted lists for JSON serialisation."""
    if isinstance(v, set):
        return sorted(v)
    if isinstance(v, dict):
        return {k: _encode_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_encode_value(item) for item in v]
    if isinstance(v, tuple):
        return [_encode_value(item) for item in v]
    return v


def _decode_op(d: dict) -> Op:
    return Op(
        kind=d["kind"],
        forward=d.get("forward", {}),
        inverse=d.get("inverse", {}),
        meta=d.get("meta", {}),
    )


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def save(ops: list[Op], path: str | Path) -> None:
    """Write a list of Ops to a JSON file."""
    payload = [
        {
            "kind": op.kind,
            "forward": _encode_value(op.forward),
            "inverse": _encode_value(op.inverse),
            "meta": _encode_value(op.meta),
        }
        for op in ops
    ]
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load(path: str | Path) -> list[Op]:
    """Read a list of Ops from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_decode_op(d) for d in raw]


# ──────────────────────────────────────────────
#  MetaGraph samples  (docs/design/metagraph_schema.md)
# ──────────────────────────────────────────────

SCHEMA_VERSION = 1

# Canonical vocabulary; encoders must reject anything else explicitly.
MOTIF_KINDS = frozenset({"entry", "linear", "merge", "loop"})

# Fixed namespace for sample_id derivation (UUIDv5 over canonical
# provenance).  Changing this invalidates every derived sample_id.
SAMPLE_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://github.com/fregata-ariel/gr/metagraph-sample"
)


def _canonicalize(v: Any) -> Any:
    """JSON-safe copy with recursively sorted dict keys (determinism)."""
    if isinstance(v, dict):
        return {k: _canonicalize(v[k]) for k in sorted(v)}
    if isinstance(v, (list, tuple)):
        return [_canonicalize(item) for item in v]
    if isinstance(v, set):
        return sorted(v)
    return v


def sample_id_for(provenance: dict) -> str:
    """
    Derive the deterministic sample_id for a provenance object.

    One-way: consumers treat sample_id as an opaque UUID and never
    parse it.  Same provenance always yields the same id, so batch
    generation is idempotent and duplicates are detectable.
    """
    canonical = json.dumps(
        _canonicalize(provenance),
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return str(uuid.uuid5(SAMPLE_NAMESPACE, canonical))


def _encode_motif(m: Motif) -> dict:
    if m.kind not in MOTIF_KINDS:
        raise ValueError(
            f"unknown motif kind {m.kind!r} at step {m.step}: "
            f"canonical vocabulary is {sorted(MOTIF_KINDS)}"
        )

    if m.kind == "loop":
        meta = {
            "header": m.meta["header"],
            "scc": sorted(m.meta["scc"]),
            "back_edges": sorted([src, dst] for src, dst in m.meta["back_edges"]),
        }
    elif m.meta:
        raise ValueError(
            f"non-loop motif at step {m.step} carries meta {m.meta!r}; "
            "schema v1 requires empty meta for non-loop kinds"
        )
    else:
        meta = {}

    return {
        "kind": m.kind,
        "node": m.node,
        "preds": list(m.preds),
        "succs": list(m.succs),
        "meta": meta,
        "step": m.step,
    }


def encode_metagraph(mg: MetaGraph) -> dict:
    """MetaGraph -> canonical JSON-ready dict (recursive)."""
    return {
        "motifs": [
            _encode_motif(m) for m in sorted(mg.motifs, key=lambda m: m.step)
        ],
        "edges": [list(e) for e in sorted(mg.edges)],
        "subgraphs": [
            {"loop_step": step, "graph": encode_metagraph(mg.subgraphs[step])}
            for step in sorted(mg.subgraphs)
        ],
    }


def decode_metagraph(data: dict) -> MetaGraph:
    """Canonical JSON dict -> MetaGraph, rebuilding Loop children."""
    subgraphs: dict[int, MetaGraph] = {}
    for entry in data["subgraphs"]:
        subgraphs[entry["loop_step"]] = decode_metagraph(entry["graph"])

    motifs: list[Motif] = []
    for md in sorted(data["motifs"], key=lambda d: d["step"]):
        kind = md["kind"]
        if kind not in MOTIF_KINDS:
            raise ValueError(
                f"unknown motif kind {kind!r} at step {md['step']}: "
                f"canonical vocabulary is {sorted(MOTIF_KINDS)}"
            )

        meta: dict[str, Any] = {}
        children: tuple[Motif, ...] = ()
        if kind == "loop":
            meta = {
                "header": md["meta"]["header"],
                "scc": list(md["meta"]["scc"]),
                "back_edges": [tuple(e) for e in md["meta"]["back_edges"]],
            }
            sub = subgraphs.get(md["step"])
            if sub is not None:
                children = sub.motifs

        motifs.append(Motif(
            kind=kind,
            node=md["node"],
            preds=tuple(md["preds"]),
            succs=tuple(md["succs"]),
            meta=meta,
            step=md["step"],
            children=children,
        ))

    loop_steps = {m.step for m in motifs if m.kind == "loop"}
    orphans = set(subgraphs) - loop_steps
    if orphans:
        raise ValueError(
            f"subgraph loop_step(s) {sorted(orphans)} have no matching "
            "loop motif at this level"
        )

    return MetaGraph(
        motifs=tuple(motifs),
        edges=tuple((e[0], e[1]) for e in data["edges"]),
        subgraphs=subgraphs,
    )


def encode_sample(
    mg: MetaGraph, provenance: dict, sample_id: str | None = None,
) -> dict:
    """
    Wrap a MetaGraph in the canonical sample envelope.

    sample_id defaults to the UUIDv5 derived from provenance; pass it
    explicitly only for hand-authored fixtures.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id if sample_id is not None
        else sample_id_for(provenance),
        "provenance": _canonicalize(provenance),
        "metagraph": encode_metagraph(mg),
    }


def decode_sample(payload: dict) -> MetaGraph:
    """Validate the envelope and return its MetaGraph."""
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION}"
        )
    return decode_metagraph(payload["metagraph"])


def save_sample(
    mg: MetaGraph,
    provenance: dict,
    path: str | Path,
    sample_id: str | None = None,
) -> None:
    """Write one MetaGraph sample to a canonical JSON file."""
    payload = encode_sample(mg, provenance, sample_id)
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_sample(path: str | Path) -> MetaGraph:
    """Read one MetaGraph sample from a canonical JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return decode_sample(payload)
