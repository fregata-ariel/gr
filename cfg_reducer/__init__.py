"""cfg_reducer — step-by-step CFG reduction with undo."""

from .types import NodeType, Op, Motif, MetaGraph, BASIC
from .engine import GraphEngine, Node
from .algorithm import ReductionAlgorithm, Scope, tarjan_scc
# NOTE: `dataset` is deliberately not imported here — it is the CLI
# entry point (`python -m cfg_reducer.dataset`) and importing it at
# package level would re-execute it under -m. Use
# `from cfg_reducer import dataset` (submodule import) instead.
from .generate import generate_cfg
from . import store, motif, metagraph, generate, model_input

__all__ = [
    "NodeType", "Op", "Motif", "MetaGraph", "BASIC",
    "GraphEngine", "Node",
    "ReductionAlgorithm", "Scope", "tarjan_scc",
    "generate_cfg",
    "store", "motif", "metagraph", "generate", "model_input",
]
