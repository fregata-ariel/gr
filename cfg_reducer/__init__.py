"""cfg_reducer — step-by-step CFG reduction with undo."""

from .types import NodeType, Op, Motif, MetaGraph, BASIC
from .engine import GraphEngine, Node
from .algorithm import ReductionAlgorithm, Scope, tarjan_scc
# NOTE: `dataset` is deliberately not imported here — it is the CLI
# entry point (`python -m cfg_reducer.dataset`) and importing it at
# package level would re-execute it under -m. Use
# `from cfg_reducer import dataset` (submodule import) instead.
from .generate import generate_cfg
from .generator_types import GeneratorSpec, CFGShape, Json, Features
from .family_registry import (
    CFGFamily, register_family, get_family, family_names, load_plugins,
)
from .generate_v2 import (
    normalize_spec, spec_to_json, spec_from_json, generate_cfg_v2, descriptor_for,
)
from . import store, motif, metagraph, generate, model_input

__all__ = [
    "NodeType", "Op", "Motif", "MetaGraph", "BASIC",
    "GraphEngine", "Node",
    "ReductionAlgorithm", "Scope", "tarjan_scc",
    "generate_cfg",
    "GeneratorSpec", "CFGShape", "Json", "Features", "CFGFamily",
    "register_family", "get_family", "family_names", "load_plugins",
    "normalize_spec", "spec_to_json", "spec_from_json", "generate_cfg_v2",
    "descriptor_for",
    "store", "motif", "metagraph", "generate", "model_input",
]
