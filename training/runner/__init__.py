"""Execution-agnostic experiment plans for the compute backend router.

T1 surface: plan types, JSON (de)serialisation, remote script generation,
and the milestone sweep builder. Nothing here executes a backend.
"""

from .plans import sweep_block, sweep_plan
from .scripts import rescore_jobs, rescore_script, train_wrapper
from .types import (
    Block,
    Plan,
    RescoreJob,
    TrainJob,
    bundle_name,
    plan_from_json,
    plan_to_json,
    validate_plan,
)

__all__ = [
    "Block",
    "Plan",
    "RescoreJob",
    "TrainJob",
    "bundle_name",
    "plan_from_json",
    "plan_to_json",
    "rescore_jobs",
    "rescore_script",
    "sweep_block",
    "sweep_plan",
    "train_wrapper",
    "validate_plan",
]
