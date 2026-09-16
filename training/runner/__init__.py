"""Execution-agnostic experiment plans for the compute backend router.

T1 surface: plan types, JSON (de)serialisation, remote script generation,
and the milestone sweep builder. T2 adds the ``Backend`` contract and the
Colab CLI adapter. T3 adds the local Docker adapter and the test doubles.
"""

from .backend import (
    Backend,
    BackendUnavailable,
    CommandResult,
    CommandRunner,
    RemoteFailure,
    SessionLost,
    filter_output,
    run_command,
)
from .colab import ColabBackend, ColabTimeouts
from .docker import DockerBackend
from .fake import DryRunBackend, FakeBackend
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
    "Backend",
    "BackendUnavailable",
    "Block",
    "ColabBackend",
    "ColabTimeouts",
    "CommandResult",
    "CommandRunner",
    "DockerBackend",
    "DryRunBackend",
    "FakeBackend",
    "Plan",
    "RemoteFailure",
    "RescoreJob",
    "SessionLost",
    "TrainJob",
    "bundle_name",
    "filter_output",
    "plan_from_json",
    "plan_to_json",
    "rescore_jobs",
    "rescore_script",
    "run_command",
    "sweep_block",
    "sweep_plan",
    "train_wrapper",
    "validate_plan",
]
