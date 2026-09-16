"""Execution-agnostic experiment plans for the compute backend router.

T1 surface: plan types, JSON (de)serialisation, remote script generation,
and the milestone sweep builder. T2 adds the ``Backend`` contract and the
Colab CLI adapter. T3 adds the local Docker adapter and the test doubles.
T4 adds the policy router and the plan executor.
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
from .executor import BUNDLE_FILES, CODE_FILES, TRAIN_OUTPUTS, PlanExecutor
from .fake import DryRunBackend, FakeBackend
from .plans import sweep_block, sweep_plan
from .router import (
    DEFAULT_POLICY,
    KINDS,
    POLICIES,
    Router,
    resolve_policy,
)
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
    "BUNDLE_FILES",
    "CODE_FILES",
    "DEFAULT_POLICY",
    "KINDS",
    "POLICIES",
    "TRAIN_OUTPUTS",
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
    "PlanExecutor",
    "RemoteFailure",
    "RescoreJob",
    "Router",
    "SessionLost",
    "TrainJob",
    "bundle_name",
    "filter_output",
    "plan_from_json",
    "plan_to_json",
    "rescore_jobs",
    "rescore_script",
    "resolve_policy",
    "run_command",
    "sweep_block",
    "sweep_plan",
    "train_wrapper",
    "validate_plan",
]
