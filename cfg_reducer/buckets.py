"""Measure realized CFG and MetaGraph features for dataset candidates."""

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
import math

from .generator_types import Features, Json
from .reducibility import is_reducible
from .structure_features import features_for
from .types import MetaGraph


@dataclass(frozen=True)
class Candidate:
    split: str
    seed: int
    sample_id: str
    requested: dict[str, Json]
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    mg: MetaGraph


def measure_realized(candidate: Candidate) -> Features:
    return features_for(candidate.mg) | {
        "num_nodes": len(candidate.nodes),
        "reducible": is_reducible(candidate.nodes, candidate.edges, entry=candidate.nodes[0]),
    }



_FEATURE_KEYS = frozenset(features_for(MetaGraph((), (), {}))) | {'num_nodes', 'reducible'}


@dataclass(frozen=True)
class CFGReference:
    dataset_id: str
    sample_id: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class BucketDimension:
    feature: str
    cuts: tuple[float, ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.feature not in _FEATURE_KEYS or any(c in self.feature for c in '|='):
            raise ValueError('unknown or invalid feature')
        if any(not label or any(c in label for c in '|=') for label in self.labels):
            raise ValueError('invalid label')
        if len(set(self.labels)) != len(self.labels):
            raise ValueError('duplicate labels')
        if any(not math.isfinite(c) for c in self.cuts) or any(
            a >= b for a, b in zip(self.cuts, self.cuts[1:])
        ):
            raise ValueError('cuts must be finite and strictly increasing')
        if self.feature == 'reducible':
            if self.cuts or self.labels != ('no', 'yes'):
                raise ValueError('reducible requires no/yes labels and no cuts')
        elif len(self.labels) != len(self.cuts) + 1:
            raise ValueError('labels must delimit cuts')


@dataclass(frozen=True)
class BucketPlan:
    dims: tuple[BucketDimension, ...]
    target_per_bucket: int
    active: tuple[tuple[str, ...], ...] | None = None

    def __post_init__(self) -> None:
        if type(self.target_per_bucket) is not int or self.target_per_bucket <= 0:
            raise ValueError('target must be a positive integer')
        if not self.dims or len({d.feature for d in self.dims}) != len(self.dims):
            raise ValueError('empty or duplicate dimensions')
        if self.active is not None:
            if not self.active or len(set(self.active)) != len(self.active):
                raise ValueError('empty or duplicate active buckets')
            for labels in self.active:
                if len(labels) != len(self.dims) or any(
                    label not in dim.labels for dim, label in zip(self.dims, labels)
                ):
                    raise ValueError('unknown active labels')

    def bucket_ids(self) -> tuple[str, ...]:
        labels = self.active if self.active is not None else product(*(d.labels for d in self.dims))
        return tuple(_bucket_id(self, row) for row in labels)


def _bucket_id(plan: BucketPlan, labels: tuple[str, ...]) -> str:
    return '|'.join(f'{dim.feature}={label}' for dim, label in zip(plan.dims, labels))


@dataclass(frozen=True)
class AcceptanceState:
    counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class AcceptDecision:
    accepted: bool
    reason: str | None
    bucket: str | None
    realized: Features


AcceptHook = Callable[[Candidate, AcceptanceState], AcceptDecision]


def bucket_for(features: Features, plan: BucketPlan) -> str:
    labels = []
    for dim in plan.dims:
        value = features.get(dim.feature)
        if value is None or (isinstance(value, (int, float)) and not math.isfinite(value)):
            raise ValueError('invalid_feature')
        if dim.feature == 'reducible':
            if type(value) is not bool:
                raise TypeError('reducible must be bool')
            index = int(value)
        else:
            if type(value) not in (int, float):
                raise TypeError('numeric feature required')
            index = bisect_left(dim.cuts, value)
        labels.append(dim.labels[index])
    return _bucket_id(plan, tuple(labels))


@dataclass(frozen=True)
class _BucketAcceptor:
    plan: BucketPlan

    def __call__(self, candidate: Candidate, state: AcceptanceState) -> AcceptDecision:
        realized = measure_realized(candidate)
        try:
            bucket = bucket_for(realized, self.plan)
        except ValueError:
            return AcceptDecision(False, 'invalid_feature', None, realized)
        reason = None
        if bucket not in self.plan.bucket_ids():
            reason = 'outside_plan'
        elif dict(state.counts).get(bucket, 0) >= self.plan.target_per_bucket:
            reason = 'bucket_full'
        return AcceptDecision(reason is None, reason, bucket, realized)


def bucket_acceptor(plan: BucketPlan) -> AcceptHook:
    """Stateless hook; .plan exposes quota metadata to the dataset builder."""
    return _BucketAcceptor(plan)


def measurement_acceptor(candidate: Candidate, state: AcceptanceState) -> AcceptDecision:
    return AcceptDecision(True, None, None, measure_realized(candidate))


def default_bucket_plan(target_per_bucket: int) -> BucketPlan:
    return BucketPlan((
        BucketDimension('mean_offset', (1.68, 2.03), ('low', 'mid', 'high')),
        BucketDimension('max_depth', (1, 2), ('0-1', '2', '3+')),
        BucketDimension('max_in_degree', (2, 3), ('le2', '3', 'ge4')),
        BucketDimension('reducible', (), ('no', 'yes')),
    ), target_per_bucket)


@dataclass(frozen=True)
class _RangesAcceptor:
    inner: AcceptHook
    ranges: tuple[tuple[str, tuple[float | None, float | None]], ...]

    @property
    def plan(self) -> BucketPlan | None:
        return getattr(self.inner, 'plan', None)

    def __call__(self, candidate: Candidate, state: AcceptanceState) -> AcceptDecision:
        decision = self.inner(candidate, state)
        for feature, (lower, upper) in self.ranges:
            value = decision.realized.get(feature)
            if value is None or not math.isfinite(value):
                return AcceptDecision(False, 'invalid_feature', decision.bucket, decision.realized)
            if (lower is not None and value < lower) or (upper is not None and value > upper):
                return AcceptDecision(False, 'outside_plan', decision.bucket, decision.realized)
        return decision


def ranges_acceptor(
    inner: AcceptHook, ranges: dict[str, tuple[float | None, float | None]],
) -> AcceptHook:
    for feature, bounds in ranges.items():
        if feature not in _FEATURE_KEYS or len(bounds) != 2:
            raise ValueError('invalid range')
        lower, upper = bounds
        if any(x is not None and not math.isfinite(x) for x in bounds):
            raise ValueError('range bounds must be finite')
        if lower is not None and upper is not None and lower > upper:
            raise ValueError('reversed range')
    return _RangesAcceptor(inner, tuple((k, tuple(v)) for k, v in sorted(ranges.items())))
