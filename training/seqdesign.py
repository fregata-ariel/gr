"""有限候補空間の検証・永続化と独立 GP の数値コア。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import json
import math
import random
from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from training.mixture_doe import Observation

Array = NDArray[np.float64]


class _CopyFields:
    def __post_init__(self) -> None:
        for field in fields(cast(Any, self)):
            value = deepcopy(getattr(self, field.name))
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
            object.__setattr__(self, field.name, value)


@dataclass(frozen=True)
class Space(_CopyFields):
    factor_names: tuple[str, ...]
    candidates: tuple[dict[str, float], ...]
    simplex: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Proposal(_CopyFields):
    proposal_id: str
    candidate_id: int
    point: int
    factors: dict[str, float]
    seeds: tuple[int, ...]
    acquisition: str
    observation_revision: int
    campaign_revision: int


@dataclass(frozen=True)
class Campaign(_CopyFields):
    version: int
    campaign_id: str
    revision: int
    observation_revision: int
    space: Space
    responses: tuple[str, ...]
    objective: dict[str, object]
    input_mode: str
    context: dict[str, object] | None
    observations: tuple[Observation, ...]
    proposals: tuple[Proposal, ...]
    registry: tuple[dict[str, object], ...]
    rng_state: dict[str, int]


@dataclass(frozen=True)
class GPFit(_CopyFields):
    X: Array
    offset: Array
    span: Array
    active: NDArray[np.bool_]
    m: float
    s: float
    a: float
    ell: Array
    v: float
    L: Array
    alpha: Array
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Posterior(_CopyFields):
    mean: Array
    covariance: Array
    L: Array
    jitter: float


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _keys(value: Any, required: set[str], optional: set[str] | frozenset[str] = frozenset()) -> None:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        raise ValueError("invalid object keys")


def _integer(value: Any, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError("expected an integer in range")


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected a finite number")
    return float(value) if value else 0.0


def _names(value: Sequence[str]) -> None:
    if not value or any(not isinstance(x, str) or not x for x in value) or len(set(value)) != len(value):
        raise ValueError("names must be nonempty and unique")


def _factors(space: Space, row: Any) -> tuple[float, ...]:
    _keys(row, set(space.factor_names))
    result = tuple(_number(row[name]) for name in space.factor_names)
    if space.simplex is not None:
        values = [_number(row[name]) for name in space.simplex]
        if min(values) < 0 or abs(sum(values) - 1) > 1e-9:
            raise ValueError("invalid simplex coordinates")
    return result


def validate_space(space: Space) -> None:
    _names(space.factor_names)
    if space.simplex is not None:
        _names(space.simplex)
        if not set(space.simplex) <= set(space.factor_names):
            raise ValueError("unknown simplex factor")
    if not space.candidates:
        raise ValueError("empty candidate space")
    keys = [_factors(space, row) for row in space.candidates]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate candidates")


def space_from_json(text: str) -> Space:
    value = json.loads(text)
    _keys(value, {"factor_names", "candidates", "simplex"})
    if not isinstance(value["factor_names"], list) or not isinstance(value["candidates"], list):
        raise ValueError("space names and candidates must be arrays")
    if value["simplex"] is not None and not isinstance(value["simplex"], list):
        raise ValueError("simplex must be an array or null")
    space = Space(tuple(value["factor_names"]), tuple(value["candidates"]),
                  None if value["simplex"] is None else tuple(value["simplex"]))
    validate_space(space)
    return replace(space, candidates=tuple(dict(zip(space.factor_names, _factors(space, row)))
                                           for row in space.candidates))


def space_to_json(space: Space) -> str:
    validate_space(space)
    return _dump(asdict(space))


def observations_from_json(text: str, responses: Sequence[str] = ()) -> tuple[Observation, ...]:
    """既存 parser の数値変換より先に入力を検証する。"""
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise ValueError("observations must be an array")
    for row in rows:
        _keys(row, {"point", "seed", "x", "y", "wf", "n_train"}, {"y_raw"})
        for name in ("point", "seed", "n_train"):
            _integer(row[name])
        if not isinstance(row["x"], list) or len(row["x"]) != 3:
            raise ValueError("x must have three components")
        for value in row["x"]:
            _number(value)
        for name in ("y", "y_raw"):
            values = row.get(name)
            if name == "y_raw" and values is None:
                continue
            if not isinstance(values, dict) or any(not isinstance(k, str) or not k for k in values):
                raise ValueError("invalid response mapping")
            for value in values.values():
                _number(value)
        if not set(responses) <= row["y"].keys():
            raise ValueError("missing response")
        if row["wf"] is not None and not 0 <= _number(row["wf"]) <= 1:
            raise ValueError("wf outside [0,1]")
    from training.mixture_doe import observations_from_json as parse
    return tuple(parse(_dump(rows)))


def observations_to_json(observations: Sequence[Observation]) -> str:
    from training.mixture_doe import observations_to_json as encode
    text = encode(observations)
    observations_from_json(text)
    return _dump(json.loads(text))


def _registry(c: Campaign) -> dict[int, Any]:
    result: dict[int, Any] = {}
    patterns: set[tuple[float, ...]] = set()
    for row in c.registry:
        _keys(row, {"point", "candidate_id", "factors"})
        point, cid, factors = cast(Any, row["point"]), cast(Any, row["candidate_id"]), row["factors"]
        _integer(point)
        if point in result:
            raise ValueError("duplicate registry point")
        if cid is not None:
            _integer(cid)
            if cid >= len(c.space.candidates):
                raise ValueError("unknown candidate")
        if factors is None:
            if c.input_mode != "mixture" or cid is not None:
                raise ValueError("unknown nominal factors")
        else:
            key = _factors(c.space, factors)
            if key in patterns:
                raise ValueError("nominal pattern registered twice")
            patterns.add(key)
            if cid is not None and key != _factors(c.space, c.space.candidates[cid]):
                raise ValueError("candidate and factors disagree")
        result[point] = row
    return result


def _check_observations(c: Campaign, rows: Sequence[Observation]) -> tuple[Observation, ...]:
    checked = observations_from_json(observations_to_json(rows), c.responses)
    registry = _registry(c)
    unique: dict[tuple[int, int], Observation] = {}
    points: dict[int, Observation] = {}
    for row in checked:
        key = row.point, row.seed
        if key in unique and unique[key] != row:
            raise ValueError("conflicting re-observation")
        if c.input_mode == "mixture":
            if min(row.x) < 0 or abs(sum(row.x) - 1) > 1e-9:
                raise ValueError("invalid realised composition")
            old = points.get(row.point)
            if old is not None and (old.x != row.x or old.n_train != row.n_train):
                raise ValueError("replicates must share x and n_train")
        elif row.point not in registry:
            raise ValueError("unknown registry point")
        unique[key] = row
        points[row.point] = row
    return tuple(unique[key] for key in sorted(unique))


def validate_campaign(c: Campaign) -> None:
    validate_space(c.space)
    _integer(c.version, 1)
    if c.version != 1 or not isinstance(c.campaign_id, str) or not c.campaign_id:
        raise ValueError("unknown schema or empty campaign id")
    _integer(c.revision)
    _integer(c.observation_revision)
    if c.observation_revision > c.revision:
        raise ValueError("invalid revisions")
    _names(c.responses)
    if set(c.responses) & {"bal", "max"}:
        raise ValueError("response collides with composite name")
    _keys(c.objective, {"kind", "responses", "direction"})
    kind, responses = c.objective["kind"], c.objective["responses"]
    if kind not in ("bal", "max", "response") or c.objective["direction"] != "minimize":
        raise ValueError("invalid objective")
    if not isinstance(responses, list) or (kind == "response" and (len(responses) != 1 or responses[0] not in c.responses)) or (kind != "response" and responses != list(c.responses)):
        raise ValueError("invalid objective responses")
    if c.input_mode not in ("mixture", "registry"):
        raise ValueError("unknown input mode")
    if c.input_mode == "mixture" and (set(c.space.factor_names) != {"lay", "str", "spa"} or set(c.space.simplex or ()) != {"lay", "str", "spa"}):
        raise ValueError("mixture requires three simplex factors")
    if c.context is not None and not isinstance(c.context, dict):
        raise ValueError("context must be an object or null")
    _dump(c.context)
    _keys(c.rng_state, {"seed", "draws"})
    if type(c.rng_state["seed"]) is not int:
        raise ValueError("seed must be an integer")
    _integer(c.rng_state["draws"])
    registry = _registry(c)
    _check_observations(c, c.observations)
    ids: set[str] = set()
    reserved: set[tuple[int, int]] = set()
    for p in c.proposals:
        _integer(p.candidate_id)
        _integer(p.point)
        _integer(p.observation_revision)
        _integer(p.campaign_revision, 1)
        if not isinstance(p.proposal_id, str) or not p.proposal_id or p.proposal_id in ids:
            raise ValueError("invalid proposal id")
        ids.add(p.proposal_id)
        if p.candidate_id >= len(c.space.candidates) or _factors(c.space, p.factors) != _factors(c.space, c.space.candidates[p.candidate_id]):
            raise ValueError("invalid proposal factors")
        if p.point not in registry or registry[p.point]["candidate_id"] != p.candidate_id:
            raise ValueError("proposal not registered")
        if p.acquisition not in ("ts", "ei", "ucb", "random", "fixed") or p.observation_revision > c.observation_revision or p.campaign_revision > c.revision or p.observation_revision >= p.campaign_revision:
            raise ValueError("invalid proposal metadata")
        if not p.seeds:
            raise ValueError("empty proposal seeds")
        for seed in p.seeds:
            _integer(seed)
            key = p.point, seed
            if key in reserved:
                raise ValueError("duplicate seed reservation")
            reserved.add(key)


def campaign_from_json(text: str) -> Campaign:
    value = json.loads(text)
    _keys(value, {f.name for f in fields(Campaign)})
    for name in ("responses", "observations", "proposals", "registry"):
        if not isinstance(value[name], list):
            raise ValueError(f"{name} must be an array")
    value["space"] = space_from_json(_dump(value["space"]))
    value["responses"] = tuple(value["responses"])
    value["observations"] = observations_from_json(_dump(value["observations"]), value["responses"])
    proposals = []
    for row in value["proposals"]:
        _keys(row, {f.name for f in fields(Proposal)})
        if not isinstance(row["seeds"], list):
            raise ValueError("seeds must be an array")
        data: dict[str, Any] = {**row, "seeds": tuple(row["seeds"])}
        proposals.append(Proposal(**data))
    value["proposals"] = tuple(proposals)
    value["registry"] = tuple(value["registry"])
    c = Campaign(**value)
    validate_campaign(c)
    return replace(c, observations=_check_observations(c, c.observations),
                   registry=tuple(sorted(c.registry, key=lambda r: r["point"])))


def campaign_to_json(c: Campaign) -> str:
    validate_campaign(c)
    value = asdict(c)
    value["observations"] = json.loads(observations_to_json(_check_observations(c, c.observations)))
    value["registry"] = sorted(c.registry, key=lambda r: r["point"])
    return _dump(value)


def observe(campaign: Campaign, observations: Sequence[Observation]) -> Campaign:
    validate_campaign(campaign)
    rows = _check_observations(campaign, (*campaign.observations, *observations))
    if rows == _check_observations(campaign, campaign.observations):
        return campaign
    registry = _registry(campaign)
    for row in rows:
        if row.point not in registry:
            registry[row.point] = {"point": row.point, "candidate_id": None, "factors": None}
    result = replace(campaign, observations=rows, registry=tuple(registry[p] for p in sorted(registry)),
                     revision=campaign.revision + 1, observation_revision=campaign.observation_revision + 1)
    validate_campaign(result)
    return result


def observation_coordinates(c: Campaign) -> Array:
    validate_campaign(c)
    registry = _registry(c)
    rows = []
    for obs in c.observations:
        factors = dict(zip(("lay", "str", "spa"), obs.x)) if c.input_mode == "mixture" else registry[obs.point]["factors"]
        rows.append([factors[name] for name in c.space.factor_names])
    return np.asarray(rows, dtype=float).reshape(-1, len(c.space.factor_names))


class DeterministicRNG(random.Random):
    """random() のみを数え、seed と消費数から再開する。"""
    def __init__(self, seed: int, draws: int = 0):
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        _integer(draws)
        super().__init__(seed)
        self.initial_seed = seed
        self.draws = 0
        for _ in range(draws):
            self.random()

    def random(self) -> float:
        self.draws += 1
        return super().random()

    @property
    def rng_state(self) -> dict[str, int]:
        return {"seed": self.initial_seed, "draws": self.draws}


def normal_draw(rng: random.Random) -> float:
    return math.sqrt(-2 * math.log(1 - rng.random())) * math.cos(2 * math.pi * rng.random())


def uniform_index(size: int, rng: random.Random) -> int:
    _integer(size, 1)
    return math.floor(size * rng.random())


def _matrix(value: ArrayLike) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("expected a finite matrix")
    return result


def rbf_kernel(X: ArrayLike, Y: ArrayLike, a: float, ell: ArrayLike) -> Array:
    x, y = _matrix(X), _matrix(Y)
    lengths = np.asarray(ell, dtype=float)
    if x.shape[1] != y.shape[1] or lengths.shape != (x.shape[1],) or not np.isfinite(lengths).all() or np.any(lengths <= 0) or not math.isfinite(a) or a <= 0:
        raise ValueError("invalid kernel parameters")
    return a * np.exp(-0.5 * np.sum(((x[:, None, :] - y[None, :, :]) / lengths) ** 2, axis=2))


def _cholesky(matrix: Array, scale: float) -> tuple[Array, float]:
    for exponent in range(-10, -3):
        jitter = scale * 10.0 ** exponent
        try:
            L = np.linalg.cholesky(matrix + jitter * np.eye(len(matrix)))
            if np.isfinite(L).all():
                return L, jitter
        except np.linalg.LinAlgError:
            pass
    raise ValueError("Cholesky failed after jitter retries")


def _likelihood(X: Array, z: Array, a: float, ell: Array, v: float) -> tuple[float, Array, Array, float]:
    L, jitter = _cholesky(rbf_kernel(X, X, a, ell) + v * np.eye(len(X)), max(1, a + v))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
    lml = float(-0.5 * z @ alpha - np.log(np.diag(L)).sum() - len(z) / 2 * math.log(2 * math.pi))
    if not math.isfinite(lml):
        raise ValueError("nonfinite log marginal likelihood")
    return lml, L, alpha, jitter


def log_marginal_likelihood(X: ArrayLike, z: ArrayLike, a: float, ell: ArrayLike, v: float) -> float:
    x, response = _matrix(X), np.asarray(z, dtype=float)
    if response.shape != (len(x),) or not np.isfinite(response).all() or not math.isfinite(v) or v < 0:
        raise ValueError("invalid response or noise")
    return _likelihood(x, response, a, np.asarray(ell, dtype=float), v)[0]


def fit_gp(X: ArrayLike, y: ArrayLike, *, candidates: ArrayLike | None = None,
           fixed: tuple[float, ArrayLike, float] | None = None) -> GPFit:
    """候補の範囲で標準化する。省略時は X を候補集合として扱う。

    fixed=(a, ell, v) は標準化単位の固定パラメタによる検証用。
    """
    raw, response = _matrix(X), np.asarray(y, dtype=float)
    grid = raw if candidates is None else _matrix(candidates)
    if not len(grid) or grid.shape[1] != raw.shape[1] or response.shape != (len(raw),) or not np.isfinite(response).all():
        raise ValueError("invalid training data or candidate space")
    offset, span = grid.min(axis=0), np.ptp(grid, axis=0)
    active = span > 0
    x = (raw[:, active] - offset[active]) / span[active]
    m = float(response.mean()) if len(response) else 0.0
    s = float(response.std()) if len(response) else 1.0
    if s < 1e-8:
        s = 1.0
    z = (response - m) / s
    groups: dict[tuple[float, ...], list[float]] = {}
    for row, value in zip(raw, response):
        groups.setdefault(tuple(row), []).append(float(value))
    df = sum(len(values) - 1 for values in groups.values())
    pe = sum(sum((value - float(np.mean(values))) ** 2 for value in values) for values in groups.values()) / df if df else None
    noise = float(np.clip(pe / s**2, 1e-10, 10)) if pe is not None else 0.01
    failures: list[dict[str, Any]] = []
    dimension = x.shape[1]
    bounds = [(1e-4, 100.0)] + [(0.02, 2.0)] * dimension + ([] if df else [(1e-10, 10.0)])
    lower, upper = np.log(np.asarray(bounds)).T

    def unpack(theta: Array) -> tuple[float, Array, float]:
        values = np.exp(theta)
        return float(values[0]), values[1:dimension + 1], noise if df else float(values[-1])

    def evaluate(theta: Array) -> float:
        try:
            return _likelihood(x, z, *unpack(theta))[0]
        except (ValueError, np.linalg.LinAlgError) as exc:
            failures.append({"lml": None, "reason": str(exc), "log_parameters": theta.tolist()})
            return -math.inf

    best_score, best = -math.inf, None
    if fixed is None:
        for length in ((0.05, 0.3, 1.0) if len(groups) >= 2 else (0.3,)):
            theta = np.log([1.0] + [length] * dimension + ([] if df else [0.01]))
            score, step = evaluate(theta), (upper - lower) / 4
            if len(groups) >= 2:
                for _ in range(30):
                    improved = False
                    for d in range(len(theta)):
                        chosen, chosen_score = theta, score
                        for direction in (-1, 1):
                            trial = theta.copy()
                            trial[d] = np.clip(theta[d] + direction * step[d], lower[d], upper[d])
                            trial_score = evaluate(trial)
                            if trial_score > chosen_score + 1e-8:
                                chosen, chosen_score = trial, trial_score
                        if chosen_score > score + 1e-8:
                            theta, score, improved = chosen, chosen_score, True
                    if not improved:
                        step /= 2
                    if np.all(step < 1e-3):
                        break
            if score > best_score:
                best_score, best = score, theta
        if best is None:
            raise ValueError("all GP trials failed")
        a, ell, v = unpack(best)
        labels = ["a"] + [f"ell_{i}" for i in np.flatnonzero(active)] + ([] if df else ["v"])
        boundary = [label for label, val, lo, hi in zip(labels, best, lower, upper) if abs(val - lo) < 1e-10 or abs(val - hi) < 1e-10]
    else:
        a, lengths, v = fixed
        ell = np.asarray(lengths, dtype=float)
        if not math.isfinite(v) or v < 0:
            raise ValueError("invalid fixed noise")
        boundary = []
    lml, L, alpha, jitter = _likelihood(x, z, a, ell, v)
    return GPFit(x, offset, span, active, m, s, a, ell, v, L, alpha,
                 {"lml": lml, "jitter": jitter, "pure_error_df": df, "pure_error_variance": pe,
                  "boundary": boundary, "failures": failures, "prior": not len(raw)})


def posterior(fit: GPFit, Xstar: ArrayLike) -> Posterior:
    raw = _matrix(Xstar)
    if raw.shape[1] != len(fit.offset):
        raise ValueError("prediction dimension mismatch")
    x = (raw[:, fit.active] - fit.offset[fit.active]) / fit.span[fit.active]
    B = rbf_kernel(fit.X, x, fit.a, fit.ell)
    V = np.linalg.solve(fit.L, B)
    mean = fit.m + fit.s * (B.T @ fit.alpha)
    covariance = rbf_kernel(x, x, fit.a, fit.ell) - V.T @ V
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if len(eigenvalues) and eigenvalues[0] < -1e-8 * max(1, fit.a):
        raise ValueError("posterior covariance has a materially negative eigenvalue")
    if np.any(eigenvalues < 0):
        covariance = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
        covariance = (covariance + covariance.T) / 2
    L, jitter = _cholesky(covariance, max(1, fit.a + fit.v))
    return Posterior(mean, fit.s**2 * covariance, fit.s * L, fit.s**2 * jitter)


def draw_joint(distribution: Posterior, rng: random.Random) -> Array:
    return distribution.mean + distribution.L @ np.asarray([normal_draw(rng) for _ in distribution.mean])
