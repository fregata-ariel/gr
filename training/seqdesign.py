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


def normal_draws(rng: random.Random, count: int) -> Array:
    """normal_draw を count 回呼んだのと同じ列を、一様乱数だけ逐次に取り出して numpy で変換する。"""
    _integer(count)
    uniform = np.asarray([rng.random() for _ in range(2 * count)], dtype=float).reshape(count, 2)
    return np.sqrt(-2 * np.log(1 - uniform[:, 0])) * np.cos(2 * math.pi * uniform[:, 1])


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


# 以下は T1 の数値 API を組み合わせる逐次設計層。
def candidate_coordinates(space: Space) -> Array:
    return np.asarray([_factors(space, row) for row in space.candidates])


def _fits(c: Campaign) -> tuple[GPFit, ...]:
    x, grid = observation_coordinates(c), candidate_coordinates(c.space)
    return tuple(fit_gp(x, [o.y[r] for o in c.observations], candidates=grid) for r in c.responses)


def composite(samples: Array, responses: Sequence[str], objective: str) -> Array:
    """最後から2番目の応答軸を元単位で合成する。"""
    if objective == "bal":
        return samples.mean(axis=-2)
    if objective == "max":
        return samples.max(axis=-2)
    return samples[..., responses.index(objective), :]


def _objective(c: Campaign) -> str:
    return cast(list[str], c.objective["responses"])[0] if c.objective["kind"] == "response" else str(c.objective["kind"])


def _samples(distributions: Sequence[Posterior], count: int, rng: random.Random,
             *, latent: bool = False) -> Array:
    # 乱数順は draw → response → coordinate。行列積だけをまとめて高速化する。
    size = len(distributions[0].mean)
    normals = normal_draws(rng, count * len(distributions) * size).reshape(count, len(distributions), size)
    result = np.empty_like(normals)
    for r, p in enumerate(distributions):
        if latent:
            # 表示区間には sampling jitter を含めない。
            values, vectors = np.linalg.eigh(p.covariance)
            factor = vectors * np.sqrt(np.maximum(values, 0))
        else:
            factor = p.L
        result[:, r, :] = p.mean + normals[:, r, :] @ factor.T
    return result


def _select(c: Campaign, k: int, acq: str, rng: random.Random,
            fits: Sequence[GPFit] | None = None) -> tuple[list[int], list[str]]:
    _integer(k, 1)
    if acq not in ("ts", "ei", "ucb", "random"):
        raise ValueError("unknown acquisition (fixed is simulate-only)")
    m = len(c.space.candidates)
    if acq == "random" or (acq == "ei" and not c.observations):
        note = ["EI: 観測0件のため random に退避"] if acq == "ei" else []
        return [uniform_index(m, rng) for _ in range(k)], note
    grid = candidate_coordinates(c.space)
    if acq == "ei":
        grid = np.concatenate((grid, observation_coordinates(c)))
    models = _fits(c) if fits is None else fits
    distributions = tuple(posterior(f, grid) for f in models)
    samples = composite(_samples(distributions, k if acq == "ts" else 1024, rng), c.responses, _objective(c))
    if acq == "ts":
        return np.argmin(samples, axis=1).tolist(), []
    if acq == "ei":
        incumbent = samples[:, m:].mean(axis=0).min()
        index = int(np.argmax(np.maximum(incumbent - samples[:, :m], 0).mean(axis=0)))
    else:
        index = int(np.argmin(samples.mean(axis=0) - 2 * samples.std(axis=0, ddof=0)))
    return [index] * k, []


def _payload(c: Campaign, proposals: Sequence[Proposal], revision: int) -> dict[str, Any]:
    return {"version": 1, "campaign_id": c.campaign_id, "campaign_revision": revision,
            "context": deepcopy(c.context), "proposals": [asdict(p) for p in proposals]}


def _reserve(c: Campaign, indices: Sequence[int], seeds: int, acq: str,
             rng_state: dict[str, int]) -> tuple[Campaign, dict[str, Any]]:
    _integer(seeds, 1)
    registry = _registry(c)
    patterns = {_factors(c.space, row["factors"]): point for point, row in registry.items()
                if row["factors"] is not None}
    maximum: dict[int, int] = {}
    for o in c.observations:
        maximum[o.point] = max(maximum.get(o.point, -1), o.seed)
    for p in c.proposals:
        maximum[p.point] = max(maximum.get(p.point, -1), *p.seeds)
    batch = []
    used_ids = {p.proposal_id for p in c.proposals}
    serial = 1
    for cid in indices:
        factors = c.space.candidates[cid]
        key = _factors(c.space, factors)
        if key in patterns:
            point = patterns[key]
        else:
            point = 100
            while point in registry:
                point += 1
            patterns[key] = point
        registry[point] = {"point": point, "candidate_id": cid, "factors": factors}
        start = maximum.get(point, -1) + 1
        maximum[point] = start + seeds - 1
        while f"q{serial:06d}" in used_ids:
            serial += 1
        batch.append(Proposal(f"q{serial:06d}", cid, point, factors, tuple(range(start, start + seeds)),
                              acq, c.observation_revision, c.revision + 1))
        serial += 1
    result = replace(c, revision=c.revision + 1, proposals=(*c.proposals, *batch),
                     registry=tuple(registry[p] for p in sorted(registry)), rng_state=rng_state)
    validate_campaign(result)
    return result, _payload(result, batch, result.revision)


def propose(campaign: Campaign, k: int = 3, seeds: int = 2,
            acq: str = "ts") -> tuple[Campaign, dict[str, Any]]:
    """候補を選び seed を予約する。入力状態と外部ファイルは変更しない。"""
    import sys
    validate_campaign(campaign)
    _integer(seeds, 1)
    rng = DeterministicRNG(**campaign.rng_state)
    indices, notes = _select(campaign, k, acq, rng)
    result = _reserve(campaign, indices, seeds, acq, rng.rng_state)
    for note in notes:
        print(note, file=sys.stderr)
    return result


def replay(c: Campaign, revision: int) -> dict[str, Any]:
    _integer(revision, 1)
    batch = tuple(p for p in c.proposals if p.campaign_revision == revision)
    if not batch:
        raise ValueError("no proposal batch at this revision")
    return _payload(c, batch, revision)


def _summaries(c: Campaign, fits: Sequence[GPFit], coordinates: Array,
               count: int = 4096) -> dict[str, dict[str, Array]]:
    distributions = tuple(posterior(f, coordinates) for f in fits)
    rng = random.Random(0)
    samples = _samples(distributions, count, rng, latent=True)
    noise = normal_draws(rng, samples.size).reshape(samples.shape)
    predictive = samples + noise * np.asarray([f.s * math.sqrt(f.v) for f in fits])[None, :, None]
    result = {}
    for name in (*c.responses, "bal", "max"):
        draws = composite(samples, c.responses, name)
        future = composite(predictive, c.responses, name)
        if name in c.responses:
            p = distributions[c.responses.index(name)]
            mean, sd = p.mean, np.sqrt(np.maximum(np.diag(p.covariance), 0))
        elif name == "bal":
            mean = np.mean([p.mean for p in distributions], axis=0)
            sd = np.sqrt(np.sum([np.diag(p.covariance) for p in distributions], axis=0)) / len(distributions)
        else:
            mean, sd = draws.mean(axis=0), draws.std(axis=0)
        result[name] = {"mean": mean, "sd": sd, "latent": np.quantile(draws, [.025, .975], axis=0),
                        "predictive": np.quantile(future, [.025, .975], axis=0)}
    return result


def _fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}" if math.isfinite(value) else "算出不能"
    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(_fmt(v) for v in value)
    return str(value)


def report(campaign: Campaign, out: Any, fig_dir: Any = None) -> None:
    """独立 RNG による日本語報告。campaign は更新しない。"""
    from pathlib import Path
    c = campaign
    validate_campaign(c)
    fits = _fits(c)
    summaries = _summaries(c, fits, candidate_coordinates(c.space))
    observed = {(o.point, o.seed) for o in c.observations}
    pending = sum((p.point, s) not in observed for p in c.proposals for s in p.seeds)
    lines = [f"# 逐次設計: {c.campaign_id}", "", f"schema: {c.version} / revision: {c.revision}",
             f"観測数: {len(c.observations)} / 点数: {len({o.point for o in c.observations})} / 予約中 seed: {pending}",
             f"input_mode: {c.input_mode}", "予測は名目候補座標、mixture の学習は実現組成、registry の学習は登録因子。",
             "", "## 実験条件", "", "```json", _dump(c.context).rstrip(), "```", "", "## GP 診断", "",
             "等分散ノイズモデルのため頂点の大きな seed 分散が内部誤差を過大評価し得る。",
             "反復なしでは短い長さと大きなノイズを識別しにくい。", "",
             "|応答|m|s|a|ell|v|pure-error df|LML|jitter|境界|", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, f in zip(c.responses, fits):
        lines.append("|" + "|".join(_fmt(v) for v in (name, f.m, f.s, f.a, f.ell, f.v,
                      f.diagnostics["pure_error_df"], f.diagnostics["lml"], f.diagnostics["jitter"], f.diagnostics["boundary"])) + "|")
        if f.diagnostics["failures"]:
            lines.append(f"試行失敗 ({name}): {_dump(f.diagnostics['failures'])}")
    if not c.observations:
        lines.append("未観測: 事前分布による報告。EI は random に退避する。")
    names = tuple(summaries)
    lines += ["", "## 全候補（潜在応答）", "", "|候補|因子|point|観測 seed 数|" + "|".join(f"{n} mean / SD" for n in names) + "|",
              "|---|---|---|---|" + "---|" * len(names)]
    registry = {_factors(c.space, row["factors"]): int(cast(Any, row["point"])) for row in c.registry if row["factors"] is not None}
    for i, factors in enumerate(c.space.candidates):
        point = registry.get(_factors(c.space, factors))
        cells = [str(i), _dump(factors).replace("\n", " ").strip(), str(point), str(sum(o.point == point for o in c.observations))]
        cells += [f"{_fmt(summaries[n]['mean'][i])} / {_fmt(summaries[n]['sd'][i])}" for n in names]
        lines.append("|" + "|".join(cells) + "|")
    lines += ["", "## 目的別推奨", "", "区間は選択候補の潜在応答であり、最適位置や最小値の信頼区間ではない。",
              "|目的|候補|潜在95%等裾区間|将来1 seed 95%予測区間|", "|---|---|---|---|"]
    for name, summary in summaries.items():
        i = int(np.argmin(summary["mean"]))
        lines.append(f"|{name}|{i}|{_fmt(summary['latent'][:, i])}|{_fmt(summary['predictive'][:, i])}|")
    lines += ["", "## 提案履歴", "", "|proposal_id|観測 revision|acq|point|名目因子|seed: 状態|", "|---|---|---|---|---|---|"]
    for p in c.proposals:
        status = ", ".join(f"{s}: {'完了' if (p.point, s) in observed else '未完了'}" for s in p.seeds)
        lines.append(f"|{p.proposal_id}|{p.observation_revision}|{p.acquisition}|{p.point}|{p.factors}|{status}|")
    if c.input_mode == "mixture" and any(o.point == 7 for o in c.observations) and {"lay", "str", "spa"} <= set(c.responses):
        from training.mixture_doe import paired_delta
        lines += ["", "## p7 との共有 seed の観測対比（paired_delta）", "", "|応答|point|平均差|95%区間|", "|---|---|---|---|"]
        for name in ("lay", "str", "spa", "bal", "max"):
            for point, (mean, low, high) in paired_delta(c.observations, name, 7).items():
                lines.append(f"|{name}|{point}|{_fmt(mean)}|{_fmt((low, high))}|")
    if fig_dir is not None and c.space.simplex is not None and len(c.space.simplex) == 3:
        _figures(c, summaries, Path(fig_dir))
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _figures(c: Campaign, summaries: dict[str, dict[str, Array]], directory: Any) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    names = ("lay", "str", "spa") if c.input_mode == "mixture" else cast(tuple[str, ...], c.space.simplex)
    def xy(rows: Array) -> Array:
        return np.column_stack((rows[:, 1] + rows[:, 2] / 2, math.sqrt(3) * rows[:, 2] / 2))
    coords = xy(np.asarray([[row[n] for n in names] for row in c.space.candidates]))
    obs = observation_coordinates(c)
    obs = xy(obs[:, [c.space.factor_names.index(n) for n in names]])
    for number, (name, summary) in enumerate(summaries.items()):
        for stat in ("mean", "sd"):
            fig, ax = plt.subplots()
            try:
                values = summary[stat]
                if len(coords) >= 3 and np.linalg.matrix_rank(coords - coords[0]) == 2:
                    try:
                        artist = ax.tricontourf(coords[:, 0], coords[:, 1], values)
                    except (ValueError, RuntimeError):
                        artist = ax.scatter(coords[:, 0], coords[:, 1], c=values)
                else:
                    artist = ax.scatter(coords[:, 0], coords[:, 1], c=values)
                fig.colorbar(artist, ax=ax)
                ax.scatter(coords[:, 0], coords[:, 1], s=3, color="gray", label="nominal")
                if len(obs):
                    ax.scatter(obs[:, 0], obs[:, 1], marker="x", color="black", label="observed")
                rec = coords[int(np.argmin(summary["mean"]))]
                ax.scatter(*rec, marker="*", color="red", label="recommended")
                ax.set(title=f"{name} {stat}", aspect="equal")
                ax.legend()
                fig.savefig(directory / f"response_{number}_{stat}.png")
            finally:
                plt.close(fig)


_FIXED = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (.5, .5, 0.),
          (.5, 0., .5), (0., .5, .5), (1/3, 1/3, 1/3),
          (2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3))


def _simulation_space() -> Space:
    rows = [(i / 20, j / 20, (20-i-j) / 20) for i in range(21) for j in range(21-i)]
    rows.extend(_FIXED[6:])
    return Space(("lay", "str", "spa"), tuple(dict(zip(("lay", "str", "spa"), row)) for row in rows),
                 ("lay", "str", "spa"))


def _empty(space: Space, objective: str, seed: int, *, registry: tuple[dict[str, Any], ...] = ()) -> Campaign:
    return Campaign(1, "simulation", 0, 0, space, ("lay", "str", "spa"),
                    {"kind": objective, "responses": ["lay", "str", "spa"], "direction": "minimize"},
                    "mixture", {"response_scale": "synthetic"}, (), (), registry, {"seed": seed, "draws": 0})


def _surface(grid: Array, variant: str) -> tuple[Array, Array]:
    q = -.4 + .12 * np.sum((grid - [.40, .25, .35]) ** 2, axis=1)
    d = grid[:, 0] - grid[:, 1]
    truth = q[None, :] + np.asarray([-.06, .04, .02])[:, None] * d
    h = np.exp(-(1 - grid[:, 1]) / .025) if variant == "vertex_sharp" else (grid[:, 1] == 1).astype(float)
    if variant != "base":
        truth = truth + np.asarray([3., 0., .9])[:, None] * h
    return truth, h


def _coverage(summaries: dict[str, dict[str, Array]], truth: Array, responses: Sequence[str],
              vertex: NDArray[np.bool_], kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
    coverage, width = {}, {}
    for name, summary in summaries.items():
        actual = composite(truth, responses, name)
        low, high = summary[kind]
        coverage[name], width[name] = {}, {}
        for label, mask in (("vertex", vertex), ("interior", ~vertex)):
            count = int(mask.sum())
            covered = int(((low <= actual) & (actual <= high) & mask).sum())
            coverage[name][label] = {"count": count, "covered": covered, "rate": covered / count if count else None}
            width[name][label] = float((high-low)[mask].mean()) if count else None
    return coverage, width


def _slice_summaries(summaries: dict[str, dict[str, Array]], indices: Sequence[int]) -> dict[str, dict[str, Array]]:
    return {name: {stat: value[..., indices] for stat, value in row.items()} for name, row in summaries.items()}


def _aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in results:
        groups.setdefault(tuple(row[k] for k in ("condition", "objective", "method", "round")), []).append(row)
    output = []
    for key, rows in groups.items():
        entry: dict[str, Any] = dict(zip(("condition", "objective", "method", "round"), key))
        metrics = {name: [row[name] for row in rows] for name in ("regret", "near_fraction")}
        for kind in ("latent", "predictive"):
            for name in rows[0]["coverage"][kind]:
                for region in ("vertex", "interior"):
                    metrics[f"coverage.{kind}.{name}.{region}"] = [row["coverage"][kind][name][region]["rate"] for row in rows]
                    metrics[f"interval_width.{kind}.{name}.{region}"] = [row["interval_width"][kind][name][region] for row in rows]
        for name, values in metrics.items():
            a = np.asarray([v for v in values if v is not None])
            entry[name] = None if not len(a) else {"mean": float(a.mean()), "median": float(np.median(a)),
                         "p10": float(np.quantile(a, .1)), "p90": float(np.quantile(a, .9)),
                         "se": float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else None}
        # 同じ repeat の random との差。独立 SE ではなく対応差の SE。
        baseline = {r["repeat"]: r for r in groups[(key[0], key[1], "random", key[3])]}
        entry["paired_vs_random"] = {}
        for metric, values in metrics.items():
            differences = []
            for row, value in zip(rows, values):
                reference: Any = baseline[row["repeat"]]
                for part in metric.split("."):
                    reference = reference[part]
                if metric.startswith("coverage."):
                    reference = reference["rate"]
                if value is not None and reference is not None:
                    differences.append(value - reference)
            diff = np.asarray(differences)
            entry["paired_vs_random"][metric] = {"mean": float(diff.mean()) if len(diff) else None,
                "se": float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else None}
        output.append(entry)
    return output


def _retrospective(obs_path: Any, space: Space, directory: Any) -> list[dict[str, Any]]:
    from pathlib import Path
    path = Path(obs_path)
    observations = observations_from_json(path.read_text(encoding="utf-8"), ("lay", "str", "spa"))
    if {(o.point, o.seed) for o in observations} != {(p, s) for p in range(1, 13) for s in range(3)} or len(observations) != 36:
        raise ValueError("retrospective requires points 1..12 with seeds 0..2")
    registry = []
    families = {"layered": "lay", "structured": "str", "spaghetti": "spa"}
    for point in range(1, 13):
        spec = json.loads((path.parent / "specs" / f"spec_p{point}.json").read_text(encoding="utf-8"))
        components: list[dict[str, Any]] = spec["params"]["components"] if spec["family"] == "mixture" else [{"family": spec["family"], "weight": 1}]
        weights = {n: 0. for n in space.factor_names}
        for component in components:
            weights[families[component["family"]]] += _number(component["weight"])
        total = sum(weights.values())
        factors = {n: v / total for n, v in weights.items()}
        key = _factors(space, factors)
        cid = next((i for i, row in enumerate(space.candidates) if _factors(space, row) == key), None)
        registry.append({"point": point, "candidate_id": cid, "factors": factors})
    output = []
    for objective in ("bal", "max"):
        c = replace(_empty(space, objective, 20260926, registry=tuple(registry)),
                    campaign_id=f"retrospective_{objective}", context={"response_scale": "excess_target"})
        for point in range(1, 13):
            c = observe(c, tuple(o for o in observations if o.point == point))
            if point not in (6, 12):
                continue
            snapshot = replace(c, rng_state={"seed": 20260926, "draws": 0})
            proposed, payload = propose(snapshot, 3, 2)
            stem = f"retrospective_{objective}_{point}"
            (directory / f"{stem}_proposal.json").write_text(_dump(payload), encoding="utf-8")
            report(proposed, directory / f"{stem}.md")
            (directory / f"{stem}_campaign.json").write_text(campaign_to_json(snapshot), encoding="utf-8")
            fits = _fits(snapshot)
            summary = _summaries(snapshot, fits, candidate_coordinates(space))
            entry: dict[str, Any] = {"objective": objective, "points": point, "proposals": payload,
                                    "posterior": {n: {s: v.tolist() for s, v in row.items()} for n, row in summary.items()}}
            entry["observed_means"] = {
                str(p): {name: float(composite(np.asarray([[o.y[r] for o in snapshot.observations if o.point == p]
                        for r in c.responses]), c.responses, name).mean())
                        for name in (*c.responses, "bal", "max")}
                for p in range(1, point + 1)}
            if point == 6:
                remaining = tuple(o for o in observations if o.point > 6)
                coordinates = np.asarray([o.x for o in remaining])
                forecast = _summaries(snapshot, fits, coordinates)
                actual = np.asarray([[o.y[r] for o in remaining] for r in c.responses])
                entry["heldout_coverage"], entry["heldout_interval_width"] = _coverage(
                    forecast, actual, c.responses, coordinates.max(axis=1) >= .95, "predictive")
            output.append(entry)
    return output


def simulate(*, surface: str = "scheffe", rounds: int = 8, k: int = 3, repeats: int = 20,
             seeds: int = 2, seed: int = 20260926, out: Any = "sim.md", obs: Any = None,
             summary_samples: int = 512) -> dict[str, Any]:
    """共通の事前生成ノイズで全条件・全手法を比較する。実機は起動しない。"""
    from pathlib import Path
    import platform
    from training.mixture_doe import Observation
    if surface != "scheffe":
        raise ValueError("unknown surface")
    for value in (rounds, k, repeats, seeds, summary_samples):
        _integer(value, 1)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    space = _simulation_space()
    grid = candidate_coordinates(space)
    fixed = [next(i for i, row in enumerate(grid) if tuple(row) == pattern) for pattern in _FIXED]
    conditions: list[tuple[str, float | str]] = [(variant, sd) for variant in ("base", "vertex_sharp", "vertex_discontinuous") for sd in (0., .005, .03)]
    conditions.append(("vertex_sharp", "heteroscedastic"))
    methods = ("ts", "ei", "ucb", "random", "fixed")
    results: list[dict[str, Any]] = []
    budget = rounds * k * seeds
    for variant, noise_condition in conditions:
        truth, h = _surface(grid, variant)
        sd = np.full_like(truth, .005 if isinstance(noise_condition, str) else noise_condition)
        if isinstance(noise_condition, str):
            sd[0] += .125 * h
        for repeat in range(repeats):
            noise_rng = random.Random(seed + repeat)
            noise = np.asarray([normal_draw(noise_rng) for _ in range(len(grid) * budget * 3)]).reshape(len(grid), budget, 3)
            noise *= sd.T[:, None, :]
            for objective in ("bal", "max"):
                true_objective = composite(truth, space.factor_names, objective)
                optimum = float(true_objective.min())
                empty = _empty(space, objective, seed + repeat)
                prior_fits = _fits(empty)
                prior = _summaries(empty, prior_fits, grid, summary_samples)
                for method in methods:
                    c, models, summaries = empty, prior_fits, prior
                    rng = DeterministicRNG(seed + repeat)
                    near = 0
                    for round_index in range(rounds):
                        notes: list[str] = []
                        if method == "fixed":
                            indices = [fixed[(round_index*k+j) % 10] for j in range(k)]
                        else:
                            indices, notes = _select(c, k, method, rng, models)
                        c, payload = _reserve(c, indices, seeds, method, rng.rng_state)
                        rows, sample_indices = [], []
                        for p in payload["proposals"]:
                            cid = p["candidate_id"]
                            for replicate in p["seeds"]:
                                y = truth[:, cid] + noise[cid, replicate]
                                rows.append(Observation(p["point"], replicate, tuple(grid[cid]),
                                                        dict(zip(c.responses, y.tolist())), None, 0))
                                sample_indices.append(cid)
                                near += int(true_objective[cid] - optimum <= .01)
                        frozen = _slice_summaries(summaries, sample_indices)
                        actual = np.asarray([[o.y[r] for o in rows] for r in c.responses])
                        pc, pw = _coverage(frozen, actual, c.responses, grid[sample_indices].max(axis=1) >= .95, "predictive")
                        c = observe(c, rows)
                        models = _fits(c)
                        summaries = _summaries(c, models, grid, summary_samples)
                        lc, lw = _coverage(summaries, truth, c.responses, grid.max(axis=1) >= .95, "latent")
                        rec = int(np.argmin(summaries[objective]["mean"]))
                        results.append({"condition": f"{variant}:{noise_condition}", "objective": objective,
                            "method": method, "repeat": repeat, "round": round_index+1, "budget": len(c.observations),
                            "diagnostics": notes,
                            "recommended": rec, "regret": float(true_objective[rec]-optimum),
                            "near_fraction": near / len(c.observations), "coverage": {"predictive": pc, "latent": lc},
                            "interval_width": {"predictive": pw, "latent": lw}})
    path = Path(out)
    config = {"surface": surface, "rounds": rounds, "k": k, "repeats": repeats, "seeds": seeds, "seed": seed,
              "summary_samples": summary_samples,
              "out": str(out), "obs": None if obs is None else str(obs), "repeat_seeds": [seed+r for r in range(repeats)],
              "report_seed": 0, "retrospective_seed": 20260926, "candidates": list(space.candidates),
              "noise_conditions": conditions, "methods": methods, "objectives": ["bal", "max"],
              "coefficients": {"center": [.4, .25, .35], "intercept": -.4, "quadratic": .12,
                               "response_linear": [-.06, .04, .02], "vertex": [3., 0., .9], "decay": .025,
                               "heteroscedastic_lay": [.005, .125], "near_threshold": .01},
              "environment": {"python": platform.python_version(), "numpy": np.__version__}}
    output: dict[str, Any] = {"version": 1, "config": config, "results": results, "summary": _aggregate(results)}
    path.parent.mkdir(parents=True, exist_ok=True)
    if obs is not None:
        output["retrospective"] = _retrospective(obs, space, path.parent)
    text = ["# 逐次設計の合成検証", "", "頂点急変・頂点不連続は厳密な Scheffé 二次ではない。",
            "予測区間は観測前に凍結。潜在区間は全候補で評価。頂点は max(x) >= 0.95。",
            "平均差は同じ repeat の random との差。1反復の SE は算出不能。過信時にも実機へ自動進行しない。", "",
            "|条件|目的|手法|round|regret 平均 / 中央 / p10 / p90|平均差 / paired SE|近傍予算比率 平均|", "|---|---|---|---|---|---|---|"]
    for row in output["summary"]:
        stats = row["regret"]
        paired = row["paired_vs_random"]["regret"]
        text.append(f"|{row['condition']}|{row['objective']}|{row['method']}|{row['round']}|"
                    + _fmt([stats[n] for n in ("mean", "median", "p10", "p90")]) + "|"
                    + _fmt([paired["mean"], paired["se"]]) + f"|{_fmt(row['near_fraction']['mean'])}|")
    text += ["", "## 校正・区間幅（反復単位の集計）", "", "各応答・合成の頂点/内部別平均、中央値、10/90百分位、SE。", "", "```json",
             _dump(output["summary"]).rstrip(), "```"]
    if obs is not None:
        text += ["", "## 遡及評価", "", "6点・12点の bal/max 提案と報告を同じディレクトリに保存。反実仮想 regret は計算しない。",
                 "```json", _dump(output["retrospective"]).rstrip(), "```"]
    path.with_suffix(".json").write_text(_dump(output), encoding="utf-8")
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return output


def _atomic_write(path: Any, text: str) -> None:
    import os
    import tempfile
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    from pathlib import Path
    import sys
    parser = argparse.ArgumentParser(description="有限候補の逐次ベイズ実験計画")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--campaign", required=True, type=Path)
    init.add_argument("--space", required=True, type=Path)
    init.add_argument("--responses", required=True)
    init.add_argument("--objective", required=True)
    init.add_argument("--seed", type=int, default=20260926)
    init.add_argument("--input-mode", choices=("mixture", "registry"), default="mixture")
    init.add_argument("--registry", type=Path)
    init.add_argument("--context", type=Path)
    ob = commands.add_parser("observe")
    ob.add_argument("--campaign", required=True, type=Path)
    ob.add_argument("--obs", required=True, type=Path)
    pr = commands.add_parser("propose")
    pr.add_argument("--campaign", required=True, type=Path)
    pr.add_argument("--k", type=int, default=3)
    pr.add_argument("--seeds", type=int, default=2)
    pr.add_argument("--acq", choices=("ts", "ei", "ucb", "random"), default="ts")
    pr.add_argument("--replay-revision", type=int)
    rep = commands.add_parser("report")
    rep.add_argument("--campaign", required=True, type=Path)
    rep.add_argument("--out", required=True, type=Path)
    rep.add_argument("--fig-dir", type=Path)
    sim = commands.add_parser("simulate")
    sim.add_argument("--surface", choices=("scheffe",), default="scheffe")
    for name, default in (("rounds", 8), ("k", 3), ("seeds", 2), ("repeats", 20), ("seed", 20260926),
                          ("summary-samples", 512)):
        sim.add_argument(f"--{name}", type=int, default=default)
    sim.add_argument("--out", required=True, type=Path)
    sim.add_argument("--obs", type=Path)
    try:
        args = parser.parse_args(argv)
        if args.command == "simulate":
            simulate(surface=args.surface, rounds=args.rounds, k=args.k, seeds=args.seeds,
                     repeats=args.repeats, seed=args.seed, out=args.out, obs=args.obs,
                     summary_samples=args.summary_samples)
            return 0
        path = args.campaign
        if args.command == "init":
            if path.exists():
                raise ValueError("campaign already exists")
            responses = tuple(args.responses.split(","))
            kind = args.objective if args.objective in ("bal", "max") else "response"
            c = Campaign(1, path.stem, 0, 0, space_from_json(args.space.read_text(encoding="utf-8")), responses,
                         {"kind": kind, "responses": list(responses) if kind != "response" else [args.objective], "direction": "minimize"},
                         args.input_mode, json.loads(args.context.read_text(encoding="utf-8")) if args.context else None,
                         (), (), (), {"seed": args.seed, "draws": 0})
            if args.registry:
                rows = json.loads(args.registry.read_text(encoding="utf-8"))
                if not isinstance(rows, list):
                    raise ValueError("registry must be an array")
                c = replace(c, registry=tuple(rows))
            _atomic_write(path, campaign_to_json(c))
        else:
            c = campaign_from_json(path.read_text(encoding="utf-8"))
            if args.command == "observe":
                updated = observe(c, observations_from_json(args.obs.read_text(encoding="utf-8"), c.responses))
                if updated is not c:
                    _atomic_write(path, campaign_to_json(updated))
            elif args.command == "propose":
                _integer(args.k, 1)
                _integer(args.seeds, 1)
                if args.replay_revision is not None:
                    payload = replay(c, args.replay_revision)
                else:
                    updated, payload = propose(c, args.k, args.seeds, args.acq)
                    _atomic_write(path, campaign_to_json(updated))
                print(_dump(payload), end="")
            else:
                # 出力先が campaign 自身なら読み取り専用の契約を壊すため拒否する。
                if args.out.resolve() == path.resolve():
                    raise ValueError("report output must differ from campaign")
                report(c, args.out, args.fig_dir)
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
    except (ValueError, TypeError, KeyError, OSError, ArithmeticError, np.linalg.LinAlgError) as exc:
        print(f"seqdesign: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
