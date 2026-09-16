"""Scheffé mixture analysis for mixed CFG-family training runs.

Design: docs/design/mixture_doe.md section 4.  The module counts the
realized family composition of a dataset, collects one NLL/token
observation per (design point, training seed), fits Scheffé mixture
polynomials with seed block effects, searches the simplex for composite
optima, and writes a Markdown report plus ternary contour figures.

Torch-free: imports only the standard library, numpy, matplotlib (Agg)
and ``cfg_reducer``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

from cfg_reducer.families.mixture import component_for  # noqa: E402
from cfg_reducer.generator_types import GeneratorSpec  # noqa: E402

FAMILIES = ("layered", "structured", "spaghetti")
TARGETS = ("lay", "str", "spa")
COMPOSITES = ("bal", "max", "min")

# Two-sided 95% t quantiles for df = 1..10; 1.96 beyond.
_T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228)


# ──────────────────────────────────────────────
#  Composition
# ──────────────────────────────────────────────

def _component_family(spec: dict[str, Any], seed: int) -> str:
    family = spec["family"]
    if family != "mixture":
        return str(family)
    child = GeneratorSpec(family, spec["num_nodes"], spec["params"])
    index = component_for(child, seed)
    component = spec["params"]["components"][index]
    return str(component["family"])


def realized_composition(dataset_dir: Path, split: str = "train") -> dict[str, int]:
    """Count the realized family of every sample in one split."""
    manifest = json.loads(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    counts = {family: 0 for family in FAMILIES}
    for sample in manifest["splits"][split]["samples"]:
        payload = json.loads(
            (dataset_dir / split / f"{sample['sample_id']}.json").read_text(
                encoding="utf-8"
            )
        )
        generator = payload["provenance"]["generator"]
        family = _component_family(generator["config"]["spec"],
                                   generator["seed"])
        if family not in counts:
            raise ValueError(f"unknown family: {family!r}")
        counts[family] += 1
    return counts


def fractions(counts: Mapping[str, int]) -> tuple[float, float, float]:
    """Counts in FAMILIES order as a composition summing to 1."""
    total = sum(counts[family] for family in FAMILIES)
    if total <= 0:
        raise ValueError("composition counts are empty")
    return (counts[FAMILIES[0]] / total, counts[FAMILIES[1]] / total,
            counts[FAMILIES[2]] / total)


# ──────────────────────────────────────────────
#  Observations
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class Observation:
    point: int
    seed: int
    x: tuple[float, float, float]
    y: dict[str, float]
    wf: float | None
    n_train: int


def nll_per_token(scores_path: Path) -> float:
    """Sum(nll) / sum(n_tokens) over a test_scores.jsonl file."""
    total_nll = 0.0
    total_tokens = 0
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total_nll += float(row["nll"])
        total_tokens += int(row["n_tokens"])
    return total_nll / total_tokens


def _run_name(prefix: str, size: int, point: int, suffix: str, seed: int) -> str:
    return f"{prefix}_s{size}_p{point}{suffix}_mask_n{size}_s{seed}"


def collect(
    runs_dir: Path,
    data_dir: Path,
    points: Sequence[int],
    seeds: Sequence[int],
    *,
    size: int = 24,
    prefix: str = "d",
) -> tuple[list[Observation], list[str]]:
    """Collect one observation per (point, seed) from a runs directory."""
    observations: list[Observation] = []
    missing: list[str] = []
    composition_cache: dict[int, tuple[tuple[float, float, float], int]] = {}
    for point in points:
        if point not in composition_cache:
            counts = realized_composition(data_dir / f"d{size}_p{point}")
            composition_cache[point] = (fractions(counts), sum(counts.values()))
        x, n_train = composition_cache[point]
        for seed in seeds:
            base = _run_name(prefix, size, point, "", seed)
            paths = {
                "lay": runs_dir / base / "test_scores.jsonl",
                "str": runs_dir / _run_name(prefix, size, point, "2str", seed)
                / "test_scores.jsonl",
                "spa": runs_dir / _run_name(prefix, size, point, "2spa", seed)
                / "test_scores.jsonl",
            }
            if not all(path.exists() for path in paths.values()):
                missing.append(base)
                continue
            y = {target: nll_per_token(path) for target, path in paths.items()}
            eval_path = runs_dir / base / "eval.json"
            wf: float | None = None
            if eval_path.exists():
                wf = float(json.loads(
                    eval_path.read_text(encoding="utf-8")
                )["well_formed_rate"])
            observations.append(
                Observation(point, seed, x, y, wf, n_train)
            )
    return observations, missing


def observations_to_json(obs: Sequence[Observation]) -> str:
    rows = [
        {
            "point": o.point,
            "seed": o.seed,
            "x": list(o.x),
            "y": dict(o.y),
            "wf": o.wf,
            "n_train": o.n_train,
        }
        for o in obs
    ]
    return json.dumps(rows, sort_keys=True, indent=2)


def observations_from_json(text: str) -> list[Observation]:
    rows = json.loads(text)
    return [
        Observation(
            int(row["point"]),
            int(row["seed"]),
            (float(row["x"][0]), float(row["x"][1]), float(row["x"][2])),
            {key: float(value) for key, value in row["y"].items()},
            None if row["wf"] is None else float(row["wf"]),
            int(row["n_train"]),
        )
        for row in rows
    ]


# ──────────────────────────────────────────────
#  Design matrix and least squares
# ──────────────────────────────────────────────

def design_matrix(
    xs: Sequence[tuple[float, float, float]],
    blocks: Sequence[int],
    model: str,
) -> tuple[np.ndarray, list[str]]:
    """Scheffé design matrix with seed block dummies."""
    if model not in ("quadratic", "special_cubic"):
        raise ValueError(f"unknown model: {model!r}")
    rows: list[list[float]] = []
    for x1, x2, x3 in xs:
        row = [x1, x2, x3, x1 * x2, x1 * x3, x2 * x3]
        if model == "special_cubic":
            row.append(x1 * x2 * x3)
        rows.append(row)
    labels = ["x1", "x2", "x3", "x1x2", "x1x3", "x2x3"]
    if model == "special_cubic":
        labels.append("x1x2x3")
    levels = sorted(set(blocks))
    for level in levels[1:]:
        labels.append(f"block_{level}")
        for row, block in zip(rows, blocks):
            row.append(1.0 if block == level else 0.0)
    return np.asarray(rows, dtype=float), labels


def _response_value(obs: Observation, response: str) -> float:
    if response in TARGETS:
        return obs.y[response]
    values = [obs.y[target] for target in TARGETS]
    if response == "bal":
        return sum(values) / len(values)
    if response == "max":
        return max(values)
    if response == "min":
        return min(values)
    raise ValueError(f"unknown response: {response!r}")


@dataclass(frozen=True)
class Fit:
    model: str
    labels: tuple[str, ...]
    coef: tuple[float, ...]
    se: tuple[float, ...]
    df_resid: int
    r2: float
    sigma: float
    lack_of_fit_f: float | None
    lack_of_fit_p: float | None
    block_levels: tuple[int, ...]


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (NR style)."""
    max_it, eps, fpmin = 200, 3.0e-14, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def f_sf(f_value: float, d1: int, d2: int) -> float:
    """Upper tail P(F_{d1,d2} > f_value)."""
    if f_value <= 0.0:
        return 1.0
    x = d2 / (d2 + d1 * f_value)
    return _betai(d2 / 2.0, d1 / 2.0, x)


def fit(
    obs: Sequence[Observation],
    response: str,
    model: str = "quadratic",
) -> Fit:
    """Least-squares Scheffé fit with block effects and lack-of-fit."""
    if not obs:
        raise ValueError("no observations to fit")
    values = [_response_value(o, response) for o in obs]
    matrix, labels = design_matrix(
        [o.x for o in obs], [o.seed for o in obs], model
    )
    y = np.asarray(values, dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(matrix, y, rcond=None)
    residual = y - matrix @ coefficients
    rss = float(residual @ residual)
    n, k = matrix.shape
    df_resid = n - k
    if df_resid > 0:
        sigma2 = rss / df_resid
        covariance = sigma2 * np.linalg.pinv(matrix.T @ matrix)
        se = np.sqrt(np.diag(covariance))
        sigma = math.sqrt(sigma2)
    else:
        se = np.full(k, math.nan)
        sigma = math.nan
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / total if total > 0 else 1.0

    groups: dict[int, list[float]] = defaultdict(list)
    for o, value in zip(obs, values):
        groups[o.point].append(value)
    pure_error = 0.0
    for group in groups.values():
        mean = sum(group) / len(group)
        pure_error += sum((value - mean) ** 2 for value in group)
    df_pe = n - len(groups)
    lack_of_fit = rss - pure_error
    df_lof = df_resid - df_pe
    f_value: float | None = None
    p_value: float | None = None
    if df_lof > 0 and df_pe > 0:
        if pure_error > 0:
            f_calc = (lack_of_fit / df_lof) / (pure_error / df_pe)
        else:
            f_calc = math.inf if lack_of_fit > 0 else 0.0
        f_value = f_calc
        p_value = f_sf(f_calc, df_lof, df_pe)

    return Fit(
        model=model,
        labels=tuple(labels),
        coef=tuple(float(c) for c in coefficients),
        se=tuple(float(s) for s in se),
        df_resid=int(df_resid),
        r2=float(r2),
        sigma=float(sigma),
        lack_of_fit_f=None if f_value is None else float(f_value),
        lack_of_fit_p=None if p_value is None else float(p_value),
        block_levels=tuple(sorted({o.seed for o in obs})),
    )


_POLY_TERMS = ("x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x1x2x3")


def predict(fit: Fit, x: tuple[float, float, float]) -> float:
    """Polynomial value at x plus the mean block effect (baseline 0)."""
    x1, x2, x3 = x
    terms = {
        "x1": x1, "x2": x2, "x3": x3,
        "x1x2": x1 * x2, "x1x3": x1 * x3, "x2x3": x2 * x3,
        "x1x2x3": x1 * x2 * x3,
    }
    value = 0.0
    block_sum = 0.0
    for label, coefficient in zip(fit.labels, fit.coef):
        if label in terms:
            value += coefficient * terms[label]
        elif label.startswith("block_"):
            block_sum += coefficient
    if fit.block_levels:
        value += block_sum / len(fit.block_levels)
    return value


# ──────────────────────────────────────────────
#  Simplex search
# ──────────────────────────────────────────────

def simplex_grid(step: float = 0.05) -> list[tuple[float, float, float]]:
    """All step multiples on the simplex (231 points at step 0.05)."""
    divisions = round(1.0 / step)
    grid: list[tuple[float, float, float]] = []
    for i in range(divisions + 1):
        for j in range(divisions + 1 - i):
            k = divisions - i - j
            grid.append((
                round(i * step, 10),
                round(j * step, 10),
                round(k * step, 10),
            ))
    return grid


def _composite(predictions: Sequence[float], composite: str) -> float:
    if composite == "bal":
        return sum(predictions) / len(predictions)
    if composite == "max":
        return max(predictions)
    if composite == "min":
        return min(predictions)
    raise ValueError(f"unknown composite: {composite!r}")


def optimize(
    fits: Mapping[str, Fit],
    composite: str,
    step: float = 0.05,
) -> tuple[tuple[float, float, float], float]:
    """Grid argmin of a composite response (ties: first grid point)."""
    if composite not in COMPOSITES:
        raise ValueError(f"unknown composite: {composite!r}")
    best_x: tuple[float, float, float] | None = None
    best_value = math.inf
    for x in simplex_grid(step):
        predictions = [predict(fits[target], x) for target in TARGETS]
        value = _composite(predictions, composite)
        if best_x is None or value < best_value:
            best_x, best_value = x, value
    assert best_x is not None
    return best_x, best_value


# ──────────────────────────────────────────────
#  Paired deltas
# ──────────────────────────────────────────────

def _t_quantile(df: int) -> float:
    if 1 <= df <= len(_T95):
        return _T95[df - 1]
    return 1.96


def paired_delta(
    obs: Sequence[Observation],
    response: str,
    ref_point: int,
) -> dict[int, tuple[float, float, float]]:
    """(mean, low, high) paired differences against ref_point, 95% CI."""
    by_point: dict[int, dict[int, float]] = defaultdict(dict)
    for o in obs:
        by_point[o.point][o.seed] = _response_value(o, response)
    if ref_point not in by_point:
        raise ValueError(f"reference point {ref_point} not in observations")
    reference = by_point[ref_point]
    result: dict[int, tuple[float, float, float]] = {}
    for point in sorted(by_point):
        if point == ref_point:
            continue
        seeds = sorted(set(by_point[point]) & set(reference))
        diffs = [by_point[point][seed] - reference[seed] for seed in seeds]
        if not diffs:
            continue
        mean = sum(diffs) / len(diffs)
        if len(diffs) == 1:
            result[point] = (mean, math.nan, math.nan)
            continue
        variance = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
        half = _t_quantile(len(diffs) - 1) * math.sqrt(variance) / math.sqrt(
            len(diffs)
        )
        result[point] = (mean, mean - half, mean + half)
    return result


# ──────────────────────────────────────────────
#  Report and figures
# ──────────────────────────────────────────────

def _ternary(x: tuple[float, float, float]) -> tuple[float, float]:
    x1, x2, x3 = x
    return x2 + x3 / 2.0, x3 * math.sqrt(3.0) / 2.0


def _plot_ternary(
    path: Path,
    grid: Sequence[tuple[float, float, float]],
    values: Sequence[float],
    observed: Sequence[tuple[float, float, float]],
    title: str,
    optimum: tuple[float, float, float] | None,
) -> None:
    points = [_ternary(x) for x in grid]
    tri = Triangulation(
        [p[0] for p in points], [p[1] for p in points]
    )
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    contour = ax.tricontourf(tri, np.asarray(values, dtype=float), levels=14)
    fig.colorbar(contour, ax=ax)
    corners = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    offsets = ((-0.06, -0.04), (0.03, -0.04), (-0.04, 0.04))
    for family, corner, offset in zip(FAMILIES, corners, offsets):
        cx, cy = _ternary(corner)
        ax.text(cx + offset[0], cy + offset[1], family, fontsize=8)
    if observed:
        obs_points = [_ternary(x) for x in observed]
        ax.plot([p[0] for p in obs_points], [p[1] for p in obs_points],
                "k.", markersize=4)
    if optimum is not None:
        ox, oy = _ternary(optimum)
        ax.plot([ox], [oy], "r*", markersize=12)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def report(
    obs: Sequence[Observation],
    fits: Mapping[str, Fit],
    composites: Mapping[str, tuple[tuple[float, float, float], float]],
    out_md: Path,
    *,
    ref_point: int = 7,
    fig_dir: Path | None = None,
    step: float = 0.05,
) -> None:
    """Write the Markdown report and optional ternary contour figures."""
    lines: list[str] = ["# Mixture DoE report", ""]

    lines.append("## 1. Observations")
    lines.append("")
    lines.append(
        "| point | seed | x_lay | x_str | x_spa | n_train | y_lay | y_str "
        "| y_spa | wf |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for o in sorted(obs, key=lambda o: (o.point, o.seed)):
        lines.append(
            f"| {o.point} | {o.seed} | {o.x[0]:.3f} | {o.x[1]:.3f} | "
            f"{o.x[2]:.3f} | {o.n_train} | {o.y['lay']:.4f} | "
            f"{o.y['str']:.4f} | {o.y['spa']:.4f} | {_fmt(o.wf, 3)} |"
        )
    lines.append("")

    by_point: dict[int, list[Observation]] = defaultdict(list)
    for o in obs:
        by_point[o.point].append(o)
    lines.append("## 2. Point means")
    lines.append("")
    header = ["point", "x_lay", "x_str", "x_spa", "n"]
    for target in TARGETS:
        header += [f"{target}_mean", f"{target}_sd"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for point in sorted(by_point):
        group = by_point[point]
        row = [str(point), f"{group[0].x[0]:.3f}", f"{group[0].x[1]:.3f}",
               f"{group[0].x[2]:.3f}", str(len(group))]
        for target in TARGETS:
            mean, sd = _mean_sd([o.y[target] for o in group])
            row += [f"{mean:.4f}", f"{sd:.4f}"]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## 3. Coefficient fits")
    lines.append("")
    for name, current in fits.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            f"model={current.model}, R2={current.r2:.4f}, "
            f"sigma={_fmt(current.sigma)}, df_resid={current.df_resid}, "
            f"lack_of_fit_F={_fmt(current.lack_of_fit_f)}, "
            f"lack_of_fit_p={_fmt(current.lack_of_fit_p)}"
        )
        lines.append("")
        lines.append("| term | coef | se | t |")
        lines.append("| --- | --- | --- | --- |")
        for label, coefficient, se in zip(
            current.labels, current.coef, current.se
        ):
            t_value = coefficient / se if se else math.nan
            lines.append(
                f"| {label} | {coefficient:.4f} | {_fmt(se)} | "
                f"{_fmt(t_value)} |"
            )
        lines.append("")

    lines.append("## 4. Paired deltas")
    lines.append("")
    if ref_point in by_point:
        for target in TARGETS:
            lines.append(f"### {target}")
            lines.append("")
            lines.append("| point | mean | low | high |")
            lines.append("| --- | --- | --- | --- |")
            for point, (mean, low, high) in sorted(
                paired_delta(obs, target, ref_point).items()
            ):
                lines.append(
                    f"| {point} | {mean:.4f} | {_fmt(low)} | {_fmt(high)} |"
                )
            lines.append("")
    else:
        lines.append(f"reference point {ref_point} not observed")
        lines.append("")

    lines.append("## 5. Optima")
    lines.append("")
    for composite in COMPOSITES:
        if composite not in composites:
            continue
        x, value = composites[composite]
        predictions = {target: predict(fits[target], x) for target in TARGETS}
        lines.append(
            f"- **{composite}**: x=({x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f}), "
            f"value={value:.4f}, "
            f"pred=({predictions['lay']:.4f}, {predictions['str']:.4f}, "
            f"{predictions['spa']:.4f})"
        )
    lines.append("")

    figure_names: list[str] = []
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)
        grid = simplex_grid(step)
        observed = [o.x for o in obs]
        for name in (*TARGETS, *COMPOSITES):
            if name in TARGETS:
                if name not in fits:
                    continue
                values = [predict(fits[name], x) for x in grid]
            else:
                if not all(target in fits for target in TARGETS):
                    continue
                values = [
                    _composite([predict(fits[t], x) for t in TARGETS], name)
                    for x in grid
                ]
            optimum = composites[name][0] if name in composites else None
            filename = f"mixture_doe_{name}.png"
            _plot_ternary(
                fig_dir / filename, grid, values, observed, name, optimum
            )
            figure_names.append(filename)
        lines.append("### Figures")
        lines.append("")
        for filename in figure_names:
            lines.append(f"![{filename}]({fig_dir / filename})")
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def _parse_points(text: str) -> list[int]:
    if "-" in text:
        start, stop = text.split("-", 1)
        return list(range(int(start), int(stop) + 1))
    return [int(item) for item in text.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.mixture_doe")
    sub = parser.add_subparsers(dest="command", required=True)

    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--runs", type=Path, required=True)
    collect_parser.add_argument("--data", type=Path, required=True)
    collect_parser.add_argument("--points", required=True)
    collect_parser.add_argument("--seeds", required=True)
    collect_parser.add_argument("--size", type=int, default=24)
    collect_parser.add_argument("--prefix", default="d")
    collect_parser.add_argument("--out", type=Path, required=True)

    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--obs", type=Path, required=True)
    fit_parser.add_argument("--out", type=Path, required=True)
    fit_parser.add_argument("--fig-dir", type=Path, default=None)
    fit_parser.add_argument("--model", default="quadratic",
                            choices=["quadratic", "special_cubic"])
    fit_parser.add_argument("--step", type=float, default=0.05)
    fit_parser.add_argument("--ref-point", type=int, default=7)

    composition_parser = sub.add_parser("composition")
    composition_parser.add_argument("--dataset", type=Path, required=True)
    composition_parser.add_argument("--split", default="train")

    args = parser.parse_args(argv)

    if args.command == "collect":
        observations, missing = collect(
            args.runs, args.data,
            _parse_points(args.points), _parse_points(args.seeds),
            size=args.size, prefix=args.prefix,
        )
        args.out.write_text(observations_to_json(observations),
                            encoding="utf-8")
        print(f"collected {len(observations)} observations")
        for name in missing:
            print(f"missing {name}")
        return 0

    if args.command == "fit":
        observations = observations_from_json(
            args.obs.read_text(encoding="utf-8")
        )
        fits = {target: fit(observations, target, args.model)
                for target in TARGETS}
        fits.update({composite: fit(observations, composite, args.model)
                     for composite in COMPOSITES})
        optima = {composite: optimize(fits, composite, args.step)
                  for composite in COMPOSITES}
        report(observations, fits, optima, args.out,
               ref_point=args.ref_point, fig_dir=args.fig_dir, step=args.step)
        print(f"wrote report to {args.out}")
        return 0

    counts = realized_composition(args.dataset, args.split)
    frac = fractions(counts)
    payload = {
        "counts": counts,
        "fractions": dict(zip(FAMILIES, frac)),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
