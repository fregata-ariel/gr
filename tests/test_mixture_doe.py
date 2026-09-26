"""Fast, torch-free tests for training/mixture_doe.py."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from cfg_reducer.dataset_v2 import main as dataset_main
from cfg_reducer.families.mixture import component_for
from cfg_reducer.generate_v2 import spec_to_json
from cfg_reducer.generator_types import GeneratorSpec
from training import mixture_doe
from training.mixture_doe import (
    COMPOSITES, FAMILIES, TARGETS, Observation, design_matrix, f_sf,
    fit, observations_from_json, observations_to_json, optimize,
    paired_delta, predict, realized_composition, report, simplex_grid,
)

TRUE_COEF = (0.7, 0.9, 1.0, -0.6, -0.4, 0.2)
BLOCKS = (0.0, 0.01, -0.01)
DESIGN = (
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
    (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
    (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
)


def surface(x: tuple[float, float, float]) -> float:
    x1, x2, x3 = x
    b = TRUE_COEF
    return (b[0] * x1 + b[1] * x2 + b[2] * x3
            + b[3] * x1 * x2 + b[4] * x1 * x3 + b[5] * x2 * x3)


def synth_obs(noise: float = 0.002) -> list[Observation]:
    rng = random.Random(0)
    obs: list[Observation] = []
    for point, x in enumerate(DESIGN, start=1):
        for seed in range(3):
            value = surface(x) + BLOCKS[seed] + rng.gauss(0.0, noise)
            obs.append(
                Observation(point, seed, x, {t: value for t in TARGETS},
                            None, 100)
            )
    return obs


def true_fit() -> mixture_doe.Fit:
    return mixture_doe.Fit(
        "quadratic", ("x1", "x2", "x3", "x1x2", "x1x3", "x2x3"),
        TRUE_COEF, (0.0,) * 6, 22, 1.0, 0.0, None, None, (0, 1, 2),
    )


def test_f_sf_values():
    assert f_sf(1.0, 1, 10) == pytest.approx(0.3409, abs=1e-3)
    assert f_sf(4.9646, 1, 10) == pytest.approx(0.050, abs=2e-3)
    assert f_sf(0.0, 3, 5) == 1.0


def test_design_matrix_labels_and_shapes():
    xs = list(DESIGN[:4])
    blocks = [0, 0, 1, 1]
    matrix, labels = design_matrix(xs, blocks, "quadratic")
    assert matrix.shape == (4, 7)
    assert labels == ["x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "block_1"]
    cubic, cubic_labels = design_matrix(xs, blocks, "special_cubic")
    assert cubic.shape == (4, 8)
    assert cubic_labels == [
        "x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x1x2x3", "block_1",
    ]
    assert cubic[:, 6].tolist() == [
        x[0] * x[1] * x[2] for x in xs
    ]
    assert matrix[:, -1].tolist() == [0.0, 0.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        design_matrix(xs, blocks, "linear")


def test_synthetic_recovery():
    fitted = fit(synth_obs(), "bal")
    for got, truth in zip(fitted.coef[:6], TRUE_COEF):
        assert got == pytest.approx(truth, abs=0.02)
    assert fitted.coef[6] == pytest.approx(BLOCKS[1], abs=0.02)
    assert fitted.coef[7] == pytest.approx(BLOCKS[2], abs=0.02)
    assert fitted.r2 > 0.99
    centroid = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    assert predict(fitted, centroid) == pytest.approx(
        surface(centroid), abs=0.01
    )
    assert fitted.lack_of_fit_p is not None
    assert fitted.block_levels == (0, 1, 2)


def test_optimize_matches_bruteforce():
    fits = {target: true_fit() for target in TARGETS}
    x_opt, value = optimize(fits, "bal")
    grid = simplex_grid(0.05)
    brute = min(grid, key=surface)
    assert x_opt == brute
    assert value == pytest.approx(surface(brute), abs=1e-9)
    with pytest.raises(ValueError):
        optimize(fits, "nope")


def test_simplex_grid_properties():
    grid = simplex_grid(0.05)
    assert len(grid) == 231
    assert all(abs(sum(x) - 1.0) < 1e-9 for x in grid)


def test_paired_delta():
    obs = []
    for seed, (a, b) in enumerate(
        [(1.0, 0.5), (1.1, 0.6), (1.2, 0.7)]
    ):
        obs.append(Observation(1, seed, (1.0, 0.0, 0.0),
                               {t: a for t in TARGETS}, None, 10))
        obs.append(Observation(2, seed, (0.0, 1.0, 0.0),
                               {t: b for t in TARGETS}, None, 10))
    deltas = paired_delta(obs, "lay", 1)
    mean, low, high = deltas[2]
    assert mean == pytest.approx(-0.5)
    assert low == pytest.approx(-0.5)
    assert high == pytest.approx(-0.5)
    single = [o for o in obs if o.seed == 0]
    one = paired_delta(single, "lay", 1)[2]
    assert one[0] == pytest.approx(-0.5)
    assert math.isnan(one[1]) and math.isnan(one[2])


def _mixture_spec() -> GeneratorSpec:
    return GeneratorSpec("mixture", 24, {"components": [
        {"family": "layered", "weight": 1},
        {"family": "structured", "weight": 1},
        {"family": "spaghetti", "weight": 2,
         "params": {"spaghetti_rate": 0}},
    ]})


def _build_dataset(tmp_path: Path, spec: GeneratorSpec, name: str,
                   span: str = "train=0:8") -> Path:
    spec_path = tmp_path / f"{name}.json"
    spec_path.write_text(json.dumps(spec_to_json(spec)))
    out = tmp_path / name
    dataset_main(["--spec", str(spec_path), "--out", str(out),
                  "--split", span, "--version", "test-mixture"])
    return out


def test_realized_composition_matches_component_for(tmp_path):
    out = _build_dataset(tmp_path, _mixture_spec(), "mixed")
    counts = realized_composition(out)
    manifest = json.loads((out / "manifest.json").read_text())
    samples = manifest["splits"]["train"]["samples"]
    assert sum(counts.values()) == len(samples)
    assert set(counts) == set(FAMILIES)
    direct = {family: 0 for family in FAMILIES}
    for sample in samples:
        payload = json.loads(
            (out / "train" / f"{sample['sample_id']}.json").read_text()
        )
        generator = payload["provenance"]["generator"]
        spec = generator["config"]["spec"]
        index = component_for(
            GeneratorSpec(spec["family"], spec["num_nodes"], spec["params"]),
            generator["seed"],
        )
        direct[spec["params"]["components"][index]["family"]] += 1
    assert counts == direct


def test_realized_composition_pure_family(tmp_path):
    out = _build_dataset(
        tmp_path, GeneratorSpec("layered", 24, {}), "pure"
    )
    counts = realized_composition(out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert counts["layered"] == manifest["splits"]["train"]["kept"]
    assert counts["structured"] == 0
    assert counts["spaghetti"] == 0


def test_realized_composition_unknown_family(tmp_path):
    dataset = tmp_path / "fake"
    (dataset / "train").mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "sample_id": "s0",
        "provenance": {"generator": {
            "seed": 0,
            "config": {"spec": {"family": "weird", "num_nodes": 8,
                                "params": {}}},
        }},
        "metagraph": {},
    }
    (dataset / "train" / "s0.json").write_text(json.dumps(payload))
    (dataset / "manifest.json").write_text(json.dumps(
        {"splits": {"train": {"samples": [{"sample_id": "s0"}]}}}
    ))
    with pytest.raises(ValueError):
        realized_composition(dataset)


def _write_scores(path: Path, nll: float, tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"nll": nll, "n_tokens": tokens}) + "\n")


def test_collect(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    data = tmp_path / "data"
    (data / "d24_p1").mkdir(parents=True)
    (data / "d24_p2").mkdir(parents=True)
    composition = {
        "d24_p1": {"layered": 3, "structured": 1, "spaghetti": 0},
        "d24_p2": {"layered": 0, "structured": 0, "spaghetti": 4},
    }

    def fake_composition(dataset_dir: Path, split: str = "train"):
        return composition[dataset_dir.name]

    monkeypatch.setattr(
        mixture_doe, "realized_composition", fake_composition
    )

    for seed, wf in ((0, 0.75), (1, 0.5)):
        base = runs / f"d_s24_p1_mask_n24_s{seed}"
        _write_scores(base / "test_scores.jsonl", 10.0, 100)
        _write_scores(
            runs / f"d_s24_p12str_mask_n24_s{seed}" / "test_scores.jsonl",
            20.0, 100,
        )
        _write_scores(
            runs / f"d_s24_p12spa_mask_n24_s{seed}" / "test_scores.jsonl",
            30.0, 100,
        )
        (base / "eval.json").write_text(
            json.dumps({"well_formed_rate": wf})
        )
    _write_scores(
        runs / "d_s24_p2_mask_n24_s0" / "test_scores.jsonl", 5.0, 100
    )
    _write_scores(
        runs / "d_s24_p22spa_mask_n24_s0" / "test_scores.jsonl", 5.0, 100
    )
    base2 = runs / "d_s24_p2_mask_n24_s1"
    _write_scores(base2 / "test_scores.jsonl", 12.0, 100)
    _write_scores(
        runs / "d_s24_p22str_mask_n24_s1" / "test_scores.jsonl", 13.0, 100
    )
    _write_scores(
        runs / "d_s24_p22spa_mask_n24_s1" / "test_scores.jsonl", 14.0, 100
    )

    obs, missing = mixture_doe.collect(
        runs, data, [1, 2], [0, 1], size=24, prefix="d"
    )
    assert len(obs) == 3
    assert [o.point for o in obs] == [1, 1, 2]
    first = obs[0]
    assert first.x == pytest.approx((0.75, 0.25, 0.0))
    assert first.n_train == 4
    assert first.y == pytest.approx({"lay": 0.1, "str": 0.2, "spa": 0.3})
    assert first.wf == pytest.approx(0.75)
    assert obs[-1].wf is None
    assert missing == ["d_s24_p2_mask_n24_s0"]


def test_observations_json_round_trip():
    obs = synth_obs()
    restored = observations_from_json(observations_to_json(obs))
    assert restored == obs


def test_observations_json_round_trip_y_raw():
    obs = [
        Observation(1, 0, (0.5, 0.25, 0.25),
                    {"lay": 1.0, "str": 2.0, "spa": 3.0}, 0.9, 10,
                    {"lay": 0.7, "str": 0.8, "spa": 0.9}),
        Observation(1, 1, (1.0, 0.0, 0.0),
                    {"lay": 1.0, "str": 2.0, "spa": 3.0}, None, 10),
    ]
    restored = observations_from_json(observations_to_json(obs))
    assert restored == obs
    assert restored[0].y_raw == pytest.approx(
        {"lay": 0.7, "str": 0.8, "spa": 0.9})
    assert restored[1].y_raw is None


def _write_named_scores(path: Path, rows: list[tuple[str, float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps({"sample_id": sid, "nll": nll, "n_tokens": n_tokens}) + "\n"
        for sid, nll, n_tokens in rows))


def _baseline_collect_fixture(tmp_path):
    runs = tmp_path / "runs"
    data = tmp_path / "data"
    baseline = tmp_path / "baseline"
    (data / "d24_p1").mkdir(parents=True)
    model = {
        "lay": [("a", 10.0, 5), ("b", 20.0, 5)],
        "str": [("a", 15.0, 5), ("b", 25.0, 5)],
        "spa": [("a", 25.0, 5), ("b", 25.0, 5)],
    }
    for target, rows in model.items():
        suffix = "" if target == "lay" else f"2{target}"
        _write_named_scores(
            runs / f"d_s24_p1{suffix}_mask_n24_s0" / "test_scores.jsonl", rows)
    for target, rows in {
        "lay": [("a", 6.0, 5), ("b", 12.0, 5)],
        "str": [("a", 10.0, 5), ("b", 15.0, 5)],
        "spa": [("a", 20.0, 5), ("b", 25.0, 5)],
    }.items():
        _write_named_scores(baseline / f"base_{target}.jsonl", rows)
    for target, rows in {
        "lay": [("a", 8.0, 5), ("b", 12.0, 5)],
        "str": [("a", 20.0, 5), ("b", 15.0, 5)],
        "spa": [("a", 15.0, 5), ("b", 25.0, 5)],
    }.items():
        _write_named_scores(baseline / f"base_p1_{target}.jsonl", rows)
    return runs, data, baseline


def test_collect_baseline_target_and_source(tmp_path, monkeypatch):
    runs, data, baseline = _baseline_collect_fixture(tmp_path)
    monkeypatch.setattr(
        mixture_doe, "realized_composition",
        lambda dataset_dir, split="train": {
            "layered": 4, "structured": 0, "spaghetti": 0})

    target, missing = mixture_doe.collect(
        runs, data, [1], [0], baseline_dir=baseline, baseline_mode="target")
    assert missing == []
    assert target[0].y == pytest.approx({"lay": 1.2, "str": 1.5, "spa": 0.5})
    assert target[0].y_raw == pytest.approx({"lay": 3.0, "str": 4.0, "spa": 5.0})

    source, _ = mixture_doe.collect(
        runs, data, [1], [0], baseline_dir=baseline, baseline_mode="source")
    assert source[0].y == pytest.approx({"lay": 1.0, "str": 0.5, "spa": 1.0})
    assert source[0].y_raw == pytest.approx({"lay": 3.0, "str": 4.0, "spa": 5.0})


def test_collect_baseline_missing_id_and_unknown_mode(tmp_path, monkeypatch):
    runs, data, baseline = _baseline_collect_fixture(tmp_path)
    monkeypatch.setattr(
        mixture_doe, "realized_composition",
        lambda dataset_dir, split="train": {
            "layered": 4, "structured": 0, "spaghetti": 0})
    _write_named_scores(baseline / "base_lay.jsonl", [("a", 6.0, 5)])
    with pytest.raises(ValueError, match="missing sample_id: b"):
        mixture_doe.collect(runs, data, [1], [0], baseline_dir=baseline,
                            baseline_mode="target")
    _write_named_scores(baseline / "base_lay.jsonl",
                        [("a", 6.0, 5), ("b", 12.0, 5), ("c", 1.0, 5)])
    with pytest.raises(ValueError, match="extra sample_id: c"):
        mixture_doe.collect(runs, data, [1], [0], baseline_dir=baseline,
                            baseline_mode="target")
    with pytest.raises(ValueError, match="baseline_mode"):
        mixture_doe.collect(runs, data, [1], [0], baseline_dir=baseline,
                            baseline_mode="bogus")


def test_report_adds_raw_columns_when_y_raw_present(tmp_path):
    obs = [
        Observation(o.point, o.seed, o.x, o.y, o.wf, o.n_train,
                    {target: o.y[target] + 1.0 for target in TARGETS})
        for o in synth_obs()
    ]
    fits = {target: fit(obs, target) for target in TARGETS}
    composites = {c: optimize(fits, c) for c in COMPOSITES}
    out_md = tmp_path / "report.md"
    report(obs, fits, composites, out_md, ref_point=7)
    text = out_md.read_text()
    for column in ("raw_lay", "raw_str", "raw_spa"):
        assert column in text


def test_report_and_figures(tmp_path):
    obs = synth_obs()
    fits = {target: fit(obs, target) for target in TARGETS}
    composites = {c: optimize(fits, c) for c in COMPOSITES}
    out_md = tmp_path / "report.md"
    fig_dir = tmp_path / "figs"
    report(obs, fits, composites, out_md, ref_point=7, fig_dir=fig_dir)
    text = out_md.read_text()
    for heading in ("## 1. Observations", "## 2. Point means",
                    "## 3. Coefficient fits", "## 4. Paired deltas",
                    "## 5. Optima"):
        assert heading in text
    for name in (*TARGETS, *COMPOSITES):
        assert (fig_dir / f"mixture_doe_{name}.png").exists()
