"""逐次設計 T1 の合成データによる契約試験。"""
from dataclasses import replace
import json
import math
import random
import subprocess
import sys

import numpy as np
import pytest

from training import seqdesign as sd
from training.mixture_doe import Observation


def campaign(mode="mixture"):
    space = sd.Space(("spa", "lay", "str"),
                     ({"lay": 1., "str": 0., "spa": 0.},
                      {"lay": 0., "str": 1., "spa": 0.}), ("lay", "str", "spa"))
    registry: tuple[dict[str, object], ...] = ()
    if mode == "registry":
        space = sd.Space(("width", "depth"), ({"width": 8., "depth": 2.}, {"width": 16., "depth": 4.}))
        registry = ({"point": 8, "candidate_id": 0, "factors": space.candidates[0]},)
    return sd.Campaign(1, "synthetic", 0, 0, space, ("loss",),
                       {"kind": "response", "responses": ["loss"], "direction": "minimize"},
                       mode, {"response_scale": "synthetic"}, (), (), registry,
                       {"seed": 42, "draws": 0})


def observation(seed=0):
    return Observation(8, seed, (.495, .268, .237), {"loss": -.4, "extra": 2.}, .87, 200,
                       {"loss": .8})


@pytest.mark.parametrize("maximum_likelihood", [False, True])
def test_function_recovery(maximum_likelihood):
    grid = np.linspace(0, 1, 41)[:, None]
    x = np.linspace(0, 1, 9)[:, None]
    y = np.sin(2 * np.pi * x[:, 0])
    fit = sd.fit_gp(x, y, candidates=grid, fixed=None if maximum_likelihood else (1., [.2], 1e-10))
    prediction = sd.posterior(fit, grid)
    heldout = np.arange(41) % 5 != 0
    assert np.sqrt(np.mean((prediction.mean[heldout] - np.sin(2 * np.pi * grid[heldout, 0]))**2)) < .10
    assert math.isfinite(fit.diagnostics["lml"])


@pytest.mark.parametrize("duplicate", [False, True])
def test_collapse(duplicate):
    x = np.linspace(0, 1, 9)[:, None]
    y = np.sin(2 * np.pi * x[:, 0])
    if duplicate:
        x, y = np.repeat(x, 2, axis=0), np.repeat(y, 2)
    p = sd.posterior(sd.fit_gp(x, y, fixed=(1., [.2], 1e-10)), x)
    assert np.max(np.abs(p.mean - y)) < 1e-5
    assert np.max(np.diag(p.covariance)) < 1e-7
    assert np.isfinite(p.L).all()


def test_pure_error():
    x = [[0.], [0.], [1.], [1.], [1.]]
    y = np.array([1., 3., 4., 5., 6.])
    fit = sd.fit_gp(x, y)
    assert fit.diagnostics["pure_error_df"] == 3
    assert fit.diagnostics["pure_error_variance"] == pytest.approx(4 / 3)
    assert fit.v == pytest.approx((4 / 3) / y.var())
    assert fit.v != fit.diagnostics["jitter"]
    constant = sd.fit_gp([[0.], [0.]], [2., 2.], candidates=[[0.], [1.]])
    assert constant.s == 1 and constant.v == 1e-10
    assert constant.a == 1 and constant.ell[0] == pytest.approx(.3)


def test_preprocessing_and_prior():
    fit = sd.fit_gp([[5., 7.], [25., 7.]], [10., 14.], candidates=[[10., 7.], [20., 7.]], fixed=(1., [.2], .01))
    np.testing.assert_allclose(fit.X, [[-.5], [1.5]])
    assert fit.m == 12 and fit.s == 2
    assert fit.active.tolist() == [True, False]
    prior = sd.fit_gp(np.empty((0, 2)), [], candidates=[[10., 7.], [20., 7.]])
    p = sd.posterior(prior, [[10., 7.], [20., 7.]])
    np.testing.assert_allclose(p.mean, 0)
    np.testing.assert_allclose(np.diag(p.covariance), 1)
    assert prior.diagnostics["prior"]
    all_constant = sd.fit_gp([[7.], [7.]], [1., 3.])
    assert all_constant.X.shape == (2, 0)
    assert np.isfinite(sd.posterior(all_constant, [[7.]]).mean).all()


def test_kernel_and_likelihood():
    x = np.array([[0., 0.], [1., 2.]])
    k = sd.rbf_kernel(x, x, 2., [1., 2.])
    np.testing.assert_allclose(k, [[2., 2 / math.e], [2 / math.e, 2.]])
    expected = -.5 / 1.1 - .5 * math.log(1.1) - .5 * math.log(2 * math.pi)
    assert sd.log_marginal_likelihood([[0.]], [1.], 1., [.2], .1) == pytest.approx(expected)


def test_joint_covariance_and_rng_resume():
    covariance = np.array([[1., .7], [.7, 2.]])
    p = sd.Posterior(np.array([2., -1.]), covariance, np.linalg.cholesky(covariance), 0.)
    rng = sd.DeterministicRNG(193)
    samples = np.array([sd.draw_joint(p, rng) for _ in range(16000)])
    np.testing.assert_allclose(samples.mean(axis=0), p.mean, atol=.035)
    np.testing.assert_allclose(np.cov(samples.T), covariance, atol=.055)
    assert rng.draws == 64000
    restored = sd.DeterministicRNG(**rng.rng_state)
    np.testing.assert_array_equal(sd.draw_joint(p, rng), sd.draw_joint(p, restored))
    source = random.Random(9)
    expected = math.sqrt(-2 * math.log(1-source.random())) * math.cos(2*math.pi*source.random())
    assert sd.normal_draw(random.Random(9)) == expected
    assert sd.uniform_index(7, random.Random(9)) == math.floor(7 * random.Random(9).random())


def test_observation_and_campaign_roundtrip():
    obs = observation()
    assert sd.observations_from_json(sd.observations_to_json([obs]), ["loss"]) == (obs,)
    c = sd.observe(campaign(), [observation(2), obs])
    encoded = sd.campaign_to_json(c)
    assert encoded.endswith("\n") and "NaN" not in encoded
    assert sd.campaign_from_json(encoded) == c
    assert sd.campaign_to_json(sd.campaign_from_json(encoded)) == encoded
    assert [o.seed for o in c.observations] == [0, 2]
    assert c.registry == ({"point": 8, "candidate_id": None, "factors": None},)
    np.testing.assert_allclose(sd.observation_coordinates(c), [[.237, .495, .268]] * 2)
    assert c.revision == c.observation_revision == 1
    assert sd.observe(c, [obs]) is c
    before = sd.campaign_to_json(c)
    for bad in (replace(obs, y={"loss": 9.}), replace(obs, seed=3, n_train=201),
                replace(obs, seed=3, x=(.5, .25, .25))):
        with pytest.raises(ValueError):
            sd.observe(c, [observation(4), bad])
        assert sd.campaign_to_json(c) == before


def test_registry_and_proposal_roundtrip():
    c = campaign("registry")
    obs = replace(observation(), x=(0., 0., 0.))
    c = sd.observe(c, [obs])
    np.testing.assert_array_equal(sd.observation_coordinates(c), [[8., 2.]])
    p = sd.Proposal("q000001", 0, 8, c.space.candidates[0], (1, 2), "ts", 1, 2)
    c = replace(c, proposals=(p,), revision=2)
    assert sd.campaign_from_json(sd.campaign_to_json(c)) == c
    with pytest.raises(ValueError, match="unknown registry point"):
        sd.observe(c, [replace(obs, point=9)])


@pytest.mark.parametrize("field,value", [("point", 1.2), ("seed", True), ("n_train", -1),
                                          ("x", [1., 0.]), ("x", [float("nan"), 0., 0.]),
                                          ("y", {"loss": float("inf")}), ("wf", 1.1),
                                          ("y", {"other": 1.}), ("y_raw", {"loss": float("nan")})])
def test_invalid_observations(field, value):
    row = json.loads(sd.observations_to_json([observation()]))[0]
    row[field] = value
    with pytest.raises(ValueError):
        sd.observations_from_json(json.dumps([row]), ["loss"])


@pytest.mark.parametrize("space", [sd.Space(("x",), ()), sd.Space(("x", "x"), ({"x": 1.},)),
                                    sd.Space(("x",), ({"x": 0.}, {"x": -0.})),
                                    sd.Space(("x",), ({"x": True},)),
                                    sd.Space(("x",), ({"x": float("nan")},)),
                                    sd.Space(("x",), ({"z": 1.},)),
                                    sd.Space(("x",), ({"x": .5},), ("x",))])
def test_invalid_spaces(space):
    with pytest.raises(ValueError):
        sd.validate_space(space)


def test_space_json_and_ownership():
    original = {"x": -0.}
    space = sd.Space(("x",), (original, {"x": 1.}))
    original["x"] = 7
    parsed = sd.space_from_json(sd.space_to_json(space))
    assert math.copysign(1, parsed.candidates[0]["x"]) == 1
    assert parsed == space
    c = campaign()
    changed = replace(c, revision=1)
    changed.context["new"] = 1
    assert "new" not in c.context
    fit = sd.fit_gp([[0.], [1.]], [0., 1.], fixed=(1., [.2], .01))
    with pytest.raises(ValueError):
        fit.X[0, 0] = 4


@pytest.mark.parametrize("mutate", [lambda d: d.update(version=2), lambda d: d.update(extra=1),
                                     lambda d: d["space"].update(extra=1),
                                     lambda d: d["objective"].update(direction="maximize"),
                                     lambda d: d["rng_state"].update(draws=-1),
                                     lambda d: d["context"].update(bad=float("nan"))])
def test_campaign_schema_rejection(mutate):
    payload = json.loads(sd.campaign_to_json(campaign()))
    mutate(payload)
    with pytest.raises(ValueError):
        sd.campaign_from_json(json.dumps(payload))


def test_numerical_failure_policy(monkeypatch):
    fit = sd.fit_gp([[0.], [1.]], [0., 1.], fixed=(1., [.2], 1e-10))
    original = np.linalg.eigh
    def negative(matrix):
        values, vectors = original(matrix)
        values[0] = -1e-9
        return values, vectors
    monkeypatch.setattr(np.linalg, "eigh", negative)
    p = sd.posterior(fit, [[0.], [1.]])
    assert np.linalg.eigvalsh(p.covariance).min() >= -1e-15
    def too_negative(matrix):
        values, vectors = original(matrix)
        values[0] = -1e-3
        return values, vectors
    monkeypatch.setattr(np.linalg, "eigh", too_negative)
    with pytest.raises(ValueError, match="negative eigenvalue"):
        sd.posterior(fit, [[0.], [1.]])
    monkeypatch.setattr(np.linalg, "eigh", original)
    cholesky = np.linalg.cholesky
    calls = []
    def retry(matrix):
        calls.append(matrix.copy())
        if len(calls) == 1:
            raise np.linalg.LinAlgError("synthetic failure")
        return cholesky(matrix)
    monkeypatch.setattr(np.linalg, "cholesky", retry)
    p = sd.posterior(fit, [[0.], [1.]])
    assert len(calls) == 2
    assert p.jitter == pytest.approx(fit.s**2 * max(1, fit.a + fit.v) * 1e-9)


def test_core_import_is_lazy():
    subprocess.run([sys.executable, "-c", "import sys; import training.seqdesign; "
                    "assert 'training.mixture_doe' not in sys.modules; "
                    "assert 'torch' not in sys.modules; assert 'matplotlib' not in sys.modules"], check=True)


# T2: 獲得、予約、CLI、検証プロトコル。
def quadratic_campaign():
    grid = sd.Space(("x",), tuple({"x": i / 10} for i in range(11)))
    registry = tuple({"point": i, "candidate_id": i, "factors": f} for i, f in enumerate(grid.candidates))
    c = replace(campaign("registry"), space=grid, registry=registry)
    rows = [Observation(i, seed, (0., 0., 0.), {"loss": (f["x"]-.3)**2}, None, 0)
            for i, f in enumerate(grid.candidates) for seed in range(2)]
    return sd.observe(c, rows)


def test_ts_optimum_and_resume(tmp_path):
    c = quadratic_campaign()
    updated, payload = sd.propose(c, 10)
    assert [p["candidate_id"] for p in payload["proposals"]] == [3] * 10
    assert [p["seeds"] for p in payload["proposals"]] == [tuple(range(s, s+2)) for s in range(2, 22, 2)]
    assert updated.observation_revision == c.observation_revision
    assert updated.revision == c.revision+1
    restored = sd.campaign_from_json(sd.campaign_to_json(updated))
    assert sd.propose(updated, 2) == sd.propose(restored, 2)
    before = sd.campaign_to_json(updated)
    sd.report(updated, tmp_path / "report.md")
    assert sd.campaign_to_json(updated) == before
    text = (tmp_path / "report.md").read_text()
    assert "将来1 seed" in text and "未完了" in text
    assert sd.replay(updated, updated.revision) == payload


def test_reservations_existing_offgrid_registry_and_new():
    c = campaign("registry")
    # candidate_id が未設定でも同じ名目パターンの point を再利用する。
    c = replace(c, registry=({"point": 8, "candidate_id": None, "factors": c.space.candidates[0]},
                             {"point": 100, "candidate_id": None, "factors": {"width": 12., "depth": 3.}}))
    c = sd.observe(c, [replace(observation(7), x=(0., 0., 0.))])
    updated, payload = sd._reserve(c, [0, 1, 1, 0], 2, "random", c.rng_state)
    assert [p["point"] for p in payload["proposals"]] == [8, 101, 101, 8]
    assert [p["seeds"] for p in payload["proposals"]] == [(8, 9), (0, 1), (2, 3), (10, 11)]
    resumed = sd.campaign_from_json(sd.campaign_to_json(updated))
    again, next_payload = sd._reserve(resumed, [1], 1, "random", resumed.rng_state)
    assert next_payload["proposals"][0]["seeds"] == (4,)
    assert next_payload["proposals"][0]["proposal_id"] == "q000005"
    assert again.revision == 3


def test_composite_in_original_units_and_draw_order():
    ps = (sd.Posterior(np.array([10., 11.]), np.eye(2)*4, np.eye(2)*2, 0.),
          sd.Posterior(np.array([-10., -11.]), np.eye(2)*9, np.eye(2)*3, 0.))
    rng = random.Random(8)
    expected = np.asarray([[sd.draw_joint(p, rng) for p in ps] for _ in range(4)])
    actual = sd._samples(ps, 4, random.Random(8))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    np.testing.assert_allclose(sd.composite(actual, ("a", "b"), "bal"), expected.mean(axis=1))
    np.testing.assert_array_equal(sd.composite(actual, ("a", "b"), "a"), actual[:, 0])
    p = sd.Posterior(np.zeros(1), np.ones((1, 1)), np.ones((1, 1)), 0.)
    samples = sd._samples((p, p), 10000, random.Random(31))
    assert sd.composite(samples, ("a", "b"), "max").mean() > .5
    assert abs(samples.mean(axis=0).max()) < .03


def test_acquisition_scores_and_ties(monkeypatch, capsys):
    c = campaign("registry")
    updated, payload = sd.propose(c, 4, 1, "ei")
    assert "random" in capsys.readouterr().err
    assert sd.propose(c, 4, 1, "random")[1]["proposals"][0]["candidate_id"] == payload["proposals"][0]["candidate_id"]
    assert updated.rng_state["draws"] == 4
    c = sd.observe(c, [replace(observation(), x=(0., 0., 0.), y={"loss": -1000.})])
    shapes = []
    def fake_posterior(f, grid):
        shapes.append(len(grid))
        return sd.Posterior(np.zeros(len(grid)), np.eye(len(grid)), np.eye(len(grid)), 0.)
    def fake_samples(ps, count, rng):
        assert count == 1024
        # EI incumbent は最後の列の標本平均2。候補0のEI=1、候補1のEI=0。
        return np.tile(np.array([[[1., 3., 2.]]])[:, :, :len(ps[0].mean)], (count, 1, 1))
    monkeypatch.setattr(sd, "posterior", fake_posterior)
    monkeypatch.setattr(sd, "_samples", fake_samples)
    indices, _ = sd._select(c, 3, "ei", random.Random(0))
    assert indices == [0]*3 and shapes == [3]
    assert sd._select(c, 2, "ucb", random.Random(0))[0] == [0, 0]
    monkeypatch.setattr(sd, "_samples", lambda ps, count, rng: np.zeros((count, 1, len(ps[0].mean))))
    for acq in ("ts", "ei", "ucb"):
        assert sd._select(c, 2, acq, random.Random(0))[0] == [0, 0]
    with pytest.raises(ValueError):
        sd.propose(c, acq="fixed")


def test_ucb_population_sd(monkeypatch):
    c = campaign("registry")
    # 母SDなら候補1、不偏SDなら候補0になるよう狭い境界を作る。
    samples = np.zeros((1024, 1, 2))
    samples[:512, 0, 0] = -1
    samples[512:, 0, 0] = 1
    samples[:, 0, 1] = -2.0005
    monkeypatch.setattr(sd, "_samples", lambda *args: samples)
    assert sd._select(c, 2, "ucb", random.Random(0))[0] == [1, 1]


def test_report_jitter_and_rng_independence():
    c = campaign("registry")
    fits = sd._fits(c)
    summary = sd._summaries(c, fits, sd.candidate_coordinates(c.space))
    assert c.rng_state == {"seed": 42, "draws": 0}
    for name in summary:
        assert np.all(summary[name]["predictive"][1] > summary[name]["predictive"][0])
    np.testing.assert_allclose(summary["loss"]["sd"], 1)


def test_cli_transaction_and_replay(tmp_path, capsys, monkeypatch):
    path, space, obs, out = (tmp_path / n for n in ("campaign.json", "space.json", "obs.json", "report.md"))
    space.write_text(sd.space_to_json(campaign().space))
    init = ["init", "--campaign", str(path), "--space", str(space), "--responses", "loss", "--objective", "loss"]
    assert sd.main(init) == 0
    initial = path.read_bytes()
    assert sd.main(init) == 2 and path.read_bytes() == initial
    obs.write_text(sd.observations_to_json([observation()]))
    assert sd.main(["observe", "--campaign", str(path), "--obs", str(obs)]) == 0
    assert sd.main(["propose", "--campaign", str(path), "--k", "2", "--acq", "random"]) == 0
    payload = json.loads(capsys.readouterr().out)
    before = path.read_bytes()
    assert sd.main(["propose", "--campaign", str(path), "--replay-revision", "2"]) == 0
    assert json.loads(capsys.readouterr().out) == payload and path.read_bytes() == before
    assert sd.main(["report", "--campaign", str(path), "--out", str(out)]) == 0
    assert path.read_bytes() == before and "全候補" in out.read_text()
    for arguments in (["propose", "--k", "0"], ["propose", "--seeds", "-1"],
                      ["propose", "--replay-revision", "99"], ["propose", "--acq", "fixed"],
                      ["report", "--out", str(path)], ["report", "--out", str(tmp_path / "absent" / "r.md")]):
        assert sd.main([*arguments, "--campaign", str(path)]) == 2
        assert path.read_bytes() == before
    obs.write_text('[{"point":1}]')
    assert sd.main(["observe", "--campaign", str(path), "--obs", str(obs)]) == 2
    assert path.read_bytes() == before
    import os
    def failed_replace(*args):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(os, "replace", failed_replace)
    capsys.readouterr()
    assert sd.main(["propose", "--campaign", str(path)]) == 2
    assert capsys.readouterr().out == "" and path.read_bytes() == before
    assert not list(tmp_path.glob(".campaign.json.*"))


def test_proposal_hashseed_subprocess(tmp_path):
    import os
    original = sd.campaign_to_json(campaign("registry"))
    outputs = []
    for value in ("1", "982"):
        path = tmp_path / f"{value}.json"
        path.write_text(original)
        result = subprocess.run([sys.executable, "-m", "training.seqdesign", "propose", "--campaign", str(path)],
                                env={**os.environ, "PYTHONHASHSEED": value}, capture_output=True, text=True, check=True)
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    assert len(json.loads(outputs[0])["proposals"]) == 3


def test_simulation_space_surface_and_coverage():
    space = sd._simulation_space()
    sd.validate_space(space)
    assert len(space.candidates) == 235
    grid = sd.candidate_coordinates(space)
    base, _ = sd._surface(grid, "base")
    sharp, h = sd._surface(grid, "vertex_sharp")
    disc, _ = sd._surface(grid, "vertex_discontinuous")
    np.testing.assert_allclose(base.mean(axis=0), -.4+.12*((grid-[.4, .25, .35])**2).sum(axis=1))
    np.testing.assert_allclose(sharp[0], base[0] + 3*h)
    assert np.count_nonzero(disc[0]-base[0]) == 1
    summaries = {"lay": {"latent": np.array([[0., 0.], [2., 2.]])}}
    coverage, width = sd._coverage(summaries, np.array([[1., 3.]]), ("lay",), np.array([True, False]), "latent")
    assert coverage["lay"]["vertex"]["rate"] == 1
    assert coverage["lay"]["interior"]["rate"] == 0
    assert width["lay"]["interior"] == 2


def test_tiny_simulate_cli(tmp_path, monkeypatch):
    # 通常試験は固定10点の小集合。全235点×20反復は slow 試験で実行する。
    small = sd.Space(("lay", "str", "spa"), tuple(dict(zip(("lay", "str", "spa"), row)) for row in sd._FIXED), ("lay", "str", "spa"))
    monkeypatch.setattr(sd, "_simulation_space", lambda: small)
    real_summaries = sd._summaries
    monkeypatch.setattr(sd, "_summaries", lambda c, fits, grid: real_summaries(c, fits, grid, count=32))
    out = tmp_path / "sim.md"
    assert sd.main(["simulate", "--rounds", "1", "--k", "1", "--seeds", "1", "--repeats", "1", "--out", str(out)]) == 0
    payload = json.loads(out.with_suffix(".json").read_text())
    assert len(payload["results"]) == 100
    assert len(payload["summary"]) == 100
    assert {r["method"] for r in payload["results"]} == {"ts", "ei", "ucb", "random", "fixed"}
    assert all(r["budget"] == 1 and r["regret"] >= 0 for r in payload["results"])
    assert all(0 <= r["near_fraction"] <= 1 for r in payload["results"])
    assert "厳密な Scheffé 二次ではない" in out.read_text()
    assert sd.main(["simulate", "--repeats", "0", "--out", str(out)]) == 2


def test_paired_report_and_collinear_figures(tmp_path):
    c = campaign()
    c = replace(c, responses=("lay", "str", "spa"), objective={"kind": "bal", "responses": ["lay", "str", "spa"], "direction": "minimize"})
    rows = [replace(observation(), point=p, y={"lay": -.4, "str": -.3, "spa": -.2}) for p in (7, 8)]
    c = sd.observe(c, rows)
    sd.report(c, tmp_path / "r.md", tmp_path / "fig")
    assert "算出不能" in (tmp_path / "r.md").read_text()
    assert len(list((tmp_path / "fig").glob("*.png"))) == 10


def test_retrospective_synthetic_fixture(tmp_path, monkeypatch):
    # 実験ファイルを必要とせず specs 正規化・6/12点分岐・元系列の不変性を検査する。
    specs = tmp_path / "specs"
    specs.mkdir()
    patterns = (*sd._FIXED, (.4, .3, .3), (.3, .4, .3))
    rows = []
    for point, pattern in enumerate(patterns, 1):
        components = [{"family": f, "weight": w*6} for f, w in zip(("layered", "structured", "spaghetti"), pattern)]
        (specs / f"spec_p{point}.json").write_text(json.dumps({"family": "mixture", "params": {"components": components}}))
        rows.extend(Observation(point, s, pattern, {"lay": -.4, "str": -.3, "spa": -.2}, None, 0) for s in range(3))
    path = tmp_path / "obs.json"
    path.write_text(sd.observations_to_json(rows))
    small = sd.Space(("lay", "str", "spa"), tuple(dict(zip(("lay", "str", "spa"), row)) for row in patterns), ("lay", "str", "spa"))
    # この試験は分岐の契約に集中。GP 最尤化は別の数値試験で検証済み。
    real_fit = sd.fit_gp
    monkeypatch.setattr(sd, "fit_gp", lambda x, y, **kw: real_fit(x, y, **kw, fixed=(1., [.3]*3, .01)))
    output = sd._retrospective(path, small, tmp_path)
    assert [(r["objective"], r["points"]) for r in output] == [("bal", 6), ("bal", 12), ("max", 6), ("max", 12)]
    for row in output:
        assert len(row["proposals"]["proposals"]) == 3
        stem = f"retrospective_{row['objective']}_{row['points']}"
        saved = sd.campaign_from_json((tmp_path / f"{stem}_campaign.json").read_text())
        assert saved.rng_state == {"seed": 20260926, "draws": 0} and not saved.proposals
        assert len(saved.observations) == row["points"]*3
        assert ("heldout_coverage" in row) == (row["points"] == 6)


def test_full_simulation_slow(tmp_path):
    import os
    if os.environ.get("GR_SLOW_TESTS") != "1":
        pytest.skip("GR_SLOW_TESTS=1 のみ全条件20反復")
    result = sd.simulate(out=tmp_path / "sim.md")
    assert len(result["results"]) == 10*2*5*20*8
