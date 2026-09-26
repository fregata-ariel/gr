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
