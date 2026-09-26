"""Tests for the ``run`` / ``sweep`` command line entry point.

The module is imported and ``main`` is called directly (no subprocesses); the
dry run and the fatal-path tests never construct a real backend, and nothing
here talks to Colab or Docker.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import training.runner.__main__ as cli
from training.runner import (
    Block,
    Plan,
    RescoreJob,
    TrainJob,
    plan_from_json,
    plan_to_json,
    sweep_plan,
)


def _small_plan() -> Plan:
    return Plan(
        plan_id="t",
        blocks=(
            Block(
                source_bundle="data/tok_x",
                train=(
                    TrainJob(
                        name="run_a",
                        epochs=2,
                        patience=1,
                        num_samples=1,
                        constrained_samples=1,
                        seed=0,
                        sample_seed=1000,
                        extra=("--ref-legal-mask",),
                    ),
                ),
                rescore=(
                    RescoreJob(
                        source_run="run_a",
                        target_bundle="data/tok_x2y",
                        out_run="run_a_x2y",
                    ),
                ),
            ),
        ),
    )


def _placeholders(repo_root: Path) -> None:
    training = repo_root / "training"
    training.mkdir(parents=True, exist_ok=True)
    (training / "train_ar.py").write_text("", encoding="utf-8")
    (training / "grammar_mask.py").write_text("", encoding="utf-8")

    tok = repo_root / "data" / "tok_x"
    tok.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "vocab.json", "meta.json"):
        (tok / name).write_text("", encoding="utf-8")

    target = repo_root / "data" / "tok_x2y"
    target.mkdir(parents=True, exist_ok=True)
    (target / "test.jsonl").write_text("", encoding="utf-8")


def _write_plan(tmp_path: Path) -> Path:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan_to_json(_small_plan()), encoding="utf-8")
    return plan_path


def _dry_run_argv(tmp_path: Path, plan_path: Path) -> list[str]:
    return [
        "run",
        "--plan", str(plan_path),
        "--repo-root", str(tmp_path),
        "--runs-dir", str(tmp_path / "runs"),
        "--backend", "local",
        "--dry-run",
    ]


def test_sweep_writes_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "plan.json"
    assert cli.main(["sweep", "--sizes", "12,16", "--out", str(out)]) == 0

    written = plan_from_json(out.read_text(encoding="utf-8"))
    assert written == sweep_plan((12, 16))
    assert written.plan_id in capsys.readouterr().out


def test_run_dry_run_prints_commands_and_creates_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _placeholders(tmp_path)
    plan_path = _write_plan(tmp_path)

    assert cli.main(_dry_run_argv(tmp_path, plan_path)) == 0

    printed = capsys.readouterr().out
    for marker in ("PUT", "RUN", "GET", "EVAL", "RUN-DONE"):
        assert marker in printed
    runs = tmp_path / "runs"
    assert not runs.exists() or not any(runs.iterdir())


def test_run_dry_run_ignores_gr_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GR_BACKEND", "local")
    _placeholders(tmp_path)
    plan_path = _write_plan(tmp_path)

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("DockerBackend must not be constructed in dry-run")

    monkeypatch.setattr(cli, "DockerBackend", _boom)
    assert cli.main(_dry_run_argv(tmp_path, plan_path)) == 0
    assert "RUN-DONE" in capsys.readouterr().out


def test_run_invalid_backend_returns_4(tmp_path: Path) -> None:
    assert cli.main(["run", "--plan", str(tmp_path / "plan.json"), "--backend", "foo"]) == 4


def test_run_missing_plan_returns_4(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    assert cli.main(["run", "--plan", str(missing), "--backend", "local"]) == 4


def test_run_unknown_plan_key_returns_4(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = json.loads(plan_to_json(_small_plan()))
    payload["bogus"] = 1
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    assert cli.main(["run", "--plan", str(plan_path), "--backend", "local"]) == 4
    assert "error:" in capsys.readouterr().err


def test_run_invalid_gr_backend_returns_4(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GR_BACKEND", "bogus")
    assert cli.main(["run", "--plan", str(tmp_path / "plan.json")]) == 4
    assert "error:" in capsys.readouterr().err


def test_run_gpu_option_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from training.runner import __main__ as cli_main

    parser = cli_main._build_parser()
    monkeypatch.delenv("GR_COLAB_GPU", raising=False)
    assert parser.parse_args(["run", "--plan", "p.json"]).gpu == "T4"
    assert parser.parse_args(["run", "--plan", "p.json", "--gpu", "L4"]).gpu == "L4"
    monkeypatch.setenv("GR_COLAB_GPU", "A100")
    assert cli_main._build_parser().parse_args(["run", "--plan", "p.json"]).gpu == "A100"
