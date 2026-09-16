import os
from pathlib import Path

import pytest

from training.runner import (
    Block,
    RescoreJob,
    TrainJob,
    rescore_jobs,
    rescore_script,
    sweep_plan,
    train_wrapper,
)

MASK_WRAPPER = """\
import sys
sys.path.insert(0, '/content')
for _m in ('train_ar', 'grammar_mask'):
    sys.modules.pop(_m, None)
import train_ar
train_ar.main(['--out', '/content/c_s32_lay_mask_n32_s0', '--epochs', '300', '--patience', '20', '--num-samples', '400', '--test', '/content/test.jsonl', '--constrained-samples', '400', '--seed', '0', '--sample-seed', '1000', '--ref-legal-mask'])
"""

BASE_WRAPPER = """\
import sys
sys.path.insert(0, '/content')
for _m in ('train_ar', 'grammar_mask'):
    sys.modules.pop(_m, None)
import train_ar
train_ar.main(['--out', '/content/c_s32_lay_mask_n32_s0', '--epochs', '300', '--patience', '20', '--num-samples', '400', '--test', '/content/test.jsonl', '--constrained-samples', '400', '--seed', '0', '--sample-seed', '1000'])
"""

POINTER_WRAPPER = """\
import sys
sys.path.insert(0, '/content')
for _m in ('train_ar', 'grammar_mask'):
    sys.modules.pop(_m, None)
import train_ar
train_ar.main(['--out', '/content/c_s32_mix_pointer_n32_s0', '--epochs', '300', '--patience', '20', '--num-samples', '400', '--test', '/content/test.jsonl', '--constrained-samples', '400', '--seed', '0', '--sample-seed', '1000', '--pointer', '--pointer-legal', '--pointer-dist-bias', '--pointer-dist-bias-mode', 'context'])
"""


def mask_job(extra: tuple[str, ...] = ("--ref-legal-mask",)) -> TrainJob:
    return TrainJob(
        name="c_s32_lay_mask_n32_s0",
        epochs=300,
        patience=20,
        num_samples=400,
        constrained_samples=400,
        seed=0,
        sample_seed=1000,
        extra=extra,
    )


def test_train_wrapper_mask_example() -> None:
    assert train_wrapper(mask_job(), "/content") == MASK_WRAPPER


def test_train_wrapper_without_extra() -> None:
    assert train_wrapper(mask_job(extra=()), "/content") == BASE_WRAPPER


def test_train_wrapper_pointer_example() -> None:
    job = TrainJob(
        name="c_s32_mix_pointer_n32_s0",
        epochs=300,
        patience=20,
        num_samples=400,
        constrained_samples=400,
        seed=0,
        sample_seed=1000,
        extra=(
            "--pointer",
            "--pointer-legal",
            "--pointer-dist-bias",
            "--pointer-dist-bias-mode",
            "context",
        ),
    )
    assert train_wrapper(job, "/content") == POINTER_WRAPPER


def test_train_wrapper_trailing_newline() -> None:
    text = train_wrapper(mask_job(), "/content")
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_train_wrapper_uses_root() -> None:
    text = train_wrapper(mask_job(), "/work")
    assert "/content" not in text
    assert "'/work/c_s32_lay_mask_n32_s0'" in text


def test_rescore_jobs_mapping() -> None:
    block = Block(
        source_bundle="data/tok_s32_lay",
        train=(TrainJob("run0", 1, 2, 3, 4, 5, 6),),
        rescore=(RescoreJob("run0", "data/tok_s32_lay2str", "out0"),),
    )
    assert rescore_jobs(block, "/work") == [{
        "run": "run0",
        "model": "/work/run0_model.pt",
        "config": "/work/run0_samples.json",
        "vocab": "/work/vocab_tok_s32_lay.json",
        "meta": "/work/meta_tok_s32_lay.json",
        "test": "/work/test_tok_s32_lay2str.jsonl",
        "out": "/work/run0__tok_s32_lay2str.jsonl",
    }]


def test_rescore_script_has_no_content_path() -> None:
    script = rescore_script("/work")
    assert "/content" not in script
    assert "/work/rescore_jobs.json" in script
    assert "RESCORE-DONE" in script
    assert "RESCORED" in script


def test_sweep_plan_counts_and_names() -> None:
    plan = sweep_plan((12, 16, 24, 32, 48))
    train_jobs = [job for block in plan.blocks for job in block.train]
    rescore_jobs_all = [job for block in plan.blocks for job in block.rescore]
    assert len(plan.blocks) == 10
    assert plan.plan_id == "sweep_12_16_24_32_48"
    assert len(train_jobs) == 30
    assert len(rescore_jobs_all) == 75
    assert plan.blocks[0].source_bundle == "data/tok_s12_lay"
    assert plan.blocks[1].source_bundle == "data/tok_s12_mix"

    names = {job.name for job in train_jobs}
    assert "c_s12_lay_mask_n12_s0" in names
    assert "c_s48_mix_mask_n48_s2" in names

    lay = next(block for block in plan.blocks if block.source_bundle == "data/tok_s12_lay")
    assert len(lay.rescore) == 6
    assert lay.train[0].extra == ("--ref-legal-mask",)
    assert lay.train[0].epochs == 300
    assert lay.train[0].patience == 20
    assert lay.train[0].num_samples == 400
    assert lay.train[0].constrained_samples == 400
    assert [job.sample_seed for job in lay.train] == [1000, 1001, 1002]

    triples = {
        (job.source_run, job.target_bundle, job.out_run)
        for job in rescore_jobs_all
    }
    assert (
        "c_s12_lay_mask_n12_s0",
        "data/tok_s12_lay2str",
        "c_s12_lay2str_mask_n12_s0",
    ) in triples
    assert (
        "c_s24_lay_mask_n24_s1",
        "data/tok_s24_lay2spa",
        "c_s24_lay2spa_mask_n24_s1",
    ) in triples
    assert (
        "c_s48_mix_mask_n48_s2",
        "data/tok_s48_mix2lay",
        "c_s48_mix2lay_mask_n48_s2",
    ) in triples


@pytest.mark.skipif(
    not os.environ.get("GR_SWEEP_WRAPPERS"),
    reason="GR_SWEEP_WRAPPERS not set to the wrapper directory",
)
def test_sweep_wrappers_match_files() -> None:
    directory = Path(os.environ["GR_SWEEP_WRAPPERS"])
    plan = sweep_plan((12, 16, 24, 32, 48))
    for block in plan.blocks:
        for job in block.train:
            path = directory / f"w_{job.name}.py"
            assert path.read_bytes() == train_wrapper(job, "/content").encode("utf-8")
