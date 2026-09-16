import json
from dataclasses import asdict

import pytest

from training.runner import (
    Block,
    Plan,
    RescoreJob,
    TrainJob,
    bundle_name,
    plan_from_json,
    plan_to_json,
    validate_plan,
)


def make_plan() -> Plan:
    return Plan(
        plan_id="p1",
        blocks=(
            Block(
                source_bundle="data/tok_s12_lay",
                train=(
                    TrainJob(
                        name="run_a",
                        epochs=300,
                        patience=20,
                        num_samples=400,
                        constrained_samples=400,
                        seed=0,
                        sample_seed=1000,
                        extra=("--ref-legal-mask",),
                    ),
                ),
                rescore=(
                    RescoreJob(
                        source_run="run_a",
                        target_bundle="data/tok_s12_lay2str",
                        out_run="out_a",
                    ),
                ),
            ),
        ),
    )


def assert_sorted_keys(obj) -> None:
    if isinstance(obj, dict):
        keys = list(obj)
        assert keys == sorted(keys)
        for value in obj.values():
            assert_sorted_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            assert_sorted_keys(value)


def test_bundle_name() -> None:
    assert bundle_name("data/tok_s32_lay") == "tok_s32_lay"
    assert bundle_name("tok_s12_mix2spa") == "tok_s12_mix2spa"


def test_json_round_trip_preserves_plan() -> None:
    plan = make_plan()
    text = plan_to_json(plan)
    assert isinstance(text, str)
    assert text.endswith("\n")
    assert not text.endswith("\n\n")
    assert plan_from_json(text) == plan


def test_json_keys_are_sorted_at_every_level() -> None:
    text = plan_to_json(make_plan())
    assert text == json.dumps(asdict(make_plan()), sort_keys=True, indent=2) + "\n"
    assert_sorted_keys(json.loads(text))


def test_json_lists_become_tuples() -> None:
    plan = plan_from_json(plan_to_json(make_plan()))
    assert isinstance(plan.blocks, tuple)
    block = plan.blocks[0]
    assert isinstance(block.train, tuple)
    assert isinstance(block.rescore, tuple)
    assert isinstance(block.train[0].extra, tuple)


def test_plan_from_json_rejects_unknown_plan_key() -> None:
    data = json.loads(plan_to_json(make_plan()))
    data["bogus"] = 1
    with pytest.raises(ValueError):
        plan_from_json(json.dumps(data))


def test_plan_from_json_rejects_unknown_nested_key() -> None:
    data = json.loads(plan_to_json(make_plan()))
    data["blocks"][0]["train"][0]["surprise"] = True
    with pytest.raises(ValueError):
        plan_from_json(json.dumps(data))


def test_validate_plan_accepts_valid_plan() -> None:
    validate_plan(make_plan())


def test_validate_plan_rejects_duplicate_run_names() -> None:
    block = Block(
        source_bundle="data/tok_s12_lay",
        train=(
            TrainJob("dup", 1, 1, 1, 1, 0, 0),
            TrainJob("dup", 1, 1, 1, 1, 1, 1),
        ),
    )
    with pytest.raises(ValueError):
        validate_plan(Plan("p", (block,)))


def test_validate_plan_rejects_dangling_source_run() -> None:
    block = Block(
        source_bundle="data/tok_s12_lay",
        train=(TrainJob("run_a", 1, 1, 1, 1, 0, 0),),
        rescore=(RescoreJob("missing", "data/tok_s12_lay2str", "out_a"),),
    )
    with pytest.raises(ValueError):
        validate_plan(Plan("p", (block,)))


def test_validate_plan_rejects_out_run_clash() -> None:
    block = Block(
        source_bundle="data/tok_s12_lay",
        train=(TrainJob("run_a", 1, 1, 1, 1, 0, 0),),
        rescore=(RescoreJob("run_a", "data/tok_s12_lay2str", "run_a"),),
    )
    with pytest.raises(ValueError):
        validate_plan(Plan("p", (block,)))


def test_validate_plan_rejects_duplicate_out_run_names() -> None:
    block = Block(
        source_bundle="data/tok_s12_lay",
        train=(TrainJob("run_a", 1, 1, 1, 1, 0, 0),),
        rescore=(
            RescoreJob("run_a", "data/tok_s12_lay2str", "same"),
            RescoreJob("run_a", "data/tok_s12_lay2spa", "same"),
        ),
    )
    with pytest.raises(ValueError):
        validate_plan(Plan("p", (block,)))
