from collections import Counter
from random import Random

import pytest

from cfg_reducer.families.structured import Stmt, Structured, lower_structure, plan_structure
from cfg_reducer.generator_types import GeneratorSpec


def counts(tree: Stmt, depth: int = 0) -> tuple[Counter[str], int]:
    depth += tree.kind in ('while', 'do_while')
    result, maximum = Counter([tree.kind]), depth
    for child in tree.children:
        child_counts, child_depth = counts(child, depth)
        result.update(child_counts)
        maximum = max(maximum, child_depth)
    return result, maximum


@pytest.mark.parametrize('loops', [0, 1, 2, 4])
@pytest.mark.parametrize('depth', [0, 1, 2, 3])
@pytest.mark.parametrize('gotos', [0, 1, 3])
def test_plan_counts_depth(loops, depth, gotos):
    spec = GeneratorSpec('structured', 48, {
        'loop_count': loops, 'target_depth': depth, 'goto_count': gotos})
    if not (depth == 0 if loops == 0 else 1 <= depth <= loops):
        with pytest.raises(ValueError, match='target_depth'):
            plan_structure(spec, Random(0))
        return
    for seed in range(5):
        tree = plan_structure(spec, Random(seed))
        actual, maximum = counts(tree)
        assert actual['while'] + actual['do_while'] == loops
        assert maximum == depth
        assert sum(actual[k] for k in ('break', 'continue', 'return')) == gotos
        if not loops:
            assert actual['return'] == gotos
        assert tree == plan_structure(spec, Random(seed))


@pytest.mark.parametrize('params', [
    {'max_layer_width': 0}, {'branch_degree': 1}, {'merge_degree': 1},
    {'loop_count': -1}, {'goto_count': -1}, {'target_depth': True},
    {'target_depth': 1.5}, {'target_depth': -1}, {'span_mode': 'bad'},
    {'max_layer_width': True}, {'branch_degree': 2.5}, {'merge_degree': False},
    {'loop_count': True}, {'goto_count': '1'}, {'spaghetti_rate': 0},
    {'max_layer_width': 1}, {'loop_count': 30},
])
def test_plan_invalid(params):
    with pytest.raises(ValueError):
        plan_structure(GeneratorSpec('structured', 24, params), Random(0))


def test_plan_defaults_and_shared_join():
    family = Structured()
    spec = GeneratorSpec('structured', 48)
    normalized = family.normalize(spec)
    assert family.normalize(normalized) == normalized
    assert len(normalized.params) == 7
    for loops in (0, 1, 2, 4):
        for seed in range(10):
            tree = plan_structure(GeneratorSpec('structured', 48, {
                'loop_count': loops, 'goto_count': 3}), Random(seed))
            assert counts(tree)[1] in (range(1, min(loops, 3) + 1) if loops else [0])
    # Repeated insertion at the single atom must stay flat in the same seq.
    tree = plan_structure(GeneratorSpec('structured', 24, {
        'loop_count': 1, 'target_depth': 1, 'goto_count': 3}), Random(0))
    body = tree.children[0].children[0]
    assert [child.kind for child in body.children] == ['if', 'if', 'if', 'atom']
    assert len(lower_structure(tree, 99).nodes) == 6 + 3 * 3 - 1
    for n in (1, 2, 8):
        with pytest.raises(ValueError):
            plan_structure(GeneratorSpec('structured', n), Random(0))
