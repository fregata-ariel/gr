from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from cfg_reducer.families.structured import Stmt, lower_structure
from cfg_reducer.reducibility import is_reducible


A = Stmt("atom")


def check_shape(tree, expected, merge_degree=3):
    shape = lower_structure(tree, merge_degree)
    edges = tuple(sorted((f"N{u:02d}", f"N{v:02d}") for u, v in expected))
    assert shape.edges == edges
    assert shape.nodes == tuple(f"N{i:02d}" for i in range(len(shape.nodes)))
    assert shape.entry == "N00"
    assert max(Counter(v for _, v in shape.edges).values()) <= merge_degree
    # The predicate also rejects any unreachable node.
    assert is_reducible(shape.nodes, shape.edges, entry=shape.entry)
    assert is_reducible(shape.nodes[::-1], shape.edges[::-1], entry=shape.entry)
    return shape


@pytest.mark.parametrize(("tree", "expected"), [
    (A, [(0, 1), (1, 2)]),
    (Stmt("seq"), [(0, 1), (1, 2)]),
    (Stmt("seq", (A, A)), [(0, 1), (1, 2), (2, 3)]),
    (Stmt("if", (A,)), [(0, 1), (1, 2), (1, 3), (3, 2), (2, 4)]),
    (Stmt("if_else", (A, A)), [(0, 1), (1, 3), (1, 4), (3, 2), (4, 2), (2, 5)]),
    (Stmt("switch", (A, A, A)), [(0, 1), (1, 3), (1, 4), (1, 5), (3, 2), (4, 2), (5, 2), (2, 6)]),
    (Stmt("while", (A,)), [(0, 1), (1, 4), (1, 3), (4, 2), (2, 1), (3, 5)]),
    (Stmt("do_while", (A,)), [(0, 1), (1, 4), (4, 2), (2, 1), (2, 3), (3, 5)]),
    (Stmt("seq", (Stmt("if", (Stmt("return"),)), A)),
     [(0, 1), (1, 2), (1, 3), (3, 4), (2, 4)]),
])
def test_template_edges(tree, expected):
    check_shape(tree, expected)


@pytest.mark.parametrize("loop", ["while", "do_while"])
@pytest.mark.parametrize("abrupt", ["break", "continue", "return"])
def test_template_edges_abrupt(loop, abrupt):
    tree = Stmt(loop, (Stmt("seq", (Stmt("if", (Stmt(abrupt),)), A)),))
    target = {"break": 3, "continue": 2, "return": 7}[abrupt]
    expected = [(0, 1), (1, 4), (4, 5), (4, 6), (6, target),
                (5, 2), (2, 1), (3, 7), (1 if loop == "while" else 2, 3)]
    check_shape(tree, expected)
    assert len(lower_structure(tree, 3).nodes) == len(lower_structure(Stmt(loop, (A,)), 3).nodes) + 2


def test_merge_router():
    # Five arms: repeatedly replace the first two sources by a new relay.
    check_shape(Stmt("switch", (A,) * 5), [
        (0, 1), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (3, 8), (4, 8), (8, 9), (5, 9), (9, 10), (6, 10),
        (10, 2), (7, 2), (2, 11),
    ], merge_degree=2)


@pytest.mark.parametrize("abrupt", ["break", "continue", "return"])
def test_nested_context_and_all_destinations(abrupt):
    guarded = Stmt("seq", (Stmt("if", (Stmt(abrupt),)), A))
    tree = Stmt("while", (Stmt("seq", (
        Stmt("do_while", (Stmt("seq", (guarded,) * 5),)), guarded,
    )),))
    shape = lower_structure(tree, 2)
    assert max(Counter(v for _, v in shape.edges).values()) <= 2
    assert is_reducible(shape.nodes, shape.edges, entry=shape.entry)


@pytest.mark.parametrize(("kind", "children"), [
    ("unknown", ()), ("atom", (A,)), ("if", ()), ("if_else", (A,)),
    ("switch", (A, A)), ("while", ()), ("do_while", (A, A)),
    ("break", (A,)), ("continue", (A,)), ("return", (A,)),
])
def test_invalid_stmt(kind, children):
    with pytest.raises(ValueError):
        Stmt(kind, children)


def test_frozen_stmt():
    with pytest.raises(FrozenInstanceError):
        setattr(A, "kind", "seq")


@pytest.mark.parametrize("kind", ["break", "continue"])
def test_abrupt_requires_loop(kind):
    with pytest.raises(ValueError, match="loop context"):
        lower_structure(Stmt("if", (Stmt(kind),)), 2)


@pytest.mark.parametrize("degree", [True, 1, 0, 2.5])
def test_invalid_merge_degree(degree):
    with pytest.raises(ValueError):
        lower_structure(A, degree)
