import pytest

from cfg_reducer.reducibility import is_reducible


@pytest.mark.parametrize(("nodes", "edges", "expected"), [
    (("s",), (), True),
    (("s", "a", "b"), (("s", "a"), ("a", "b")), True),
    (("s", "a", "b", "x"), (("s", "a"), ("s", "b"), ("a", "x"), ("b", "x")), True),
    (("s",), (("s", "s"),), True),
    (("s", "h", "a", "b"), (("s", "h"), ("h", "a"), ("a", "b"), ("b", "a"), ("b", "h")), True),
    (("s", "p", "h", "a"), (("s", "p"), ("s", "h"), ("p", "h"), ("h", "a"), ("a", "h")), True),
    (("s", "a", "b"), (("s", "a"), ("s", "b"), ("a", "b"), ("b", "a")), False),
])
def test_reducibility_cases(nodes, edges, expected):
    for ns in (nodes, nodes[::-1]):
        for es in (edges, edges[::-1]):
            assert is_reducible(ns, es, entry="s") is expected


def test_nested_multi_entry():
    nodes = ("s", "h", "a", "b")
    edges = (("s", "h"), ("h", "a"), ("h", "b"), ("a", "b"), ("b", "a"), ("a", "h"))
    for ns in (nodes, nodes[::-1]):
        for es in (edges, edges[::-1]):
            assert not is_reducible(ns, es, entry="s")


@pytest.mark.parametrize(("nodes", "edges", "entry"), [
    ((), (), "s"),
    (("s",), (), "missing"),
    (("s",), (("s", "missing"),), "s"),
    (("s",), (("missing", "s"),), "s"),
    (("s", "isolated"), (), "s"),
])
def test_invalid_cfg(nodes, edges, entry):
    for ns in (nodes, nodes[::-1]):
        for es in (edges, edges[::-1]):
            with pytest.raises(ValueError):
                is_reducible(ns, es, entry=entry)
