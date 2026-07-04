from cfg_reducer.gen import build_cfg
from main import build_cfg as main_build_cfg


def _edge_set(engine) -> set[tuple[str, str]]:
    return {
        (src, dst)
        for src in engine.node_ids()
        for dst in engine.successors(src)
    }


def test_build_cfg_is_importable_from_cfg_reducer_gen_and_main_reexport():
    assert build_cfg is main_build_cfg


def test_build_cfg_is_seed_deterministic():
    first = build_cfg(num_nodes=10, edge_prob=0.2, seed=7)
    second = build_cfg(num_nodes=10, edge_prob=0.2, seed=7)

    assert _edge_set(first) == _edge_set(second)


def test_build_cfg_creates_requested_number_of_nodes():
    engine = build_cfg(num_nodes=9, edge_prob=0.2, seed=3)

    assert len(engine.node_ids()) == 9
