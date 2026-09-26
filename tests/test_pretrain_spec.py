import json
from pathlib import Path

import pytest

from cfg_reducer.dataset_v2 import DatasetSpec, dataset_spec_from_json
from cfg_reducer.generate_v2 import spec_from_json
from experiments.pretrain.make_spec import main, params_for_n, CHOICES, WEIGHTS


@pytest.mark.parametrize('n,loops,gotos', [(3, 0, 0), (6, 0, 0), (8, 1, 0), (12, 1, 1),
                                        (48, 7, 4), (192, 28, 19), (256, 38, 25)])
def test_rule_table(n, loops, gotos):
    components = params_for_n(n)['components']
    assert [c['weight'] for c in components] == [1, 1, 1]
    assert components[0]['params'] == dict(loop_count=3*n//20, goto_count=n//10,
        max_layer_width=3, edge_prob=0.18, span_mode='uniform')
    for c in components[1:]:
        assert c['params']['loop_count'] == loops
        assert c['params']['goto_count'] == gotos
        assert 3 + 3 * loops + 2 * gotos <= 11 * n // 10
    spec_from_json({'family': 'mixture', 'num_nodes': n, 'params': params_for_n(n)})


def test_make_specs(tmp_path):
    invalid: tuple = (0, 2, True, 3.5)
    for n in invalid:
        with pytest.raises(ValueError):
            params_for_n(n)
    main(['--out', str(tmp_path)])
    assert len(list(tmp_path.glob('*.json'))) == 12
    for n in (*CHOICES, 192, 256):
        spec = spec_from_json(json.loads((tmp_path / f'n{n}.json').read_bytes()))
        if n in (12, 48):
            expected = Path(__file__).parents[1] / f'experiments/sweep/spec_mixed_n{n}.json'
            assert spec == spec_from_json(json.loads(expected.read_bytes()))
    mixed = dataset_spec_from_json(json.loads((tmp_path / 'mixed.json').read_bytes()))
    assert isinstance(mixed, DatasetSpec)
    assert mixed.num_nodes.choices == CHOICES and mixed.num_nodes.weights == WEIGHTS
    assert len(mixed.specs) == len(CHOICES)
