"""Mixture normalization, RNG isolation, and unchanged dataset integration."""

from collections import Counter
import json
import os
from random import Random
import subprocess
import sys

import pytest

from cfg_reducer.dataset_v2 import main
from cfg_reducer.families.mixture import Mixture, component_for
from cfg_reducer.family_registry import family_names, get_family
from cfg_reducer.generate_v2 import descriptor_for, normalize_spec, spec_to_json
from cfg_reducer.generator_types import CFGShape, GeneratorSpec, Json


def mixture_spec():
    return GeneratorSpec('mixture', 24, {'components': [
        {'family': 'layered', 'weight': 1},
        {'family': 'structured', 'weight': 1},
        {'family': 'spaghetti', 'weight': 2, 'params': {'spaghetti_rate': 0}},
    ]})


def test_normalize():
    assert 'mixture' in family_names()
    spec = mixture_spec()
    normalized = normalize_spec(spec)
    assert normalize_spec(normalized) == normalized
    assert 'params' not in spec.params['components'][0]
    components = normalized.params['components']
    assert isinstance(components, list)
    for component in components:
        assert isinstance(component, dict)
        assert isinstance(component['family'], str)
        params: dict[str, Json] = (
            {} if component['family'] != 'spaghetti' else {'spaghetti_rate': 0})
        assert component['params'] == normalize_spec(
            GeneratorSpec(component['family'], 24, params)).params
        assert type(component['weight']) is float
    assert [c['weight'] for c in components if isinstance(c, dict)] == [1., 1., 2.]
    assert descriptor_for(spec).name == 'cfg_v2:mixture'


@pytest.mark.parametrize('params', [
    {}, {'components': []}, {'components': None}, {'components': {}},
    {'components': 'layered'}, {'components': [None]},
    {'components': [{'family': 'layered'}]},
    {'components': [{'weight': 1}]},
    {'components': [{'family': 'layered', 'weight': 1, 'unknown': 0}]},
    {'components': [{'family': 'layered', 'weight': 1}], 'unknown': 0},
    *[{'components': [{'family': family, 'weight': 1}]}
      for family in ('unknown', 'mixture', 1, None, True)],
    *[{'components': [{'family': 'layered', 'weight': weight}]}
      for weight in (0, -1, float('nan'), float('inf'), -float('inf'),
                     True, None, '1', [], {}, 10 ** 400)],
    *[{'components': [{'family': 'layered', 'weight': 1, 'params': params}]}
      for params in (None, [], 1, {'unknown': 1}, {'loop_count': True})],
])
def test_invalid(params):
    with pytest.raises(ValueError):
        normalize_spec(GeneratorSpec('mixture', 24, params))


def test_wrong_family():
    with pytest.raises(ValueError):
        Mixture().normalize(GeneratorSpec('layered', 24))


@pytest.mark.parametrize('seed', [True, 1.0, '1'])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        component_for(mixture_spec(), seed)


def test_component_and_generation_agree():
    spec = normalize_spec(mixture_spec())
    for seed in range(200):
        rng = Random(seed)
        u = rng.random()
        expected_index = 0 if u < .25 else 1 if u < .5 else 2
        child_seed = rng.getrandbits(64)
        index = component_for(spec, seed)
        assert index == expected_index
        components = spec.params['components']
        assert isinstance(components, list)
        component = components[index]
        assert isinstance(component, dict)
        assert isinstance(component['family'], str)
        assert isinstance(component['params'], dict)
        child = GeneratorSpec(component['family'], spec.num_nodes, component['params'])
        expected = get_family(child.family).generate(child, Random(child_seed))
        parent = Random(seed)
        actual = Mixture().generate(spec, parent)
        assert actual == expected == Mixture().generate(spec, Random(seed))
        assert parent.getstate() == rng.getstate()


def test_weights():
    counts = Counter(component_for(mixture_spec(), seed) for seed in range(1000))
    for index, expected in enumerate((.25, .25, .5)):
        assert abs(counts[index] / 1000 - expected) <= .05
    for weights in ((1e308, 1e308, 1e308), (5e-324, 5e-324, 5e-324)):
        spec = GeneratorSpec('mixture', 24, {'components': [
            {'family': 'layered', 'weight': weight} for weight in weights]})
        assert {component_for(spec, seed) for seed in range(30)} == {0, 1, 2}


def test_plugin_shape_preserved_and_component_for_does_not_generate(monkeypatch):
    from cfg_reducer import family_registry

    shape = CFGShape(('z', 'a'), (('a', 'z'),), 'a')

    class Plugin:
        name = 'mixture_test'

        def normalize(self, spec):
            assert spec.num_nodes == 2
            return spec

        def generate(self, spec, rng):
            return shape

    monkeypatch.setattr(family_registry, '_families', dict(family_registry._families))
    plugin = Plugin()
    family_registry.register_family(plugin)
    spec = GeneratorSpec('mixture', 2, {'components': [
        {'family': plugin.name, 'weight': 1}]})
    assert Mixture().generate(spec, Random(0)) is shape

    def fail(*args):
        raise AssertionError('component_for must not generate')

    monkeypatch.setattr(plugin, 'generate', fail)
    assert component_for(spec, 0) == 0


def test_hashseed():
    script = '''
import json
from dataclasses import asdict
from random import Random
from cfg_reducer.families.mixture import Mixture, component_for
from cfg_reducer.generator_types import GeneratorSpec
spec = GeneratorSpec('mixture', 24, {'components': [
    {'family': family, 'weight': weight}
    for family, weight in [('layered', 1), ('structured', 1), ('spaghetti', 2)]]})
print(json.dumps([(component_for(spec, seed), asdict(Mixture().generate(spec, Random(seed))))
                  for seed in range(20)], sort_keys=True, allow_nan=False))
'''
    outputs = [subprocess.check_output([sys.executable, '-c', script],
               env=os.environ | {'PYTHONHASHSEED': value}) for value in ('1', '77')]
    assert outputs[0] == outputs[1]


def test_dataset_cli(tmp_path):
    spec = mixture_spec()
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec_to_json(spec)))
    out = tmp_path / 'dataset'
    main(['--spec', str(path), '--out', str(out), '--split', 'train=0:20',
          '--version', 'test-mixture'])
    manifest = json.loads((out / 'manifest.json').read_text())
    config = {'spec': spec_to_json(normalize_spec(spec))}
    assert manifest['generator']['name'] == 'cfg_v2:mixture'
    assert manifest['generator']['config'] == config
    samples = manifest['splits']['train']['samples']
    assert samples
    for sample in samples:
        assert type(sample['realized']['reducible']) is bool
        payload = json.loads((out / 'train' / f"{sample['sample_id']}.json").read_text())
        assert payload['provenance']['generator']['name'] == 'cfg_v2:mixture'
        assert payload['provenance']['generator']['config'] == config
