#!/usr/bin/env python3
"""Frozen C11 evaluator. Independent NumPy forward/scoring; never fits or generates.

Only the pinned C9 profiling validators are imported; C9/C10 numerical/fitting
code is never imported. Qualification does not open any confirmation panel.
"""
import argparse
import csv
import ctypes
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import signal
import struct
import sys
import time

sys.dont_write_bytecode = True
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np

R = Path(__file__).resolve().parent
RESEARCH = R.parent
RUNS = Path('/home/stepan/Projects/code/.tofy-runs')
R8 = RESEARCH / '2026-09-09T104547Z-tofy-looped-frozen-features'
R9 = RESEARCH / '2026-09-09T113612Z-tofy-looped-learned-readout'
R10 = RESEARCH / '2026-09-09T124206Z-tofy-looped-role-selector-witness'
C8 = RUNS / 'looped-frozen-features-20260909T115455-IST'
C9 = RUNS / 'looped-learned-readout-20260909T124221-IST'
C10 = R10 / 'evidence-01'
REG_SHA = 'ce687312edae9ed9edb914217d0c6383fca8b6c9ece1c3a72a6d8e12a68d6a2b'
DEPENDENCY = '1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a'
PARENTS = dict(c8='ebae6f94562879ed343895dc09e9243ab616900f5f8f5e64b54f7ac408cf9e60',
               c9='ea0fb1e773400363e52e3b6966600fde5d8e0ffe37a14547e49772d2186fe477',
               c10='76565de6508b903ff19537a7f63352f1d649debef89abf1773b93096f8b727e8')
CORES = dict(initial='4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802',
             final='a983626a888279cbb31651c6abcd1ebf4f808c8927a29a8349d03ffc35bd4e5a')
PRODUCERS = dict(c10=('82ac8cb6ea3a06ae13d836d526c47994cd675d18', 'a49151b2cd1294aa86fc72433755feb92e3d6b855796e8ee157e2402b51ac748'),
                 c9=('06a8d76fada203c5b1a11a45782ed777df27ff4c', 'fecbc1b91099ddc9b39ac9b6361ee8072c26563f6b710e06c29ec7b3adeef20e'))
HISTORY = ('b7f33c8d3968f55bd1298aeda53f91849d40275b42a3aaf53facadf995e7b938',
           '0613fecd567a0887a94a7698e5a5b7c7d9776f6b8f6ee249f56f5da6a8ce77d5',
           '68a010fc546f727782bf7de178ec8850d48253306aca185a1d337609c53bc697',
           '6379faa5b6b9df51876852bbf0c7dbbda60bb4a8d72ef20a9fecb96d7c1e2f00',
           '09559eb4dd1b933de724da81aef3bb80547e010a00e38e39de31c9f72fab1c6d',
           '361092818769307bcd82f1f30e10808e73857be7a96844f1293f2021d0410d87')
ARMS = ('c10_true', 'c10_null', 'c9_spatial', 'c9_cls', 'c9_null')
FIELDS = ('input_index', 'episode_id', 'partition', 'input_sha256', 'query_sha256', 'label_sha256', 'correct_action')
SHAPES = dict(spatial={'queries': (2, 128), 'output.weight': (4, 256), 'output.bias': (4,)},
              cls={'hidden.weight': (10, 128), 'hidden.bias': (10,), 'output.weight': (4, 10), 'output.bias': (4,)})
SNAPSHOT = {}


class Invalid(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise Invalid(reason)


def exact(actual, expected, label='value'):
    require(type(actual) is type(expected), f'{label}: wrong type')
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f'{label}: wrong fields')
        for k, v in expected.items():
            exact(actual[k], v, f'{label}.{k}')
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f'{label}: wrong length')
        for i, (a, e) in enumerate(zip(actual, expected)):
            exact(a, e, f'{label}[{i}]')
    else:
        require(actual == expected, f'{label}: differs')


def subset(actual, expected, label='record'):
    for k, v in expected.items():
        require(k in actual, f'{label}: missing {k}')
        exact(actual[k], v, f'{label}.{k}')


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest(path, expected=None):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), f'missing/nonregular input: {path}')
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    if expected is not None:
        require(valid_sha(expected) and value == expected, f'hash differs: {path}')
    require(str(path) not in SNAPSHOT or SNAPSHOT[str(path)] == value, f'input changed during analysis: {path}')
    SNAPSHOT[str(path)] = value
    return value


def read(path):
    digest(path)
    def bad(value):
        raise Invalid(f'nonfinite JSON: {value}')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=bad, object_pairs_hook=unique)


def rows(path):
    digest(path)
    return [json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(Invalid(v)))
            for line in Path(path).read_text().splitlines()]


def tree(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), f'bad root: {root}')
    found = set()
    for path in root.rglob('*'):
        require(not path.is_symlink(), f'linked artifact: {path}')
        if path.is_file():
            found.add(str(path.relative_to(root)))
        else:
            require(path.is_dir(), f'nonregular artifact: {path}')
    return found


def checked_files(root, files, *, includes_manifest=False):
    root = Path(root)
    require(tree(root) == set(files) | (set() if includes_manifest else {'manifest.json'}), 'artifact population differs')
    for rel, info in files.items():
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts, 'manifest path escape')
        path = root / rel
        if isinstance(info, dict):
            require(type(info['bytes']) is int and path.stat().st_size == info['bytes'], f'wrong byte count: {path}')
            info = info['sha256']
        digest(path, info)


def manifest(root, expected=None):
    root = Path(root)
    if expected is None:
        pin = root.with_suffix('.manifest.sha256')
        digest(pin)
        expected = pin.read_text().strip()
    digest(root / 'manifest.json', expected)
    result = read(root / 'manifest.json')
    checked_files(root, result['files'])
    return result, expected


def binary_array(path, dtype, shape):
    digest(path)
    require(Path(path).stat().st_size == math.prod(shape) * np.dtype(dtype).itemsize, f'array byte count: {path}')
    array = np.fromfile(path, dtype=dtype).reshape(shape)
    require(np.isfinite(array).all(), f'nonfinite array: {path}')
    return array


def descriptor(file, shape, index, *, dtype='F32LE'):
    size = math.prod(shape) * 4
    return dict(file=file, dtype=dtype, shape=list(shape), byte_offset=index * size, byte_length=size)


def safe_tensors(path):
    digest(path)
    raw = Path(path).read_bytes()
    require(len(raw) >= 8, 'short safetensors header')
    length, = struct.unpack('<Q', raw[:8])
    require(0 < length <= len(raw) - 8, 'safetensors header length')
    header = json.loads(raw[8:8 + length])
    data, output, intervals = raw[8 + length:], {}, []
    for name, spec in header.items():
        if name == '__metadata__':
            continue
        require(spec['dtype'] == 'F32' and all(type(v) is int and v > 0 for v in spec['shape']), 'safetensors dtype/shape')
        start, end = spec['data_offsets']
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(data)
                and end - start == math.prod(spec['shape']) * 4, 'safetensors offset')
        output[name] = np.frombuffer(data[start:end], dtype='<f4').reshape(spec['shape'])
        require(np.isfinite(output[name]).all(), 'nonfinite parameter')
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        require(start == cursor, 'overlapping/gapped parameter payload')
        cursor = end
    require(cursor == len(data), 'unclaimed parameter bytes')
    return output


def imported_parameters(item, core, arm):
    root = Path(item['root'])
    require(root.is_absolute() and tree(root) == {'manifest.json', 'parameters.f32'}, 'import file boundary')
    digest(root / 'manifest.json', item['manifest_sha256'])
    m = read(root / 'manifest.json')
    kind, parent = ('cls' if arm == 'c9_cls' else 'spatial'), ('c10' if arm.startswith('c10_') else 'c9')
    original = C10 / f'selector-{core}-{arm[4:]}.npz' if parent == 'c10' else C9 / f"fit-{core}-{dict(c9_spatial='spatial', c9_cls='cls', c9_null='spatial-null')[arm]}" / 'final.safetensors'
    subset(m, dict(schema='looped-imported-readout-source-v1', arm=arm, head_kind=kind,
                   core_checkpoint=core, core_checkpoint_sha256=CORES[core],
                   source_kind='c10_role_ridge' if parent == 'c10' else 'c9_adamw',
                   parent_manifests=PARENTS,
                   producer=dict(source_revision=PRODUCERS[parent][0], implementation_sha256=PRODUCERS[parent][1]),
                   recipe=dict(name='c10_role_ridge_c8_affine_v1' if parent == 'c10' else 'c9_adamw1000_v1',
                               original_optimizer_updates=0 if parent == 'c10' else 1000,
                               privileged_role_supervision=parent == 'c10',
                               cast='f64_to_f32_once_no_rescale' if parent == 'c10' else 'identity_f32')))
    exact(m['original_artifact']['path'], str(original))
    digest(original, m['original_artifact']['sha256'])
    raw = (root / 'parameters.f32').read_bytes()
    exact(m['artifact'], dict(file='parameters.f32', bytes=len(raw), sha256=digest(root / 'parameters.f32')))
    require(len(m['tensors']) == len(SHAPES[kind]), 'import tensor count')
    parameters, offset = {}, 0
    for spec, (name, shape) in zip(m['tensors'], SHAPES[kind].items()):
        size = math.prod(shape) * 4
        block = raw[offset:offset + size]
        exact(spec, dict(name=name, dtype='F32LE', shape=list(shape), byte_offset=offset,
                         byte_length=size, sha256=sha_bytes(block)), 'import tensor')
        require(len(block) == size, 'short import tensor')
        parameters[name] = np.frombuffer(block, dtype='<f4').reshape(shape)
        require(np.isfinite(parameters[name]).all(), 'nonfinite imported parameter')
        offset += size
    require(offset == len(raw), 'trailing parameter bytes')
    if parent == 'c10':
        with np.load(original, allow_pickle=False) as source:
            mapped = {name: np.asarray(source[name.replace('.', '_')], dtype='<f4') for name in SHAPES[kind]}
    else:
        mapped = safe_tensors(original)
        require(set(mapped) == set(parameters), 'foreign C9 tensor population')
    for name, value in parameters.items():
        require(mapped[name].shape == value.shape and mapped[name].tobytes() == value.tobytes(), 'export changed original parameter function')
    return dict(root=root, manifest=m, manifest_sha256=item['manifest_sha256'], kind=kind, parameters=parameters)


def forward(parameters, features, kind):
    x = np.asarray(features, dtype=np.float32)
    attention = None
    if kind == 'spatial':
        require(x.ndim == 3 and x.shape[1:] == (64, 128), 'spatial feature shape')
        scores = np.matmul(parameters['queries'][None, :, :], x.transpose(0, 2, 1)) * np.float32(1 / math.sqrt(128))
        attention = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attention = attention / attention.sum(axis=-1, keepdims=True, dtype=np.float32)
        pooled = np.matmul(attention, x).reshape(len(x), 256)
    else:
        require(x.ndim == 2 and x.shape[1:] == (128,), 'CLS feature shape')
        hidden = x @ parameters['hidden.weight'].T + parameters['hidden.bias']
        # Stable F32 sigmoid, including large negative finite preactivations.
        sigmoid = np.empty_like(hidden)
        positive = hidden >= 0
        sigmoid[positive] = 1 / (1 + np.exp(-hidden[positive]))
        negative_exp = np.exp(hidden[~positive])
        sigmoid[~positive] = negative_exp / (1 + negative_exp)
        pooled = hidden * sigmoid
    logits = pooled @ parameters['output.weight'].T + parameters['output.bias']
    for value in (logits, pooled, attention):
        require(value is None or (value.dtype == np.float32 and np.isfinite(value).all()), 'nonfinite/non-F32 CPU forward')
    return dict(logits=logits, pooled=pooled, attention=attention)


def parity(actual, reference, *, attention=False, actions=False):
    require(actual.shape == reference.shape and np.isfinite(actual).all() and np.isfinite(reference).all(), 'parity shape/nonfinite')
    error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    band = (1e-5 if attention else 1e-4) + 1e-5 * np.abs(reference.astype(np.float64))
    require(np.all(error <= band), 'elementwise numerical parity failed')
    if actions:
        require(np.array_equal(actual.argmax(axis=-1), reference.argmax(axis=-1)), 'argmax parity failed')
    return dict(max_absolute_error=float(error.max(initial=0)), max_fraction_of_tolerance=float((error / band).max(initial=0)), identical_argmax=True if actions else None)


def cross_entropy(logits, labels):
    values = logits.astype(np.float64)
    peak = values.max(axis=1)
    return peak + np.log(np.exp(values - peak[:, None]).sum(axis=1)) - values[np.arange(len(labels)), labels]


def wilson(correct, count):
    require(0 <= correct <= count and count > 0, 'Wilson count')
    z = 1.959963984540054
    p, denom = correct / count, 1 + z * z / count
    center = (p + z * z / (2 * count)) / denom
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denom
    return [max(0., center - half), min(1., center + half)]


def endpoint(logits, labels):
    predictions = logits.argmax(axis=1)
    return dict(correct=predictions == labels, ce=cross_entropy(logits, labels))


def endpoint_summary(value):
    correct = int(value['correct'].sum())
    return dict(rows=len(value['correct']), correct=correct, accuracy=float(value['correct'].mean()),
                mean_ce=float(value['ce'].mean()), wilson95=wilson(correct, len(value['correct'])))


def localization(attention, roles):
    require(attention.shape == (len(roles), 2, 64) and np.isfinite(attention).all(), 'attention shape/nonfinite')
    require(np.all(attention >= 0) and np.all(attention <= 1) and np.allclose(attention.sum(axis=2), 1, atol=1e-5, rtol=1e-5), 'attention probability boundary')
    hits = attention.argmax(axis=2) == roles
    masses = attention[np.arange(len(roles))[:, None], np.arange(2)[None, :], roles]
    ties = (attention == attention.max(axis=2, keepdims=True)).sum(axis=2)
    return dict(agent=hits[:, 0], goal=hits[:, 1], joint=hits.all(axis=1)), dict(
        roles=['agent', 'goal'], mean_target_mass=masses.mean(axis=0).tolist(),
        minimum_target_mass=masses.min(axis=0).tolist(), maximum_argmax_ties=ties.max(axis=0).tolist(),
        rows_with_argmax_ties=(ties > 1).sum(axis=0).tolist(), localization_source='retained CUDA attention.f32; first argmax on ties')


def bootstrap_indices(panel, draws=10000):
    require(panel is None or type(panel) is int and panel in range(3), 'bootstrap panel')
    rng = np.random.Generator(np.random.PCG64(1919 if panel is None else 1920 + panel))
    if panel is not None:
        return rng.integers(0, 256, size=(draws, 256))
    return (rng.integers(0, 256, size=(draws, 3, 256)) + np.arange(3)[None, :, None] * 256).reshape(draws, 768)


def interval(values):
    return np.quantile(values, [.025, .975], method='linear').tolist()


def best_constant_draws(labels, indices):
    sampled = labels[indices]
    return np.stack([(sampled == action).sum(axis=1) for action in range(4)], axis=1).max(axis=1) / indices.shape[1]


def comparisons(endpoints, labels, role_hits, panel):
    indices = bootstrap_indices(panel)
    constants = best_constant_draws(labels, indices)
    constant_accuracy = float(np.bincount(labels, minlength=4).max() / len(labels))
    output = dict(seed=1919 if panel is None else 1920 + panel, draws=10000, sampling='stratified whole-query' if panel is None else 'whole-query',
                  quantiles='linear pointwise 95%', shared_across_all_cores_and_heads=True,
                  constant=dict(correct=int(np.bincount(labels, minlength=4).max()), accuracy=constant_accuracy,
                                action=int(np.bincount(labels, minlength=4).argmax()), reselected_each_draw=True, bootstrap95=interval(constants)),
                  endpoints={}, paired={}, roles={}, paired_roles={})
    draws = {}
    for key, value in endpoints.items():
        accuracy = value['correct'][indices].mean(axis=1)
        ce = value['ce'][indices].mean(axis=1)
        draws[key] = dict(accuracy=accuracy, mean_ce=ce)
        output['endpoints'][key] = dict(**endpoint_summary(value), accuracy_bootstrap95=interval(accuracy), mean_ce_bootstrap95=interval(ce),
                                      advantage_over_constant=dict(estimate=float(value['correct'].mean()) - constant_accuracy, bootstrap95=interval(accuracy - constants)))
    # All pairs within each core, plus every final-minus-initial endpoint.
    keys = list(endpoints)
    pairs = [(a, b) for a, b in itertools.combinations(keys, 2) if a.split('/')[0] == b.split('/')[0]]
    pairs += [('final/' + arm, 'initial/' + arm) for arm in ('c10_true', 'c10_null', 'c9_spatial', 'c9_cls', 'c9_null', 'native', 'oracle_role')]
    for a, b in pairs:
        require(a in endpoints and b in endpoints, 'missing paired endpoint')
        output['paired'][a + ' minus ' + b] = {metric: dict(estimate=float(
            endpoints[a]['correct'].mean() - endpoints[b]['correct'].mean() if metric == 'accuracy' else endpoints[a]['ce'].mean() - endpoints[b]['ce'].mean()),
            bootstrap95=interval(draws[a][metric] - draws[b][metric])) for metric in ('accuracy', 'mean_ce')}
    for key, value in role_hits.items():
        output['roles'][key] = {name: dict(correct=int(hit.sum()), accuracy=float(hit.mean()), bootstrap95=interval(hit[indices].mean(axis=1))) for name, hit in value.items()}
    role_pairs = [(c + '/c10_true', c + '/c10_null') for c in CORES]
    role_pairs += [('final/' + arm, 'initial/' + arm) for arm in ('c10_true', 'c10_null')]
    for left, right in role_pairs:
        output['paired_roles'][left + ' minus ' + right] = {
            role: dict(estimate=float(role_hits[left][role].mean() - role_hits[right][role].mean()),
                       bootstrap95=interval((role_hits[left][role].astype(float) - role_hits[right][role])[indices].mean(axis=1)))
            for role in ('agent', 'goal', 'joint')}
    return output


def registered_decision(panels):
    require(set(panels) == {'0', '1', '2'}, 'decision requires exactly three panels')
    components = {}
    for panel, result in panels.items():
        require(result['geometry_correct'] == 256 and type(result['geometry_correct']) is int, 'geometry integrity failure')
        for core in CORES:
            e, r = result['statistics']['endpoints'], result['statistics']['roles']
            true = e[core + '/c10_true']
            pair = result['statistics']['paired'][core + '/c10_true minus ' + core + '/c9_spatial']['accuracy']
            controls = dict(oracle_role=e[core + '/oracle_role']['accuracy'] >= .90,
                            c10_null_role=r[core + '/c10_null']['joint']['accuracy'] <= .10,
                            c10_null_action=e[core + '/c10_null']['accuracy'] <= .50,
                            c9_null_action=e[core + '/c9_null']['accuracy'] <= .50)
            thresholds = dict(true_joint_role=r[core + '/c10_true']['joint']['accuracy'] >= .95,
                              true_action=true['accuracy'] >= .90,
                              advantage_constant=true['advantage_over_constant']['bootstrap95'][0] > .25,
                              advantage_c9_spatial=pair['bootstrap95'][0] > .25)
            components[panel + '/' + core] = dict(controls=controls, thresholds=thresholds,
                controls_pass=all(controls.values()), thresholds_pass=all(thresholds.values()))
    control_pass = all(v['controls_pass'] for v in components.values())
    threshold_pass = all(v['thresholds_pass'] for v in components.values())
    decision = 'inconclusive_failed_control' if not control_pass else ('confirmed_on_registered_synthetic_populations' if threshold_pass else 'not_supported')
    return dict(decision=decision, controls_pass=control_pass, thresholds_pass=threshold_pass, components=components,
                pooled_cannot_rescue=True, source_numerical_data_profile_integrity_pass=True)


def projected_rows(records, count, panel):
    require(len(records) == count, 'projected row count')
    tag = 0x46454154555245 if panel is None else 0x43554441434f4e46 + panel * 0x10000
    partition = 'fit' if panel is None else 'confirmation_eval'
    for i, row in enumerate(records):
        require(set(row) == set(FIELDS), 'privileged/unknown projected row fields')
        subset(row, dict(input_index=i, episode_id=tag + i, partition=partition))
        require(type(row['correct_action']) is int and row['correct_action'] in range(4), 'label dtype/range')
        for key in ('input_sha256', 'query_sha256', 'label_sha256'):
            require(valid_sha(row[key]), 'invalid row hash')
        exact(row['label_sha256'], sha_bytes(struct.pack('<I', row['correct_action'])), 'label hash')
    for key in ('query_sha256', 'input_sha256', 'episode_id'):
        require(len({r[key] for r in records}) == count, 'duplicate projected identity')
    labels = np.array([r['correct_action'] for r in records], dtype=np.int64)
    require(np.bincount(labels, minlength=4).min() > 0, 'missing action class')
    return labels


def cache(item, core, panel, config, source_rows=None):
    root = Path(item['root'])
    m, sha = manifest(root, item['manifest_sha256'])
    count = 512 if panel is None else 256
    expected_root = C9 / 'caches' / ('fit-' + core) if panel is None else Path(config['campaign']) / 'caches' / str(panel) / core
    exact(str(root), str(expected_root), 'cache path')
    subset(m, dict(schema='looped-learned-readout-cache-v1' if panel is None else 'looped-imported-readout-cache-v1',
                   rows=count, partition='fit' if panel is None else 'confirmation_eval', core_checkpoint_sha256=CORES[core]))
    require(set(m['files']) == {'cls.f32', 'current.f32', 'rows.jsonl'}, 'cache contains privileged/nonfeature files')
    source = m['source']
    parent = C8 / ('features-' + core) if panel is None else Path(config['confirmation']['features'][f'{panel}/{core}'])
    expected_source = dict(root=str(parent), data_seed=20260915 if panel is None else 20260917 + panel,
                           episode_id_base=0x46454154555245 if panel is None else 0x43554441434f4e46 + panel * 0x10000,
                           feature_schema='looped-known-features-v1',
                           source_revision='2f2aaa711eb8f39eb823cc4e354288c7b4fbcf42' if panel is None else config['source'],
                           binary_sha256='83150acaf7275bab63204d897972194422b2eca01137c9b7752ade456105ea8e' if panel is None else config['binaries']['looped_agent_probe'])
    if panel is not None:
        expected_source['confirmation_panel'] = panel
    exact(source, dict(expected_source, manifest_sha256=source['manifest_sha256']), 'cache source')
    require(set(m['source_arrays']) == {'cls', 'current'}, 'cache source array boundary')
    source_manifest, _ = manifest(parent, source['manifest_sha256'])
    if source_rows is None:
        source_rows = rows(parent / 'known-features-rows.jsonl')[:count]
    require(len(source_rows) == count, 'source row count')
    records = rows(root / 'rows.jsonl')
    labels = projected_rows(records, count, panel)
    exact(records, [{key: r[key] for key in FIELDS} for r in source_rows], 'cache source identity')
    arrays = {}
    for name, shape in (('cls', (128,)), ('current', (64, 128))):
        file = 'known-features-' + name + '.f32'
        size = count * math.prod(shape) * 4
        exact(m['source_arrays'][name], dict(file=file, sha256=source_manifest['files'][file], byte_offset=0, byte_length=size), 'source array binding')
        with (parent / file).open('rb') as stream:
            raw = stream.read(size)
        require(len(raw) == size and sha_bytes(raw) == m['files'][name + '.f32']['sha256'], 'cache bytes differ from source prefix')
        for i, row in enumerate(source_rows):
            exact(row['arrays'][name], descriptor(file, shape, i), 'source array row')
        arrays[name] = binary_array(root / (name + '.f32'), '<f4', (count, *shape))
    return dict(root=root, manifest=m, manifest_sha256=sha, rows=records, labels=labels, arrays=arrays)


def geometry(records, panel=None, *, smoke=False):
    count = 1 if smoke else (512 if panel is None else 256)
    require(len(records) == count, 'geometry population')
    roles, labels = [], []
    for i, row in enumerate(records):
        expected = dict(schema='looped-known-features-v1', input_index=i, layout_index=i,
                        partition='fit' if panel is None else 'confirmation_eval',
                        episode_id=(0x46454154555245 if panel is None else 0x43554441434f4e46 + panel * 0x10000) + i,
                        data_seed=20260915 if panel is None else 20260917 + panel,
                        episode_seed=20260915 if panel is None else 20260917 + panel,
                        permutation_id=0, split='KnownMapping', condition='factual', support_cleared=False,
                        min_distance=1, max_distance=1, oracle_distance=1, evaluation_loops=4, inferred_controls=[0, 1, 2, 3])
        subset(row, expected, 'geometry row')
        support = row['observed_support_action_ids']
        require(len(support) == len(set(support)) == 3 and all(type(v) is int and v in range(4) for v in support), 'support identities')
        cells = row['visible_cells']
        require(len(cells) == 64 and all(type(v) is int and v in range(4) for v in cells), 'visible geometry dtype')
        require(cells.count(2) == cells.count(3) == 1, 'role uniqueness')
        require(all(cells[r * 8 + c] == 1 for r in range(8) for c in range(8) if r in (0, 7) or c in (0, 7)), 'maze border')
        a, g = cells.index(2), cells.index(3)
        dy, dx = g // 8 - a // 8, g % 8 - a % 8
        require(abs(dy) + abs(dx) == 1, 'nonadjacent registered roles')
        label = [(-1, 0), (1, 0), (0, -1), (0, 1)].index((dy, dx))
        exact(row['correct_action'], label, 'coordinate action')
        policy = [float(k == label) for k in range(4)]
        require(row['target_policy'] == policy and row['target_rewards'] == policy and row['target_value'] == 1., 'geometry targets')
        targets = []
        for delta in (-8, 8, -1, 1):
            target = cells.copy()
            dest = a + delta
            if cells[dest] != 1:
                target[a] = 0
                target[dest] = 4 if dest == g else 2
            targets.append(target)
        exact(row['target_cells'], targets, 'independent successor geometry')
        # The serialized input is patch-major: 64 uniform pixels per patch.
        query = np.repeat(np.array(cells, dtype='<u4'), 64)
        exact(row['query_sha256'], sha_bytes(query.tobytes()), 'query image hash')
        target_images = np.repeat(np.array(targets, dtype='<u4'), 64, axis=1)
        target_bytes = target_images.tobytes() + np.array(policy + policy + [1.], dtype='<f4').tobytes()
        exact(row['targets_sha256'], sha_bytes(target_bytes), 'target image hash')
        exact(row['label_sha256'], sha_bytes(struct.pack('<I', label)), 'geometry label hash')
        roles.append([a, g]); labels.append(label)
    if not smoke:
        require(np.bincount(labels, minlength=4).min() > 0, 'geometry missing action class')
    return np.array(roles, dtype=np.int64), np.array(labels, dtype=np.int64)


def timestamp(value):
    result = dt.datetime.fromisoformat(value)
    require(result.tzinfo is not None, 'timestamp missing zone')
    return result.timestamp()


def finite_number(value, minimum=0, maximum=math.inf):
    return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum


def arguments(argv):
    require(isinstance(argv, list) and len(argv) > 1 and all(isinstance(x, str) for x in argv), 'argv shape')
    result, index = {}, 1
    while index < len(argv):
        key = argv[index]
        require(key.startswith('--'), 'positional argument')
        value = True
        if index + 1 < len(argv) and not argv[index + 1].startswith('--'):
            value = argv[index + 1]; index += 1
        key = key[2:]
        if key == 'known-features-exclude':
            result.setdefault(key, []).append(value)
        else:
            require(key not in result, 'duplicate CLI option')
            result[key] = value
        index += 1
    return result


def core_arguments(root, core, panel, *, smoke=False, audit=False):
    exclusion = read(C8 / 'exclusions.json')
    paths = list(exclusion['paths'])
    if not smoke:
        paths += [str(C8 / 'audit-features/known-features-input-audit.jsonl'), str(C9 / 'fresh-audit/known-features-input-audit.jsonl')]
    expected = {'mode': 'known-features-smoke' if smoke else ('known-features-audit' if audit else 'known-features'),
                'known-mapping': True, 'seed': '0', 'loops': '4', 'batch': '1', 'effective-batch': '1',
                'max-seconds': '60' if audit else '120', 'updates': '1', 'eval-episodes': '768' if smoke else '256',
                'data-seed': '20260915' if smoke else str(20260917 + panel), 'profile-eval': 'true',
                'known-features-exclude': paths, 'output-dir': str(root), 'device': 'cpu' if audit else 'cuda:0'}
    if not audit:
        expected['checkpoint'] = str(C8 / ('features-' + core) / 'initial.safetensors')
    if not smoke:
        expected['known-features-confirmation-panel'] = str(panel)
    return expected


def exited(root):
    state = read(root.with_suffix('.exit.json'))
    subset(state, dict(accepted=True, returncode=0, pid_gone=True, model_pid_gone=True, group_gone=True,
                       owned_survivors=[], group_survivors=[], cleanup_error=None, failure=None,
                       report_status='complete_pending_analysis'), 'external exit')
    require(state.get('launch_record_error') is None and state.get('integrity_error') is None, 'external launch/postcheck failure')
    require(finite_number(state['headroom_mib'], 512) and finite_number(state['max_temperature_c'], 0, 84.999999), 'GPU telemetry bounds')
    for pid, tick in state.get('owned_process_start_ticks', {}).items():
        path = Path('/proc') / str(pid) / 'stat'
        if path.exists():
            require(path.read_text().rsplit(')', 1)[1].split()[19] != str(tick), 'owned process survived')
    require(state.get('owned_pids') and state.get('owned_process_start_ticks'), 'owned-process identity evidence missing')
    return state


def common(root, config, *, head, expected_name, audit=False):
    root = Path(root)
    require(root == Path(config['campaign']) / expected_name, 'unregistered invocation root')
    mfest, root_manifest_sha = manifest(root)
    m, r, launch = read(root / 'metadata.json'), read(root / 'report.json'), read(root / 'launch.json')
    state, process = exited(root), read(root.with_suffix('.process.json'))
    exact(state['manifest_sha256'], root_manifest_sha, 'external exit manifest')
    subset(r, dict(status='complete_pending_analysis', optimizer_updates=0))
    name = 'learned_readout_probe' if head else 'looped_agent_probe'
    subset(m['provenance'], dict(source_revision=config['source'], candle_graph_revision=DEPENDENCY, binary_sha256=config['binaries'][name]))
    require({'cudnn', 'profiling'} <= set(m['provenance']['features'].split(',')), 'compiled profiler/backend features')
    require(all(token in m['provenance']['build_command'] for token in ('--release', '--locked', 'cudnn,profiling,serde_json/float_roundtrip')), 'build recipe')
    argv = m['exact_args']; args = arguments(argv)
    exact(argv, launch['exact_args'], 'launch argv')
    exact(argv[0], str(Path(config['campaign']) / name), 'binary path')
    digest(argv[0], config['binaries'][name])
    subset(args, {'output-dir': str(root), 'device': 'cpu' if audit else 'cuda:0'})
    require(process['command'][-len(argv):] == argv and process['binary_sha256'] == config['binaries'][name], 'external process binary/argv')
    subset(process['environment'], dict(NVIDIA_TF32_OVERRIDE='0', TOFY_PERF_TRACE=str(root.with_suffix('.host.json')), NSYS_NVTX_PROFILER_REGISTER_ONLY='0'))
    digest(process['invocation'], process['invocation_sha256'])
    invocation = read(process['invocation'])
    subset(invocation, dict(name=expected_name, binary=name))
    exact(argv[1:], invocation['arguments'] + ['--output-dir', str(root), '--device', 'cpu' if audit else 'cuda:0'], 'registered invocation argv')
    for file, sha in invocation['frozen_files'].items():
        digest(file, sha)
    bound = 60 if head or audit else 120
    require(args['max-seconds'] == str(bound) and finite_number(r['elapsed_seconds'], 0, bound), 'model time budget')
    require(finite_number(state['reported_model_elapsed_seconds'], 0, bound)
            and finite_number(state['model_phase_seconds'], 0, bound + 5)
            and finite_number(state['finalization_seconds'], 0, 60), 'supervised stage time budget')
    require(timestamp(process['started_local']) <= timestamp(state['finished_local']), 'process chronology')
    return dict(root=root, manifest=mfest, manifest_sha256=root_manifest_sha, metadata=m, report=r, args=args,
                exit=state, process=process, started=timestamp(process['started_local']), finished=timestamp(state['finished_local']))


def profile_helper():
    path = R9 / 'readout_analysis.py'
    digest(path, '98a912a64b4a5774344caf840ec7808964a61a280760adec8021467cdfbac275')
    spec = importlib.util.spec_from_file_location('c11_frozen_profile_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profiles(run, helper, kind=None, count=None):
    root = run['root']
    if kind is None:
        result = helper.km.validate_profiles(run, [])
        trace = rows(root.with_suffix('.profiles') / 'evaluation-000001/trace.jsonl')
        helper.p8.validate_feature_trace(trace)
        require(trace[0]['capture_contract']['gradients'] == 'none' and not any(r['kind'] == 'gradient' for r in trace), 'inference gradient evidence')
    else:
        profile_root, capture = root.with_suffix('.profiles'), 'evaluation-000001'
        declared = read(root / 'profiles.json')
        require(Path(declared['root']) == profile_root and tree(profile_root) == set(declared['files']), 'profile declaration population')
        require({name.split('/')[0] for name in declared['files']} == {capture}, 'profile capture cadence')
        for name, sha in declared['files'].items():
            digest(profile_root / name, sha)
        host = declared['host_trace']
        exact(host['path'], str(root.with_suffix('.host.json')), 'host trace path')
        digest(host['path'], host['sha256'])
        original = helper.km.profile_bundle(profile_root / capture)
        bound_root = root.with_suffix('.bound') / capture
        bound = helper.km.profile_bundle(bound_root)
        p = original['provenance']
        subset(p, dict(measured_region_device_synchronized=True, device='cuda', capture_step=1))
        require(p['capture_contract']['measurement_scope'] == 'profiled_work', 'profile scope')
        require(digest(bound_root / 'trace.jsonl') == digest(profile_root / capture / 'trace.jsonl') and bound['gpu']['provenance']['binding'] == 'bound', 'profile trace binding')
        trace = rows(profile_root / capture / 'trace.jsonl')
        helper.head_trace_contract(trace, kind, training=False, count=count)
        exact(p['tags'], trace[0]['tags']); exact(p['capture_contract'], trace[0]['capture_contract'])
        nm = read(bound_root / 'nsight/capture-manifest.json')
        subset(nm, dict(schema='candle-graph/nsight-capture/1', source_revisions=dict(tofy=run['metadata']['provenance']['source_revision'], candle_graph=DEPENDENCY)))
        require(nm['run']['id'] == p['run_id'] and nm['correlation']['id'] == p['correlation_id']
                and run['metadata']['provenance']['gpu'] in nm['hardware']['devices'], 'Nsight run/hardware')
        helper.km.checked_files(bound_root / 'nsight', nm['artifacts'], size_key='size_bytes', exact=False)
        command = shlex.split(nm['commands'][0])
        require(all(x in command for x in ('--trace=cuda,nvtx,osrt,cudnn,cublas', '--sample=process-tree', '--cpuctxsw=process-tree'))
                and command[-len(run['metadata']['exact_args']):] == run['metadata']['exact_args'], 'Nsight command planes')
        required = {'tofy.looped/capture', 'tofy.looped/readout-eval-000000000001/forward'}
        with (bound_root / 'nsight/stats_nvtx_sum.csv').open() as stream:
            labels = {row['Range'].removeprefix(':') for row in csv.DictReader(stream)}
        require(set(p['capture_contract']['required_semantic_labels']) == set(nm['required_semantic_labels']) == required and required <= labels, 'Nsight actual semantic ranges')
        result = dict(host_sha256=host['sha256'], captures={capture: dict(bundle_sha256=digest(profile_root / capture / 'bundle.json'),
                    bound_bundle_sha256=digest(bound_root / 'bundle.json'), gaps=bound['gaps'], automatic_gpu_correlation_complete=bound['gpu']['correlation']['complete'])})
    binding = read(root.with_suffix('.binding.json'))
    require(binding['status'] == 'complete' and type(binding['pid']) is int and finite_number(binding['elapsed_seconds']), 'binder did not complete')
    operation = root.parent / 'operations' / (root.name + '-bind')
    state, process = read(operation.with_suffix('.exit.json')), read(operation.with_suffix('.process.json'))
    subset(state, dict(accepted=True, returncode=0, error=None, cleanup_error=None, pid_gone=True, group_gone=True, owned_survivors=[]), 'binder external exit')
    require(process['pid'] == binding['pid'] and process['command'][-1] == root.name, 'binder external identity')
    require(not (Path('/proc') / str(binding['pid'])).exists(), 'binder PID still present')
    for pid, tick in state['owned_process_start_ticks'].items():
        path = Path('/proc') / str(pid) / 'stat'
        if path.exists():
            require(path.read_text().rsplit(')', 1)[1].split()[19] != str(tick), 'binder child survived')
    require(timestamp(process['started_local']) <= timestamp(binding['finished_local']) <= timestamp(state['finished_local']), 'binder external chronology')
    summary = read(root.with_suffix('.bound') / 'summary.json')
    require(len(summary) == 1 and summary[0]['health']['structurally_valid'] is True
            and summary[0]['health']['capture_complete'] is True and summary[0]['raw_application_labels_verified'] is True, 'bound profile summary unhealthy')
    run['binding'] = binding
    run['finished'] = max(run['finished'], timestamp(state['finished_local']))
    return result


def head_outputs(root, cached, kind):
    count, records, labels = len(cached['rows']), cached['rows'], cached['labels']
    logits = binary_array(root / 'logits.f32', '<f4', (count, 4))
    actual_labels = binary_array(root / 'labels.u32', '<u4', (count, 2))
    ids = binary_array(root / 'episode-ids.u64', '<u8', (count,))
    require(np.array_equal(actual_labels, np.column_stack((labels, labels))) and np.array_equal(ids, [r['episode_id'] for r in records]), 'raw true/fitted labels or IDs differ')
    pooled = binary_array(root / 'pooled.f32', '<f4', (count, 10 if kind == 'cls' else 256))
    attention = binary_array(root / 'attention.f32', '<f4', (count, 2, 64)) if kind == 'spatial' else None
    require(kind == 'spatial' or not (root / 'attention.f32').exists(), 'CLS foreign attention artifact')
    predictions = rows(root / 'predictions.jsonl')
    require(len(predictions) == count, 'prediction row count')
    for i, (record, identity) in enumerate(zip(predictions, records)):
        subset(record, dict(identity=identity, true_action=int(labels[i]), fitted_action=int(labels[i]), prediction=int(logits[i].argmax()),
                            logits=descriptor('logits.f32', (4,), i),
                            labels=dict(file='labels.u32', dtype='U32LE', byte_offset=i * 8, byte_length=8, order=['true', 'fitted']),
                            episode_id=dict(file='episode-ids.u64', dtype='U64LE', byte_offset=i * 8, byte_length=8),
                            pooled=descriptor('pooled.f32', (pooled.shape[1],), i),
                            attention=descriptor('attention.f32', (2, 64), i) if kind == 'spatial' else None), 'prediction record')
    return dict(logits=logits, attention=attention, pooled=pooled)


def load_head(root, cached, imported, core, arm, panel, config, helper):
    name = f'qual-{core}-{arm}' if panel is None else f'head-{panel}-{core}-{arm}'
    run = common(root, config, head=True, expected_name=name)
    m, r, args, kind, count = run['metadata'], run['report'], run['args'], imported['kind'], len(cached['rows'])
    shared = dict(schema='looped-imported-readout-evaluation-v1', head_kind=kind,
                  core_checkpoint_sha256=CORES[core], seed=0,
                  physical_batch=count, effective_batch=count, accumulation=1, optimizer=None,
                  optimizer_updates=0, executed_core_forwards=0, core_optimizer_updates=0, cached_extraction_loops=4,
                  import_source=dict(root=str(imported['root']), manifest_sha256=imported['manifest_sha256'], manifest=imported['manifest']),
                  import_qualification=panel is None, confirmation_panel=panel,
                  evidence_class='implementation_smoke' if panel is None else 'frozen_imported_readout_confirmation',
                  partition='fit' if panel is None else 'confirmation_eval',
                  cuda_gemm_reduced_precision_f32=False, nvidia_tf32_override='0')
    subset(m, shared, 'head metadata'); subset(r, shared, 'head report')
    for document in (m, r):
        require(not {'permuted_labels', 'label_permutation', 'label_permutation_sha256', 'head_source'} & set(document), 'legacy label/head provenance leaked into imported report')
    subset(m, dict(core_checkpoint=core,
                   feature_preprocessing='none; raw cached F32', profile_updates=[],
                   model=dict(type='learned_readout', head_kind=kind, parameters=1334 if kind == 'cls' else 1284, cached_width=128, head_recurrence=False),
                   cache=dict(root=str(cached['root']), manifest_sha256=cached['manifest_sha256'], manifest=cached['manifest'])))
    subset(r, dict(input_rows=count, head_forwards=1, frozen_feature_gradient_checks=0, frozen_feature_bytes_unchanged=True,
                   canonical_parameter_bytes_unchanged=True, saved_canonical_parameter_roundtrip=True,
                   cache_manifest_sha256=cached['manifest_sha256'], core_source=cached['manifest']['source'], method_promotion=False))
    expected_args = {'mode': 'evaluate-imported', 'kind': kind, 'core-checkpoint': core,
                     'import-dir': str(imported['root']), 'import-manifest-sha256': imported['manifest_sha256'],
                     'cache-dir': str(cached['root']), 'cache-manifest-sha256': cached['manifest_sha256'],
                     'max-seconds': '60', 'output-dir': str(run['root']), 'device': 'cuda:0'}
    expected_args.update({'import-qualification': True} if panel is None else {'confirmation-panel': str(panel)})
    exact(args, expected_args, 'imported inference CLI boundary')
    initial = safe_tensors(run['root'] / 'initial.safetensors')
    final = safe_tensors(run['root'] / 'final.safetensors')
    require(set(initial) == set(final) == set(imported['parameters']), 'saved parameter population')
    for key, parameter in imported['parameters'].items():
        require(initial[key].shape == final[key].shape == parameter.shape
                and initial[key].tobytes() == final[key].tobytes() == parameter.tobytes(), 'imported canonical parameter changed')
    subset(r, dict(initial_checkpoint_sha256=digest(run['root'] / 'initial.safetensors'), final_checkpoint_sha256=digest(run['root'] / 'final.safetensors')))
    require(r['initial_checkpoint_sha256'] == r['final_checkpoint_sha256'], 'head file changed during inference')
    actual = head_outputs(run['root'], cached, kind)
    reference = forward(imported['parameters'], cached['arrays']['cls' if kind == 'cls' else 'current'], kind)
    numerical = {key: parity(actual[key], reference[key], attention=key == 'attention', actions=key == 'logits')
                 for key in ('logits', 'pooled', 'attention') if actual[key] is not None}
    value = endpoint(actual['logits'], cached['labels'])
    summary = endpoint_summary(value)
    subset(r['predictions'], dict(rows=count, true_correct=summary['correct'], fitted_correct=summary['correct']))
    for key, expected in (('true_accuracy', summary['accuracy']), ('fitted_accuracy', summary['accuracy']), ('true_ce', summary['mean_ce']), ('fitted_ce', summary['mean_ce'])):
        require(finite_number(r['predictions'][key]) and math.isclose(r['predictions'][key], expected, rel_tol=1e-10, abs_tol=1e-10), 'reported metric differs from raw logits')
    evidence = dict(root=str(run['root']), manifest_sha256=run['manifest_sha256'], parameter_sha256=imported['manifest']['artifact']['sha256'],
                    canonical_parameter_bytes_unchanged=True, numerical=numerical, metrics=summary, profiles=profiles(run, helper, kind, count))
    return run, value, actual['attention'], evidence


def load_features(root, core, panel, config, helper, expected_audit, *, smoke=False):
    run = common(root, config, head=False, expected_name='qual-core-smoke' if smoke else f'features-{panel}-{core}')
    r, m, args = run['report'], run['metadata'], run['args']
    exact(args, core_arguments(run['root'], core, panel, smoke=smoke), 'frozen core CLI boundary')
    count, seed, tag = (1, 20260915, 0x46454154555245) if smoke else (256, 20260917 + panel, 0x43554441434f4e46 + panel * 0x10000)
    subset(r, dict(schema='looped-known-features-v1', input_rows=count, model_forwards=count, optimizer_updates=0,
                   physical_batch=1, effective_batch=1, accumulation=1, loops=4, layouts=count,
                   data_seed=seed, episode_id_base=tag, feature_width=128, parameters=992393,
                   checkpoint_sha256=CORES[core], initial_checkpoint_sha256=CORES[core], final_checkpoint_sha256=CORES[core],
                   first_forward_profile='evaluation-000001', ordinary_heads_per_forward=dict(policy=1, value=1, reward=1, successor=4)))
    subset(args, dict(mode='known-features-smoke' if smoke else 'known-features', **{'known-mapping': True, 'seed': '0', 'loops': '4', 'batch': '1',
                   'effective-batch': '1', 'data-seed': str(seed), 'eval-episodes': '768' if smoke else '256', 'updates': '1', 'profile-eval': 'true'}))
    checkpoint = C8 / ('features-' + core) / 'initial.safetensors'
    exact(args['checkpoint'], str(checkpoint)); digest(checkpoint, CORES[core])
    subset(m['provenance']['checkpoint'], dict(path=str(checkpoint), sha256=CORES[core]))
    require(digest(run['root'] / 'initial.safetensors') == digest(run['root'] / 'final.safetensors') == CORES[core], 'core weights changed')
    if not smoke:
        exact(args['known-features-confirmation-panel'], str(panel))
        subset(m['known_features'], dict(confirmation_panel=panel, partition='confirmation_eval', cuda_gemm_reduced_precision_f32=False, nvidia_tf32_override='0'))
    own_audit = rows(run['root'] / 'known-features-input-audit.jsonl')
    exact(own_audit, expected_audit, 'extraction panel audit')
    records = rows(run['root'] / 'known-features-rows.jsonl')
    exact([{k: v for k, v in row.items() if k != 'arrays'} for row in records], own_audit, 'extracted row identity')
    roles, labels = geometry(own_audit, panel, smoke=smoke)
    arrays = {}
    for name, shape in (('cls', (128,)), ('current', (64, 128)), ('policy', (4,))):
        file = 'known-features-' + name + '.f32'
        arrays[name] = binary_array(run['root'] / file, '<f4', (count, *shape))
        for i, row in enumerate(records):
            require(set(row['arrays']) == {'cls', 'current', 'policy'}, 'raw feature array population')
            exact(row['arrays'][name], descriptor(file, shape, i), 'feature descriptor')
    core_parameters = safe_tensors(checkpoint)
    native = arrays['cls'].astype(np.float64) @ core_parameters['policy_head.weight'].astype(np.float64).T + core_parameters['policy_head.bias'].astype(np.float64)
    numerical = parity(arrays['policy'], native, actions=True)
    run['profiles'] = profiles(run, helper)
    return run, records, arrays, roles, labels, dict(native_parity=numerical, profiles=run['profiles'])


def oracle_policy(core, current, roles):
    path = C10 / ('policy-' + core + '.json')
    digest(path, dict(initial='f6c81ad896f3073963e7449238741fbabbf598e642689f70b405b96b4a8a2369',
                      final='9cb314510c8805805d581284521eb274bda26a31fef0929eaf703359edf066ae')[core])
    policy = read(path)
    arrays = {key: np.asarray(policy[key], dtype=np.float64) for key in ('mean', 'scale', 'coefficients', 'label_mean')}
    for key, shape in dict(mean=(256,), scale=(256,), coefficients=(256, 4), label_mean=(4,)).items():
        require(arrays[key].shape == shape and np.isfinite(arrays[key]).all(), 'frozen oracle policy array')
    require(np.all(arrays['scale'] > 0), 'oracle standardization scale')
    pooled = current[np.arange(len(current))[:, None], roles].reshape(len(current), 256).astype(np.float64)
    return ((pooled - arrays['mean']) / arrays['scale']) @ arrays['coefficients'] + arrays['label_mean']


def authority(name, config):
    path, pin = R / (name + '-authorization.json'), R / (name + '-authorization.sha256')
    digest(pin); digest(path, pin.read_text().strip())
    value = read(path)
    subset(value, dict(accepted=True, campaign=config['campaign'], source=config['source'], binaries=config['binaries'], dependency=DEPENDENCY))
    timestamp(value['created_local'])
    return value, digest(path)


def authority_closure(config, authority_value, stage):
    static = authority_value['frozen_files']
    require(all(config['frozen_files'].get(file) == sha for file, sha in static.items()), 'static authority inputs absent/changed in analyzer config')
    qualifier = {str(R / ('qualification-authorization' + suffix)) for suffix in ('.json', '.sha256')}
    later = {str(R / (name + suffix)) for name in ('confirmation-authorization', 'panel-seal', 'cache-seal') for suffix in ('.json', '.sha256')}
    extras = set(config['frozen_files']) - set(static)
    require(extras <= qualifier | (later if stage == 'confirmation' else set()), 'unregistered dynamic analyzer binding')
    # Later artifacts remain unopened until the accepted qualification report
    # and confirmation authority are validated by analyze().
    for file in set(static) | (extras & qualifier):
        digest(file, config['frozen_files'][file])


def parents_and_config(config, stage):
    require(Path(config['campaign']).is_absolute() and Path(config['campaign']).parent == RUNS, 'campaign location')
    require(isinstance(config['source'], str) and len(config['source']) == 40, 'execution revision')
    exact(config['dependency'], DEPENDENCY)
    require(set(config['binaries']) == {'learned_readout_probe', 'looped_agent_probe'} and all(valid_sha(v) for v in config['binaries'].values()), 'execution binaries')
    digest(R / 'registration.md', REG_SHA)
    a, authority_sha = authority('qualification', config)
    authority_closure(config, a, stage)
    for file in (R / 'registration.md', Path(__file__), R / 'analyze_cuda_readout_tests.py', R / 'parameter-seal.json'):
        require(str(file) in config['frozen_files'], 'analyzer/registration/tests/parameters not frozen')
    parent_summary = {}
    for key, research, root in (('c8', R8, C8), ('c9', R9, C9), ('c10', R10, C10)):
        path = research / 'completed-campaign.manifest.json'
        digest(path, PARENTS[key]); m = read(path)
        exact(m['campaign'], str(root), 'parent root')
        checked_files(root, m['files'], includes_manifest=True)
        for file, sha in m.get('bindings', {}).items():
            digest(file, sha)
        parent_summary[key] = dict(manifest_sha256=PARENTS[key], files=len(m['files']))
    subset(a, dict(registration_sha256=REG_SHA, imports_sha256={key: item['manifest_sha256'] for key, item in config['imports'].items()}))
    for file, sha in a['frozen_files'].items():
        digest(file, sha)
    seal = read(R / 'parameter-seal.json')
    exact(seal['imports'], config['imports'], 'parameter seal imports')
    require(set(config['imports']) == {core + '/' + arm for core in CORES for arm in ARMS}, 'ten fixed imports required')
    imported = {key: imported_parameters(item, *key.split('/')) for key, item in config['imports'].items()}
    return a, authority_sha, imported, parent_summary


def frozen_seal(name):
    path, pin = R / (name + '-seal.json'), R / (name + '-seal.sha256')
    digest(pin); digest(path, pin.read_text().strip())
    value = read(path)
    exact(value['accepted'], True, name + ' seal acceptance')
    for file, sha in value['frozen_files'].items():
        digest(file, sha)
    return value, digest(path)


def historical_sets():
    exclusion = read(C8 / 'exclusions.json')
    paths = [Path(p) for p in exclusion['paths']] + [C8 / 'audit-features/known-features-input-audit.jsonl', C9 / 'fresh-audit/known-features-input-audit.jsonl']
    require(len(paths) == len(set(paths)) == 6, 'historical source count')
    result = [set(), set(), set()]
    for path, expected in zip(paths, HISTORY):
        digest(path, expected)
        for row in rows(path):
            values = [row['query_sha256'], row['input_sha256'], row.get('episode_id', row.get('id'))]
            require(valid_sha(values[0]) and valid_sha(values[1]) and type(values[2]) is int, 'historical identity')
            for group, value in zip(result, values):
                group.add(value)
    require(len(result[0]) == 5688, 'historical query union')
    return result


def cumulative_budget(config, runs):
    clock = read(R / 'campaign-clock.json')
    began = timestamp(clock['started_local'])
    model, final, counted = 0., 0., []
    # Include retained failed launches/retries too; they consumed the budget.
    for path in Path(config['campaign']).glob('*.exit.json'):
        if path.name.startswith('audit-'):
            continue
        state = read(path)
        require(finite_number(state['model_phase_seconds']) and finite_number(state['finalization_seconds']), 'invalid cumulative runtime record')
        model += state['model_phase_seconds']; final += state['finalization_seconds']; counted.append(str(path))
    for path in Path(config['campaign']).glob('*.binding.json'):
        binding = read(path)
        require(finite_number(binding['elapsed_seconds']), 'invalid cumulative binder runtime')
        final += binding['elapsed_seconds']
    now = dt.datetime.now().astimezone().timestamp()
    require(0 <= model <= 360 and 0 <= final <= 600 and 0 <= now - began <= 1800, 'cumulative registered budget')
    require(all(v['started'] >= began for v in runs), 'invocation predates campaign clock')
    return dict(model_seconds=model, profiler_and_supervisor_finalization_seconds=final, campaign_seconds=now - began,
                campaign_started_local=clock['started_local'], counted_launch_records=counted)


def observed_blas():
    paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if 'openblas' in line.lower() or 'mkl_rt' in line.lower()}
    require(paths, 'unverified BLAS runtime')
    result = []
    for path in sorted(paths):
        library = ctypes.CDLL(path)
        names = ('scipy_openblas_get_num_threads64_', 'scipy_openblas_get_num_threads',
                 'openblas_get_num_threads64_', 'openblas_get_num_threads', 'MKL_Get_Max_Threads')
        getter = next((getattr(library, name) for name in names if hasattr(library, name)), None)
        require(getter is not None, 'BLAS thread getter unavailable')
        getter.restype = ctypes.c_int
        count = getter()
        require(count == 1, 'analysis must use one BLAS thread')
        result.append(dict(library=path, sha256=digest(path), num_threads=count))
    return result


def analyze(config, stage):
    blas = observed_blas()
    a, a_sha, imported, parent_summary = parents_and_config(config, stage)
    helper = profile_helper()
    base = dict(schema='tofy-cuda-readout-analysis-v1', stage=stage, accepted=True, campaign=config['campaign'],
                source=config['source'], dependency=DEPENDENCY, binaries=config['binaries'], registration_sha256=REG_SHA,
                qualification_authorization_sha256=a_sha, parameter_seal_sha256=digest(R / 'parameter-seal.json'),
                parents=parent_summary, no_fitting=True, blas=blas, numpy_forward_precision='F32; F64 native/oracle affine and scoring',
                caveats=['C10 used privileged synthetic role supervision; fixed inference reads features only.',
                         'Initial-core success cannot receive C7 training credit.',
                         'Same 120 adjacent role configurations; new nuisance/support robustness, not structural extrapolation.',
                         'No architecture, optimizer, policy-only learnability, ARC, planner or world-model promotion.',
                         'Pointwise paired bootstrap and Wilson intervals assume independent queries and are not simultaneous bounds.',
                         'All-correct empirical bootstrap can be degenerate; no population 100% reliability claim.',
                         'Affine ridge softmax CE is descriptive; probability calibration was not established.',
                         'Fixed single null permutations are coarse controls, not a general chance distribution.',
                         'Core extraction and cached-head inference are separate CUDA stages; no integrated controller claim.',
                         'Profiled work is not production timing; retained bundle gaps remain limitations.'])
    qualification = config['qualification']
    require(set(qualification['heads']) == set(imported), 'qualification head population')
    fit_seal_path = R9 / 'fit-cache-seal.json'
    digest(fit_seal_path, 'dad74c5ca75623566fa51ee3259e32b1deb437efaf71438144b79c26505b2f78')
    fit_seal = read(fit_seal_path)
    runs, qual_evidence = [], {}
    # Always revalidate the old qualification artifacts before reading new panels.
    for core in CORES:
        cached = cache(fit_seal[core], core, None, config)
        for arm in ARMS:
            key = core + '/' + arm
            run, _, _, evidence = load_head(qualification['heads'][key], cached, imported[key], core, arm, None, config, helper)
            require(run['started'] >= timestamp(a['created_local']), 'qualification predates source/parameter/analyzer freeze')
            runs.append(run); qual_evidence[key] = evidence
    old_row = rows(C8 / 'audit-features/known-features-input-audit.jsonl')[:1]
    smoke, _, smoke_arrays, _, _, smoke_evidence = load_features(qualification['core_smoke'], 'initial', None, config, helper, old_row, smoke=True)
    require(smoke['started'] >= timestamp(a['created_local']), 'core smoke predates qualification authorization')
    for name, shape in (('cls', (128,)), ('current', (64, 128)), ('policy', (4,))):
        path = C8 / 'features-initial' / ('known-features-' + name + '.f32')
        with path.open('rb') as stream:
            old = np.frombuffer(stream.read(math.prod(shape) * 4), dtype='<f4').reshape(1, *shape)
        smoke_evidence[name + '_old_core_parity'] = parity(smoke_arrays[name], old, actions=name == 'policy')
    runs.append(smoke)
    base['qualification'] = dict(heads=qual_evidence, core_smoke=smoke_evidence, classification='implementation_smoke', completed_local=dt.datetime.fromtimestamp(max(v['finished'] for v in runs)).astimezone().isoformat())
    if stage == 'qualification':
        base['budget'] = cumulative_budget(config, runs)
        return base
    ca, ca_sha = authority('confirmation', config)
    subset(ca, dict(qualification_authorization_sha256=a_sha, parameter_seal_sha256=base['parameter_seal_sha256']))
    digest(ca['qualification_report']['path'], ca['qualification_report']['sha256'])
    qr = read(ca['qualification_report']['path'])
    subset(qr, dict(accepted=True, stage='qualification', campaign=config['campaign'], source=config['source'], binaries=config['binaries'], parameter_seal_sha256=base['parameter_seal_sha256']))
    require(timestamp(a['created_local']) <= timestamp(qr['created_local']) <= timestamp(ca['created_local'])
            and max(v['finished'] for v in runs) <= timestamp(qr['created_local']), 'qualification report/confirmation chronology')
    for file, sha in qr['input_hashes'].items():
        digest(file, sha)
    for file, sha in config['frozen_files'].items():
        digest(file, sha)
    base['confirmation_authorization_sha256'] = ca_sha
    confirmation = config['confirmation']
    require(set(confirmation['audits']) == {'0', '1', '2'}
            and set(confirmation['features']) == set(confirmation['caches']) == {f'{p}/{c}' for p in range(3) for c in CORES}
            and set(confirmation['heads']) == {f'{p}/{c}/{arm}' for p in range(3) for c in CORES for arm in ARMS}, 'confirmation population')
    panel_seal, panel_sha = frozen_seal('panel')
    cache_seal, _ = frozen_seal('cache')
    require(panel_seal['historical_unique_queries'] == 5688 and set(panel_seal['panels']) == {'0', '1', '2'}, 'panel seal population')
    require(cache_seal['frozen_files'].get(str(R / 'panel-seal.json')) == panel_sha, 'cache seal not bound to all-panel audit barrier')
    exact(cache_seal['caches'], confirmation['caches'], 'six frozen caches')
    history, audits, audit_runs = historical_sets(), {}, []
    for panel in range(3):
        run = common(confirmation['audits'][str(panel)], config, head=False, audit=True, expected_name=f'audit-{panel}')
        exact(run['args'], core_arguments(run['root'], 'initial', panel, audit=True), 'panel audit CLI boundary')
        subset(run['args'], {'mode': 'known-features-audit', 'known-features-confirmation-panel': str(panel), 'data-seed': str(20260917 + panel), 'eval-episodes': '256'})
        subset(run['report'], dict(model_forwards=0, input_rows=256, query_overlap=0, excluded_unique_queries=5688))
        require(len(run['report']['exclusions']) == 6 and timestamp(ca['created_local']) <= run['started'], 'audit authorization/exclusions')
        own = rows(run['root'] / 'known-features-input-audit.jsonl')
        geometry(own, panel)
        for group, key in zip(history, ('query_sha256', 'input_sha256', 'episode_id')):
            values = {r[key] for r in own}
            require(len(values) == 256 and group.isdisjoint(values), 'within/historical/cross-panel collision; no replacement permitted')
            group.update(values)
        exact(panel_seal['panels'][str(panel)]['manifest_sha256'], run['manifest_sha256'])
        audits[panel] = own; audit_runs.append(run); runs.append(run)
    require(max(v['finished'] for v in audit_runs) <= timestamp(panel_seal['created_local']) <= timestamp(cache_seal['created_local']), 'all-audits seal chronology')
    features, cached_panels, extraction_runs = {}, {}, []
    for panel in range(3):
        for core in CORES:
            key = f'{panel}/{core}'
            run, records, arrays, roles, labels, evidence = load_features(confirmation['features'][key], core, panel, config, helper, audits[panel])
            require(run['started'] >= timestamp(panel_seal['created_local']), 'feature extraction predates all-panel seal')
            features[key] = dict(arrays=arrays, roles=roles, labels=labels, evidence=evidence)
            cached_panels[key] = cache(confirmation['caches'][key], core, panel, config, records)
            extraction_runs.append(run); runs.append(run)
    require(max(v['finished'] for v in extraction_runs) <= timestamp(cache_seal['created_local']), 'cache seal predates completed six extractions')
    panels, all_values, all_roles, all_labels = {}, {}, {}, []
    for panel in range(3):
        values, role_hits, head_evidence = {}, {}, {}
        labels = features[f'{panel}/initial']['labels']
        require(np.array_equal(labels, features[f'{panel}/final']['labels']), 'cores differ in labels')
        all_labels.append(labels)
        for core in CORES:
            f = features[f'{panel}/{core}']
            cached = cached_panels[f'{panel}/{core}']
            values[core + '/native'] = endpoint(f['arrays']['policy'], labels)
            values[core + '/oracle_role'] = endpoint(oracle_policy(core, f['arrays']['current'], f['roles']), labels)
            for arm in ARMS:
                key = core + '/' + arm
                run, value, attention, evidence = load_head(confirmation['heads'][f'{panel}/{key}'], cached, imported[key], core, arm, panel, config, helper)
                require(run['started'] >= timestamp(cache_seal['created_local']), 'head evaluated before all-six-cache freeze')
                runs.append(run); values[key] = value; head_evidence[key] = evidence
                if arm.startswith('c10_'):
                    role_hits[key], evidence['role_attention'] = localization(attention, f['roles'])
        statistics = comparisons(values, labels, role_hits, panel)
        panels[str(panel)] = dict(geometry_correct=256, label_counts=np.bincount(labels, minlength=4).tolist(), statistics=statistics,
                                  heads=head_evidence, features={c: features[f'{panel}/{c}']['evidence'] for c in CORES})
        for key, value in values.items():
            all_values.setdefault(key, []).append(value)
        for key, value in role_hits.items():
            all_roles.setdefault(key, []).append(value)
    pooled_values = {key: {metric: np.concatenate([v[metric] for v in value]) for metric in ('correct', 'ce')} for key, value in all_values.items()}
    pooled_roles = {key: {role: np.concatenate([v[role] for v in value]) for role in ('agent', 'goal', 'joint')} for key, value in all_roles.items()}
    base.update(panels=panels, pooled=comparisons(pooled_values, np.concatenate(all_labels), pooled_roles, None),
                registered=registered_decision(panels), budget=cumulative_budget(config, runs),
                classification='completed_confirmation_evidence', retained_C9_failed_arms=True)
    return base


def main():
    require(__debug__, 'guards require Python optimization disabled')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--stage', choices=('qualification', 'confirmation'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and not args.output.exists(), 'analysis output must be absolute and never reused')
    started = time.monotonic()
    def timeout(signum, frame):
        raise Invalid('registered 120-second CPU analysis deadline')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    try:
        digest(args.config, args.sha256)
        config = read(args.config)
        result = analyze(config, args.stage)
        for file, sha in list(SNAPSHOT.items()):
            digest(file, sha)
        result.update(created_local=dt.datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic() - started,
                      pid=os.getpid(), input_hashes=dict(sorted(SNAPSHOT.items())))
        require(result['elapsed_seconds'] <= 120, 'analysis runtime ceiling')
        result['budget']['campaign_seconds'] = dt.datetime.now().astimezone().timestamp() - timestamp(result['budget']['campaign_started_local'])
        require(0 <= result['budget']['campaign_seconds'] <= 1800, 'campaign elapsed during final integrity verification')
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(dict(accepted=True, stage=args.stage, output=str(args.output), sha256=sha_bytes(args.output.read_bytes()),
                              decision=result.get('registered', {}).get('decision'), elapsed_seconds=result['elapsed_seconds'])))
    except Exception as error:
        if not args.output.exists():
            with args.output.open('x') as stream:
                json.dump(dict(accepted=False, stage=args.stage, failure=repr(error), created_local=dt.datetime.now().astimezone().isoformat(),
                               elapsed_seconds=time.monotonic() - started, pid=os.getpid()), stream, indent=2, allow_nan=False)
                stream.write('\n')
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
