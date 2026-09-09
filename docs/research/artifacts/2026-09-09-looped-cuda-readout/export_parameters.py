#!/usr/bin/env python3
"""Translate ten already sealed C9/C10 heads once to canonical F32; never fit."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
import struct
import numpy as np
from campaign_io import R, C8, C9, C10, R9, PARENTS, CORES, ARMS, campaign, digest, read, require, save, now, verify_parents, verified_manifest, unique_object

SHAPES = {
    'spatial': [('queries', (2, 128)), ('output.weight', (4, 256)), ('output.bias', (4,))],
    'cls': [('hidden.weight', (10, 128)), ('hidden.bias', (10,)), ('output.weight', (4, 10)), ('output.bias', (4,))],
}

def safetensors(path, kind):
    data = path.read_bytes()
    require(len(data) >= 8, 'truncated safetensors')
    header_size, = struct.unpack('<Q', data[:8])
    require(header_size <= len(data) - 8, 'invalid safetensor header')
    header = json.loads(data[8:8 + header_size], object_pairs_hook=unique_object,
                        parse_constant=lambda value: require(False, f'nonfinite tensor header: {value}'))
    payload = data[8 + header_size:]
    require(set(header) == {name for name, _ in SHAPES[kind]}, 'foreign tensor names')
    result, intervals = {}, []
    for name, shape in SHAPES[kind]:
        item = header[name]
        start, end = item['data_offsets']
        require(item['dtype'] == 'F32' and type(item['shape']) is list
                and all(type(size) is int for size in item['shape'])
                and item['shape'] == list(shape), 'foreign tensor dtype/shape')
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(payload)
                and end-start == int(np.prod(shape))*4, 'foreign tensor offsets')
        result[name] = np.frombuffer(payload[start:end], dtype='<f4').copy().reshape(shape)
        require(np.isfinite(result[name]).all(), 'nonfinite foreign tensor')
        intervals.append((start, end))
    intervals.sort()
    require(intervals[0][0] == 0 and intervals[-1][1] == len(payload)
            and all(a[1] == b[0] for a, b in zip(intervals, intervals[1:])), 'foreign payload gaps/overlaps')
    return result

def c10_values(path):
    with np.load(path, allow_pickle=False) as data:
        values = {name: data[{'queries': 'queries', 'output.weight': 'output_weight',
                              'output.bias': 'output_bias'}[name]] for name, _ in SHAPES['spatial']}
    for name, shape in SHAPES['spatial']:
        value = values[name]
        require(value.dtype == np.dtype('<f8') and value.shape == shape and np.isfinite(value).all(),
                'original C10 tensor must be finite F64 with registered shape')
    return values

def encode(values, kind):
    require(set(values) == {name for name, _ in SHAPES[kind]}, 'tensor population mismatch')
    payload, table = bytearray(), []
    for name, shape in SHAPES[kind]:
        require(np.asarray(values[name]).dtype.kind == 'f', 'tensor must contain floating-point values')
        value = np.asarray(values[name], dtype='<f4')
        require(value.shape == shape and np.isfinite(value).all(), 'tensor shape/nonfinite value')
        data = value.tobytes(order='C')
        table.append(dict(name=name, dtype='F32LE', shape=list(shape), byte_offset=len(payload),
                          byte_length=len(data), sha256=hashlib.sha256(data).hexdigest()))
        payload.extend(data)
    require(len(payload) == (5136 if kind == 'spatial' else 5336), 'canonical payload length')
    return bytes(payload), table

def main():
    c = campaign()
    require(not (c / 'imports').exists() and not (c / 'imports').is_symlink()
            and not list(R.glob('parameter-seal*')), 'never reuse exports')
    parents = verify_parents()
    source_inputs = dict(parents['frozen_files'])
    output = {}
    for core in CORES:
        for arm in ARMS:
            kind = 'cls' if arm == 'c9_cls' else 'spatial'
            if arm.startswith('c10'):
                original = C10 / f'selector-{core}-{arm.removeprefix("c10_")}.npz'
                values = c10_values(original)
                producer = dict(source_revision='82ac8cb6ea3a06ae13d836d526c47994cd675d18',
                                implementation_sha256='a49151b2cd1294aa86fc72433755feb92e3d6b855796e8ee157e2402b51ac748')
                recipe = dict(name='c10_role_ridge_c8_affine_v1', original_optimizer_updates=0,
                              privileged_role_supervision=True, cast='f64_to_f32_once_no_rescale')
                source_kind = 'c10_role_ridge'
            else:
                family = {'c9_spatial': 'spatial', 'c9_cls': 'cls', 'c9_null': 'spatial-null'}[arm]
                root = C9 / f'fit-{core}-{family}'
                verified_manifest(root)
                report = read(root / 'report.json')
                require(report['optimizer_updates'] == 1000 and report['core_checkpoint_sha256'] == CORES[core]
                        and report['head_kind'] == kind and report['permuted_labels'] is (arm == 'c9_null'), 'C9 original arm mismatch')
                original = root / 'final.safetensors'
                values = safetensors(original, kind)
                producer = dict(source_revision='06a8d76fada203c5b1a11a45782ed777df27ff4c',
                                implementation_sha256='fecbc1b91099ddc9b39ac9b6361ee8072c26563f6b710e06c29ec7b3adeef20e')
                recipe = dict(name='c9_adamw1000_v1', original_optimizer_updates=1000,
                              privileged_role_supervision=False, cast='identity_f32')
                source_kind = 'c9_adamw'
            source_inputs[str(original)] = digest(original)
            payload, table = encode(values, kind)
            root = c / 'imports' / core / arm
            root.mkdir(parents=True)
            with (root / 'parameters.f32').open('xb') as stream:
                stream.write(payload)
            manifest = dict(schema='looped-imported-readout-source-v1', source_kind=source_kind,
                head_kind=kind, core_checkpoint=core, core_checkpoint_sha256=CORES[core], arm=arm,
                producer=producer, recipe=recipe, parent_manifests={name: sha for name, (_, sha) in PARENTS.items()},
                original_artifact=dict(path=str(original), sha256=source_inputs[str(original)]),
                artifact=dict(file='parameters.f32', sha256=digest(root / 'parameters.f32'), bytes=len(payload)), tensors=table)
            save(root / 'manifest.json', manifest)
            output[f'{core}/{arm}'] = dict(root=str(root), manifest_sha256=digest(root / 'manifest.json'))
    # Rehash every original after conversion, then bind all exports outside the tree.
    from campaign_io import verify_files
    verify_files(source_inputs)
    save(R / 'parameter-seal.json', dict(created_local=now(), imports=output, source_inputs=source_inputs,
                                        parent_verification=parents, exporter_sha256=digest(__file__)))
    save(R / 'parameter-seal-verification.json', dict(accepted=True, manifest_sha256=digest(R / 'parameter-seal.json'), created_local=now()))
    print(json.dumps(dict(imports=len(output), manifest_sha256=digest(R / 'parameter-seal.json'))))

if __name__ == '__main__':
    main()
