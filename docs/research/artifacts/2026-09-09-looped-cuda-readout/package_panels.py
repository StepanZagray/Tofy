#!/usr/bin/env python3
"""Freeze three audited populations and six projected feature caches; no generation."""
import argparse
import datetime
import hashlib
import math
import os
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from campaign_io import *
from supervise_stage import qualified, panel_binding

def rows(path):
    return [read_line(line) for line in path.read_text().splitlines()]

def read_line(line):
    return json.loads(line, parse_constant=lambda value: require(False, 'nonfinite row'))

def moment(text):
    value = datetime.datetime.fromisoformat(text)
    require(value.tzinfo is not None, 'missing chronology timezone')
    return value

def stage(root, expected_stage):
    state = read(root.with_suffix('.exit.json'))
    require(state['accepted'] is True and state['returncode'] == 0 and state['failure'] is None
            and state['pid_gone'] is True and state['model_pid_gone'] is True and state['group_gone'] is True
            and not state['owned_survivors'] and state['cleanup_error'] is None, 'stage failed or unclean')
    for pid, start in state['owned_process_start_ticks'].items():
        path = Path('/proc') / str(pid) / 'stat'
        if path.exists():
            require(int(path.read_text().rsplit(')', 1)[1].split()[19]) != start, 'owned process alive')
    proc = read(root.with_suffix('.process.json'))
    require(proc['stage'] == expected_stage, 'source stage differs')
    verified_manifest(root, state['manifest_sha256'])
    return proc, state

def history():
    old = read(C8 / 'exclusions.json')
    paths = [Path(path) for path in old['paths']] + [C8 / 'audit-features' / 'known-features-input-audit.jsonl',
                                                   C9 / 'fresh-audit' / 'known-features-input-audit.jsonl']
    pins = dict(old['sha256'])
    pins[str(paths[-2])] = '09559eb4dd1b933de724da81aef3bb80547e010a00e38e39de31c9f72fab1c6d'
    pins[str(paths[-1])] = '361092818769307bcd82f1f30e10808e73857be7a96844f1293f2021d0410d87'
    verify_files(pins)
    queries, inputs, ids = set(), set(), set()
    for path in paths:
        for row in rows(path):
            queries.add(row['query_sha256'])
            inputs.add(row['input_sha256'])
            ids.add(historical_episode(row))
    require(len(queries) == 5688 and None not in ids, 'historical population differs')
    return queries, inputs, ids, pins

def historical_episode(row):
    require(all(type(row[key]) is int and row[key] >= 0 for key in ('episode_id','id') if key in row),
            'invalid historical episode identity type')
    if 'episode_id' in row and 'id' in row:
        require(row['episode_id'] == row['id'], 'conflicting historical episode identity')
    value = row.get('episode_id', row.get('id'))
    require(type(value) is int and value >= 0, 'invalid historical episode identity')
    return value

def validate_rows(values, panel):
    require(type(panel) is int and panel in range(3) and len(values) == 256, 'panel/count differs')
    seen = [set(), set(), set()]
    labels = [0] * 4
    for index, row in enumerate(values):
        integer_identity = dict(confirmation_panel=panel, layout_index=index, input_index=index,
            data_seed=20260917+panel, episode_seed=20260917+panel,
            episode_id=0x43554441434f4e46+panel*0x10000+index,
            evaluation_loops=4, permutation_id=0, min_distance=1, max_distance=1, oracle_distance=1)
        require(all(type(row.get(key)) is int and row[key] == value for key, value in integer_identity.items()),
                'typed panel identity differs')
        require(row.get('inferred_controls') == [0, 1, 2, 3]
                and all(type(value) is int for value in row['inferred_controls'])
                and row.get('split') == 'KnownMapping', 'known controls differ')
        for key in ('input_sha256', 'query_sha256', 'label_sha256', 'targets_sha256'):
            value = row.get(key)
            require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                    'invalid identity digest')
        require(row['schema'] == 'looped-known-features-v1' and row['partition'] == 'confirmation_eval'
                and row['data_seed'] == 20260917 + panel and row['episode_seed'] == 20260917 + panel
                and type(row['episode_id']) is int and row['episode_id'] == 0x43554441434f4e46 + panel*0x10000 + index
                and type(row['input_index']) is int and row['input_index'] == index
                and row['evaluation_loops'] == 4 and row['condition'] == 'factual'
                and row['permutation_id'] == 0 and row['min_distance'] == row['max_distance'] == row['oracle_distance'] == 1
                and row['support_cleared'] is False, 'panel identity/task differs')
        cells = row['visible_cells']
        require(type(cells) is list and len(cells) == 64 and all(type(x) is int and x in range(5) for x in cells)
                and cells.count(2) == cells.count(3) == 1, 'invalid role geometry')
        agent, goal = cells.index(2), cells.index(3)
        delta = (goal%8-agent%8, goal//8-agent//8)
        directions = ((0,-1), (0,1), (-1,0), (1,0))
        require(delta in directions and type(row['correct_action']) is int
                and row['correct_action'] == directions.index(delta), 'geometry action differs')
        query = np.repeat(np.asarray(cells, dtype='<u4'), 64).tobytes()
        require(hashlib.sha256(query).hexdigest() == row['query_sha256'], 'query image/hash mismatch')
        require(hashlib.sha256(row['correct_action'].to_bytes(4, 'little')).hexdigest() == row['label_sha256'], 'label hash mismatch')
        labels[row['correct_action']] += 1
        for group, key in zip(seen, ('query_sha256', 'input_sha256', 'episode_id')):
            require(row[key] not in group, 'within-panel collision')
            group.add(row[key])
    require(min(labels) > 0, 'missing action class')
    return seen, labels

def seal_panels():
    c = campaign()
    require(not (R / 'panel-seal.json').exists() and not (R / 'panel-seal.sha256').exists(), 'never reuse panel seal')
    frozen = qualified(c)
    authority = read(R / 'confirmation-authorization.json')
    authorized_at = moment(authority['created_local'])
    queries, inputs, ids, old = history()
    frozen.update(old)
    seen = [queries, inputs, ids]
    panels = {}
    for panel in range(3):
        root = c / f'audit-{panel}'
        proc, state = stage(root, 'audit')
        require(moment(proc['started_local']) >= authorized_at, 'new panel predates qualification')
        values = rows(root / 'known-features-input-audit.jsonl')
        local, labels = validate_rows(values, panel)
        for historic, new in zip(seen, local):
            require(historic.isdisjoint(new), 'historical/cross-panel collision; do not replace queries')
            historic.update(new)
        report = read(root / 'report.json')
        require(report['optimizer_updates'] == report['model_forwards'] == report['query_overlap'] == 0
                and report['excluded_unique_queries'] == 5688 and len(report['exclusions']) == 6,
                'audit report differs')
        for path in [root / 'manifest.json', root / 'known-features-input-audit.jsonl', root / 'report.json',
                     root.with_suffix('.exit.json'), root.with_suffix('.process.json')]:
            frozen[str(path)] = digest(path)
        panels[str(panel)] = dict(root=str(root), rows=256, label_counts=labels,
                                 query_overlap=0, input_overlap=0, episode_overlap=0,
                                 manifest_sha256=digest(root / 'manifest.json'))
    verify_files(frozen)
    save(R / 'panel-seal.json', dict(accepted=True, created_local=now(), panels=panels,
                                    historical_unique_queries=5688, new_unique_queries=768, frozen_files=frozen))
    (R / 'panel-seal.sha256').write_text(digest(R / 'panel-seal.json') + '\n')
    print(json.dumps(dict(accepted=True, panels=panels, manifest_sha256=digest(R / 'panel-seal.json'))))

def seal_caches():
    c = campaign()
    require(not (R / 'cache-seal.json').exists() and not (R / 'cache-seal.sha256').exists()
            and not (c / 'caches').exists(), 'never reuse confirmation caches/seal')
    panel_seal, frozen = panel_binding(c)  # Exact three panels, authority, chronology and immutable bindings.
    pin = (R / 'panel-seal.sha256').read_text().strip()
    verify_files({str(R / 'panel-seal.json'): pin})
    require(panel_seal['accepted'] is True, 'panels not sealed')
    frozen.update(panel_seal['frozen_files'])
    frozen[str(R / 'panel-seal.json')] = pin
    binding = read(c / 'binary.json')
    result = {}
    for panel in range(3):
        audit_rows = rows(c / f'audit-{panel}' / 'known-features-input-audit.jsonl')
        validate_rows(audit_rows, panel)
        for core in CORES:
            source = c / f'features-{panel}-{core}'
            proc, state = stage(source, 'extract')
            require(moment(proc['started_local']) >= moment(panel_seal['created_local']), 'extraction precedes panel seal')
            report, metadata = read(source / 'report.json'), read(source / 'metadata.json')
            require(report['model_forwards'] == 256 and report['optimizer_updates'] == 0
                    and report['checkpoint_sha256'] == CORES[core], 'feature extraction population/core differs')
            require(metadata['provenance']['source_revision'] == binding['source']
                    and metadata['provenance']['binary_sha256'] == binding['binaries']['looped_agent_probe'], 'feature executable differs')
            for name in ('initial.safetensors', 'final.safetensors'):
                require(digest(source / name) == CORES[core], 'frozen core changed')
            values = rows(source / 'known-features-rows.jsonl')
            validate_rows(values, panel)
            require([{k:v for k,v in row.items() if k != 'arrays'} for row in values] == audit_rows,
                    'feature/audit complete row stream differs')
            destination = c / 'caches' / str(panel) / core
            destination.mkdir(parents=True)
            with (destination / 'rows.jsonl').open('x') as stream:
                for row in values:
                    stream.write(json.dumps({key: row[key] for key in FIELDS}, sort_keys=True) + '\n')
            arrays = {}
            for key, shape in [('cls', [128]), ('current', [64,128]), ('policy', [4])]:
                width = math.prod(shape)
                name = f'known-features-{key}.f32'
                data = (source / name).read_bytes()
                require(len(data) == 256*width*4 and np.isfinite(np.frombuffer(data, dtype='<f4')).all(), 'feature array shape/nonfinite')
                for index, row in enumerate(values):
                    descriptor = row['arrays'][key]
                    require(type(descriptor.get('byte_offset')) is int and type(descriptor.get('byte_length')) is int
                        and type(descriptor.get('shape')) is list and all(type(x) is int for x in descriptor['shape'])
                        and descriptor == dict(file=name, dtype='F32LE', shape=shape,
                        byte_offset=index*width*4, byte_length=width*4), 'feature array descriptor mismatch')
                frozen[str(source / name)] = digest(source / name)
                if key != 'policy':
                    with (destination / f'{key}.f32').open('xb') as stream:
                        stream.write(data)
                    arrays[key] = dict(file=name, sha256=digest(source / name), byte_offset=0, byte_length=len(data))
            for path in [source / 'manifest.json', source / 'known-features-rows.jsonl', source.with_suffix('.process.json'), source.with_suffix('.exit.json')]:
                frozen[str(path)] = digest(path)
            files = {name: dict(sha256=digest(destination / name), bytes=(destination / name).stat().st_size)
                     for name in ('cls.f32', 'current.f32', 'rows.jsonl')}
            save(destination / 'manifest.json', dict(schema='looped-imported-readout-cache-v1', partition='confirmation_eval',
                rows=256, core_checkpoint_sha256=CORES[core], files=files, source_arrays=arrays,
                source=dict(root=str(source), manifest_sha256=digest(source / 'manifest.json'), source_revision=binding['source'],
                    binary_sha256=binding['binaries']['looped_agent_probe'], data_seed=20260917+panel,
                    episode_id_base=0x43554441434f4e46+panel*0x10000, confirmation_panel=panel,
                    feature_schema='looped-known-features-v1')))
            verified_manifest(destination, digest(destination / 'manifest.json'))
            result[f'{panel}/{core}'] = dict(root=str(destination), manifest_sha256=digest(destination / 'manifest.json'))
    verify_files(frozen)
    save(R / 'cache-seal.json', dict(accepted=True, created_local=now(), caches=result, frozen_files=frozen))
    (R / 'cache-seal.sha256').write_text(digest(R / 'cache-seal.json') + '\n')
    print(json.dumps(dict(caches=result, manifest_sha256=digest(R / 'cache-seal.json'))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('audit-seal', 'cache-seal'), required=True)
    args = parser.parse_args()
    (seal_panels if args.stage == 'audit-seal' else seal_caches)()
