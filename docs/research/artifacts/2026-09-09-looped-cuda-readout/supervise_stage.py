#!/usr/bin/env python3
"""Exact, zero-update C11 invocation with the tested C8 process lifecycle."""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
sys.dont_write_bytecode = True
from campaign_io import *

def exact(actual, expected, label='config'):
    require(type(actual) is type(expected), f'{label} type mismatch')
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f'{label} fields mismatch')
        for key, value in expected.items():
            exact(actual[key], value, f'{label}.{key}')
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f'{label} length mismatch')
        for i, value in enumerate(expected):
            exact(actual[i], value, f'{label}[{i}]')
    else:
        require(actual == expected, f'{label} value mismatch')

def subset(actual, expected):
    for key, value in expected.items():
        require(key in actual, f'missing {key}')
        exact(actual[key], value, key)

def qualification_authority(c, binding):
    path = R / 'qualification-authorization.json'
    pin = R / 'qualification-authorization.sha256'
    verify_files({str(path): pin.read_text().strip()})
    authority = read(path)
    subset(authority, dict(accepted=True, campaign=str(c), source=binding['source'],
                          binaries=binding['binaries'], dependency=DEPENDENCY))
    moment(authority['created_local'])
    required = {str(R / 'registration.md'): REGISTRATION_SHA, str(LIFECYCLE): LIFECYCLE_SHA}
    paths = [R / name for name in ('campaign_io.py', 'export_parameters.py', 'supervise_stage.py',
                                   'parameter-seal.json', 'parameter-seal-verification.json')]
    paths += [c / 'binary.json', R9 / 'fit-cache-seal.json', C8 / 'exclusions.json']
    required.update({str(file): digest(file) for file in paths})
    required.update({str(c / name): sha for name, sha in binding['binaries'].items()})
    required.update({str(research / 'completed-campaign.manifest.json'): sha
                     for research, sha in PARENTS.values()})
    subset(authority['frozen_files'], required)
    verify_files(authority['frozen_files'])
    require(moment(read(R / 'parameter-seal.json')['created_local']) <= moment(authority['created_local']),
            'qualification authority predates fixed exports')
    return authority

def qualified(c):
    path = R / 'confirmation-authorization.json'
    pin = R / 'confirmation-authorization.sha256'
    verify_files({str(path): pin.read_text().strip()})
    authority = read(path)
    binding = read(c / 'binary.json')
    first = qualification_authority(c, binding)
    shared = dict(campaign=str(c), source=binding['source'], binaries=binding['binaries'],
                  parameter_seal_sha256=digest(R / 'parameter-seal.json'))
    subset(authority, dict(accepted=True, dependency=DEPENDENCY,
                          qualification_authorization_sha256=digest(R / 'qualification-authorization.json'), **shared))
    report = authority['qualification_report']
    verify_files({report['path']: report['sha256']})
    value = read(report['path'])
    subset(value, dict(accepted=True, **shared))
    require(moment(first['created_local']) <= moment(value['created_local']) <= moment(authority['created_local']),
            'qualification/report/confirmation chronology differs')
    return {str(path): digest(path), str(pin): digest(pin), report['path']: report['sha256'],
            str(R / 'qualification-authorization.json'): digest(R / 'qualification-authorization.json')}

def panel_binding(c):
    frozen = qualified(c)
    path, pin = R / 'panel-seal.json', R / 'panel-seal.sha256'
    verify_files({str(path): pin.read_text().strip()})
    seal = read(path)
    subset(seal, dict(accepted=True, historical_unique_queries=5688, new_unique_queries=768))
    require(set(seal['panels']) == {'0', '1', '2'}, 'all three panels not audited')
    for panel, item in seal['panels'].items():
        subset(item, dict(root=str(c / f'audit-{panel}'), rows=256, query_overlap=0, input_overlap=0, episode_overlap=0))
    require(moment(read(R / 'confirmation-authorization.json')['created_local']) <= moment(seal['created_local']),
            'panel seal predates qualification')
    subset(seal['frozen_files'], frozen)
    verify_files(seal['frozen_files'])
    frozen.update(seal['frozen_files'])
    frozen.update({str(path): digest(path), str(pin): digest(pin)})
    return seal, frozen

def import_binding(core, arm):
    c = campaign()
    verification = read(R / 'parameter-seal-verification.json')
    require(verification['accepted'] is True, 'parameter exports not accepted')
    verify_files({str(R / 'parameter-seal.json'): verification['manifest_sha256']})
    seal = read(R / 'parameter-seal.json')
    moment(seal['created_local'])
    require(moment(seal['created_local']) <= moment(verification['created_local']), 'export verification predates exports')
    require(set(seal['imports']) == {f'{c}/{a}' for c in CORES for a in ARMS}, 'ten fixed imports required')
    require(seal['parent_verification']['accepted'] is True, 'parents were not verified')
    verify_files(seal['source_inputs'])
    item = seal['imports'][f'{core}/{arm}']
    root = Path(item['root'])
    require(root == c / 'imports' / core / arm, 'import root mismatch')
    verify_files({str(root / 'manifest.json'): item['manifest_sha256']})
    manifest = read(root / 'manifest.json')
    subset(manifest, dict(schema='looped-imported-readout-source-v1', arm=arm, core_checkpoint=core,
                         core_checkpoint_sha256=CORES[core], head_kind='cls' if arm == 'c9_cls' else 'spatial',
                         source_kind='c10_role_ridge' if arm.startswith('c10_') else 'c9_adamw'))
    require(regular_files(root) == {'manifest.json', 'parameters.f32'}, 'import artifact population differs')
    frozen = {str(R / 'parameter-seal.json'): verification['manifest_sha256'],
              str(R / 'parameter-seal-verification.json'): digest(R / 'parameter-seal-verification.json'),
              str(root / 'manifest.json'): item['manifest_sha256'],
              str(root / 'parameters.f32'): manifest['artifact']['sha256'],
              manifest['original_artifact']['path']: manifest['original_artifact']['sha256']}
    frozen.update(seal['source_inputs'])
    verify_files(frozen)
    return item, frozen

def registered(name, binding):
    c = campaign()
    head_names = {f'qual-{core}-{arm}': (None, core, arm) for core in CORES for arm in ARMS}
    head_names.update({f'head-{panel}-{core}-{arm}': (panel, core, arm)
                       for panel in range(3) for core in CORES for arm in ARMS})
    if name in head_names:
        panel, core, arm = head_names[name]
        item, frozen = import_binding(core, arm)
        if panel is None:
            cache_item = read(R9 / 'fit-cache-seal.json')[core]
            require(Path(cache_item['root']) == C9 / 'caches' / f'fit-{core}', 'fit cache root mismatch')
        else:
            panels, panel_files = panel_binding(c)
            frozen.update(panel_files)
            pin = R / 'cache-seal.sha256'
            verify_files({str(R / 'cache-seal.json'): pin.read_text().strip()})
            cache_seal = read(R / 'cache-seal.json')
            require(cache_seal['accepted'] is True and set(cache_seal['caches']) == {
                f'{p}/{k}' for p in range(3) for k in CORES}, 'all six caches not sealed')
            require(moment(panels['created_local']) <= moment(cache_seal['created_local']), 'cache seal predates panels')
            require(cache_seal['frozen_files'][str(R / 'panel-seal.json')] == digest(R / 'panel-seal.json'),
                    'cache seal belongs to different panels')
            verify_files(cache_seal['frozen_files'])
            frozen.update(cache_seal['frozen_files'])
            cache_item = cache_seal['caches'][f'{panel}/{core}']
            require(Path(cache_item['root']) == c / 'caches' / str(panel) / core, 'confirmation cache root mismatch')
            frozen.update({str(R / 'cache-seal.json'): digest(R / 'cache-seal.json'), str(pin): digest(pin)})
        cache_root = Path(cache_item['root'])
        manifest = verified_manifest(cache_root, cache_item['manifest_sha256'])
        subset(manifest, dict(core_checkpoint_sha256=CORES[core], rows=512 if panel is None else 256,
                              partition='fit' if panel is None else 'confirmation_eval'))
        frozen[str(cache_root / 'manifest.json')] = cache_item['manifest_sha256']
        frozen.update({str(cache_root / file): info['sha256'] for file, info in manifest['files'].items()})
        kind = 'cls' if arm == 'c9_cls' else 'spatial'
        arguments = ['--mode', 'evaluate-imported', '--kind', kind, '--core-checkpoint', core,
            '--import-dir', item['root'], '--import-manifest-sha256', item['manifest_sha256'],
            '--cache-dir', str(cache_root), '--cache-manifest-sha256', cache_item['manifest_sha256'], '--max-seconds', '60']
        arguments += ['--import-qualification'] if panel is None else ['--confirmation-panel', str(panel)]
        expected = dict(schema='looped-imported-readout-evaluation-v1', status='complete_pending_analysis',
            optimizer_updates=0, executed_core_forwards=0, core_optimizer_updates=0,
            head_kind=kind, core_checkpoint_sha256=CORES[core], input_rows=512 if panel is None else 256,
            physical_batch=512 if panel is None else 256, effective_batch=512 if panel is None else 256,
            accumulation=1, partition='fit' if panel is None else 'confirmation_eval')
        return dict(name=name, stage='qualification' if panel is None else 'head_eval', binary='learned_readout_probe',
            arguments=arguments, frozen_files=frozen, expected_report=expected, model_seconds=60, finalization_seconds=60)
    feature_names = {f'features-{panel}-{core}': (panel, core) for panel in range(3) for core in CORES}
    audit_names = {f'audit-{panel}': panel for panel in range(3)}
    require(name == 'qual-core-smoke' or name in feature_names or name in audit_names, 'unregistered invocation')
    smoke = name == 'qual-core-smoke'
    audit = name in audit_names
    panel, core = (None, 'initial') if smoke else ((audit_names[name], 'initial') if audit else feature_names[name])
    frozen = {} if smoke else qualified(c)
    mode = 'known-features-smoke' if smoke else ('known-features-audit' if audit else 'known-features')
    arguments = ['--mode', mode, '--known-mapping', '--seed', '0', '--loops', '4', '--batch', '1',
        '--effective-batch', '1', '--max-seconds', '60' if audit else '120', '--updates', '1',
        '--eval-episodes', '768' if smoke else '256', '--data-seed', '20260915' if smoke else str(20260917 + panel),
        '--profile-eval', 'true']
    if not audit:
        checkpoint = C8 / f'features-{core}' / 'initial.safetensors'
        arguments += ['--checkpoint', str(checkpoint)]
        frozen[str(checkpoint)] = CORES[core]
    exclusions = read(C8 / 'exclusions.json')
    paths = exclusions['paths'][:]
    frozen.update(exclusions['sha256'])
    if not smoke:
        arguments += ['--known-features-confirmation-panel', str(panel)]
        paths += [str(C8 / 'audit-features' / 'known-features-input-audit.jsonl'),
                  str(C9 / 'fresh-audit' / 'known-features-input-audit.jsonl')]
        frozen[paths[-2]] = '09559eb4dd1b933de724da81aef3bb80547e010a00e38e39de31c9f72fab1c6d'
        frozen[paths[-1]] = '361092818769307bcd82f1f30e10808e73857be7a96844f1293f2021d0410d87'
    if not smoke and not audit:
        _, panel_files = panel_binding(c)
        frozen.update(panel_files)
    for path in paths:
        arguments += ['--known-features-exclude', path]
    expected = dict(status='complete_pending_analysis', optimizer_updates=0, input_rows=1 if smoke else 256,
                    model_forwards=0 if audit else (1 if smoke else 256))
    return dict(name=name, stage='qualification' if smoke else ('audit' if audit else 'extract'),
        binary='looped_agent_probe', arguments=arguments, frozen_files=frozen, expected_report=expected,
        model_seconds=60 if audit else 120, finalization_seconds=60)

def valid_cleanup(state):
    subset(state, dict(returncode=0, failure=None, pid_gone=True, model_pid_gone=True, group_gone=True,
                       owned_survivors=[], group_survivors=[], cleanup_error=None, launch_record_error=None))
    require(type(state['model_pid']) is int and state['model_pid'] > 1
            and type(state['pgid']) is int and state['pgid'] > 1, 'invalid recorded process identity')
    owned = state['owned_pids']
    require(type(owned) is list and all(type(pid) is int and pid > 1 for pid in owned)
            and len(set(owned)) == len(owned) and state['model_pid'] in owned and state['pgid'] in owned,
            'model/profiler not in owned process set')
    for pid in owned:
        require(not Path(f'/proc/{pid}').exists(), 'recorded process still exists')
    for key in ('headroom_mib', 'max_temperature_c', 'model_phase_seconds', 'elapsed_seconds', 'finalization_seconds'):
        value = state[key]
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f'invalid {key}')
    require(state['headroom_mib'] >= 512 and state['max_temperature_c'] < 85, 'registered GPU bound violated')

def main():
    require(__debug__, 'launch guards require Python optimization disabled')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    verify_files({str(args.config): args.sha256})
    config = read(args.config)
    c = campaign()
    binding = read(c / 'binary.json')
    require(set(binding['binaries']) == {'looped_agent_probe', 'learned_readout_probe'}, 'binary population differs')
    authority = qualification_authority(c, binding)
    require(binding['features'] == FEATURES, 'wrong compiled features')
    for repo, revision in [(REPO, binding['source']), (REPO.parent / 'candle_graph', DEPENDENCY)]:
        require(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == revision,
                'runtime checkout revision differs')
        require(not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all'], text=True).strip(),
                'runtime checkout is dirty')
        subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', 'HEAD', '@{upstream}'], check=True)
    prescribed = registered(config['name'], binding)
    exact(config, prescribed)
    verify_files(config['frozen_files'])
    name = config['name']
    root = c / name
    require(not list(c.glob(name + '*')), 'never reuse invocation root or evidence siblings')
    binary = c / config['binary']
    verify_files({str(binary): binding['binaries'][config['binary']], str(LIFECYCLE): LIFECYCLE_SHA})
    spec = importlib.util.spec_from_file_location('c11_lifecycle', LIFECYCLE)
    lifecycle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lifecycle)
    signal.signal(signal.SIGTERM, lifecycle.terminated)
    signal.signal(signal.SIGINT, lifecycle.terminated)
    audit = config['stage'] == 'audit'
    command = [str(binary), *config['arguments'], '--output-dir', str(root), '--device', 'cpu' if audit else 'cuda:0']
    if not audit:
        nsdir = root.with_suffix('.nsight')
        nsdir.mkdir()
        command = [NSYS, 'profile', '--trace=cuda,nvtx,osrt,cudnn,cublas', '--sample=process-tree',
            '--cpuctxsw=process-tree', '--backtrace=lbr', '--capture-range=nvtx', '--nvtx-capture=tofy.looped/capture',
            '--capture-range-end=repeat:1:defer', '--wait=all', '--kill=none', '--output', str(nsdir / 'capture'), *command]
    env = os.environ.copy()
    env.update(TOFY_PERF_TRACE=str(root.with_suffix('.host.json')), NSYS_NVTX_PROFILER_REGISTER_ONLY='0',
               NVIDIA_TF32_OVERRIDE='0', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def sample_gpu(timeout):
        require(Path('/sys/class/power_supply/ACAD/online').read_text().strip() == '1', 'AC is offline')
        value = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits', '--id=0'], capture_output=True, text=True, check=True, timeout=timeout)
        used, total, temp = [int(x.strip()) for x in value.stdout.strip().split(',')]
        require(total-used >= 512 and temp < 85, 'registered GPU reserve/temperature bound violated')
        return used, total, temp
    sample_gpu(5)
    def record_child(pid, pgid):
        record = dict(pid=pid, pgid=pgid, supervisor_pid=os.getpid(), command=command,
            environment={key: env[key] for key in ('TOFY_PERF_TRACE', 'NSYS_NVTX_PROFILER_REGISTER_ONLY', 'NVIDIA_TF32_OVERRIDE')},
            binary_sha256=digest(binary), stage=config['stage'], invocation=str(args.config), invocation_sha256=args.sha256,
            lifecycle_sha256=LIFECYCLE_SHA, started_local=now())
        save(root.with_suffix('.process.json'), record)
        print(json.dumps(dict(name=name, pid=pid, supervisor_pid=os.getpid(), started_local=record['started_local'])), flush=True)
    with root.with_suffix('.stdout.log').open('x') as output, root.with_suffix('.telemetry.jsonl').open('x') as telemetry:
        state = lifecycle.supervise_child(command, root, env, output, telemetry, record_child, sample_gpu,
            model_seconds=config['model_seconds'], finalization_seconds=config['finalization_seconds'])
    try:
        state['manifest_sha256'] = lifecycle.verify(root)
        report = read(root / 'report.json')
        subset(report, config['expected_report'])
        subset(read(root / 'metadata.json')['provenance'], dict(source_revision=binding['source'],
            binary_sha256=binding['binaries'][config['binary']], candle_graph_revision=DEPENDENCY))
        state['report_status'] = report['status']
        state['reported_model_elapsed_seconds'] = lifecycle.reported_model_elapsed(report, config['model_seconds'])
        verify_files(config['frozen_files'])
        verify_files(authority['frozen_files'])
        valid_cleanup(state)
        state['accepted'] = True
    except Exception as error:
        state.update(accepted=False, integrity_error=repr(error))
    state['classification'] = ('implementation_smoke' if config['stage'] == 'qualification' else 'complete_pending_analysis') if state['accepted'] else 'failed_infrastructure_or_integrity'
    state['finished_local'] = now()
    save(root.with_suffix('.exit.json'), state)
    print(json.dumps(state), flush=True)
    require(state['accepted'], 'C11 invocation failed; exclude its retained root from evidence')

if __name__ == '__main__':
    main()
