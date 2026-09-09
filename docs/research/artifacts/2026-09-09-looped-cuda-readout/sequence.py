#!/usr/bin/env python3
"""Run one closed C11 stage; every child is bounded, recorded, and reaped."""
import argparse
import datetime
import importlib.util
import math
import os
from pathlib import Path
import signal
import sys
import time
sys.dont_write_bytecode = True
from campaign_io import *
from supervise_stage import registered

CPU_LIFECYCLE = R10 / 'supervise_cpu.py'
CPU_SHA = '6796e12fb06a87f713f76677baf543758943716287e8b4a9f2ef68a2ec7a0cd7'
PYTHON = '/home/stepan/venvs/tensorboard/bin/python3'

def tracked(command, name, timeout):
    require(Path(name).name == name and name not in ('.', '..'), 'unsafe operation name')
    require(duration(timeout, 'operation timeout') > 0, 'operation deadline exhausted')
    verify_files({str(CPU_LIFECYCLE): CPU_SHA})
    spec = importlib.util.spec_from_file_location('c11_outer_lifecycle', CPU_LIFECYCLE)
    lifecycle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lifecycle)
    signal.signal(signal.SIGINT, lifecycle.stop_signal)
    signal.signal(signal.SIGTERM, lifecycle.stop_signal)
    root = campaign() / 'operations' / name
    root.parent.mkdir(exist_ok=True)
    require(not list(root.parent.glob(name + '.*')), 'never reuse operation evidence')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', NVIDIA_TF32_OVERRIDE='0')
    def record(pid, start):
        save(root.with_suffix('.process.json'), dict(pid=pid, supervisor_pid=os.getpid(), process_start_ticks=start,
            command=command, cwd=str(REPO), timeout_seconds=timeout, started_local=now()))
        print(json.dumps(dict(operation=name, pid=pid, supervisor_pid=os.getpid(), started_local=now())), flush=True)
    with root.with_suffix('.stdout.log').open('x') as log:
        state = lifecycle.supervise(command, REPO, env, log, record, timeout=timeout)
    state['accepted'] = (state['returncode'] == 0 and state['error'] is None and state['cleanup_error'] is None
                         and state['pid_gone'] and state['group_gone'] and not state['owned_survivors'])
    state['finished_local'] = now()
    save(root.with_suffix('.exit.json'), state)
    require(state['accepted'], f'operation failed; inspect {root.with_suffix(".stdout.log")}')
    return state

def duration(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f'invalid nonnegative runtime: {name}')
    return float(value)

def admit_model(config, budget):
    model = duration(config['model_seconds'], 'configured model budget')
    final = duration(config['finalization_seconds'], 'configured finalization budget')
    if config['stage'] != 'audit':
        require(model > 0 and final > 0 and 360-budget['model_seconds'] >= model
                and 600-budget['finalization_seconds'] >= final, 'insufficient remaining phase budget')
    limit = min(model+final+30, 1800-budget['wall_seconds'])
    require(limit > 0, 'campaign wall deadline exhausted')
    return limit

def binder_deadline(budget):
    limit = min(120, 600-budget['finalization_seconds'], 1800-budget['wall_seconds'])
    require(limit > 0, 'binder deadline exhausted')
    return limit

def budgets():
    c = campaign()
    clock = read(R / 'campaign-clock.json')
    began = datetime.datetime.fromisoformat(clock['started_local'])
    require(began.tzinfo is not None, 'campaign clock missing timezone')
    elapsed = (datetime.datetime.now().astimezone() - began).total_seconds()
    model, final = 0.0, 0.0
    for path in c.glob('*.exit.json'):
        if path.stem.startswith('audit-'):
            continue
        state = read(path)
        model += duration(state['model_phase_seconds'], f'{path.name} model')
        final += duration(state['finalization_seconds'], f'{path.name} finalization')
    for path in c.glob('*.binding.json'):
        final += duration(read(path)['elapsed_seconds'], f'{path.name} binding')
    require(0 <= model <= 360 and 0 <= final <= 600 and 0 <= elapsed <= 1800, 'registered cumulative runtime exhausted')
    return dict(model_seconds=model, finalization_seconds=final, wall_seconds=elapsed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('qualification', 'audits', 'extractions', 'heads'), required=True)
    args = parser.parse_args()
    c = campaign()
    binding = read(c / 'binary.json')
    if args.stage == 'qualification':
        require(not (R / 'campaign-clock.json').exists(), 'never restart qualification clock')
        names = ['qual-core-smoke'] + [f'qual-{core}-{arm}' for core in CORES for arm in ARMS]
        save(R / 'campaign-clock.json', dict(started_local=now(), pid=os.getpid(), source=binding['source']))
    elif args.stage == 'audits':
        names = [f'audit-{panel}' for panel in range(3)]
    elif args.stage == 'extractions':
        names = [f'features-{panel}-{core}' for panel in range(3) for core in CORES]
    else:
        names = [f'head-{panel}-{core}-{arm}' for panel in range(3) for core in CORES for arm in ARMS]
    (c / 'invocations').mkdir(exist_ok=True)
    for name in names:
        budget = budgets()
        config = registered(name, binding)
        path = c / 'invocations' / f'{name}.json'
        save(path, config)
        limit = admit_model(config, budget)
        tracked([PYTHON, '-B', str(R / 'supervise_stage.py'), '--config', str(path), '--sha256', digest(path)],
                name + '-supervisor', limit)
        state = read((c / name).with_suffix('.exit.json'))
        require(state['accepted'] is True, 'model invocation not accepted')
        if config['stage'] != 'audit':
            budget = budgets()
            tracked([PYTHON, '-B', str(c / 'bind_nsight.py'), name], name + '-bind',
                    binder_deadline(budget))
            bundles = read((c / name).with_suffix('.bound') / 'summary.json')
            require(len(bundles) == 1 and bundles[0]['health']['structurally_valid'] is True
                    and bundles[0]['health']['capture_complete'] is True
                    and bundles[0]['raw_application_labels_verified'] is True, 'missing/unhealthy first-forward capture')
        print(json.dumps(dict(completed=name, budget=budgets())), flush=True)
    if args.stage in ('audits', 'extractions'):
        stage = 'audit-seal' if args.stage == 'audits' else 'cache-seal'
        tracked([PYTHON, '-B', str(R / 'package_panels.py'), '--stage', stage], stage, min(120, 1800-budgets()['wall_seconds']))
    save(c / f'{args.stage}-sequence.json', dict(accepted=True, finished_local=now(), names=names, budget=budgets(), pid=os.getpid()))

if __name__ == '__main__':
    main()
