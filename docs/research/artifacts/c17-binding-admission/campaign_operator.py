"""Registered C17 sequencing; generic, hash-pinned C12 process supervision."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import importlib.util
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.dont_write_bytecode = True
R = Path(__file__).resolve().parent
R12 = R.parent / '2026-09-09T143629Z-tofy-looped-grounded-policy-learning'
R15 = R.parent / '2026-09-09T173927Z-tofy-looped-demonstration-grounding'
REPO = Path('/home/stepan/Projects/code/Tofy-binding-admission')
PYTHON = '/home/stepan/venvs/tensorboard/bin/python3'
DEPENDENCY = '1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a'
PINNED = {
    'supervise.py': '0c1eb08878efbb9f178828e04fcce74712d826cffc23d9aa2a2a363b51f151c9',
    'driver.py': '8de46c86aa3d2a95e97616cb0908026521ee536a75729934d9691c36d68c2078',
    'bind_nsight.py': '82870b0247b142533b3702d5332bca43aa10f9785c9b96a8acd3ec405ce8be5b',
}
for _name, _sha in PINNED.items():
    with (R12 / _name).open('rb') as _f:
        if hashlib.file_digest(_f, 'sha256').hexdigest() != _sha:
            raise ValueError(f'C12 lifecycle changed: {_name}')
sys.path.insert(0, str(R12))
from supervise import read, save, digest, verify_files, require, now, root_manifest
from driver import tracked
sys.path.remove(str(R12))

ARTIFACTS = ('registration.md', 'campaign_operator.py', 'operator_tests.py', 'data.py', 'data_tests.py', 'analysis.py', 'analysis_tests.py', 'independent_review.py', 'independent_review_tests.py')
SOURCE_FILES = ('Cargo.toml', 'Cargo.lock', 'build.rs', 'src/p2/looped_agent/binding.rs', 'src/p2/looped_agent/model.rs', 'src/p2/looped_agent/mod.rs', 'src/p2/looped_agent/profile.rs', 'src/p2/optimizer.rs', 'examples/action_binding_probe.rs', 'examples/action_binding/engine.rs', 'examples/grounded_policy/evidence.rs')
COUNTS = {'fit': 1536, 'heldout': 768, 'cached_visual': 1024}

def clean_revision():
    revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    require(not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=all'], text=True).strip(), 'dirty checkout')
    subprocess.run(['git', '-C', str(REPO), 'merge-base', '--is-ancestor', 'HEAD', '@{upstream}'], check=True)
    return revision

def remaining(deadline, ceiling):
    seconds = min(ceiling, deadline - time.monotonic())
    require(seconds > 0, 'campaign budget exhausted')
    return seconds

def campaign_deadline(campaign):
    clock = read(campaign / 'clock.json')
    require(clock['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'campaign host rebooted')
    require(type(clock['started_monotonic']) in (int,float) and math.isfinite(clock['started_monotonic']) and 0 <= clock['started_monotonic'] <= time.monotonic(), 'campaign clock invalid')
    return clock['started_monotonic'] + 1800

def timing_components(report, records, supervisor):
    count = report['optimizer_updates']
    require(type(count) is int and count in (2,5) and len(records) == count, 'registered disposable count required')
    def duration(value):
        require(type(value) in (int,float) and math.isfinite(value) and value >= 0, 'invalid duration')
        return float(value)
    cumulative=[]
    for index,row in enumerate(records,1):
        require(type(row['update']) is int and row['update'] == index, 'timing update order differs')
        cumulative.append(duration(row['elapsed_seconds']))
    ordinary=[b-a for a,b in zip(cumulative,cumulative[1:])]
    require(all(x>0 for x in ordinary), 'nonmonotonic update times')
    whole=duration(report['updates_elapsed_seconds'])
    reported=duration(report['elapsed_seconds'])
    checkpoint=duration(report['checkpoint_seconds'])
    require(cumulative[-1]<=reported and checkpoint<=reported, 'timing records/checkpoint exceed model lifetime')
    selected_residual=whole-sum(ordinary)
    setup_residual=reported-whole
    wrapper_residual=duration(supervisor['model_phase_seconds'])-reported
    require(selected_residual>0 and setup_residual>=0 and wrapper_residual>=0, 'inconsistent timing boundaries')
    return dict(ordinary_seconds=max(ordinary), selected_update_residual_seconds=selected_residual, setup_and_postloop_residual_seconds=setup_residual, supervisory_residual_seconds=wrapper_residual, checkpoint_seconds=checkpoint, observed_ordinary_updates=len(ordinary))

def admission(trials):
    require(len(trials)==2 and [t['report']['optimizer_updates'] for t in trials]==[2,5], 'two- and five-update trials required')
    components=[timing_components(t['report'],t['records'],t['supervisor']) for t in trials]
    maxima={key:max(row[key] for row in components) for key in components[0] if key!='observed_ordinary_updates'}
    predicted=1.25*(1147*maxima['ordinary_seconds']+3*maxima['selected_update_residual_seconds']+maxima['setup_and_postloop_residual_seconds']+maxima['supervisory_residual_seconds'])+maxima['checkpoint_seconds']
    return dict(accepted=predicted<=600, reserved_training_seconds=predicted, model_limit_seconds=600, ordinary_updates=1147, selected_updates=3, reserve_multiplier=1.25, measured_components=components, componentwise_maxima=maxima, interpretation='Empirical stage/cadence forecast, not a runtime upper-bound theorem; selected residual includes cold work and finalization, not isolated kernel time.')

def timing_trial(campaign,name,report):
    import json
    with (campaign/name/'updates.jsonl').open() as source: records=[json.loads(line) for line in source]
    return dict(report=report, records=records, supervisor=read(campaign/f'{name}.exit.json'))

def prepare(campaign, binary, dataset):
    require(campaign.is_absolute() and not campaign.exists() and '.' not in campaign.name, 'new absolute dotless campaign required')
    revision = clean_revision()
    build = read(R / 'build.json')
    require(build['source_revision'] == revision and build['binary_sha256'] == digest(binary), 'build evidence source/binary differs')
    process = read(R / 'operations/build-cuda.process.json')
    require(read(R / 'operations/build-cuda.exit.json')['accepted'] is True and process['command'] == build['command'], 'build execution not accepted')
    require(build['features'] == 'cudnn,profiling,serde_json/float_roundtrip' and build['features'] in build['command'] and build['build_command_env'] != 'unknown', 'build feature provenance missing')
    files = dict(read(R / 'source-freeze.json')['files'])
    for name in ARTIFACTS:
        path = R / name
        require(digest(path) == digest(REPO / 'docs/research/artifacts/c17-binding-admission' / name), 'operator snapshot differs')
        files[str(path)] = digest(path)
    for name in SOURCE_FILES:
        files[str(REPO / name)] = digest(REPO / name)
    for path in [R / 'source-freeze.json', R / 'build.json', dataset, *[R / 'operations' / f'build-cuda.{suffix}' for suffix in ('process.json', 'exit.json', 'stdout.log')], *[R12 / x for x in PINNED]]:
        files[str(path)] = digest(path)
    require(digest(dataset) == 'bc1413652b6bf38cadde28c8f1fd3d88d9cf781ab55413300bc155e341c47295', 'registered canonical dataset differs')
    require(digest(R15 / 'completed-campaign.manifest.json') == '6007293c5f51d263b0e3f40dcd819124efa8ffdaf2809202eda0c60142ffd40d', 'C15 seal differs')
    outer = read(R15 / 'completed-campaign.manifest.json')
    for rel, row in outer['files'].items():
        path = Path(outer['campaign']) / rel
        require(digest(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'C15 inventory changed')
    verify_files(files)
    campaign.mkdir()
    (campaign / 'invocations').mkdir()
    shutil.copy2(binary, campaign / 'action_binding_probe')
    files[str(campaign / 'action_binding_probe')] = digest(campaign / 'action_binding_probe')
    spec = dict(schema='looped-action-binding-launch-v1', campaign=str(campaign), repository=str(REPO), source=revision, binary=str(campaign / 'action_binding_probe'), binary_sha256=digest(campaign / 'action_binding_probe'), dependency_revision=DEPENDENCY, dataset=str(dataset), dataset_sha256=digest(dataset), frozen_files=files, created_local=now())
    save(campaign / 'launch-spec.json', spec)
    return spec

def healthy_capture(campaign, name, count):
    rows = read(campaign / f'{name}.bound/summary.json')
    require(len(rows) == count and all(x['health']['structurally_valid'] and x['health']['capture_complete'] and x['raw_application_labels_verified'] and x['gpu']['status'] == 'available' and x['gpu']['provenance_binding'] == 'bound' for x in rows), 'incomplete bound capture')

def invoke(spec, name, deadline, mode, batch, updates=0, cohort=None, loops=4, cleared=False, query_cleared=False):
    campaign = Path(spec['campaign'])
    verify_files(spec['frozen_files'])
    cfg = dict(schema='looped-action-binding-config-v1', source_revision=spec['source'], registration=str(R / 'registration.md'), registration_sha256=digest(R / 'registration.md'), dataset=spec['dataset'], dataset_sha256=spec['dataset_sha256'], mode=mode, output_dir=str(campaign / name), physical_batch=batch, updates=updates, max_seconds=600 if mode == 'train' else 120, checkpoint=None, checkpoint_sha256=None, cohort=cohort, loops=loops, cleared=cleared, query_cleared=query_cleared)
    require(remaining(deadline, 1800) >= cfg['max_seconds'] + 130, 'insufficient remaining registered invocation window')
    frozen = dict(spec['frozen_files'])
    if mode == 'eval_final':
        checkpoint = campaign / 'train-seed0/final.safetensors'
        require(read(campaign / 'train-seed0.exit.json')['accepted'] is True, 'training incomplete')
        cfg.update(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))
        frozen[str(checkpoint)] = digest(checkpoint)
        frozen[str(campaign / 'train-seed0/manifest.json')] = root_manifest(campaign / 'train-seed0')[0]
    cfgpath = campaign / 'invocations' / f'{name}.json'
    save(cfgpath, cfg)
    frozen[str(cfgpath)] = digest(cfgpath)
    expected = dict(status='complete_pending_analysis', optimizer_updates=updates, physical_batch=batch, input_rows=updates*512 if updates else 4 if mode == 'qualify' else COUNTS[cohort], loops=loops, cleared=cleared, query_cleared=query_cleared, executed_vision_core_forwards=0, parameter_count=1580804)
    authpath = campaign / 'invocations' / f'{name}-authority.json'
    save(authpath, dict(schema='looped-grounded-policy-launch-v1', accepted=True, created_local=now(), config_sha256=digest(cfgpath), mode=mode, campaign=str(campaign), name=name, binary=spec['binary'], binary_sha256=spec['binary_sha256'], repository=spec['repository'], source=spec['source'], frozen_files=frozen, expected_report=expected))
    operation = tracked(spec, [PYTHON, str(R12 / 'supervise.py'), '--config', str(cfgpath), '--sha256', digest(cfgpath), '--authority', str(authpath), '--authority-sha256', digest(authpath)], name + '-model', remaining(deadline, cfg['max_seconds'] + 130), allow_failure=mode == 'batch_smoke')
    state = read(campaign / f'{name}.exit.json')
    if not state['accepted']:
        require(mode == 'batch_smoke' and state['capacity_failure'] is True, 'non-capacity failure stops campaign')
        require(operation['pid_gone'] and operation['group_gone'] and not operation['owned_survivors'] and not operation['group_survivors'] and operation['cleanup_error'] is None, 'failed trial cleanup incomplete')
        return None
    require(operation['accepted'] is True, 'outer lifecycle rejected invocation')
    tracked(spec, [PYTHON, str(R12 / 'bind_nsight.py'), '--root', str(campaign / name)], name + '-bind', remaining(deadline, 180))
    healthy_capture(campaign, name, 3 if mode == 'train' else 1)
    require(root_manifest(campaign / name)[0] == state['manifest_sha256'], 'model manifest differs')
    report = read(campaign / name / 'report.json')
    if mode in ('qualify', 'eval_initial', 'eval_final'):
        require(report['changes']['all_parameters_unchanged'] is True and report['starting_parameter_sha256'] == report['evaluated_parameter_sha256'], 'evaluation changed weights')
    if mode == 'batch_smoke':
        require(report['restored_changes']['all_parameters_unchanged'] is True, 'smoke restoration failed')
    print(f'Completed {name}', flush=True)
    return report

def training_integrity(campaign, data, batch):
    report = read(campaign / 'train-seed0/report.json')
    require(report['optimizer_updates'] == 1150 and report['input_rows'] == 588800 and report['effective_batch'] == 512 and report['physical_batch'] == batch and report['accumulation'] == math.ceil(512/batch), 'training budget/batch differs')
    require(not report['changes']['all_parameters_unchanged'] and report['changes']['changed_body_names'] and report['changes']['changed_head_names'], 'training did not update body/head')
    import json
    with (campaign / 'train-seed0/updates.jsonl').open() as source:
        rows = [json.loads(line) for line in source]
    require(len(rows) == 1150, 'update count differs')
    for i, row in enumerate(rows, 1):
        metrics = row['metrics']
        require(row['update'] == i and metrics['rows'] == 512 and metrics['physical_batch'] == batch and metrics['microbatches'] == math.ceil(512/batch), 'update identity differs')
        for key in ('input_gradient_norm', 'shared_core_gradient_norm', 'body_gradient_norm', 'head_gradient_norm', 'pre_clip_norm', 'clip_scale'):
            require(type(metrics[key]) in (int,float) and math.isfinite(metrics[key]) and metrics[key] > 0, 'invalid gradient family/clip')
        require(metrics['clip_scale'] <= 1 and math.isfinite(metrics['mean_ce']), 'invalid clipping/loss')
    count = 0
    with (campaign / 'train-seed0/training-stream.jsonl').open() as source:
        for count, line in enumerate(source, 1):
            row = json.loads(line)
            update, slot = divmod(count-1, 512)
            require(update < 1150, 'excess training rows')
            index = data['updates'][update][slot]
            require(row == dict(update=update+1, slot=slot, dataset_index=index, row=data['fit'][index]), 'training stream differs')
    require(count == 588800, 'training stream truncated')
    return report

def receipt(spec, names, streams, batch):
    campaign = Path(spec['campaign'])
    files = dict(spec['frozen_files'])
    train = training_integrity(campaign, read(spec['dataset']), batch)
    init_digest = train['initial_parameter_sha256']
    checkpoints = {stage: dict(sha256=digest(campaign / 'train-seed0' / f'{stage}.safetensors'), parameter_sha256=train[f'{stage}_parameter_sha256']) for stage in ('initial', 'final')}
    for name in names:
        root = campaign / name
        state = read(root.with_suffix('.exit.json'))
        require(state['accepted'] and state['bindings_unchanged'], 'invocation integrity')
        require(root_manifest(root)[0] == state['manifest_sha256'], 'manifest changed')
        healthy_capture(campaign, name, 3 if name == 'train-seed0' else 1)
        report = read(root / 'report.json')
        require(report['initial_parameter_sha256'] == init_digest, 'initializer differs between invocations')
        for suffix in ('model', 'bind'):
            operation = read(campaign / 'operations' / f'{name}-{suffix}.exit.json')
            require(operation['accepted'] and operation['pid_gone'] and operation['group_gone'] and not operation['owned_survivors'] and not operation['group_survivors'] and operation['cleanup_error'] is None, 'operation cleanup incomplete')
        for path in [root.with_suffix('.exit.json'), root.with_suffix('.process.json'), root / 'manifest.json', root / 'report.json', *[p for suffix in ('.bound','.nsight') for p in root.with_suffix(suffix).rglob('*') if p.is_file()]]:
            files[str(path)] = digest(path)
        if name in streams:
            stream = streams[name]
            require(report['starting_parameter_sha256'] == checkpoints[stream['stage']]['parameter_sha256'] and report['evaluated_parameter_sha256'] == report['starting_parameter_sha256'] and report['changes']['all_parameters_unchanged'], 'evaluated checkpoint differs')
            path = root / 'evaluation-rows.jsonl'
            stream.update(rows=str(path), sha256=digest(path), checkpoint_sha256=checkpoints[stream['stage']]['sha256'], parameter_sha256=checkpoints[stream['stage']]['parameter_sha256'])
            files[str(path)] = digest(path)
    for name in ('initial.safetensors', 'final.safetensors', 'training-stream.jsonl', 'updates.jsonl'):
        path = campaign / 'train-seed0' / name
        files[str(path)] = digest(path)
    require(read(campaign / 'admission.json')['accepted'], 'training admission failed')
    for name in ('admission.json','clock.json','launch-spec.json'):
        files[str(campaign / name)] = digest(campaign / name)
    verify_files(files)
    receipt_path = campaign / 'integrity.json'
    save(receipt_path, dict(schema='looped-action-binding-integrity-v1', accepted=True, checks={k:True for k in ('data','source','build','initialization','completed_training','gradients','device','profiles','checkpoints','cleanup')}, source_revision=spec['source'], binary_sha256=spec['binary_sha256'], dependency_revision=DEPENDENCY, dataset_sha256=spec['dataset_sha256'], registration_sha256=digest(R / 'registration.md'), checkpoints=checkpoints, frozen_files=files, streams=streams, physical_batch=batch, accumulation=math.ceil(512/batch), created_local=now()))
    files[str(receipt_path)] = digest(receipt_path)
    config_streams = {name:{key:stream[key] for key in ('rows','sha256','cohort','stage','loops','cleared','query_cleared')} for name,stream in streams.items()}
    save(campaign / 'analysis-config.json', dict(schema='looped-action-binding-analysis-config-v1', dataset=spec['dataset'], dataset_sha256=spec['dataset_sha256'], registration=dict(path=str(R / 'registration.md'), sha256=digest(R / 'registration.md')), streams=config_streams, integrity_receipt=str(receipt_path), integrity_receipt_sha256=digest(receipt_path), frozen_files=files))

def execute(spec):
    start = time.monotonic()
    deadline = start + 1500  # Reserve120+120+60seconds for scoring and sealing.
    campaign = Path(spec['campaign'])
    verify_files(spec['frozen_files'])
    require(clean_revision() == spec['source'], 'source changed')
    save(campaign / 'clock.json', dict(started_local=now(), started_monotonic=start, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), maximum_wall_seconds=1800, analysis_and_seal_reserve_seconds=300))
    names, streams = [], {}
    def run(name, mode, batch, **kwargs):
        result = invoke(spec, name, deadline, mode, batch, **kwargs)
        if result is not None:
            names.append(name)
            if mode.startswith('eval_'):
                streams[name] = dict(cohort=kwargs['cohort'], stage='initial' if mode == 'eval_initial' else 'final', loops=kwargs.get('loops',4), cleared=kwargs.get('cleared',False), query_cleared=kwargs.get('query_cleared',False))
        return result
    qualification = run('qualify-b4', 'qualify', 4)
    selected = None
    for batch in (512,256,128,64,32,16,8,4,2,1):
        smoke = run(f'smoke-b{batch}', 'batch_smoke', batch, updates=2)
        if smoke is None: continue
        confirm = run(f'confirm-b{batch}', 'batch_smoke', batch, updates=5)
        if confirm is None: continue
        require(smoke['initial_parameter_sha256'] == confirm['initial_parameter_sha256'] == qualification['initial_parameter_sha256'], 'qualification initialization differs')
        decision = admission([timing_trial(campaign,f'smoke-b{batch}',smoke), timing_trial(campaign,f'confirm-b{batch}',confirm)])
        decision.update(physical_batch=batch, accumulation=math.ceil(512/batch), created_local=now())
        save(campaign / 'admission.json', decision)
        require(decision['accepted'], 'registered training budget cannot be met')
        selected = batch
        break
    require(selected is not None, 'no stable physical batch')
    for cohort in ('fit','heldout'):
        run(f'initial-{cohort}-l4', 'eval_initial', selected, cohort=cohort)
    run('train-seed0', 'train', selected, updates=1150)
    for loops in (1,2,4,8):
        for cohort in ('fit','heldout'):
            run(f'final-{cohort}-l{loops}', 'eval_final', selected, cohort=cohort, loops=loops)
    for cohort in ('fit','heldout'):
        run(f'final-{cohort}-effects-zero', 'eval_final', selected, cohort=cohort, cleared=True)
        run(f'final-{cohort}-query-zero', 'eval_final', selected, cohort=cohort, query_cleared=True)
    run('final-cached-visual-l4', 'eval_final', selected, cohort='cached_visual')
    receipt(spec, names, streams, selected)
    remaining(deadline,1800)
    save(campaign / 'execution.json', dict(accepted=True, elapsed_seconds=time.monotonic()-start, names=names, physical_batch=selected, accumulation=math.ceil(512/selected), finished_local=now()))

def analyze(spec):
    campaign = Path(spec['campaign'])
    deadline = campaign_deadline(campaign)
    require(read(campaign / 'execution.json')['accepted'] is True, 'execution incomplete')
    verify_files(spec['frozen_files'])
    require(clean_revision() == spec['source'], 'analysis source changed')
    tracked(spec, [PYTHON, str(R/'analysis.py'), '--config', str(campaign/'analysis-config.json'), '--output', str(campaign/'analysis.json')], 'analysis', remaining(deadline-180,120))
    tracked(spec, [PYTHON, str(R/'independent_review.py'), '--config', str(campaign/'analysis-config.json'), '--report', str(campaign/'analysis.json'), '--output', str(campaign/'independent-review.json')], 'independent-review', remaining(deadline-60,120))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare','execute','analyze'))
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--spec-sha256')
    args = parser.parse_args()
    if args.stage == 'prepare':
        require(args.binary is not None and args.dataset is not None, 'binary and dataset required')
        spec = prepare(args.campaign, args.binary, args.dataset)
        print(dict(source=spec['source'], binary_sha256=spec['binary_sha256'], spec_sha256=digest(args.campaign / 'launch-spec.json')))
    else:
        require(digest(args.campaign / 'launch-spec.json') == args.spec_sha256, 'launch specification differs')
        spec = read(args.campaign / 'launch-spec.json')
        (execute if args.stage == 'execute' else analyze)(spec)
        print(f'C17 registered {args.stage} stage complete.')

if __name__ == '__main__':
    main()
