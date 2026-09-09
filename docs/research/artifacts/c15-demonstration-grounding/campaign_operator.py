"""C15 fixed frozen-frame campaign, reusing the tested C12 lifecycle."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
sys.dont_write_bytecode = True
R = Path(__file__).resolve().parent
R12 = R.parent / '2026-09-09T143629Z-tofy-looped-grounded-policy-learning'
PARENT = Path('/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST')
REPO = Path('/home/stepan/Projects/code/Tofy-demonstration-grounding')
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

NAMES = ('qualify-b4', 'frames-initial', 'frames-final')
REFERENCES = {'initial': 'frozen-seen', 'final': 'final-seen-factual'}
CHECKPOINTS = {
    'initial': ('4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802', 'a43ad78fe7a6b1594d23c205fb191c8ac81610deaaff25198cb421d423edc678'),
    'final': ('dc732f17a38f7bc6510dc6480a61dc4862fcb7c89ae1560f3a06515b8785d93f', '37675528ef00055f16a17e826648078ff042492787087f3604d58cf0efc60fdf'),
}

def clean_revision():
    revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    require(not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=all'], text=True).strip(), 'dirty checkout')
    subprocess.run(['git', '-C', str(REPO), 'merge-base', '--is-ancestor', 'HEAD', '@{upstream}'], check=True)
    return revision

def prepare(campaign, binary):
    require(campaign.is_absolute() and not campaign.exists() and '.' not in campaign.name, 'new absolute dotless campaign required')
    revision = clean_revision()
    build = read(R / 'build.json')
    require(build['source_revision'] == revision and build['binary_sha256'] == digest(binary), 'build evidence source/binary differs')
    process = read(R / 'operations/build-cuda.process.json')
    build_exit = read(R / 'operations/build-cuda.exit.json')
    require(build_exit['accepted'] is True and process['command'] == build['command'], 'build execution not accepted')
    require(build['features'] == 'cudnn,profiling,serde_json/float_roundtrip' and build['features'] in build['command'] and build['build_command_env'] != 'unknown', 'build feature provenance missing')
    files = dict(read(PARENT / 'campaign-spec.json')['frozen_files'])
    for name in ('registration.md', 'campaign_operator.py', 'operator_tests.py', 'analysis.py', 'analysis_tests.py', 'independent_review.py', 'independent_review_tests.py'):
        path = R / name
        require(digest(path) == digest(REPO / 'docs/research/artifacts/c15-demonstration-grounding' / name), 'operator snapshot differs')
        files[str(path)] = digest(path)
    for name in ('Cargo.toml', 'Cargo.lock', 'build.rs', 'src/p2/looped_agent/model.rs', 'src/p2/looped_agent/profile.rs', 'src/p2/looped_agent/task.rs', 'src/p2/looped_agent/grounded_policy.rs', 'examples/grounded_policy_probe.rs', 'examples/grounded_policy/frames.rs', 'examples/grounded_policy/engine.rs', 'examples/grounded_policy/evidence.rs', 'examples/grounded_policy/data.rs'):
        files[str(REPO / name)] = digest(REPO / name)
    extra = [R / 'build.json', R / 'operations/build-cuda.process.json', R / 'operations/build-cuda.exit.json', R / 'operations/build-cuda.stdout.log', R12 / 'completed-campaign.manifest.json', R12 / 'completed-campaign-verification.json', PARENT / 'parity-b4/qualification-rows.jsonl', PARENT / 'audit/seen-factual-audit.jsonl', PARENT / 'audit/manifest.json']
    extra += [PARENT / name / 'evaluation-rows.jsonl' for name in REFERENCES.values()]
    extra += [PARENT / 'train-seed0' / name for name in ('final-core.safetensors', 'final-head.safetensors')]
    for path in extra:
        files[str(path)] = digest(path)
    require(digest(R12 / 'completed-campaign.manifest.json') == '0a4ac9a8d3f46ebfca141360384a18761ea4c9fd175c43176788dd422ec7ab6f', 'C12 outer seal differs')
    # Verify the parent evidence inventory once, not merely its manifest filename.
    outer = read(R12 / 'completed-campaign.manifest.json')
    for rel, row in outer['files'].items():
        path = PARENT / rel
        require(digest(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'parent evidence changed')
    verify_files(files)
    campaign.mkdir()
    (campaign / 'invocations').mkdir()
    shutil.copy2(binary, campaign / 'grounded_policy_probe')
    files[str(campaign / 'grounded_policy_probe')] = digest(campaign / 'grounded_policy_probe')
    for name, template, mode in (
        ('qualify-b4', 'parity-b4', 'qualify'),
        ('frames-initial', 'frozen-seen', 'eval_frames_initial'),
        ('frames-final', 'final-seen-factual', 'eval_frames_final'),
    ):
        cfg = read(PARENT / 'invocations' / f'{template}.json')
        cfg.update(source_revision=revision, registration=str(R / 'registration.md'), registration_sha256=digest(R / 'registration.md'), mode=mode, output_dir=str(campaign / name), max_seconds=120)
        pair = CHECKPOINTS['final' if name == 'frames-final' else 'initial']
        require((cfg['core_sha256'], cfg['head_sha256']) == pair and cfg['updates'] == 0, 'registered checkpoint/update identity differs')
        path = campaign / 'invocations' / f'{name}.json'
        save(path, cfg)
        files[str(path)] = digest(path)
    spec = dict(schema='looped-demonstration-grounding-launch-v1', campaign=str(campaign), repository=str(REPO), source=revision, binary=str(campaign / 'grounded_policy_probe'), binary_sha256=digest(campaign / 'grounded_policy_probe'), dependency_revision=DEPENDENCY, frozen_files=files, created_local=now())
    save(campaign / 'launch-spec.json', spec)
    return spec

def healthy_capture(campaign, name):
    rows = read(campaign / f'{name}.bound/summary.json')
    require(len(rows) == 1 and all(x['health']['structurally_valid'] and x['health']['capture_complete'] and x['raw_application_labels_verified'] and x['gpu']['status'] == 'available' and x['gpu']['provenance_binding'] == 'bound' for x in rows), 'incomplete bound capture')

def remaining(deadline, ceiling):
    seconds = min(ceiling, deadline - time.monotonic())
    require(seconds > 0, 'campaign budget exhausted')
    return seconds

def invoke(spec, name, deadline):
    campaign = Path(spec['campaign'])
    verify_files(spec['frozen_files'])
    cfgpath = campaign / 'invocations' / f'{name}.json'
    cfg = read(cfgpath)
    pair = CHECKPOINTS['final' if name == 'frames-final' else 'initial']
    require(cfg['max_seconds'] == 120 and cfg['updates'] == 0 and (cfg['core_sha256'], cfg['head_sha256']) == pair, 'registered invocation limits differ')
    expected = dict(status='complete_pending_analysis', optimizer_updates=0, physical_batch=cfg['physical_batch'], input_rows=4 if name == 'qualify-b4' else 1024)
    if name != 'qualify-b4':
        expected.update(classification='exploratory_grounding', cohort='seen', cleared=False, frame_count=7)
    authpath = campaign / 'invocations' / f'{name}-authority.json'
    save(authpath, dict(schema='looped-grounded-policy-launch-v1', accepted=True, created_local=now(), config_sha256=digest(cfgpath), mode=cfg['mode'], campaign=str(campaign), name=name, binary=spec['binary'], binary_sha256=spec['binary_sha256'], repository=spec['repository'], source=spec['source'], frozen_files=spec['frozen_files'], expected_report=expected))
    tracked(spec, [PYTHON, str(R12 / 'supervise.py'), '--config', str(cfgpath), '--sha256', digest(cfgpath), '--authority', str(authpath), '--authority-sha256', digest(authpath)], name + '-model', remaining(deadline, 600))
    require(read(campaign / f'{name}.exit.json')['accepted'] is True, 'model invocation not accepted')
    tracked(spec, [PYTHON, str(R12 / 'bind_nsight.py'), '--root', str(campaign / name)], name + '-bind', remaining(deadline, 240))
    healthy_capture(campaign, name)
    require(root_manifest(campaign / name)[0] == read(campaign / f'{name}.exit.json')['manifest_sha256'], 'model manifest differs')

def qualification(new, old):
    a = [json.loads(x) for x in Path(new).read_text().splitlines()]
    b = [json.loads(x) for x in Path(old).read_text().splitlines()]
    require(len(a) == len(b) == 4, 'qualification row count')
    errors = {}
    for i, (row, ref) in enumerate(zip(a, b)):
        require(set(row) == set(ref) == {'index', 'logits', 'attention', 'pooled', 'current', 'cls'} and row['index'] == ref['index'] == i, 'qualification identity')
        for key, shape in [('logits', (4,)), ('attention', (2, 64)), ('pooled', (256,)), ('current', (64, 128)), ('cls', (128,))]:
            x, y = np.asarray(row[key], dtype=np.float64), np.asarray(ref[key], dtype=np.float64)
            require(x.shape == y.shape == shape and np.isfinite(x).all() and np.isfinite(y).all(), 'qualification shape/nonfinite')
            error = np.abs(x-y)
            require(np.all(error <= 1e-5 + 1e-5 * np.abs(y)), 'qualification numeric mismatch')
            errors[key] = max(errors.get(key, 0), float(error.max(initial=0)))
            if key in ('logits', 'attention'):
                require(np.array_equal(x.argmax(axis=-1), y.argmax(axis=-1)), 'qualification winner changed')
    return dict(accepted=True, rows=4, maximum_absolute_errors=errors, created_local=now())

def receipt(spec):
    campaign = Path(spec['campaign'])
    files = dict(spec['frozen_files'])
    arms = {}
    for name in NAMES:
        root = campaign / name
        state = read(root.with_suffix('.exit.json'))
        require(state['accepted'] and state['bindings_unchanged'], 'invocation integrity')
        for suffix in ('model', 'bind'):
            operation = read(campaign / 'operations' / f'{name}-{suffix}.exit.json')
            require(operation['accepted'] and operation['pid_gone'] and operation['group_gone'] and not operation['owned_survivors'] and not operation['group_survivors'] and operation['cleanup_error'] is None, 'operation cleanup incomplete')
        require(root_manifest(root)[0] == state['manifest_sha256'], 'manifest changed')
        healthy_capture(campaign, name)
        for suffix in ('.exit.json', '.process.json'):
            files[str(root.with_suffix(suffix))] = digest(root.with_suffix(suffix))
        files[str(root / 'manifest.json')] = digest(root / 'manifest.json')
        files[str(root / 'report.json')] = digest(root / 'report.json')
        for path in root.with_suffix('.bound').rglob('*'):
            if path.is_file():
                files[str(path)] = digest(path)
        cfg = read(campaign / 'invocations' / f'{name}.json')
        report = read(root / 'report.json')
        require(report['optimizer_updates'] == 0, 'optimizer executed')
        if name != 'qualify-b4':
            require(report['changes']['all_parameters_unchanged'] is True and report['changes']['unused_heads_unchanged'] is True, 'parameter drift')
            files[str(root / 'evaluation-rows.jsonl')] = digest(root / 'evaluation-rows.jsonl')
            arms[name.removeprefix('frames-')] = {key: cfg[key] for key in ('core_sha256', 'head_sha256')}
    require(read(campaign / 'qualification.json')['accepted'] is True, 'qualification not accepted')
    files[str(campaign / 'qualification.json')] = digest(campaign / 'qualification.json')
    verify_files(files)
    save(campaign / 'integrity.json', dict(schema='looped-demonstration-grounding-integrity-v1', accepted=True, checks={key: True for key in ('source', 'checkpoints', 'zero_updates', 'unchanged_parameters', 'qualification', 'profiles', 'cleanup')}, frozen_files=files, source_revision=spec['source'], binary_sha256=spec['binary_sha256'], dependency_revision=DEPENDENCY, arms=arms, created_local=now()))
    config = dict(schema='looped-demonstration-grounding-analysis-v1', registration={'path': str(R / 'registration.md'), 'sha256': digest(R / 'registration.md')}, integrity={'path': str(campaign / 'integrity.json'), 'sha256': digest(campaign / 'integrity.json')}, frozen_files=files, arms={})
    for arm, ref in REFERENCES.items():
        config['arms'][arm] = {kind: {'path': str(path), 'sha256': digest(path)} for kind, path in [('rows', campaign / f'frames-{arm}/evaluation-rows.jsonl'), ('reference', PARENT / ref / 'evaluation-rows.jsonl')]}
    save(campaign / 'analysis-config.json', config)

def execute(spec):
    start = time.monotonic()
    campaign = Path(spec['campaign'])
    verify_files(spec['frozen_files'])
    require(clean_revision() == spec['source'], 'source changed')
    save(campaign / 'clock.json', dict(started_local=now()))
    invoke(spec, 'qualify-b4', start + 1200)
    save(campaign / 'qualification.json', qualification(campaign / 'qualify-b4/qualification-rows.jsonl', PARENT / 'parity-b4/qualification-rows.jsonl'))
    for name in NAMES[1:]:
        require(time.monotonic()-start < 1200, 'campaign budget exceeded')
        invoke(spec, name, start + 1200)
    receipt(spec)
    require(time.monotonic()-start < 1200, 'campaign budget exceeded')
    save(campaign / 'execution.json', dict(accepted=True, elapsed_seconds=time.monotonic()-start, finished_local=now()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'execute'))
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--spec-sha256')
    args = parser.parse_args()
    if args.stage == 'prepare':
        require(args.binary is not None, 'binary required')
        spec = prepare(args.campaign, args.binary)
        print(json.dumps(dict(source=spec['source'], binary_sha256=spec['binary_sha256'], spec_sha256=digest(args.campaign / 'launch-spec.json'))))
    else:
        require(digest(args.campaign / 'launch-spec.json') == args.spec_sha256, 'launch specification differs')
        execute(read(args.campaign / 'launch-spec.json'))
        print('C15 fixed frozen-frame campaign completed; analysis remains separate.')

if __name__ == '__main__':
    main()
