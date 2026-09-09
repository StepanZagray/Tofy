"""Seal C19 completed evidence, including valid negative scientific outcomes."""
import argparse
import importlib.util
import json
from pathlib import Path
import signal
import sys

sys.dont_write_bytecode = True
R = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('c19_seal_operator', R/'campaign_operator.py')
op = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(op)
CHECKS = {'source', 'build', 'checkpoints', 'zero_updates', 'unchanged_parameters', 'qualification', 'profiles', 'cleanup'}
GATES = {'factual_all_actions', 'factual_minimum_true_margin', 'uniform_control'}


def bound(value):
    op.require(set(value) == {'path', 'sha256'}, 'invalid artifact binding')
    path = Path(value['path'])
    op.require(path.is_absolute() and op.digest(path) == value['sha256'], 'bound artifact changed')
    return path


def merge_bindings(target, extra):
    for path, digest in extra.items():
        op.require(path not in target or target[path] == digest, 'conflicting frozen binding')
        target[path] = digest


def result_contract(report, review, config, receipt, config_sha, report_sha):
    op.require(report['schema'] == 'looped-native-binding-analysis-v1' and report['accepted'] is True, 'analysis rejected')
    op.require(review['schema'] == 'looped-native-binding-independent-review-v1' and review['accepted'] is True, 'review rejected')
    op.require(report['config_sha256'] == review['config_sha256'] == config_sha and review['report_sha256'] == report_sha, 'scored config/report changed')
    op.require(config['schema'] == 'looped-native-binding-analysis-config-v1' and set(config['streams']) == {'factual', 'uniform'}, 'analysis config differs')
    op.require(receipt['schema'] == 'looped-native-binding-integrity-v1' and receipt['accepted'] is True, 'integrity receipt rejected')
    op.require(set(receipt['checks']) == CHECKS and all(v is True for v in receipt['checks'].values()), 'runtime integrity gate failed')
    gates = report['gates']
    op.require(set(gates) == GATES and all(type(v) is bool for v in gates.values()) and all(type(v) is bool for v in review['gates'].values()) and review['gates'] == gates, 'scientific gates differ')
    decision = 'supported_frozen_native_composition' if all(gates.values()) else 'frozen_native_composition_not_supported'
    op.require(report['decision'] == review['decision'] == decision, 'scientific decision differs')
    op.require(report['registration_sha256'] == config['registration']['sha256'], 'registration differs')
    for key in ('source_revision', 'binary_sha256', 'binder_checkpoint_sha256'):
        op.require(report[key] == receipt[key], 'scored runtime identity differs')
    op.require(receipt['checkpoint_files'] == op.CHECKPOINTS and receipt['dependency_revision'] == op.DEPENDENCY, 'fixed parent identity differs')
    op.require(type(receipt['vision_loops']) is int and receipt['vision_loops'] == 4 and type(receipt['binder_loops']) is int and receipt['binder_loops'] == 4, 'frozen depth differs')


def execution_contract(execution, selection):
    op.require(execution['accepted'] is True and selection['accepted'] is True, 'execution/capacity selection incomplete')
    candidates = selection['candidates']
    op.require(0 < len(candidates) <= len(op.BATCHES), 'missing capacity population')
    confirmed, failed = [], []
    expected = ['audit', 'qualify-b4']
    for i, row in enumerate(candidates):
        batch = row['physical_batch']
        op.require(type(batch) is int and batch == op.BATCHES[i] and type(row['confirmed']) is bool, 'adaptive/invalid candidate sequence')
        if row['confirmed']:
            confirmed.append(batch)
            expected += [f'smoke-b{batch}', f'confirm-b{batch}']
        else:
            op.require(i == len(candidates)-1, 'capacity search continued after failure')
            failed = selection['failed_roots']
            op.require(failed in ([f'smoke-b{batch}'], [f'confirm-b{batch}']), 'unrecognized capacity failure')
            if failed[0].startswith('confirm-'):
                expected.append(f'smoke-b{batch}')
    op.require(confirmed and selection['physical_batch'] == execution['physical_batch'] == confirmed[-1], 'largest confirmed batch differs')
    op.require(selection['failed_roots'] == execution['failed_capacity_roots'] == failed, 'failed capacity roots differ')
    op.require(selection['limit'] == ('observed_capacity_failure' if failed else 'configured_panel_ceiling'), 'capacity stop reason differs')
    op.require(failed or len(candidates) == len(op.BATCHES), 'search stopped before configured ceiling')
    expected += ['factual', 'uniform']
    op.require(execution['names'] == expected and len(set(expected)) == len(expected), 'successful invocation population differs')
    op.require(selection['optimizer_updates'] == 0 and selection['accumulation'] == 1 and selection['scientific_rows'] == 768 and selection['actual_scientific_batch'] == min(confirmed[-1], 768), 'scientific batch/update contract differs')
    return expected, failed


def scan_process_record(value, pids, groups):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('pid', 'supervisor_pid', 'model_pid', 'pgid') and item is not None:
                op.require(type(item) is int and item > 0, 'invalid recorded process identity')
                (groups if key == 'pgid' else pids).add(item)
            elif key == 'owned_pids':
                op.require(type(item) is list and all(type(p) is int and p > 0 for p in item), 'invalid owned PID list')
                pids.update(item)
            elif key in ('owned_survivors', 'group_survivors'):
                op.require(item == [], 'owned survivors recorded')
            elif key == 'cleanup_error':
                op.require(item is None, 'cleanup error recorded')
            elif key in ('pid_gone', 'model_pid_gone', 'group_gone'):
                op.require(item is True, 'process cleanup incomplete')
            scan_process_record(item, pids, groups)
    elif isinstance(value, list):
        for item in value:
            scan_process_record(item, pids, groups)


def cleanup_contract(state):
    op.require(state['pid_gone'] is True and state['group_gone'] is True and state['owned_survivors'] == [] and state['group_survivors'] == [] and state['cleanup_error'] is None, 'mandatory process cleanup failed')


def verify_processes_gone(pids, groups):
    for pid in pids | groups:
        op.require(not Path(f'/proc/{pid}').exists(), f'process still exists: {pid}')
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            continue
        op.require(int(stat[2]) not in groups, f'process group survivor: {path.name}')


def seal(campaign):
    op.require(campaign.is_absolute() and campaign.is_dir(), 'absolute campaign required')
    deadline = op.campaign_deadline(campaign)
    op.remaining(deadline, 60)
    report_path, config_path = campaign/'analysis.json', campaign/'analysis-config.json'
    report, review = op.read(report_path), op.read(campaign/'independent-review.json')
    config = op.read(config_path)
    receipt_path = bound(config['integrity_receipt'])
    receipt = op.read(receipt_path)
    result_contract(report, review, config, receipt, op.digest(config_path), op.digest(report_path))
    op.require(config['frozen_files'] == receipt['frozen_files'] | {str(receipt_path): op.digest(receipt_path)}, 'config/receipt binding set differs')
    for value in [config[key] for key in ('registration', 'history', 'audit', 'checkpoint', 'integrity_receipt')] + list(config['streams'].values()):
        path = bound(value)
        op.require(config['frozen_files'].get(str(path)) == value['sha256'], 'unfrozen analysis input')
    execution, selection = op.read(campaign/'execution.json'), op.read(campaign/'capacity-selection.json')
    names, failed = execution_contract(execution, selection)
    invocations = {p.stem for p in (campaign/'invocations').glob('*.json') if not p.name.endswith('-authority.json')}
    op.require(invocations == set(names + failed), 'unaccounted invocation config')
    launch = op.read(campaign/'launch-spec.json')
    op.require(launch['campaign'] == str(campaign) and launch['source'] == report['source_revision'] and launch['binary_sha256'] == report['binary_sha256'], 'launch identity differs')
    bindings = dict(launch['frozen_files'])
    merge_bindings(bindings, config['frozen_files'])
    merge_bindings(bindings, op.read(R/'source-freeze.json')['files'])
    for path in (config_path, R/'source-freeze.json', R/'seal_completed.py'):
        merge_bindings(bindings, {str(path): op.digest(path)})
    captures = 0
    for name in names:
        state, cfg = op.read(campaign/f'{name}.exit.json'), op.read(campaign/'invocations'/f'{name}.json')
        op.require(state['accepted'] is True and state['bindings_unchanged'] is True and op.root_manifest(campaign/name)[0] == state['manifest_sha256'], 'invocation manifest changed after scoring')
        op.require(type(cfg['updates']) is int and cfg['updates'] == 0, 'unexpected optimizer updates')
        if name != 'audit':
            op.healthy_capture(campaign, name)
            op.report_integrity(op.read(campaign/name/'report.json'), cfg['mode'], cfg['physical_batch'], receipt['parameter_identity'])
            captures += 1
    for name in failed:
        state = op.read(campaign/f'{name}.exit.json')
        op.require(state['accepted'] is False and state['capacity_failure'] is True, 'failed root is not diagnosed capacity evidence')
    for name in names + failed:
        cleanup_contract(op.read(campaign/f'{name}.exit.json'))
        op.read(campaign/f'{name}.process.json')
        operations = ['model'] + (['bind'] if name in names and name != 'audit' else [])
        for operation in operations:
            prefix = campaign/'operations'/f'{name}-{operation}'
            op.read(Path(str(prefix)+'.process.json'))
            state = op.read(Path(str(prefix)+'.exit.json'))
            cleanup_contract(state)
            op.require(state['accepted'] is (name in names), 'operation completion status differs')
    pids, groups = set(), set()
    for base in (campaign, R/'operations'):
        for path in base.rglob('*.json'):
            if path.name.startswith('outer-seal.'):
                op.require(base != campaign, 'active seal wrapper must be external to campaign')
                continue
            if path.name.endswith(('.process.json', '.exit.json')):
                scan_process_record(op.read(path), pids, groups)
                if base != campaign:
                    merge_bindings(bindings, {str(path): op.digest(path)})
    for path in (R/'operations').glob('*.stdout.log'):
        if not path.name.startswith('outer-seal.'):
            merge_bindings(bindings, {str(path): op.digest(path)})
    for name in ('analysis', 'independent-review'):
        state = op.read(campaign/'operations'/f'{name}.exit.json')
        op.require(state['accepted'] is True, 'scoring process failed')
        cleanup_contract(state)
        op.read(campaign/'operations'/f'{name}.process.json')
    op.require(type(report['pid']) is int and report['pid'] > 0, 'invalid analysis process identity')
    pids.add(report['pid'])
    verify_processes_gone(pids, groups)
    op.verify_files(bindings)
    op.remaining(deadline, 60)
    op.save(campaign/'lifecycle.json', dict(state='complete', classification='completed_frozen_native_confirmation', decision=report['decision'], created_local=op.now(), optimizer_updates=0, all_owned_processes_gone=True, analysis_sha256=op.digest(report_path), independent_review_sha256=op.digest(campaign/'independent-review.json')))
    files = {}
    for path in sorted(campaign.rglob('*')):
        op.require(not path.is_symlink(), 'artifact symlink')
        if path.is_file():
            files[str(path.relative_to(campaign))] = dict(bytes=path.stat().st_size, sha256=op.digest(path))
    manifest = dict(schema='tofy-native-binding-evidence-v1', campaign=str(campaign), created_local=op.now(), classification='completed_frozen_native_confirmation', source=report['source_revision'], binary_sha256=report['binary_sha256'], decision=report['decision'], optimizer_updates=0, files=files, external_bindings=bindings, pids_verified_gone=sorted(pids), process_groups_verified_gone=sorted(groups), cuda_bundles=captures, successful_invocations=names, failed_capacity_roots=failed, integrity_scope='Point-in-time verification, not immutable storage; own external outer-seal wrapper excluded until its supervising receipt closes.')
    output = R/'completed-campaign.manifest.json'
    op.save(output, manifest)
    op.require({str(p.relative_to(campaign)) for p in campaign.rglob('*') if p.is_file()} == set(files), 'inventory changed')
    for rel, row in files.items():
        path = campaign/rel
        op.require(path.stat().st_size == row['bytes'] and op.digest(path) == row['sha256'], 'artifact changed while sealing')
    op.verify_files(bindings)
    verify_processes_gone(pids, groups)
    op.remaining(deadline, 60)
    digest = op.digest(output)
    with (R/'completed-campaign.manifest.sha256').open('x') as handle:
        handle.write(digest+'\n')
    result = dict(accepted=True, created_local=op.now(), manifest_sha256=digest, files=len(files), bytes=sum(row['bytes'] for row in files.values()), bindings=len(bindings), pids_gone=len(pids), process_groups_gone=len(groups), cuda_bundles=captures, optimizer_updates=0, decision=report['decision'])
    op.save(R/'completed-campaign-verification.json', result)
    state = op.read(R/'operator-state.json')
    state.update(state='complete', decision=report['decision'], manifest_sha256=digest, active_processes=[], updated_local=op.now())
    (R/'operator-state.json').write_text(json.dumps(state, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path)
    args = parser.parse_args()
    def expired(*_):
        raise TimeoutError('sealing exceeds60seconds')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    try:
        campaign = args.campaign or Path((R/'campaign-path.txt').read_text().strip())
        print(json.dumps(seal(campaign)))
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
