"""Synthetic C19 lifecycle and sealing counterexamples; no model/data execution."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


op = load('c19_operator_under_test', 'campaign_operator.py')
seal = load('c19_sealer_under_test', 'seal_completed.py')


def report(mode='evaluate', batch=512):
    n = 768 if mode == 'evaluate' else 4 if mode == 'qualify' else batch
    identity = {'core': {'parameter_sha256': 'a'*64}, 'head': {'parameter_sha256': 'b'*64},
                'binder': {'parameter_sha256': op.BINDER_PARAMETERS}}
    return dict(optimizer_updates=0, input_rows=n, panel_rows=768, physical_batch=batch,
                actual_physical_batch=min(batch, n), microbatches=math.ceil(n/batch), tail_batch=(n-1)%batch+1,
                unique_input_rows=min(n, 768), repeated_input_rows=max(0, n-768),
                core_forward_batches=math.ceil(n/batch), selector_forward_batches=7*math.ceil(n/batch),
                binder_forward_batches=math.ceil(n/batch), core_loops=4, binder_loops=4,
                privileged_role_warm_start=True, mean_ce=1.2,
                changes=dict(all_parameters_unchanged=True, unused_heads_unchanged=True),
                parameter_digests_before=identity, parameter_digests_after=copy.deepcopy(identity),
                current_frame_input_bitwise_equal=True, current_frame_max_absolute_difference=0.0)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class OperatorTests(unittest.TestCase):
    def test_actual_scientific_tails_and_full_capacity_smokes(self):
        for batch in op.BATCHES:
            for mode in ('evaluate', 'batch_smoke'):
                with self.subTest(batch=batch, mode=mode):
                    op.report_integrity(report(mode, batch), mode, batch)
        smoke, scientific = report('batch_smoke', 1024), report('evaluate', 1024)
        self.assertEqual((smoke['input_rows'], smoke['repeated_input_rows']), (1024, 256))
        self.assertEqual((scientific['input_rows'], scientific['actual_physical_batch']), (768, 768))
        self.assertEqual(report('evaluate', 512)['tail_batch'], 256)
        for key, value in [('input_rows', 768), ('actual_physical_batch', 768), ('repeated_input_rows', 0), ('tail_batch', 768)]:
            bad = dict(smoke, **{key: value})
            with self.assertRaises(ValueError):
                op.report_integrity(bad, 'batch_smoke', 1024)

    def test_frozen_identity_parity_and_finiteness_fail_closed(self):
        good = report()
        for key, value in [('optimizer_updates', 1), ('selector_forward_batches', 16), ('current_frame_input_bitwise_equal', False), ('current_frame_max_absolute_difference', 1e-12), ('mean_ce', math.nan), ('mean_ce', True)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                op.report_integrity(dict(good, **{key: value}), 'evaluate', 512)
        for component in ('core', 'head', 'binder'):
            bad = copy.deepcopy(good)
            bad['parameter_digests_after'][component]['parameter_sha256'] = 'c'*64
            with self.assertRaises(ValueError):
                op.report_integrity(bad, 'evaluate', 512)
        foreign = copy.deepcopy(good['parameter_digests_before'])
        foreign['head']['parameter_sha256'] = 'c'*64
        with self.assertRaises(ValueError):
            op.report_integrity(good, 'evaluate', 512, foreign)

    def test_admission_reserves_separate_binding_without_launch(self):
        with patch.object(op, 'verify_files'), patch.object(op, 'tracked') as launch, patch.object(op.time, 'monotonic', return_value=100):
            for mode, available in [('audit', 249), ('qualify', 369), ('evaluate', 250), ('batch_smoke', 360)]:
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'window'):
                    op.invoke({'campaign': '/nonexistent-c19-fixture', 'frozen_files': {}}, 'fixture', mode, 32, 100+available)
            launch.assert_not_called()

    def test_shared_clock_is_1200_seconds_and_rejects_reboot_future_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clock = dict(started_monotonic=time.monotonic()-10, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            write(root/'clock.json', clock)
            self.assertEqual(op.campaign_deadline(root), clock['started_monotonic']+1200)
            for change in [{'boot_id': 'wrong'}, {'started_monotonic': time.monotonic()+100}, {'started_monotonic': math.nan}, {'started_monotonic': True}]:
                write(root/'clock.json', dict(clock, **change))
                with self.assertRaises(ValueError):
                    op.campaign_deadline(root)
        with self.assertRaises(ValueError):
            op.remaining(time.monotonic()-1, 120)

    def test_capture_requires_one_bound_complete_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = dict(health=dict(structurally_valid=True, capture_complete=True), raw_application_labels_verified=True, gpu=dict(status='available', provenance_binding='bound'))
            path = root/'trial.bound/summary.json'
            write(path, [row]); op.healthy_capture(root, 'trial')
            for rows in ([], [row, row], [dict(row, gpu=dict(status='available', provenance_binding='unbound'))], [dict(row, raw_application_labels_verified=False)]):
                write(path, rows)
                with self.assertRaises(ValueError):
                    op.healthy_capture(root, 'trial')

    def test_receipt_rejects_model_and_premise_cleanup_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = dict(accepted=True, pid_gone=True, group_gone=True, owned_survivors=[], group_survivors=[], cleanup_error=None)
            write(root/'qualify-b4/report.json', report('qualify', 4))
            write(root/'audit.exit.json', dict(accepted=True, bindings_unchanged=True, manifest_sha256='a'*64))
            write(root/'audit.process.json', {})
            write(root/'invocations/audit.json', dict(mode='audit', physical_batch=1))
            (root/'audit').mkdir()
            for name in ('audit-model', 'primary-premise', 'independent-premise'):
                write(root/'operations'/f'{name}.exit.json', good)
            for name in ('primary-premise', 'independent-premise'):
                write(root/f'{name}.json', {'accepted': True})
            with patch.object(op, 'root_manifest', return_value=('a'*64, {}, [])), patch.object(op, 'digest', return_value='a'*64), patch.object(op, 'verify_files'):
                op.receipt({'campaign': str(root), 'frozen_files': {}, 'source': 's', 'binary_sha256': 'b'}, ['audit'], 32)
                for name in ('audit-model', 'primary-premise'):
                    for change in ({'pid_gone': False}, {'group_survivors': [123]}, {'cleanup_error': 'failure'}):
                        path = root/'operations'/f'{name}.exit.json'
                        write(path, dict(good, **change))
                        with self.subTest(name=name, change=change), self.assertRaisesRegex(ValueError, 'cleanup|premise'):
                            op.receipt({'campaign': str(root), 'frozen_files': {}}, ['audit'], 32)
                        write(path, good)


class SealerTests(unittest.TestCase):
    def contract(self, positive):
        gates = dict(factual_all_actions=positive, factual_minimum_true_margin=True, uniform_control=True)
        decision = 'supported_frozen_native_composition' if positive else 'frozen_native_composition_not_supported'
        identity = dict(source_revision='a'*40, binary_sha256='b'*64, binder_checkpoint_sha256='c'*64)
        report_value = dict(schema='looped-native-binding-analysis-v1', accepted=True, config_sha256='cfg', registration_sha256='reg', gates=gates, decision=decision, **identity)
        review = dict(schema='looped-native-binding-independent-review-v1', accepted=True, config_sha256='cfg', report_sha256='report', gates=copy.deepcopy(gates), decision=decision)
        config = dict(schema='looped-native-binding-analysis-config-v1', streams={'factual': {}, 'uniform': {}}, registration={'sha256': 'reg'})
        receipt = dict(schema='looped-native-binding-integrity-v1', accepted=True, checks={k: True for k in seal.CHECKS}, checkpoint_files=op.CHECKPOINTS, dependency_revision=op.DEPENDENCY, vision_loops=4, binder_loops=4, **identity)
        return report_value, review, config, receipt

    def test_valid_negative_is_completed_evidence_and_bindings_are_exact(self):
        for positive in (True, False):
            args = self.contract(positive)
            seal.result_contract(*args, 'cfg', 'report')
            for bad in ('other-config', ''):
                with self.assertRaises(ValueError):
                    seal.result_contract(*args, bad, 'report')
            args[1]['decision'] = 'invented-promotion'
            with self.assertRaises(ValueError):
                seal.result_contract(*args, 'cfg', 'report')

    def test_integrity_and_scientific_gate_types_cannot_fail_open(self):
        for mutation in ('missing-integrity', 'false-integrity', 'missing-gates', 'float-gate'):
            args = self.contract(True)
            if mutation == 'missing-integrity': args[3]['checks'].pop('cleanup')
            elif mutation == 'false-integrity': args[3]['checks']['cleanup'] = False
            elif mutation == 'missing-gates': args[0]['gates'] = {}
            else: args[0]['gates']['uniform_control'] = 1.0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                seal.result_contract(*args, 'cfg', 'report')
        with self.assertRaises(ValueError):
            seal.merge_bindings({'file': 'a'}, {'file': 'b'})

    def population(self, failure=None):
        rows = [dict(physical_batch=b, confirmed=True) for b in op.BATCHES]
        names = ['audit', 'qualify-b4'] + [f'{kind}-b{b}' for b in op.BATCHES for kind in ('smoke', 'confirm')]
        selected, failed = 1024, []
        if failure:
            rows = rows[:2] + [dict(physical_batch=128, confirmed=False)]
            names = names[:6] + (['smoke-b128'] if failure == 'confirm' else [])
            selected, failed = 64, [f'{failure}-b128']
        names += ['factual', 'uniform']
        execution = dict(accepted=True, names=names, physical_batch=selected, failed_capacity_roots=failed)
        selection = dict(accepted=True, candidates=rows, physical_batch=selected, failed_roots=failed, limit='observed_capacity_failure' if failed else 'configured_panel_ceiling', optimizer_updates=0, accumulation=1, scientific_rows=768, actual_scientific_batch=min(selected, 768))
        return execution, selection

    def test_all_successful_and_failed_capacity_roots_are_accounted(self):
        for failure in (None, 'smoke', 'confirm'):
            args = self.population(failure)
            names, failed = seal.execution_contract(*args)
            self.assertEqual(names, args[0]['names'])
            self.assertEqual(failed, args[0]['failed_capacity_roots'])
            args[0]['names'].pop(2)
            with self.assertRaises(ValueError):
                seal.execution_contract(*args)
        args = self.population()
        args[1]['candidates'].pop(2)
        with self.assertRaises(ValueError):
            seal.execution_contract(*args)

    def test_process_cleanup_collects_exact_ids_and_rejects_survivors(self):
        good = dict(pid=10, supervisor_pid=11, model_pid=12, pgid=10, owned_pids=[13, 14], pid_gone=True, group_gone=True, model_pid_gone=True, owned_survivors=[], group_survivors=[], cleanup_error=None)
        pids, groups = set(), set()
        seal.scan_process_record(good, pids, groups)
        self.assertEqual(pids, {10, 11, 12, 13, 14}); self.assertEqual(groups, {10})
        for change in ({'pid_gone': False}, {'group_gone': False}, {'owned_survivors': [13]}, {'group_survivors': [15]}, {'cleanup_error': 'error'}, {'owned_pids': [True]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                seal.scan_process_record(dict(good, **change), set(), set())

    def test_binding_rejects_changed_bytes_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'artifact'; path.write_text('fixture')
            value = dict(path=str(path), sha256=op.digest(path))
            self.assertEqual(seal.bound(value), path)
            path.write_text('changed')
            with self.assertRaises(ValueError): seal.bound(value)
            link = Path(tmp)/'link'; link.symlink_to(path)
            with self.assertRaises(ValueError): seal.bound(dict(path=str(link), sha256=op.digest(path)))

    def test_mandatory_cleanup_cannot_be_omitted_or_truthy(self):
        good = dict(pid_gone=True, group_gone=True, owned_survivors=[], group_survivors=[], cleanup_error=None)
        seal.cleanup_contract(good)
        for key in good:
            missing = dict(good); missing.pop(key)
            with self.assertRaises(KeyError): seal.cleanup_contract(missing)
        with self.assertRaises(ValueError): seal.cleanup_contract(dict(good, pid_gone=1))


if __name__ == '__main__':
    unittest.main()
