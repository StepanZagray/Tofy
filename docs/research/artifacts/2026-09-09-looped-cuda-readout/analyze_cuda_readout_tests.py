#!/usr/bin/env python3
"""CPU-only synthetic evaluator fixtures. No generator, fit, or new-panel reads."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('c11_analysis', Path(__file__).with_name('analyze_cuda_readout.py'))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
np = a.np


def write_json(path, value):
    path.write_text(json.dumps(value))


def seal(root):
    files = {p.name: {'sha256': a.sha_bytes(p.read_bytes()), 'bytes': p.stat().st_size}
             for p in root.iterdir() if p.name != 'manifest.json'}
    write_json(root / 'manifest.json', {'files': files})
    return a.sha_bytes((root / 'manifest.json').read_bytes())


def tensor_file(path, values, *, mutate=None):
    header, raw, offset = {}, b'', 0
    for key, value in values.items():
        data = np.asarray(value, dtype='<f4').tobytes()
        header[key] = dict(dtype='F32', shape=list(value.shape), data_offsets=[offset, offset + len(data)])
        raw += data; offset += len(data)
    if mutate:
        mutate(header)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + raw)


def projected(count=4):
    return [dict(input_index=i, episode_id=0x46454154555245 + i, partition='fit', correct_action=i % 4,
                 input_sha256=a.sha_bytes(('i' + str(i)).encode()), query_sha256=a.sha_bytes(('q' + str(i)).encode()),
                 label_sha256=a.sha_bytes(struct.pack('<I', i % 4))) for i in range(count)]


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        a.SNAPSHOT.clear()
        self.temp = tempfile.TemporaryDirectory(prefix='c11-analyzer-fixture-')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()
        self.assertFalse(self.root.exists())

    def test_spatial_forward_matches_uniform_attention_analytic_case(self):
        p = {'queries': np.zeros((2, 128), np.float32), 'output.weight': np.zeros((4, 256), np.float32), 'output.bias': np.arange(4, dtype=np.float32)}
        p['output.weight'][0, 0] = 1
        x = np.arange(2 * 64 * 128, dtype=np.float32).reshape(2, 64, 128) / 1024
        got = a.forward(p, x, 'spatial')
        np.testing.assert_array_equal(got['attention'], np.full((2, 2, 64), 1 / 64, np.float32))
        np.testing.assert_allclose(got['pooled'], np.tile(x.mean(axis=1), (1, 2)), atol=1e-6, rtol=0)
        np.testing.assert_array_equal(got['logits'][:, 0], x[:, :, 0].mean(axis=1))

    def test_cls_forward_silu_and_stable_extreme_values(self):
        p = {'hidden.weight': np.zeros((10, 128), np.float32), 'hidden.bias': np.array([-1000, 0, 1, 1000, 0, 0, 0, 0, 0, 0], np.float32),
             'output.weight': np.eye(4, 10, dtype=np.float32), 'output.bias': np.zeros(4, np.float32)}
        got = a.forward(p, np.zeros((2, 128), np.float32), 'cls')
        np.testing.assert_allclose(got['logits'][0], [0, 0, 1 / (1 + np.exp(-1)), 1000], atol=1e-6)
        self.assertIsNone(got['attention'])

    def test_stable_cross_entropy_large_logits(self):
        logits = np.array([[10000, 9999, -10000, 0], [-10000, 10000, 0, 1]], np.float32)
        got = a.cross_entropy(logits, np.array([0, 1]))
        np.testing.assert_allclose(got, [np.log1p(np.exp(-1)), 0], atol=1e-11)

    def test_attention_has_independent_tighter_gate(self):
        reference = np.zeros((1, 4), np.float32)
        changed = reference + np.float32(5e-5)
        a.parity(changed, reference)
        with self.assertRaisesRegex(a.Invalid, 'numerical parity'):
            a.parity(changed, reference, attention=True)

    def test_exact_argmax_even_inside_logit_tolerance(self):
        with self.assertRaisesRegex(a.Invalid, 'argmax'):
            a.parity(np.array([[0, 1e-6]], np.float32), np.array([[1e-6, 0]], np.float32), actions=True)

    def test_nonfinite_and_shape_parity_fail_closed(self):
        for changed in (np.array([[np.nan]], np.float32), np.zeros((2, 1), np.float32)):
            with self.assertRaises(a.Invalid):
                a.parity(changed, np.zeros((1, 1), np.float32))

    def test_localization_uses_actual_attention_and_first_tie(self):
        actual = np.zeros((2, 2, 64), np.float32)
        actual[:, :, 7] = 1
        actual[0, 0, 7] = .5; actual[0, 0, 3] = .5
        hits, summary = a.localization(actual, np.array([[3, 7], [7, 7]]))
        self.assertTrue(hits['joint'].all())
        self.assertEqual(summary['rows_with_argmax_ties'], [1, 0])
        self.assertEqual(summary['minimum_target_mass'], [.5, 1])

    def test_invalid_attention_normalization_rejected(self):
        with self.assertRaisesRegex(a.Invalid, 'probability'):
            a.localization(np.ones((1, 2, 64), np.float32), np.array([[1, 2]]))

    def test_projected_boundary_rejects_privileged_field_and_bool_label(self):
        good = projected()
        np.testing.assert_array_equal(a.projected_rows(good, 4, None), [0, 1, 2, 3])
        for key, value in [('role_indices', [1, 2]), ('correct_action', True)]:
            bad = copy.deepcopy(good); bad[0][key] = value
            with self.assertRaises(a.Invalid):
                a.projected_rows(bad, 4, None)

    def test_projected_duplicate_identity_and_wrong_label_hash(self):
        for key, value in [('query_sha256', projected()[1]['query_sha256']), ('label_sha256', '0' * 64), ('episode_id', True)]:
            bad = projected(); bad[0][key] = value
            with self.assertRaises(a.Invalid):
                a.projected_rows(bad, 4, None)

    def test_exact_report_int_is_not_bool(self):
        for value in (False, 0., '0'):
            with self.assertRaisesRegex(a.Invalid, 'wrong type'):
                a.subset({'optimizer_updates': value}, {'optimizer_updates': 0})

    def test_authority_static_closure_allows_self_pin_without_circularity(self):
        static = self.root / 'analyzer.py'; static.write_bytes(b'frozen')
        authority = self.root / 'qualification-authorization.json'; authority.write_bytes(b'authority')
        files = {str(static): a.sha_bytes(static.read_bytes()), str(authority): a.sha_bytes(authority.read_bytes())}
        with mock.patch.object(a, 'R', self.root):
            a.authority_closure(dict(frozen_files=files), dict(frozen_files={str(static): files[str(static)]}), 'qualification')
            bad = dict(files); bad[str(static)] = '0' * 64
            with self.assertRaisesRegex(a.Invalid, 'static authority'):
                a.authority_closure(dict(frozen_files=bad), dict(frozen_files={str(static): files[str(static)]}), 'qualification')

    def test_future_bindings_unopened_before_qualification_barrier(self):
        future = str(self.root / 'panel-seal.json')  # Deliberately does not exist.
        with mock.patch.object(a, 'R', self.root):
            with self.assertRaisesRegex(a.Invalid, 'dynamic'):
                a.authority_closure(dict(frozen_files={future: '0' * 64}), dict(frozen_files={}), 'qualification')
            a.authority_closure(dict(frozen_files={future: '0' * 64}), dict(frozen_files={}), 'confirmation')
            self.assertFalse(Path(future).exists())
            with self.assertRaisesRegex(a.Invalid, 'dynamic'):
                a.authority_closure(dict(frozen_files={str(self.root / 'unknown.json'): '0' * 64}), dict(frozen_files={}), 'confirmation')

    def test_manifest_nonfinite_size_and_identity_tampering(self):
        (self.root / 'value').write_bytes(b'a')
        pin = seal(self.root)
        a.manifest(self.root, pin)
        (self.root / 'value').write_bytes(b'b')
        with self.assertRaisesRegex(a.Invalid, 'hash differs'):
            a.manifest(self.root, pin)

    def test_manifest_foreign_file_and_symlink_rejected(self):
        (self.root / 'value').write_bytes(b'a')
        pin = seal(self.root)
        (self.root / 'extra').write_bytes(b'b')
        with self.assertRaisesRegex(a.Invalid, 'population'):
            a.manifest(self.root, pin)
        (self.root / 'extra').unlink()
        (self.root / 'extra').symlink_to(self.root / 'value')
        with self.assertRaisesRegex(a.Invalid, 'linked'):
            a.manifest(self.root, pin)

    def test_binary_shape_and_nonfinite(self):
        p = self.root / 'a.f32'; p.write_bytes(np.array([1, np.inf], dtype='<f4').tobytes())
        with self.assertRaisesRegex(a.Invalid, 'nonfinite'):
            a.binary_array(p, '<f4', (2,))
        with self.assertRaisesRegex(a.Invalid, 'byte count'):
            a.binary_array(p, '<f4', (1,))

    def test_safetensors_offset_dtype_shape_nonfinite(self):
        p = self.root / 'a.safetensors'
        values = {'x': np.zeros((2,), np.float32), 'y': np.ones((2,), np.float32)}
        tensor_file(p, values); self.assertEqual(set(a.safe_tensors(p)), {'x', 'y'})
        mutations = [lambda h: h['x'].update(dtype='F64'), lambda h: h['x'].update(shape=[True, 2]),
                     lambda h: h['y'].update(data_offsets=[0, 8]), lambda h: h['x'].update(data_offsets=[0, 12])]
        for change in mutations:
            a.SNAPSHOT.clear(); tensor_file(p, values, mutate=change)
            with self.assertRaises(a.Invalid):
                a.safe_tensors(p)
        a.SNAPSHOT.clear(); tensor_file(p, {'x': np.array([np.nan], np.float32)})
        with self.assertRaisesRegex(a.Invalid, 'nonfinite'):
            a.safe_tensors(p)

    def import_fixture(self):
        original = self.root / 'selector-initial-true.npz'
        values = {'queries': np.zeros((2, 128), np.float64), 'output_weight': np.zeros((4, 256), np.float64), 'output_bias': np.arange(4, dtype=np.float64)}
        np.savez(original, **values)
        imported = self.root / 'import'; imported.mkdir()
        raw, tensors, offset = b'', [], 0
        for name, shape in a.SHAPES['spatial'].items():
            block = values[name.replace('.', '_')].astype('<f4').tobytes()
            tensors.append(dict(name=name, dtype='F32LE', shape=list(shape), byte_offset=offset, byte_length=len(block), sha256=a.sha_bytes(block)))
            raw += block; offset += len(block)
        (imported / 'parameters.f32').write_bytes(raw)
        m = dict(schema='looped-imported-readout-source-v1', arm='c10_true', head_kind='spatial', core_checkpoint='initial',
                 core_checkpoint_sha256=a.CORES['initial'], source_kind='c10_role_ridge', parent_manifests=a.PARENTS,
                 producer=dict(source_revision=a.PRODUCERS['c10'][0], implementation_sha256=a.PRODUCERS['c10'][1]),
                 recipe=dict(name='c10_role_ridge_c8_affine_v1', original_optimizer_updates=0, privileged_role_supervision=True, cast='f64_to_f32_once_no_rescale'),
                 original_artifact=dict(path=str(original), sha256=a.sha_bytes(original.read_bytes())),
                 artifact=dict(file='parameters.f32', sha256=a.sha_bytes(raw), bytes=len(raw)), tensors=tensors)
        write_json(imported / 'manifest.json', m)
        return imported, m

    def test_import_original_cast_canonical_offsets_and_producer_identity(self):
        root, m = self.import_fixture()
        def load(document):
            a.SNAPSHOT.clear(); write_json(root / 'manifest.json', document)
            return a.imported_parameters(dict(root=str(root), manifest_sha256=a.sha_bytes((root / 'manifest.json').read_bytes())), 'initial', 'c10_true')
        with mock.patch.object(a, 'C10', self.root):
            self.assertEqual(load(m)['parameters']['queries'].shape, (2, 128))
            for field, value in [('dtype', 'F64LE'), ('byte_offset', True), ('byte_offset', 4), ('shape', [128, 2]), ('sha256', '0' * 64)]:
                bad = copy.deepcopy(m); bad['tensors'][0][field] = value
                with self.assertRaises(a.Invalid):
                    load(bad)
            for key, value in [('arm', 'c10_null'), ('core_checkpoint_sha256', a.CORES['final'])]:
                bad = copy.deepcopy(m); bad[key] = value
                with self.assertRaises(a.Invalid):
                    load(bad)
            bad = copy.deepcopy(m); bad['producer']['source_revision'] = '0' * 40
            with self.assertRaises(a.Invalid):
                load(bad)

    def test_raw_prediction_offsets_labels_attention_boundary(self):
        records, count = projected(), 4
        labels = np.arange(4)
        logits = np.eye(4, dtype=np.float32)
        (self.root / 'logits.f32').write_bytes(logits.tobytes())
        (self.root / 'labels.u32').write_bytes(np.column_stack((labels, labels)).astype('<u4').tobytes())
        (self.root / 'episode-ids.u64').write_bytes(np.array([r['episode_id'] for r in records], dtype='<u8').tobytes())
        (self.root / 'pooled.f32').write_bytes(np.zeros((count, 10), np.float32).tobytes())
        predictions = [dict(identity=r, true_action=i, fitted_action=i, prediction=i, logits=a.descriptor('logits.f32', (4,), i),
                            labels=dict(file='labels.u32', dtype='U32LE', byte_offset=i*8, byte_length=8, order=['true', 'fitted']),
                            episode_id=dict(file='episode-ids.u64', dtype='U64LE', byte_offset=i*8, byte_length=8),
                            pooled=a.descriptor('pooled.f32', (10,), i), attention=None) for i, r in enumerate(records)]
        def write_rows(values):
            a.SNAPSHOT.clear(); (self.root / 'predictions.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in values))
        write_rows(predictions)
        a.head_outputs(self.root, dict(rows=records, labels=labels), 'cls')
        for field, value in [('dtype', 'F64LE'), ('byte_offset', 4), ('shape', [2, 2])]:
            bad = copy.deepcopy(predictions); bad[0]['logits'][field] = value; write_rows(bad)
            with self.assertRaises(a.Invalid):
                a.head_outputs(self.root, dict(rows=records, labels=labels), 'cls')
        bad = copy.deepcopy(predictions); bad[0]['true_action'] = False; write_rows(bad)
        with self.assertRaises(a.Invalid):
            a.head_outputs(self.root, dict(rows=records, labels=labels), 'cls')

    def test_constant_reselected_each_draw(self):
        labels = np.array([0, 0, 1, 1])
        indices = np.array([[0, 0, 0, 1], [2, 2, 3, 3], [0, 1, 2, 3]])
        np.testing.assert_array_equal(a.best_constant_draws(labels, indices), [1, 1, .5])
        self.assertNotEqual(a.best_constant_draws(labels, indices)[1], (labels[indices[1]] == 0).mean())

    def test_registered_bootstrap_seeds_and_stratification(self):
        expected = np.random.Generator(np.random.PCG64(1921)).integers(0, 256, size=(2, 256))
        np.testing.assert_array_equal(a.bootstrap_indices(1, 2), expected)
        pooled = a.bootstrap_indices(None, 2)
        self.assertEqual(pooled.shape, (2, 768))
        for panel in range(3):
            block = pooled[:, panel*256:(panel+1)*256]
            self.assertTrue(((block >= panel*256) & (block < (panel+1)*256)).all())

    def test_wilson_not_degenerate_on_all_correct(self):
        lo, hi = a.wilson(256, 256)
        self.assertLess(lo, 1); self.assertAlmostEqual(hi, 1)

    def test_installed_blas_is_observably_single_threaded(self):
        observed = a.observed_blas()
        self.assertTrue(observed)
        self.assertTrue(all(item['num_threads'] == 1 and a.valid_sha(item['sha256']) for item in observed))

    def decision_fixture(self):
        panels = {}
        for panel in range(3):
            e, r, pairs = {}, {}, {}
            for core in a.CORES:
                for arm in (*a.ARMS, 'native', 'oracle_role'):
                    e[core + '/' + arm] = dict(accuracy=1. if arm in ('c10_true', 'oracle_role') else .25,
                        advantage_over_constant=dict(bootstrap95=[.6, .8]))
                r[core + '/c10_true'] = dict(joint=dict(accuracy=1.))
                r[core + '/c10_null'] = dict(joint=dict(accuracy=.02))
                pairs[core + '/c10_true minus ' + core + '/c9_spatial'] = dict(accuracy=dict(bootstrap95=[.6, .8]))
            panels[str(panel)] = dict(geometry_correct=256, statistics=dict(endpoints=e, roles=r, paired=pairs))
        return panels

    def test_success_requires_both_cores_all_three_panels(self):
        panels = self.decision_fixture()
        self.assertEqual(a.registered_decision(panels)['decision'], 'confirmed_on_registered_synthetic_populations')
        panels['2']['statistics']['endpoints']['final/c10_true']['accuracy'] = .89
        result = a.registered_decision(panels)
        self.assertEqual(result['decision'], 'not_supported')
        self.assertFalse(result['components']['2/final']['thresholds']['true_action'])
        self.assertTrue(result['pooled_cannot_rescue'])

    def test_each_null_and_positive_control_overrides_success(self):
        changes = [('endpoints', 'initial/oracle_role', 'accuracy', .89), ('endpoints', 'final/c10_null', 'accuracy', .51),
                   ('endpoints', 'initial/c9_null', 'accuracy', .51)]
        for group, key, metric, value in changes:
            panels = self.decision_fixture(); panels['1']['statistics'][group][key][metric] = value
            self.assertEqual(a.registered_decision(panels)['decision'], 'inconclusive_failed_control')
        panels = self.decision_fixture(); panels['0']['statistics']['roles']['initial/c10_null']['joint']['accuracy'] = .101
        self.assertEqual(a.registered_decision(panels)['decision'], 'inconclusive_failed_control')

    def test_strict_advantage_threshold_and_missing_panel(self):
        panels = self.decision_fixture()
        panels['0']['statistics']['endpoints']['initial/c10_true']['advantage_over_constant']['bootstrap95'][0] = .25
        self.assertEqual(a.registered_decision(panels)['decision'], 'not_supported')
        del panels['2']
        with self.assertRaises(a.Invalid):
            a.registered_decision(panels)

    def test_paired_statistics_shared_draws_and_all_contrasts(self):
        labels = np.arange(256) % 4
        endpoints = {core + '/' + arm: dict(correct=np.ones(256, bool) if arm == 'c10_true' else labels == 0,
                     ce=np.zeros(256) if arm == 'c10_true' else np.ones(256))
                     for core in a.CORES for arm in (*a.ARMS, 'native', 'oracle_role')}
        roles = {core + '/' + arm: {role: np.full(256, arm == 'c10_true') for role in ('agent', 'goal', 'joint')}
                 for core in a.CORES for arm in ('c10_true', 'c10_null')}
        result = a.comparisons(endpoints, labels, roles, 0)
        self.assertEqual(len(result['paired']), 49)
        self.assertEqual(result['paired']['final/c10_true minus initial/c10_true']['accuracy']['bootstrap95'], [0, 0])
        draws = a.bootstrap_indices(0)
        expected = a.interval(1 - a.best_constant_draws(labels, draws))
        self.assertEqual(result['endpoints']['initial/c10_true']['advantage_over_constant']['bootstrap95'], expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
