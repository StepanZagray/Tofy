import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import importlib.util
_spec = importlib.util.spec_from_file_location('c15_operator', Path(__file__).with_name('campaign_operator.py'))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
qualification = _module.qualification


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        row = {'index': 0, 'logits': [0., 1., 0., 0.],
               'attention': [[1.] + [0.] * 63, [0., 1.] + [0.] * 62],
               'pooled': [0.] * 256, 'current': [[0.] * 128 for _ in range(64)], 'cls': [0.] * 128}
        self.rows = [dict(copy.deepcopy(row), index=i) for i in range(4)]
        self.old = self.write('old', self.rows)

    def write(self, name, rows):
        path = self.root / name
        path.write_text(''.join(json.dumps(x) + '\n' for x in rows))
        return path

    def test_exact_and_allowed_roundoff(self):
        self.assertTrue(qualification(self.old, self.old)['accepted'])
        rows = copy.deepcopy(self.rows)
        rows[1]['current'][5][9] += 9e-6
        result = qualification(self.write('new', rows), self.old)
        self.assertEqual(result['maximum_absolute_errors']['current'], 9e-6)

    def test_numeric_and_discrete_drift_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['pooled'][0] = 2e-5
        with self.assertRaises(ValueError):
            qualification(self.write('numeric', rows), self.old)
        rows = copy.deepcopy(self.rows)
        rows[0]['logits'] = [0., 0., 1., 0.]
        with self.assertRaises(ValueError):
            qualification(self.write('winner', rows), self.old)

    def test_shape_nonfinite_and_identity_rejected(self):
        for name, change in [('shape', lambda r: r[0]['cls'].pop()),
                             ('nan', lambda r: r[0]['cls'].__setitem__(0, float('nan'))),
                             ('order', lambda r: r[0].__setitem__('index', 2))]:
            rows = copy.deepcopy(self.rows)
            change(rows)
            with self.assertRaises(ValueError):
                qualification(self.write(name, rows), self.old)

    def test_bound_gpu_profile_required(self):
        directory = self.root / 'example.bound'
        directory.mkdir()
        row = {'health': {'structurally_valid': True, 'capture_complete': True},
               'raw_application_labels_verified': True,
               'gpu': {'status': 'available', 'provenance_binding': 'bound'}}
        path = directory / 'summary.json'
        path.write_text(json.dumps([row]))
        _module.healthy_capture(self.root, 'example')
        for key, value in [('status', 'unavailable'), ('provenance_binding', 'unbound')]:
            changed = copy.deepcopy(row)
            changed['gpu'][key] = value
            path.write_text(json.dumps([changed]))
            with self.assertRaises(ValueError):
                _module.healthy_capture(self.root, 'example')

    def test_absolute_deadline_clamps_stage_budget(self):
        with patch.object(_module.time, 'monotonic', return_value=1190):
            self.assertEqual(_module.remaining(1200, 600), 10)
            self.assertEqual(_module.remaining(1800, 240), 240)
            with self.assertRaises(ValueError):
                _module.remaining(1190, 600)


if __name__ == '__main__':
    unittest.main()
