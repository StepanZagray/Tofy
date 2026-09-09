"""Counterexample tests for C16 lifecycle-only decisions; no model execution."""
import importlib.util
import math
from pathlib import Path
import tempfile
import time
import unittest

spec = importlib.util.spec_from_file_location('c16_operator_under_test', Path(__file__).with_name('campaign_operator.py'))
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)

class OperatorTests(unittest.TestCase):
    def test_admission_boundary_and_overhead(self):
        report = dict(optimizer_updates=5, updates_elapsed_seconds=2, checkpoint_seconds=1)
        self.assertAlmostEqual(op.admission(report)['reserved_training_seconds'],530)
        self.assertTrue(op.admission(report)['accepted'])
        self.assertFalse(op.admission(dict(report,checkpoint_seconds=80))['accepted'])
        with self.assertRaises(ValueError): op.admission(dict(report,optimizer_updates=2))
        for value in (-1,math.nan,math.inf,True):
            with self.assertRaises(ValueError): op.admission(dict(report,updates_elapsed_seconds=value))

    def test_remaining_hard_deadline(self):
        self.assertLessEqual(op.remaining(time.monotonic()+2,120),2)
        with self.assertRaises(ValueError): op.remaining(time.monotonic()-1,120)

    def test_shared_monotonic_clock_rejects_reboot_and_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            clock=dict(started_monotonic=time.monotonic()-10,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            import json
            path=root/'clock.json'
            path.write_text(json.dumps(clock))
            self.assertAlmostEqual(op.campaign_deadline(root),clock['started_monotonic']+1800)
            path.write_text(json.dumps(dict(clock,boot_id='different-boot')))
            with self.assertRaises(ValueError): op.campaign_deadline(root)
            path.write_text(json.dumps(dict(clock,started_monotonic=time.monotonic()+100)))
            with self.assertRaises(ValueError): op.campaign_deadline(root)

    def test_capture_rejects_missing_gpu_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            bundle=root/'trial.bound'
            bundle.mkdir()
            row=dict(health=dict(structurally_valid=True,capture_complete=True),raw_application_labels_verified=True,gpu=dict(status='available',provenance_binding='bound'))
            path=bundle/'summary.json'
            import json
            path.write_text(json.dumps([row]))
            op.healthy_capture(root,'trial',1)
            for key,bad in [('status','unavailable'),('provenance_binding','unbound')]:
                original=row['gpu'][key]
                row['gpu'][key]=bad
                path.write_text(json.dumps([row]))
                with self.assertRaises(ValueError): op.healthy_capture(root,'trial',1)
                row['gpu'][key]=original
            path.write_text(json.dumps([row]))
            with self.assertRaises(ValueError): op.healthy_capture(root,'trial',3)

    def test_hash_handles_large_binary_without_json_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'binary'
            with path.open('wb') as out: out.truncate(129*1024*1024)
            self.assertEqual(len(op.digest(path)),64)
            link=Path(tmp)/'link'
            link.symlink_to(path)
            with self.assertRaises(ValueError): op.digest(link)

    def test_import_restores_search_path(self):
        import sys
        self.assertNotIn(str(op.R12),sys.path)

if __name__=='__main__': unittest.main()
