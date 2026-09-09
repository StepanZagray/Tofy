"""Counterexample tests for C18 lifecycle-only decisions; no model execution."""
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
    def trial(self,count,ordinary=.06,selected=4.5,setup=.9,wrapper=.3):
        return dict(report=dict(optimizer_updates=count,updates_elapsed_seconds=selected+(count-1)*ordinary,elapsed_seconds=setup+selected+(count-1)*ordinary,checkpoint_seconds=.01),records=[dict(update=i,elapsed_seconds=setup+selected+(i-1)*ordinary) for i in range(1,count+1)],supervisor=dict(model_phase_seconds=wrapper+setup+selected+(count-1)*ordinary))

    def test_admission_counts_captures_once_and_uses_both_trials(self):
        trials=[self.trial(2,selected=4.8),self.trial(5)]
        result=op.admission(trials,1150)
        self.assertTrue(result['accepted'])
        self.assertAlmostEqual(result['reserved_training_seconds'],1.25*(1147*.06+3*4.8+.9+.3)+.01)
        self.assertEqual(result['ordinary_updates']+result['selected_updates'],1150)
        self.assertFalse(op.admission([self.trial(2,ordinary=1),self.trial(5,ordinary=1)],1150)['accepted'])
        self.assertFalse(op.admission([self.trial(2,selected=200),self.trial(5,selected=200)],1150)['accepted'])
        with self.assertRaises(ValueError): op.admission(trials[::-1],1150)

    def test_larger_batch_forecast_keeps_only_reachable_captures(self):
        for batch in op.BATCHES:
            count=math.ceil(588800/batch)
            captures=op.capture_updates(count)
            self.assertEqual(len(set(captures)),3)
            self.assertEqual(sorted(captures),captures)
            self.assertTrue(all(1<=x<=count for x in captures))
            forecast=op.admission([self.trial(2),self.trial(5)],count)
            self.assertEqual(forecast['ordinary_updates']+forecast['selected_updates'],count)
        for count in (0,1,2,3):
            with self.assertRaises(ValueError):op.capture_updates(count)
        for count in (False,0,1,2):
            with self.assertRaises(ValueError):op.admission([self.trial(2),self.trial(5)],count)

    def test_admission_rejects_invalid_timing_boundaries(self):
        for value in (-1,math.nan,math.inf,True):
            trial=self.trial(5)
            trial['report']['updates_elapsed_seconds']=value
            with self.assertRaises(ValueError): op.timing_components(**trial)
        for mutation in ('nonmonotonic','order','setup','wrapper','truncated','offset','checkpoint'):
            t=self.trial(5)
            if mutation=='nonmonotonic': t['records'][2]['elapsed_seconds']=t['records'][1]['elapsed_seconds']
            elif mutation=='order': t['records'][2]['update']=9
            elif mutation=='setup': t['report']['elapsed_seconds']=0
            elif mutation=='wrapper': t['supervisor']['model_phase_seconds']=0
            elif mutation=='offset':
                for row in t['records']: row['elapsed_seconds']+=100
            elif mutation=='checkpoint': t['report']['checkpoint_seconds']=100
            else: t['records'].pop()
            with self.assertRaises(ValueError): op.timing_components(**t)

    def test_smoke_rejects_zero_input_gradient_and_partial_candidate(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'smoke').mkdir()
            report=dict(physical_batch=1024,effective_batch=1024,accumulation=1,input_rows=2048,optimizer_updates=2)
            metrics=dict(rows=1024,requested_physical_batch=1024,physical_batch=1024,microbatches=1,tail_batch=1024,mean_ce=1.0,clip_scale=1.0,pre_clip_norm=1.0,input_gradient_norm=1.0,shared_core_gradient_norm=1.0,body_gradient_norm=1.0,head_gradient_norm=1.0)
            path=root/'smoke/updates.jsonl'
            def write():path.write_text(''.join(json.dumps(dict(update=i,metrics=metrics,elapsed_seconds=float(i)))+'\n' for i in (1,2)))
            (root/'smoke.exit.json').write_text('{}');write();op.timing_trial(root,'smoke',report)
            for key,value in [('input_gradient_norm',0),('shared_core_gradient_norm',float('nan')),('head_gradient_norm',True),('clip_scale',2),('rows',512),('physical_batch',512),('requested_physical_batch',512),('tail_batch',512),('microbatches',2)]:
                old=metrics[key];metrics[key]=value;write()
                with self.assertRaises(ValueError):op.timing_trial(root,'smoke',report)
                metrics[key]=old

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
