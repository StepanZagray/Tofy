#!/usr/bin/env python3
import importlib.util,unittest
from pathlib import Path
p=Path(__file__).resolve().parent/'driver.py';spec=importlib.util.spec_from_file_location('c12_driver',p);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
class Admission(unittest.TestCase):
 def report(self,prediction):return dict(optimizer_updates=5,updates_elapsed_seconds=prediction*5/1150,checkpoint_seconds=.1)
 def test_reserve_boundary_and_53_minutes_rejected(self):
  audit=dict(training_generation_and_audit_seconds=20)
  self.assertTrue(driver.admission(self.report(3130),audit)['accepted'])
  self.assertFalse(driver.admission(self.report(3180),audit)['accepted'])
 def test_generator_overhead_cannot_be_ignored(self):
  self.assertFalse(driver.admission(self.report(1000),dict(training_generation_and_audit_seconds=151))['accepted'])
 def test_nonfinite_negative_and_wrong_confirmation_rejected(self):
  for value in (float('nan'),float('inf'),-1):
   with self.assertRaises(ValueError):driver.admission(self.report(value),dict(training_generation_and_audit_seconds=1))
  report=self.report(1000);report['optimizer_updates']=2
  with self.assertRaises(ValueError):driver.admission(report,dict(training_generation_and_audit_seconds=1))
 def test_outer_cleanup_includes_group_survivors(self):
  state=dict(returncode=0,error=None,cleanup_error=None,pid_gone=True,group_gone=True,owned_survivors=[],group_survivors=[])
  self.assertTrue(driver.clean_operation(state));state['group_survivors']=[42];self.assertFalse(driver.clean_operation(state))
if __name__=='__main__':unittest.main()
