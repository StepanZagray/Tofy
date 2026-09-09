#!/usr/bin/env python3
import importlib.util,json,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('c12_operator',HERE/'supervise.py');op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
class Guards(unittest.TestCase):
 def state(self,**extra):return dict(failure="RuntimeError('model/profiler exited 1')",group_gone=True,owned_survivors=[],cleanup_error=None,pid_gone=True,model_pid_gone=True,group_survivors=[],bindings_unchanged=True,postcheck_kind='backend_failure',**extra)
 def test_capacity_allows_only_diagnosed_oom_or_reserve(self):
  for text in ('CUDA_ERROR_OUT_OF_MEMORY','CUDA out of memory','registered 512 MiB GPU reserve breached'):
   state=self.state()
   if 'reserve' in text:state['failure']="RuntimeError('registered 512 MiB GPU reserve breached')"
   self.assertTrue(op.capacity_only(state,text))
  for text in ('nonfinite CUDA_ERROR_OUT_OF_MEMORY','AC is offline CUDA out of memory','temperature CUDA out of memory','illegal memory access','device-side assert','binding mismatch','generic exit1'):
   self.assertFalse(op.capacity_only(self.state(),text))
 def test_capacity_never_hides_unreaped_process(self):
  state=self.state();state['owned_survivors']=[42]
  self.assertFalse(op.capacity_only(state,'CUDA out of memory'))
  state=self.state();state['group_gone']=False
  self.assertFalse(op.capacity_only(state,'CUDA out of memory'))
 def test_oom_string_cannot_hide_timeout_cancellation_or_surviving_descendant(self):
  for failure in ("TimeoutError('registered profiler-finalization deadline exceeded')", "KeyboardInterrupt('operator requested termination')", 'profiler exited with surviving owned descendants', "RuntimeError('launch model PID is outside the owned child tree')"):
   state=self.state();state['failure']=failure
   self.assertFalse(op.capacity_only(state,'CUDA_ERROR_OUT_OF_MEMORY'))
 def test_capacity_rejects_postcheck_or_binding_failure(self):
  for key,value in [('pid_gone',False),('model_pid_gone',False),('group_survivors',[42]),('bindings_unchanged',False),('postcheck_kind','integrity_failure')]:
   state=self.state();state[key]=value
   self.assertFalse(op.capacity_only(state,'CUDA out of memory'))
 def test_cleanup_requires_every_barrier(self):
  good=dict(failure=None,returncode=0,pid_gone=True,model_pid_gone=True,group_gone=True,owned_survivors=[],group_survivors=[],cleanup_error=None)
  op.cleanup_valid(good)
  for key,bad in [('failure','x'),('returncode',1),('pid_gone',False),('model_pid_gone',False),('group_gone',False),('owned_survivors',[42]),('group_survivors',[42]),('cleanup_error','x')]:
   state=dict(good);state[key]=bad
   with self.assertRaises(ValueError):op.cleanup_valid(state)
 def test_root_manifest_accepts_empty_cpu_host_and_rejects_changed_file(self):
  with tempfile.TemporaryDirectory() as temp:
   base=Path(temp);root=base/'audit';root.mkdir();host=base/'audit.host.json';host.write_text('[]')
   profiles=dict(root=str(base/'audit.profiles'),files={},host_trace=dict(path=str(host),sha256=op.digest(host)))
   (root/'profiles.json').write_text(json.dumps(profiles))
   (root/'report.json').write_text('{}')
   files={p.name:op.digest(p) for p in root.iterdir()}
   manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(schema='looped-grounded-policy-artifacts-v1',files=files)))
   root.with_suffix('.manifest.sha256').write_text(op.digest(manifest)+'\n')
   self.assertEqual(op.root_manifest(root)[2],[])
   (root/'report.json').write_text('{"changed":true}')
   with self.assertRaises(ValueError):op.root_manifest(root)
 def test_profile_paths_and_unlisted_artifacts_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   base=Path(temp);root=base/'audit';root.mkdir();host=base/'audit.host.json';host.write_text('[]')
   profiles=dict(root=str(base/'audit.profiles'),files={},host_trace=dict(path=str(host),sha256=op.digest(host)))
   def seal():
    (root/'profiles.json').write_text(json.dumps(profiles))
    files={'profiles.json':op.digest(root/'profiles.json')}
    manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(schema='looped-grounded-policy-artifacts-v1',files=files)))
    root.with_suffix('.manifest.sha256').write_text(op.digest(manifest)+'\n')
   seal();op.root_manifest(root)
   profiles['root']=str(base/'external');seal()
   with self.assertRaises(ValueError):op.root_manifest(root)
   profiles['root']=str(base/'audit.profiles');Path(profiles['root']).mkdir();(Path(profiles['root'])/'unlisted').write_text('x');seal()
   with self.assertRaises(ValueError):op.root_manifest(root)
 def test_symlink_binding_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'a').write_text('x');(root/'b').symlink_to(root/'a')
   with self.assertRaises(ValueError):op.digest(root/'b')
if __name__=='__main__':unittest.main()
