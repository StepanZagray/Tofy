#!/usr/bin/env python3
"""Temporary CPU children only; no witness, experiment data, or real repositories."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('cpu_supervisor',Path(__file__).with_name('supervise_cpu.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='r10-cpu-guard-')
        self.root=Path(self.temp.name)
        self.handlers={sig:signal.signal(sig,m.stop_signal) for sig in (signal.SIGTERM,signal.SIGINT)}

    def tearDown(self):
        for sig,handler in self.handlers.items():signal.signal(sig,handler)
        self.temp.cleanup()
        self.assertFalse(self.root.exists())

    def run_child(self,source,record=None,timeout=1):
        observed=[]
        def recorded(pid,start):
            observed.append((pid,start))
            if record is not None:record(pid,start)
        with (self.root/'child.log').open('w') as log:
            state=m.supervise([sys.executable,'-B','-c',source],self.root,dict(os.environ),log,recorded,
                              timeout=timeout,grace=.1)
        self.assertTrue(state['pid_gone'],state)
        self.assertTrue(state['group_gone'],state)
        self.assertEqual(state['owned_survivors'],[],state)
        self.assertIsNone(state['cleanup_error'],state)
        for pid in state['owned_pids']:
            self.assertFalse(Path('/proc',str(pid)).exists(),f'fixture PID remains: {pid}')
        self.assertTrue(observed)
        self.assertIsInstance(observed[0][1],int)
        return state

    def test_normal_completion(self):
        state=self.run_child('print("complete")')
        self.assertEqual(state['returncode'],0)
        self.assertIsNone(state['error'])

    def test_nonzero_exit(self):
        state=self.run_child('raise SystemExit(17)')
        self.assertEqual(state['returncode'],17)
        self.assertIn('exited 17',state['error'])

    def test_timeout_reaps_term_ignoring_child(self):
        state=self.run_child('import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)',timeout=.15)
        self.assertIn('deadline',state['error'])
        self.assertEqual(state['returncode'],-signal.SIGKILL)
        self.assertLess(state['elapsed_seconds'],1)

    def test_escaped_descendant_is_reaped_and_cannot_pass(self):
        state=self.run_child('import subprocess,sys;subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],start_new_session=True)')
        self.assertIn('active owned descendants',state['error'])
        self.assertGreaterEqual(len(state['owned_pids']),2)

    def test_same_group_descendant_is_reaped_and_cannot_pass(self):
        state=self.run_child('import subprocess,sys;subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])')
        self.assertIn('active owned descendants',state['error'])
        self.assertGreaterEqual(len(state['owned_pids']),2)

    def test_cancellation_during_record_is_deferred_until_child_is_owned(self):
        state=self.run_child('import time;time.sleep(60)',record=lambda pid,start:os.kill(os.getpid(),signal.SIGTERM))
        self.assertIn('cancellation during spawn',state['error'])

    def test_cancellation_during_wait(self):
        state=self.run_child('import os,signal,time;time.sleep(.03);os.kill(os.getppid(),signal.SIGINT);time.sleep(60)')
        self.assertIn('received signal',state['error'])

    def test_second_cancellation_cannot_interrupt_cleanup(self):
        state=self.run_child('import os,signal,time;parent=os.getppid();signal.signal(signal.SIGTERM,lambda *args:os.kill(parent,signal.SIGTERM));time.sleep(60)',timeout=.15)
        self.assertIn('deadline',state['error'])
        self.assertEqual(state['returncode'],-signal.SIGKILL)

    def test_record_failure_does_not_leak_process(self):
        def fail(pid,start):raise OSError('fixture process record failure')
        state=self.run_child('import time;time.sleep(60)',record=fail)
        self.assertIn('fixture process record failure',state['error'])

    def test_missing_postrun_input_still_writes_failed_exit(self):
        cwd=self.root/'repo';cwd.mkdir()
        frozen=self.root/'frozen.txt';frozen.write_text('frozen fixture')
        output=self.root/'result'
        config=self.root/'config.json'
        source='1'*40
        config.write_text(json.dumps(dict(cwd=str(cwd),source=source,output=str(output),timeout_seconds=90,
            frozen_files={str(frozen):m.sha(frozen)},
            command=[sys.executable,'-B','-c','from pathlib import Path;Path('+repr(str(frozen))+').unlink()'])))
        def git(command,**kwargs):
            if command[-2:]==['rev-parse','HEAD']:return source+'\n'
            if command[-2:]==['status','--porcelain']:return ''
            raise AssertionError(command)
        with patch.object(sys,'argv',['supervise_cpu.py','--config',str(config),'--sha256',m.sha(config)]), \
                patch.object(m.subprocess,'check_output',side_effect=git),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:m.main()
            self.assertEqual(error.exception.code,1)
        exit=json.loads(output.with_suffix('.exit.json').read_text())
        self.assertFalse(exit['accepted'])
        self.assertFalse(exit['inputs_unchanged'])
        self.assertIn('FileNotFoundError',exit['provenance_error'])
        for pid in exit['owned_pids']:self.assertFalse(Path('/proc',str(pid)).exists())


if __name__=='__main__':unittest.main(verbosity=2)
