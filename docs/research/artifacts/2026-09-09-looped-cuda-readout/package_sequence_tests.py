#!/usr/bin/env python3
"""Synthetic package/sequence guards. Never import campaign helpers or generate episodes."""
import contextlib
import copy
import datetime
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import shutil
import types
import unittest
from unittest import mock

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream)


def forbidden(*args, **kwargs):
    raise AssertionError('external campaign access or child launch attempted')


def load_source(name, filename):
    common = types.ModuleType('campaign_io')
    common.__dict__.update(json=json, R=HERE, R10=Path('/forbidden-r10'), REPO=Path('/forbidden-repo'),
        C8=Path('/forbidden-c8'), C9=Path('/forbidden-c9'),
        CORES={'initial': 'a'*64, 'final': 'b'*64}, ARMS=['c10_true','c10_null','c9_spatial','c9_cls','c9_null'],
        FIELDS=['input_index','episode_id','partition','input_sha256','query_sha256','label_sha256','correct_action'],
        require=require, digest=digest, read=lambda p: json.loads(Path(p).read_text()), save=save,
        campaign=forbidden, verify_files=forbidden, verified_manifest=forbidden,
        now=lambda: datetime.datetime.now().astimezone().isoformat())
    operator = types.ModuleType('supervise_stage')
    operator.qualified = operator.panel_binding = operator.registered = forbidden
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {'campaign_io':common, 'supervise_stage':operator}):
        spec.loader.exec_module(module)
    return module


PACKAGE = load_source('c11_package_under_test', 'package_panels.py')
SEQUENCE = load_source('c11_sequence_under_test', 'sequence.py')


def synthetic_rows(panel):
    """Hand-built arrays; no task/model/data generator and no real campaign reads."""
    values=[]
    for index in range(256):
        cells=[1]*64
        for y in range(1,7):
            for x in range(1,7):
                cells[y*8+x]=0
        # Distinct nuisance patterns across rows and panels; no random sampling.
        bits=index+256*panel
        for bit, position in enumerate([9,10,11,12,13,14,17,18,20,21]):
            cells[position]=(bits>>bit)&1
        action=index%4
        cells[27]=2
        cells[[19,35,26,28][action]]=3
        query=PACKAGE.np.repeat(PACKAGE.np.asarray(cells,dtype='<u4'),64).tobytes()
        values.append(dict(schema='looped-known-features-v1',partition='confirmation_eval',
            confirmation_panel=panel,layout_index=index,input_index=index,
            data_seed=20260917+panel,episode_seed=20260917+panel,
            episode_id=0x43554441434f4e46+panel*0x10000+index,
            evaluation_loops=4,condition='factual',permutation_id=0,
            min_distance=1,max_distance=1,oracle_distance=1,support_cleared=False,
            inferred_controls=[0,1,2,3],split='KnownMapping',visible_cells=cells,
            correct_action=action,query_sha256=hashlib.sha256(query).hexdigest(),
            label_sha256=hashlib.sha256(action.to_bytes(4,'little')).hexdigest(),
            input_sha256=hashlib.sha256(f'synthetic-input-{panel}-{index}'.encode()).hexdigest(),
            targets_sha256='c'*64))
    return values


class PackageSequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='.package-sequence-fixture-',dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.c=self.root/'campaign'; self.c.mkdir()
        self.r=self.root/'research'; self.r.mkdir()

    def test_typed_identity_labels_controls_and_collisions(self):
        values=synthetic_rows(0)
        seen,labels=PACKAGE.validate_rows(values,0)
        self.assertEqual([len(group) for group in seen],[256]*3)
        self.assertEqual(labels,[64]*4)
        for key,bad in [('confirmation_panel',1),('layout_index',999),('data_seed',20260917.0),
            ('evaluation_loops',4.0),('permutation_id',False),('min_distance',True),
            ('inferred_controls',[1,0,2,3]),('inferred_controls',[False,1,2,3]),
            ('correct_action',True),('correct_action',1),('input_sha256','not-a-hash'),
            ('query_sha256','a'*64),('label_sha256','a'*64)]:
            with self.subTest(key=key,bad=bad):
                corrupt=copy.deepcopy(values); corrupt[0][key]=bad
                with self.assertRaises(AssertionError): PACKAGE.validate_rows(corrupt,0)
        duplicate=copy.deepcopy(values); duplicate[1]['input_sha256']=duplicate[0]['input_sha256']
        with self.assertRaises(AssertionError): PACKAGE.validate_rows(duplicate,0)

    def test_historical_aliases_reject_conflicting_or_untyped_ids(self):
        self.assertEqual(PACKAGE.historical_episode({'id':17}),17)
        self.assertEqual(PACKAGE.historical_episode({'episode_id':18}),18)
        self.assertEqual(PACKAGE.historical_episode({'id':19,'episode_id':19}),19)
        for row in ({},{'id':True},{'id':1.0},{'id':-1},{'id':1,'episode_id':2},{'id':False,'episode_id':0}):
            with self.assertRaises(AssertionError): PACKAGE.historical_episode(row)

    def write_audits(self):
        for panel in range(3):
            root=self.c/f'audit-{panel}';root.mkdir()
            (root/'known-features-input-audit.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in synthetic_rows(panel)))
            save(root/'manifest.json',{})
            save(root/'report.json',dict(optimizer_updates=0,model_forwards=0,query_overlap=0,
                excluded_unique_queries=5688,exclusions=[{}]*6))
            save(root.with_suffix('.process.json'),{})
            save(root.with_suffix('.exit.json'),{})

    def package_context(self):
        stack=contextlib.ExitStack()
        stack.enter_context(mock.patch.multiple(PACKAGE,R=self.r,campaign=lambda:self.c,
            qualified=lambda c:{},history=lambda:(set(),set(),set(),{}),verify_files=lambda files:None,
            stage=lambda root,kind: ({'started_local':'2026-09-09T10:01:00+00:00'},{})))
        save(self.r/'confirmation-authorization.json',dict(created_local='2026-09-09T10:00:00+00:00'))
        return stack

    def test_missing_third_audit_does_not_publish_a_seal(self):
        self.write_audits()
        (self.c/'audit-2'/'known-features-input-audit.jsonl').unlink()
        with self.package_context(),self.assertRaises(FileNotFoundError): PACKAGE.seal_panels()
        self.assertFalse((self.r/'panel-seal.json').exists())
        self.assertFalse((self.r/'panel-seal.sha256').exists())

    def test_cross_panel_collision_fails_without_repair_or_reseeding(self):
        self.write_audits()
        path=self.c/'audit-1'/'known-features-input-audit.jsonl'
        values=synthetic_rows(1); values[0]['input_sha256']=synthetic_rows(0)[0]['input_sha256']
        path.write_text(''.join(json.dumps(row)+'\n' for row in values))
        before=path.read_bytes()
        with self.package_context(),self.assertRaisesRegex(AssertionError,'cross-panel collision'):
            PACKAGE.seal_panels()
        self.assertEqual(path.read_bytes(),before)
        self.assertFalse((self.r/'panel-seal.json').exists())

    def test_missing_or_unbound_panel_seal_rejected_before_cache_creation(self):
        with mock.patch.multiple(PACKAGE,R=self.r,campaign=lambda:self.c,panel_binding=forbidden):
            with self.assertRaisesRegex(AssertionError,'external campaign access'):
                PACKAGE.seal_caches()
        self.assertFalse((self.c/'caches').exists())
        self.assertFalse((self.r/'cache-seal.json').exists())

    def cache_context(self):
        self.write_audits()
        cores={core:hashlib.sha256(core.encode()).hexdigest() for core in ('initial','final')}
        binding=dict(source='d'*40,binaries={'looped_agent_probe':'e'*64})
        save(self.c/'binary.json',binding)
        panel_seal=dict(accepted=True,created_local='2026-09-09T10:00:00+00:00',frozen_files={},
            panels={str(i):dict(rows=256) for i in range(3)})
        save(self.r/'panel-seal.json',panel_seal)
        (self.r/'panel-seal.sha256').write_text(digest(self.r/'panel-seal.json'))
        for panel in range(3):
            for core in cores:
                root=self.c/f'features-{panel}-{core}';root.mkdir()
                values=synthetic_rows(panel)
                for row in values:
                    row['arrays']={}
                for key,shape in [('cls',[128]),('current',[64,128]),('policy',[4])]:
                    width=math.prod(shape)
                    name=f'known-features-{key}.f32'
                    array=PACKAGE.np.full((256,width),panel+(0.125 if core=='initial' else 0.25),dtype='<f4')
                    (root/name).write_bytes(array.tobytes())
                    for index,row in enumerate(values):
                        row['arrays'][key]=dict(file=name,dtype='F32LE',shape=shape,
                            byte_offset=index*width*4,byte_length=width*4)
                (root/'known-features-rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in values))
                save(root/'report.json',dict(model_forwards=256,optimizer_updates=0,checkpoint_sha256=cores[core]))
                save(root/'metadata.json',dict(provenance=dict(source_revision=binding['source'],binary_sha256=binding['binaries']['looped_agent_probe'])))
                for name in ('initial.safetensors','final.safetensors'):
                    (root/name).write_bytes(core.encode())  # Synthetic hash fixture, not a model file.
                for path in (root/'manifest.json',root.with_suffix('.process.json'),root.with_suffix('.exit.json')):
                    save(path,{})
        def verify(files):
            for path,pin in files.items():
                require(digest(path)==pin,'frozen fixture changed')
        def verify_cache(root,pin):
            require(digest(root/'manifest.json')==pin,'bad synthetic cache manifest')
            manifest=json.loads((root/'manifest.json').read_text())
            for name,item in manifest['files'].items():
                require(digest(root/name)==item['sha256'] and (root/name).stat().st_size==item['bytes'],'bad synthetic cache payload')
        stack=contextlib.ExitStack()
        stack.enter_context(mock.patch.multiple(PACKAGE,R=self.r,CORES=cores,campaign=lambda:self.c,
            panel_binding=lambda c:(panel_seal,{}),verify_files=verify,verified_manifest=verify_cache,
            stage=lambda root,kind:({'started_local':'2026-09-09T10:01:00+00:00'},{})))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return stack

    def test_six_caches_preserve_rows_offsets_bytes_and_core_source_pairing(self):
        with self.cache_context():
            PACKAGE.seal_caches()
        seal=json.loads((self.r/'cache-seal.json').read_text())
        self.assertEqual(set(seal['caches']),{f'{panel}/{core}' for panel in range(3) for core in ('initial','final')})
        for key,entry in seal['caches'].items():
            panel,core=key.split('/');root=Path(entry['root']);source=self.c/f'features-{panel}-{core}'
            manifest=json.loads((root/'manifest.json').read_text())
            self.assertEqual(manifest['source']['root'],str(source))
            self.assertEqual(manifest['source']['confirmation_panel'],int(panel))
            for name in ('cls','current'):
                self.assertEqual((root/f'{name}.f32').read_bytes(),(source/f'known-features-{name}.f32').read_bytes())
            values=PACKAGE.rows(root/'rows.jsonl')
            self.assertEqual(len(values),256)
            self.assertEqual(set(values[0]),set(PACKAGE.FIELDS))
            self.assertEqual(values[0]['partition'],'confirmation_eval')

    def test_cache_failures_never_publish_complete_seal(self):
        with self.cache_context():
            # Each failure uses restored fixture data and removes only test-owned partial outputs.
            root=self.c/'features-0-initial'
            row_path=root/'known-features-rows.jsonl';original=row_path.read_bytes()
            report_path=root/'report.json';original_report=report_path.read_bytes()
            metadata_path=root/'metadata.json';original_metadata=metadata_path.read_bytes()
            array_path=root/'known-features-current.f32';original_array=array_path.read_bytes()
            cases=['wrong-core','wrong-binary','float-offset','float-extraction-id','nonfinite','missing-sixth']
            for case in cases:
                with self.subTest(case=case):
                    if case=='wrong-core':
                        value=json.loads(original_report);value['checkpoint_sha256']='f'*64;report_path.write_text(json.dumps(value))
                    elif case=='wrong-binary':
                        value=json.loads(original_metadata);value['provenance']['binary_sha256']='f'*64;metadata_path.write_text(json.dumps(value))
                    elif case=='nonfinite':
                        array_path.write_bytes(np.asarray([float('nan')],dtype='<f4').tobytes()+original_array[4:])
                    elif case in ('float-offset','float-extraction-id'):
                        values=PACKAGE.rows(row_path)
                        if case=='float-offset':values[0]['arrays']['current']['byte_offset']=0.0
                        else:values[0]['input_index']=False
                        row_path.write_text(''.join(json.dumps(row)+'\n' for row in values))
                    else:(self.c/'features-2-final'/'report.json').unlink()
                    with self.assertRaises((AssertionError,FileNotFoundError)):PACKAGE.seal_caches()
                    self.assertFalse((self.r/'cache-seal.json').exists())
                    self.assertFalse((self.r/'cache-seal.sha256').exists())
                    row_path.write_bytes(original);report_path.write_bytes(original_report)
                    metadata_path.write_bytes(original_metadata);array_path.write_bytes(original_array)
                    if (self.c/'caches').exists():shutil.rmtree(self.c/'caches')

    def test_budget_components_cannot_cancel_or_coerce(self):
        save(self.r/'campaign-clock.json',dict(started_local=datetime.datetime.now().astimezone().isoformat()))
        save(self.c/'one.exit.json',dict(model_phase_seconds=400,finalization_seconds=2))
        path=self.c/'two.exit.json'
        for bad in (-100,float('nan'),float('inf'),None,True,'1'):
            path.write_text(json.dumps(dict(model_phase_seconds=bad,finalization_seconds=0)))
            with mock.patch.multiple(SEQUENCE,R=self.r,campaign=lambda:self.c):
                with self.assertRaises(AssertionError): SEQUENCE.budgets()
        (self.c/'one.exit.json').write_text(json.dumps(dict(model_phase_seconds=10,finalization_seconds=2)))
        path.write_text(json.dumps(dict(model_phase_seconds=20,finalization_seconds=3)))
        with mock.patch.multiple(SEQUENCE,R=self.r,campaign=lambda:self.c):
            result=SEQUENCE.budgets()
        self.assertEqual((result['model_seconds'],result['finalization_seconds']),(30.0,5.0))

    def test_phase_admission_stops_before_spending_remaining_budget(self):
        config=dict(stage='head',model_seconds=60,finalization_seconds=60)
        budget=dict(model_seconds=300,finalization_seconds=500,wall_seconds=100)
        self.assertEqual(SEQUENCE.admit_model(config,budget),150)
        for changed in ({'model_seconds':301},{'model_seconds':359},{'finalization_seconds':541},{'wall_seconds':1800}):
            with self.assertRaises(AssertionError): SEQUENCE.admit_model(config,budget|changed)
        self.assertEqual(SEQUENCE.binder_deadline(budget|{'finalization_seconds':599}),1)
        with self.assertRaises(AssertionError): SEQUENCE.binder_deadline(budget|{'finalization_seconds':600})
        # CPU audits consume wall time; they are not falsely charged as GPU model work.
        self.assertEqual(SEQUENCE.admit_model(config|{'stage':'audit'},budget|{'model_seconds':360,'finalization_seconds':600}),150)

    def test_main_budget_failure_never_calls_tracked_child(self):
        save(self.c/'binary.json',dict(source='synthetic'))
        config=dict(stage='head',model_seconds=60,finalization_seconds=60)
        with (mock.patch.multiple(SEQUENCE,R=self.r,campaign=lambda:self.c,
            budgets=lambda:dict(model_seconds=359,finalization_seconds=0,wall_seconds=1),
            registered=lambda name,binding:config),mock.patch.object(SEQUENCE,'tracked') as launch,
            mock.patch.object(sys,'argv',['sequence.py','--stage','heads'])):
            with self.assertRaisesRegex(AssertionError,'remaining phase budget'): SEQUENCE.main()
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
