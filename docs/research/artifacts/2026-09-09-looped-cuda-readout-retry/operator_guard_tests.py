"""Synthetic C11 operator contracts; no real exports, panels, processes, or models."""
import contextlib
import copy
import datetime
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import campaign_io as ci
import export_parameters as ep
import supervise_stage as ss


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp(seconds=-60):
    return (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=seconds)).isoformat()


def seal_audit(root):
    names=('known-features-input-audit.jsonl','launch.json','metadata.json','profiles.json','report.json')
    put(root/'manifest.json',dict(schema='looped-agent-artifacts-v1',files={name:sha(root/name) for name in names}))
    root.with_suffix('.manifest.sha256').write_text(sha(root/'manifest.json')+'\n')


def audit_artifacts(root,config,binding,command):
    root.mkdir(parents=True,exist_ok=True)
    panel=int(command[command.index('--known-features-confirmation-panel')+1])
    artifact=root/'known-features-input-audit.jsonl'
    artifact.write_text('{"synthetic":true}\n'*256)  # Verifier fixture, not generated query data.
    root.with_suffix('.host.json').write_text('[\n\n]')
    put(root/'profiles.json',dict(root=str(root.with_suffix('.profiles')),files={},
        host_trace=dict(path=str(root.with_suffix('.host.json')),sha256=sha(root.with_suffix('.host.json'))),
        nsight='external capture and export; bind in a separate bundle after profiler exit'))
    put(root/'report.json',dict(schema='looped-known-features-v1',status='complete_pending_analysis',
        evidence_class='data_audit',optimizer_updates=0,model_forwards=0,input_rows=256,
        fit_rows=0,eval_rows=256,layouts=256,confirmation_panel=panel,partition='confirmation_eval',
        data_seed=20260917+panel,episode_id_base=0x43554441434f4e46+panel*0x10000,elapsed_seconds=.01,
        artifacts=[dict(file=artifact.name,bytes=artifact.stat().st_size,sha256=sha(artifact))]))
    put(root/'metadata.json',dict(provenance=dict(source_revision=binding['source'],
        binary_sha256=binding['binaries']['looped_agent_probe'],candle_graph_revision=ci.DEPENDENCY,checkpoint=None),
        exact_args=command,seed=0,data_seed=20260917+panel,physical_batch=1,effective_batch=1,accumulation=1,
        known_features=dict(schema='looped-known-features-v1',optimizer_updates=0,layouts=256,
            fit_layouts=0,eval_layouts=256,loops=4,condition='factual',implementation_smoke=False,
            confirmation_panel=panel,partition='confirmation_eval',cuda_gemm_reduced_precision_f32=None)))
    put(root/'launch.json',dict(exact_args=command,source_revision=binding['source'],
        binary_sha256=binding['binaries']['looped_agent_probe']))
    seal_audit(root)


class TempCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='c11-operator-guard-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)


class AuditVerification(TempCase):
    def fixture(self,name='audit-0'):
        root=self.root/name
        arguments=['--mode','known-features-audit','--known-mapping','--seed','0','--loops','4',
            '--batch','1','--effective-batch','1','--eval-episodes','256','--data-seed','20260917',
            '--known-features-confirmation-panel','0']
        config=dict(name=name,stage='audit',binary='looped_agent_probe',arguments=arguments)
        binding=dict(source='s'*40,binaries={'looped_agent_probe':'b'*64})
        command=[str(root.parent/'looped_agent_probe'),*arguments,'--output-dir',str(root),'--device','cpu']
        audit_artifacts(root,config,binding,command)
        return root,config,binding,command

    def test_preserved_capture_reproduces_old_failure_and_passes_zero_work_verifier(self):
        root=Path('/home/stepan/Projects/code/.tofy-runs/looped-cuda-readout-20260909T143306-IST/audit-0')
        expected='9c11215890d8bd9068cd665f862656053673807baffd8f9854cd65b70159e11b'
        self.assertEqual(sha(root/'manifest.json'),expected)
        paths=list(root.iterdir())+[root.with_suffix('.host.json'),root.with_suffix('.manifest.sha256')]
        before={str(path):sha(path) for path in paths}
        ci.verify_files({str(ss.LIFECYCLE):ss.LIFECYCLE_SHA})
        spec=ss.importlib.util.spec_from_file_location('c11_old_verify_repro',ss.LIFECYCLE)
        lifecycle=ss.importlib.util.module_from_spec(spec);spec.loader.exec_module(lifecycle)
        with self.assertRaises(AssertionError):lifecycle.verify(root)
        self.assertEqual(ci.read(root.with_suffix('.host.json')),[])
        launch=ci.read(root/'launch.json');command=launch['exact_args']
        config=dict(name=root.name,stage='audit',binary='looped_agent_probe',arguments=command[1:-4])
        binding=dict(source=launch['source_revision'],binaries={'looped_agent_probe':launch['binary_sha256']})
        self.assertEqual(ss.verify_audit(root,config,binding,command),expected)
        self.assertEqual({str(path):sha(path) for path in paths},before)

    def test_audit_empty_trace_and_absent_or_empty_profile_directory_are_valid(self):
        root,config,binding,command=self.fixture()
        self.assertEqual(ss.verify_audit(root,config,binding,command),sha(root/'manifest.json'))
        root.with_suffix('.profiles').mkdir()
        self.assertEqual(ss.verify_audit(root,config,binding,command),sha(root/'manifest.json'))

    def test_resealed_tampering_wrong_mode_and_unsupported_artifacts_rejected(self):
        cases=('updates','forwards','bool-zero','wrong-source','checkpoint','cuda-command','gpu-mode',
            'profile-files','profile-path','host-path','host-hash','host-nonempty','host-object',
            'host-missing','host-changed','extra-array','extra-checkpoint','nonempty-profiles','nsight','missing-artifact','changed-artifact')
        for index,case in enumerate(cases):
            with self.subTest(case=case):
                root,config,binding,command=self.fixture(f'audit-case-{index}')
                report=ci.read(root/'report.json');metadata=ci.read(root/'metadata.json');profiles=ci.read(root/'profiles.json')
                if case=='updates':report['optimizer_updates']=1
                if case=='forwards':report['model_forwards']=1
                if case=='bool-zero':report['model_forwards']=False
                if case=='wrong-source':metadata['provenance']['source_revision']='different'
                if case=='checkpoint':metadata['provenance']['checkpoint']='model.safetensors'
                if case=='cuda-command':command[-1]='cuda:0'
                if case=='gpu-mode':
                    command[2]='known-features';config['arguments'][1]='known-features'
                if case=='profile-files':profiles['files']={'fake/trace.jsonl':'a'*64}
                if case=='profile-path':profiles['root']=str(self.root/'wrong.profiles')
                if case=='host-path':profiles['host_trace']['path']=str(self.root/'wrong.host.json')
                if case=='host-hash':profiles['host_trace']['sha256']='a'*64
                if case in ('host-nonempty','host-object'):
                    put(root.with_suffix('.host.json'),[{'fake':'event'}] if case=='host-nonempty' else {})
                    profiles['host_trace']['sha256']=sha(root.with_suffix('.host.json'))
                if case=='host-missing':root.with_suffix('.host.json').unlink()
                if case=='host-changed':root.with_suffix('.host.json').write_text('[]\n')
                if case=='extra-array':(root/'known-features-current.f32').write_bytes(b'')
                if case=='extra-checkpoint':(root/'initial.safetensors').write_bytes(b'')
                if case=='nonempty-profiles':
                    root.with_suffix('.profiles').mkdir();(root.with_suffix('.profiles')/'trace.jsonl').write_text('[]')
                if case=='nsight':root.with_suffix('.nsight').mkdir()
                put(root/'report.json',report);put(root/'metadata.json',metadata);put(root/'profiles.json',profiles)
                seal_audit(root)
                if case=='missing-artifact':(root/'known-features-input-audit.jsonl').unlink()
                if case=='changed-artifact':(root/'known-features-input-audit.jsonl').write_text('changed')
                with self.assertRaises((RuntimeError,OSError)):ss.verify_audit(root,config,binding,command)


class Files(TempCase):
    def test_json_duplicate_and_nonfinite_rejected(self):
        for raw in ('{"accepted":false,"accepted":true}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            path = self.root / 'data.json'
            path.write_text(raw)
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                ci.read(path)

    def test_file_binding_rejects_symlink_ancestor_and_wrong_hash(self):
        real = self.root / 'real'
        real.mkdir()
        path = real / 'data'
        path.write_bytes(b'fixed')
        ci.verify_files({str(path): sha(path)})
        (self.root / 'alias').symlink_to(real, target_is_directory=True)
        for candidate, expected in [(self.root / 'alias' / 'data', sha(path)), (path, '0'*64)]:
            with self.subTest(candidate=candidate), self.assertRaises(RuntimeError):
                ci.verify_files({str(candidate): expected})

    def test_manifest_population_sizes_and_path_escape(self):
        path = self.root / 'data'
        path.write_bytes(b'x')
        manifest = self.root / 'manifest.json'
        base = {'files': {'data': {'sha256': sha(path), 'bytes': 1}}}
        put(manifest, base)
        ci.verified_manifest(self.root, sha(manifest))
        for files in ({'data': {'sha256': sha(path), 'bytes': True}},
                      {'../data': {'sha256': sha(path), 'bytes': 1}}, {}):
            put(manifest, {'files': files})
            with self.subTest(files=files), self.assertRaises(RuntimeError):
                ci.verified_manifest(self.root, sha(manifest))

    def test_parent_seal_types_external_binding_and_date(self):
        research, parent = self.root / 'research', self.root / 'parent'
        parent.mkdir()
        (parent / 'data').write_bytes(b'x')
        binding = self.root / 'producer.py'
        binding.write_bytes(b'fixed source')
        doc = dict(campaign=str(parent), created_local=timestamp(),
                   files={'data': dict(bytes=1, sha256=sha(parent/'data'))},
                   bindings={str(binding): sha(binding)})
        manifest = research / 'completed-campaign.manifest.json'
        for change in ('valid', 'boolean_size', 'naive_date', 'future_date', 'changed_binding'):
            current = copy.deepcopy(doc)
            if change == 'boolean_size': current['files']['data']['bytes'] = True
            if change == 'naive_date': current['created_local'] = '2026-01-01T00:00:00'
            if change == 'future_date': current['created_local'] = timestamp(100)
            if change == 'changed_binding': current['bindings'][str(binding)] = '0'*64
            put(manifest, current)
            with self.subTest(change=change), patch.multiple(ci, PARENTS={'c10': (research, sha(manifest))}, C10=parent):
                if change == 'valid':
                    result = ci.verify_parents()
                    self.assertEqual(result['frozen_files'][str(binding)], sha(binding))
                else:
                    with self.assertRaises(RuntimeError): ci.verify_parents()


class Tensors(TempCase):
    # Literal independent layout, deliberately serialized in reverse name order.
    layouts = {'spatial': [('queries', [2,128]), ('output.weight', [4,256]), ('output.bias', [4])],
               'cls': [('hidden.weight', [10,128]), ('hidden.bias', [10]), ('output.weight', [4,10]), ('output.bias', [4])]}

    def native(self, kind):
        payload, header, expected = bytearray(), {}, {}
        bits = (0x80000000, 0x00000001, 0x3f800000, 0xbf400000)
        for index, (name, shape) in enumerate(reversed(self.layouts[kind])):
            size = int(np.prod(shape))
            raw = b''.join(struct.pack('<I', bits[(i+index)%4]) for i in range(size))
            start = len(payload)
            payload.extend(raw)
            header[name] = dict(dtype='F32', shape=shape, data_offsets=[start,len(payload)])
            expected[name] = raw
        return header, bytes(payload), expected

    def foreign(self, header, payload):
        path = self.root / 'foreign.safetensors'
        raw = json.dumps(header).encode() if isinstance(header, dict) else header
        path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)
        return path

    def test_foreign_f32_exact_bits_and_canonical_offsets_both_families(self):
        for kind in ('spatial', 'cls'):
            header, payload, expected = self.native(kind)
            values = ep.safetensors(self.foreign(header, payload), kind)
            canonical, table = ep.encode(values, kind)
            wanted = b''.join(expected[name] for name, _ in self.layouts[kind])
            self.assertEqual(canonical, wanted)
            self.assertEqual(len(canonical), 5136 if kind == 'spatial' else 5336)
            for item in table:
                raw = expected[item['name']]
                self.assertEqual(canonical[item['byte_offset']:item['byte_offset']+item['byte_length']], raw)
                self.assertEqual(item['sha256'], hashlib.sha256(raw).hexdigest())
                self.assertEqual(item['dtype'], 'F32LE')

    def test_foreign_names_dtype_shape_and_offsets_corruption(self):
        header, payload, _ = self.native('spatial')
        for mutation in ('names','dtype','shape','boolean','offset','overlap','trailing','nonfinite','duplicate'):
            h, raw = copy.deepcopy(header), payload
            if mutation == 'names': h['extra'] = h['queries']
            if mutation == 'dtype': h['queries']['dtype'] = 'F64'
            if mutation == 'shape': h['queries']['shape'] = [128,2]
            if mutation == 'boolean': h['output.bias']['data_offsets'][0] = False
            if mutation == 'offset': h['queries']['data_offsets'][1] += 4
            if mutation == 'overlap': h['queries']['data_offsets'] = [0,1024]
            if mutation == 'trailing': raw += b'\0'*4
            if mutation == 'nonfinite': raw = struct.pack('<I', 0x7fc00000) + raw[4:]
            if mutation == 'duplicate':
                h = json.dumps(h).encode()[:-1] + b',"queries":' + json.dumps(h['queries']).encode() + b'}'
            with self.subTest(mutation=mutation), self.assertRaises((RuntimeError, ValueError)):
                ep.safetensors(self.foreign(h, raw), 'spatial')

    def test_truncated_foreign_header(self):
        for raw in (b'a', struct.pack('<Q', 100) + b'{}'):
            path = self.root / 'truncated'
            path.write_bytes(raw)
            with self.assertRaises(RuntimeError): ep.safetensors(path, 'spatial')

    def test_c10_f64_fortran_once_cast_and_finite_payload(self):
        arrays = {'queries': np.full((2,128), 1.0+2**-25, order='F'),
                  'output_weight': np.full((4,256), -.75, order='F'), 'output_bias': np.array([0.,-0.,2.,-2.])}
        path = self.root / 'selector.npz'
        np.savez(path, **arrays)
        payload, _ = ep.encode(ep.c10_values(path), 'spatial')
        expected = (struct.pack('<f',1.0)*256 + struct.pack('<f',-.75)*1024
                    + struct.pack('<4f',0.,-0.,2.,-2.))
        self.assertEqual(payload, expected)
        for bad in (np.full((2,128), 1, dtype='<f4'), np.full((2,128), np.nan), np.zeros((128,2))):
            np.savez(path, **dict(arrays, queries=bad))
            with self.assertRaises(RuntimeError): ep.c10_values(path)
        arrays['queries'][:] = 1e300
        np.savez(path, **arrays)
        with np.errstate(over='ignore'), self.assertRaises(RuntimeError):
            ep.encode(ep.c10_values(path), 'spatial')

    def test_encoder_rejects_coercion_and_missing_tensors(self):
        h, p, _ = self.native('spatial')
        values = ep.safetensors(self.foreign(h,p), 'spatial')
        for bad in (dict(values, queries=np.zeros((2,128), dtype=int)), {'queries': values['queries']}):
            with self.assertRaises(RuntimeError): ep.encode(bad,'spatial')


class Admission(TempCase):
    def setUp(self):
        super().setUp()
        self.r, self.c = self.root/'r11', self.root/'campaign'
        self.r.mkdir(); self.c.mkdir()
        self.c8, self.c9, self.r9 = (self.root / name for name in ('c8','c9','r9'))
        self.binding = dict(source='s'*40, binaries={'looped_agent_probe':'', 'learned_readout_probe':''}, features=ci.FEATURES)
        for name in self.binding['binaries']:
            (self.c/name).write_bytes(name.encode())
            self.binding['binaries'][name] = sha(self.c/name)
        put(self.c/'binary.json', self.binding)
        originals = self.root/'original'
        originals.write_bytes(b'synthetic fixed head source')
        imports = {}
        for core in ci.CORES:
            for arm in ci.ARMS:
                root = self.c/'imports'/core/arm
                root.mkdir(parents=True)
                (root/'parameters.f32').write_bytes(b'synthetic payload: not executed')
                put(root/'manifest.json', dict(schema='looped-imported-readout-source-v1',arm=arm,core_checkpoint=core,
                    head_kind='cls' if arm=='c9_cls' else 'spatial',
                    source_kind='c10_role_ridge' if arm.startswith('c10_') else 'c9_adamw',core_checkpoint_sha256=ci.CORES[core],
                    artifact={'sha256':sha(root/'parameters.f32')},
                    original_artifact={'path':str(originals),'sha256':sha(originals)}))
                imports[f'{core}/{arm}'] = dict(root=str(root),manifest_sha256=sha(root/'manifest.json'))
        put(self.r/'parameter-seal.json', dict(created_local=timestamp(-60),imports=imports,
            source_inputs={str(originals):sha(originals)},parent_verification={'accepted':True}))
        put(self.r/'parameter-seal-verification.json',dict(accepted=True,created_local=timestamp(-50),
            manifest_sha256=sha(self.r/'parameter-seal.json')))
        self.caches = {}
        for core in ci.CORES:
            root = self.c9/'caches'/f'fit-{core}'
            self.caches[core] = self.cache(root)
        put(self.r9/'fit-cache-seal.json',self.caches)
        put(self.c8/'exclusions.json',{'paths':[],'sha256':{}})
        for name in ('registration.md','campaign_io.py','export_parameters.py','supervise_stage.py','lifecycle.py'):
            (self.r/name).write_text(name)
        parent = self.root/'parent-research'
        put(parent/'completed-campaign.manifest.json',{})
        self.parents = {'c10':(parent,sha(parent/'completed-campaign.manifest.json'))}
        replacements = dict(R=self.r,C8=self.c8,C9=self.c9,R9=self.r9,PARENTS=self.parents,
                            LIFECYCLE=self.r/'lifecycle.py',LIFECYCLE_SHA=sha(self.r/'lifecycle.py'),
                            REGISTRATION_SHA=sha(self.r/'registration.md'))
        self.addCleanup(patch.stopall)
        patch.multiple(ss,**replacements).start()
        patch.object(ss,'campaign',return_value=self.c).start()
        self.authority = dict(accepted=True,campaign=str(self.c),source=self.binding['source'],
                             binaries=self.binding['binaries'],dependency=ci.DEPENDENCY,created_local=timestamp(-40))
        files = [self.r/name for name in ('registration.md','campaign_io.py','export_parameters.py','supervise_stage.py',
                                          'lifecycle.py','parameter-seal.json','parameter-seal-verification.json')]
        files += [self.c/'binary.json',self.r9/'fit-cache-seal.json',self.c8/'exclusions.json',parent/'completed-campaign.manifest.json']
        files += [self.c/name for name in self.binding['binaries']]
        self.authority['frozen_files'] = {str(path):sha(path) for path in files}
        self.seal('qualification-authorization',self.authority)
        shared = dict(campaign=str(self.c),source=self.binding['source'],binaries=self.binding['binaries'],
                      parameter_seal_sha256=sha(self.r/'parameter-seal.json'))
        report = dict(accepted=True,created_local=timestamp(-30),**shared)
        put(self.r/'qualification-report.json',report)
        self.confirmation = dict(accepted=True,created_local=timestamp(-20),dependency=ci.DEPENDENCY,
            qualification_authorization_sha256=sha(self.r/'qualification-authorization.json'),
            qualification_report={'path':str(self.r/'qualification-report.json'),'sha256':sha(self.r/'qualification-report.json')},**shared)
        self.seal('confirmation-authorization',self.confirmation)
        self.called = False

    def cache(self, root):
        root.mkdir(parents=True)
        (root/'synthetic').write_bytes(b'not model features')
        fit=root.name.startswith('fit-')
        core='final' if root.name.endswith('final') else 'initial'
        put(root/'manifest.json',{'core_checkpoint_sha256':ci.CORES[core],'rows':512 if fit else 256,
            'partition':'fit' if fit else 'confirmation_eval',
            'files':{'synthetic':{'sha256':sha(root/'synthetic'),'bytes':18}}})
        return dict(root=str(root),manifest_sha256=sha(root/'manifest.json'))

    def seal(self,name,value):
        put(self.r/f'{name}.json',value)
        (self.r/f'{name}.sha256').write_text(sha(self.r/f'{name}.json')+'\n')

    def panels(self):
        doc = dict(accepted=True,created_local=timestamp(-10),historical_unique_queries=5688,new_unique_queries=768,
            panels={str(i):dict(root=str(self.c/f'audit-{i}'),rows=256,query_overlap=0,input_overlap=0,episode_overlap=0) for i in range(3)},
            frozen_files=ss.qualified(self.c))
        self.seal('panel-seal',doc)
        return doc

    def full_caches(self):
        self.panels()
        doc = dict(accepted=True,created_local=timestamp(-5),caches={f'{i}/{core}':self.cache(self.c/'caches'/str(i)/core)
            for i in range(3) for core in ci.CORES},frozen_files={str(self.r/'panel-seal.json'):sha(self.r/'panel-seal.json')})
        self.seal('cache-seal',doc)
        return doc

    def test_exact_qualification_and_zero_update_full_stage_commands(self):
        for core in ci.CORES:
            for arm in ci.ARMS:
                value = ss.registered(f'qual-{core}-{arm}',self.binding)
                self.assertEqual(value['expected_report']['optimizer_updates'],0)
                self.assertEqual(value['expected_report']['input_rows'],512)
                self.assertEqual(value['arguments'][0:2],['--mode','evaluate-imported'])
                self.assertIn('--import-qualification',value['arguments'])
        self.full_caches()
        for i in range(3):
            for core in ci.CORES:
                value = ss.registered(f'head-{i}-{core}-c9_cls',self.binding)
                self.assertEqual(value['expected_report']['input_rows'],256)
                self.assertEqual(value['arguments'][-2:],['--confirmation-panel',str(i)])

    def test_unregistered_path_injection_rejected(self):
        for name in ('../escape','qual-initial-c10_true;true','head-3-final-c9_cls','fit-initial-c10_true'):
            with self.subTest(name=name),self.assertRaises(RuntimeError): ss.registered(name,self.binding)

    def test_confirmation_requires_pinned_matching_source_campaign_and_parameters(self):
        ss.qualified(self.c)
        for key in ('accepted','source','campaign','parameter_seal_sha256','qualification_authorization_sha256'):
            value=copy.deepcopy(self.confirmation)
            value[key] = False if key=='accepted' else 'different'
            self.seal('confirmation-authorization',value)
            with self.subTest(key=key),self.assertRaises(RuntimeError): ss.qualified(self.c)

    def test_authority_requires_operator_parent_and_registration_pins(self):
        ss.qualification_authority(self.c,self.binding)
        keys = [str(self.r/key) for key in ('supervise_stage.py','registration.md','parameter-seal.json')]
        keys += [str(self.parents['c10'][0]/'completed-campaign.manifest.json')]
        for key in keys:
            value=copy.deepcopy(self.authority)
            del value['frozen_files'][key]
            self.seal('qualification-authorization',value)
            with self.subTest(key=key),self.assertRaises(RuntimeError): ss.qualification_authority(self.c,self.binding)

    def test_report_and_confirmation_dates_fail_closed(self):
        for date in ('2026-01-01T00:00:00',timestamp(-100),timestamp(100)):
            value=copy.deepcopy(self.confirmation);value['created_local']=date
            self.seal('confirmation-authorization',value)
            with self.subTest(date=date),self.assertRaises(RuntimeError): ss.qualified(self.c)

    def test_all_panels_required_before_extraction(self):
        with self.assertRaises(FileNotFoundError): ss.registered('features-0-initial',self.binding)
        doc=self.panels();del doc['panels']['2'];self.seal('panel-seal',doc)
        with self.assertRaises(RuntimeError): ss.registered('features-0-initial',self.binding)

    def test_wrong_cache_core_even_with_matching_resealed_bytes_is_rejected(self):
        path=Path(self.caches['initial']['root'])/'manifest.json'
        document=ci.read(path);document['core_checkpoint_sha256']=ci.CORES['final'];put(path,document)
        seal=ci.read(self.r9/'fit-cache-seal.json');seal['initial']['manifest_sha256']=sha(path)
        put(self.r9/'fit-cache-seal.json',seal)
        with self.assertRaises(RuntimeError):ss.registered('qual-initial-c10_true',self.binding)

    def test_all_caches_and_correct_panel_barrier_required(self):
        doc=self.full_caches()
        for mutation in ('accepted','missing','mixed','predates'):
            value=copy.deepcopy(doc)
            if mutation=='accepted':value['accepted']=False
            if mutation=='missing':del value['caches']['2/final']
            if mutation=='mixed':value['frozen_files'][str(self.r/'panel-seal.json')]='0'*64
            if mutation=='predates':value['created_local']=timestamp(-100)
            self.seal('cache-seal',value)
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):ss.registered('head-0-initial-c10_true',self.binding)

    def test_import_hash_and_ten_head_population_guard(self):
        seal = ci.read(self.r/'parameter-seal.json')
        malformed = copy.deepcopy(seal)
        del malformed['imports']['final/c9_null']
        put(self.r/'parameter-seal.json',malformed)
        verification = ci.read(self.r/'parameter-seal-verification.json')
        verification['manifest_sha256'] = sha(self.r/'parameter-seal.json')
        put(self.r/'parameter-seal-verification.json',verification)
        with self.assertRaises(RuntimeError):ss.registered('qual-initial-c10_true',self.binding)
        put(self.r/'parameter-seal.json',seal)
        verification['manifest_sha256']=sha(self.r/'parameter-seal.json')
        put(self.r/'parameter-seal-verification.json',verification)
        path=self.c/'imports'/'initial'/'c10_true'/'parameters.f32'
        path.write_bytes(b'corrupt')
        with self.assertRaises(RuntimeError):ss.registered('qual-initial-c10_true',self.binding)

    def clean_state(self):
        return dict(returncode=0,failure=None,pid_gone=True,model_pid_gone=True,group_gone=True,
            model_pid=1073741824,pgid=1073741824,owned_pids=[1073741824],owned_survivors=[],group_survivors=[],
            cleanup_error=None,launch_record_error=None,headroom_mib=4096,max_temperature_c=45,
            model_phase_seconds=.02,elapsed_seconds=.03,finalization_seconds=.01)

    def test_cleanup_rejects_malformed_pid_survivors_temperature_and_nonfinite(self):
        ss.valid_cleanup(self.clean_state())
        mutations={'returncode':False,'model_pid':True,'pid_gone':'yes','group_survivors':[1073741824],
                   'owned_survivors':[1073741824],'cleanup_error':'timeout','launch_record_error':'bad json',
                   'owned_pids':[],'max_temperature_c':85,'headroom_mib':511,'elapsed_seconds':float('nan')}
        for key,value in mutations.items():
            with self.subTest(key=key),self.assertRaises(RuntimeError):ss.valid_cleanup(dict(self.clean_state(),**{key:value}))

    def run_mock(self,config_change=None,report_change=None,state_change=None,audit=False):
        config=ss.registered('audit-0' if audit else 'qual-initial-c10_true',self.binding)
        if audit:config['frozen_files']={}  # Population authority is covered separately; no real exclusions opened.
        prescribed=copy.deepcopy(config)
        if config_change:config_change(config)
        path=self.c/'invocations'/'case.json';put(path,config)
        def supervise(command,root,env,output,telemetry,record,sample,**kwargs):
            self.called=True
            self.command,self.environment=command,env
            root.mkdir()
            record(1073741824,1073741824)
            if audit:audit_artifacts(root,config,self.binding,command)
            report=ci.read(root/'report.json') if audit else dict(config['expected_report'],elapsed_seconds=.01)
            if report_change:report.update(report_change)
            put(root/'report.json',report)
            if audit:seal_audit(root)
            else:put(root/'metadata.json',{'provenance':dict(source_revision=self.binding['source'],
                binary_sha256=self.binding['binaries']['learned_readout_probe'],candle_graph_revision=ci.DEPENDENCY)})
            return dict(self.clean_state(),**(state_change or {}))
        def gpu_verify(root):
            self.gpu_verifier_called=True
            self.assertFalse(audit,'CPU audit reached unchanged GPU verifier')
            return 'a'*64
        self.gpu_verifier_called=False
        fake=types.SimpleNamespace(supervise_child=supervise,terminated=lambda *x:None,
            verify=gpu_verify,reported_model_elapsed=lambda report,limit:report['elapsed_seconds'])
        loader=types.SimpleNamespace(exec_module=lambda module:module.__dict__.update(fake.__dict__))
        original_read=Path.read_text
        def read_text(path,*args,**kwargs):
            return '1' if str(path)=='/sys/class/power_supply/ACAD/online' else original_read(path,*args,**kwargs)
        def git(command,**kwargs):
            return '' if 'status' in command else (ci.DEPENDENCY if 'candle_graph' in command[2] else self.binding['source'])+'\n'
        with patch.object(ss.sys,'argv',['supervise_stage.py','--config',str(path),'--sha256',sha(path)]),\
             (patch.object(ss,'registered',return_value=prescribed) if audit else contextlib.nullcontext()),\
             patch.object(ss.subprocess,'check_output',side_effect=git),\
             patch.object(ss.subprocess,'run',return_value=types.SimpleNamespace(stdout='100,10000,40',returncode=0)),\
             patch.object(ss.importlib.util,'spec_from_file_location',return_value=types.SimpleNamespace(loader=loader)),\
             patch.object(ss.importlib.util,'module_from_spec',return_value=types.SimpleNamespace()),\
             patch.object(ss.signal,'signal'),patch.object(Path,'read_text',read_text),contextlib.redirect_stdout(io.StringIO()):
            ss.main()
        return ci.read((self.c/config['name']).with_suffix('.exit.json'))

    def test_mock_launch_preserves_exact_command_environment_and_cleanup(self):
        state=self.run_mock()
        self.assertTrue(self.called and state['accepted'])
        self.assertEqual(state['classification'],'implementation_smoke')
        self.assertEqual(self.command[0],ci.NSYS)
        self.assertIn('--capture-range-end=repeat:1:defer',self.command)
        self.assertEqual(self.command[-2:],['--device','cuda:0'])
        self.assertEqual(self.environment['NVIDIA_TF32_OVERRIDE'],'0')
        self.assertEqual(self.environment['TOFY_PERF_TRACE'],str(self.c/'qual-initial-c10_true.host.json'))

    def test_mock_cpu_audit_reaches_postcheck_without_gpu_verifier(self):
        state=self.run_mock(audit=True)
        self.assertTrue(state['accepted'] and self.called)
        self.assertFalse(self.gpu_verifier_called)
        self.assertEqual(self.command[0],str(self.c/'looped_agent_probe'))
        self.assertEqual(self.command[-2:],['--device','cpu'])
        self.assertEqual(ci.read(self.c/'audit-0.host.json'),[])
        self.assertEqual(state['manifest_sha256'],sha(self.c/'audit-0'/'manifest.json'))

    def test_mock_launch_rejects_extra_argument_before_child(self):
        with self.assertRaises(RuntimeError):self.run_mock(config_change=lambda c:c['arguments'].extend(['--updates','1']))
        self.assertFalse(self.called)

    def test_mock_launch_rejects_root_reuse_before_child(self):
        (self.c/'qual-initial-c10_true.stdout.log').write_text('old failed attempt')
        with self.assertRaises(RuntimeError):self.run_mock()
        self.assertFalse(self.called)

    def test_mock_launch_rejects_changed_operator_source_before_child(self):
        (self.r/'campaign_io.py').write_text('changed after source freeze')
        with self.assertRaises(RuntimeError):self.run_mock()
        self.assertFalse(self.called)

    def test_mock_launch_rejects_nonzero_core_updates(self):
        with self.assertRaises(RuntimeError):self.run_mock(report_change={'core_optimizer_updates':1})
        self.assertFalse(ci.read(self.c/'qual-initial-c10_true.exit.json')['accepted'])

    def test_mock_launch_nonzero_update_is_failed_evidence(self):
        with self.assertRaises(RuntimeError):self.run_mock(report_change={'optimizer_updates':1})
        state=ci.read(self.c/'qual-initial-c10_true.exit.json')
        self.assertFalse(state['accepted'])
        self.assertEqual(state['classification'],'failed_infrastructure_or_integrity')

    def test_mock_launch_cleanup_failure_is_failed_evidence(self):
        with self.assertRaises(RuntimeError):self.run_mock(state_change={'group_survivors':[1073741824]})
        self.assertFalse(ci.read(self.c/'qual-initial-c10_true.exit.json')['accepted'])


if __name__=='__main__':
    unittest.main()
