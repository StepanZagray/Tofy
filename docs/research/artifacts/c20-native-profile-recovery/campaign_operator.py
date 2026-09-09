"""C20 unchanged native campaign after producer profile repair; hash-pinned generic process/profile lifecycle."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse, hashlib, importlib.util, json, math, shutil, subprocess, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent
R12=R.parent/'2026-09-09T143629Z-tofy-looped-grounded-policy-learning'
R15=R.parent/'2026-09-09T173927Z-tofy-looped-demonstration-grounding'
R18=R.parent/'2026-09-09T194505Z-tofy-looped-action-equivariance'
REPO=Path('/home/stepan/Projects/code/Tofy-native-binding-profile-fix')
PYTHON='/home/stepan/venvs/tensorboard/bin/python3'
DEPENDENCY='1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a'
PINNED={'supervise.py':'0c1eb08878efbb9f178828e04fcce74712d826cffc23d9aa2a2a363b51f151c9','driver.py':'8de46c86aa3d2a95e97616cb0908026521ee536a75729934d9691c36d68c2078','bind_nsight.py':'82870b0247b142533b3702d5332bca43aa10f9785c9b96a8acd3ec405ce8be5b'}
for _name,_sha in PINNED.items():
    with (R12/_name).open('rb') as _f:
        if hashlib.file_digest(_f,'sha256').hexdigest()!=_sha:raise ValueError('generic lifecycle changed')
sys.path.insert(0,str(R12))
from supervise import read,save,digest,verify_files,require,now,root_manifest
from driver import tracked
sys.path.remove(str(R12))
ARTIFACTS=('registration.md','campaign_operator.py','operator_tests.py','analysis.py','analysis_tests.py','numpy_binding.py','numpy_binding_tests.py','independent_review.py','independent_review_tests.py','seal_completed.py')
SOURCE_FILES=('Cargo.toml','Cargo.lock','build.rs','src/p2/looped_agent/model.rs','src/p2/looped_agent/grounded_policy.rs','src/p2/looped_agent/binding.rs','src/p2/looped_agent/native_binding.rs','src/p2/looped_agent/mod.rs','src/p2/looped_agent/profile.rs','src/p2/looped_agent/task.rs','examples/native_binding_probe.rs','examples/native_binding/data.rs','examples/native_binding/engine.rs','examples/grounded_policy/evidence.rs')
BATCHES=(32,64,128,256,512,1024)
CORE=Path('/home/stepan/Projects/code/.tofy-runs/looped-frozen-features-20260909T115455-IST/features-initial/initial.safetensors')
HEAD=Path('/home/stepan/Projects/code/.tofy-runs/looped-cuda-readout-retry-20260909T151301-IST/imports/initial/c10_true/parameters.f32')
IMPORT=HEAD.parent/'manifest.json'
BINDER=Path('/home/stepan/Projects/code/.tofy-runs/looped-action-equivariant-20260909T212250-IST/equivariant-train-seed0/final.safetensors')
CHECKPOINTS={str(CORE):'4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802',str(HEAD):'a43ad78fe7a6b1594d23c205fb191c8ac81610deaaff25198cb421d423edc678',str(IMPORT):'d2f9a7a99917d180646a5e28e26acd92cbd3fe53919fa009da3e11747e0c158e',str(BINDER):'d2deeba7b0fafc2386c9a19d39bc99b7534b91a0155d66e77792a5c4037bd717'}
FAILED_MANIFEST=Path('/home/stepan/Research/_runs/2026-09-09T204022Z-tofy-looped-native-binding/failed-campaign.manifest.json')
FAILED_MANIFEST_SHA='cd938e9aa40a2010b43bf57cca3b1add47dea8968f6e62edf145ee719478a8a6'
ORIGINAL_PANEL=Path('/home/stepan/Projects/code/.tofy-runs/looped-native-binding-20260909T222118-/audit/panel-rows.jsonl')
ORIGINAL_PANEL_SHA='87459d3e5307c935dc4d65f1388d8f14419eac4b2f0044bc17efc07b81573e0b'
BINDER_PARAMETERS='86dd13d3998598d54acc1d97167c75d09d1d00f12b29b9073154d19a29a9e491'

def binding(path):return dict(path=str(path),sha256=digest(path))
def clean_revision():
    revision=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    require(not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=all'],text=True).strip(),'dirty native checkout')
    subprocess.run(['git','-C',str(REPO),'merge-base','--is-ancestor','HEAD','@{upstream}'],check=True)
    return revision

def remaining(deadline,ceiling):
    value=min(ceiling,deadline-time.monotonic());require(math.isfinite(value) and value>0,'campaign deadline exhausted');return value

def campaign_deadline(campaign):
    c=read(campaign/'clock.json')
    require(c['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'host rebooted')
    t=c['started_monotonic'];require(type(t) in (int,float) and math.isfinite(t) and 0<=t<=time.monotonic(),'invalid campaign clock')
    return t+1200

def verify_parent(run,expected):
    p=run/'completed-campaign.manifest.json';require(digest(p)==expected,'parent seal changed')
    manifest=read(p)
    for rel,row in manifest['files'].items():
        path=Path(manifest['campaign'])/rel
        require(path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'parent artifact changed')

def prepare(campaign,binary):
    require(campaign.is_absolute() and not campaign.exists() and '.' not in campaign.name,'new absolute dotless campaign required')
    revision=clean_revision();build=read(R/'build.json')
    require(build['source_revision']==revision and build['binary_sha256']==digest(binary),'build source/binary differs')
    require(read(R/'operations/build-cuda.exit.json')['accepted'] is True and read(R/'operations/build-cuda.process.json')['command']==build['command'],'build execution not bound')
    require(build['features']=='cudnn,profiling,serde_json/float_roundtrip' and build['features'] in build['command'],'device features differ')
    files=dict(read(R/'source-freeze.json')['files'])
    for name in ARTIFACTS:
        p=R/name;require(digest(p)==digest(REPO/'docs/research/artifacts/c20-native-profile-recovery'/name),'source archive differs');files[str(p)]=digest(p)
    for name in SOURCE_FILES:
        p=REPO/name;files[str(p)]=digest(p)
    for p in [R/'source-freeze.json',R/'build.json',*[R/'operations'/f'build-cuda.{x}' for x in ('process.json','exit.json','stdout.log')]]:files[str(p)]=digest(p)
    verify_parent(R18,'6535a3283029f2d5ad211ad70030fb3e2ad21b1dca1530e985406a94465c7b4c')
    verify_parent(R15,'6007293c5f51d263b0e3f40dcd819124efa8ffdaf2809202eda0c60142ffd40d')
    require(files.get(str(FAILED_MANIFEST))==FAILED_MANIFEST_SHA and digest(FAILED_MANIFEST)==FAILED_MANIFEST_SHA,'failed C19 inventory must be bound before retry')
    failed=read(FAILED_MANIFEST);verify_files(failed['external_bindings'])
    for rel,row in failed['files'].items():
        p=Path(failed['campaign'])/rel
        require(p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],'failed C19 artifact changed')
    verify_files(CHECKPOINTS);verify_files(files)
    for p,h in CHECKPOINTS.items():require(files.get(p)==h,'checkpoint not source-frozen')
    for row in read(R/'history.json')['history']:require(files.get(row['path'])==row['sha256'],'history not frozen')
    campaign.mkdir();(campaign/'invocations').mkdir();shutil.copy2(binary,campaign/'native_binding_probe')
    files[str(campaign/'native_binding_probe')]=digest(campaign/'native_binding_probe')
    spec=dict(schema='looped-native-binding-launch-v1',campaign=str(campaign),repository=str(REPO),source=revision,dependency_revision=DEPENDENCY,binary=str(campaign/'native_binding_probe'),binary_sha256=digest(campaign/'native_binding_probe'),frozen_files=files,created_local=now())
    save(campaign/'launch-spec.json',spec);return spec

def healthy_capture(campaign,name):
    rows=read(campaign/f'{name}.bound/summary.json')
    require(len(rows)==1 and all(r['health']['structurally_valid'] and r['health']['capture_complete'] and r['raw_application_labels_verified'] and r['gpu']['status']=='available' and r['gpu']['provenance_binding']=='bound' for r in rows),'native capture incomplete')

def report_integrity(report,mode,batch,identity=None):
    rows=768 if mode=='evaluate' else 4 if mode=='qualify' else batch
    require(report['optimizer_updates']==0 and report['input_rows']==rows and report['panel_rows']==768,'native row/update count differs')
    require(report['physical_batch']==batch and report['actual_physical_batch']==min(batch,rows) and report['microbatches']==math.ceil(rows/batch) and report['tail_batch']==(rows-1)%batch+1,'actual native batching differs')
    require(report['unique_input_rows']==min(rows,768) and report['repeated_input_rows']==max(0,rows-768),'native repetition differs')
    n=math.ceil(rows/batch)
    require(report['core_forward_batches']==n and report['selector_forward_batches']==7*n and report['binder_forward_batches']==n,'actual forward counts differ')
    require(report['core_loops']==4 and report['binder_loops']==4 and report['privileged_role_warm_start'] is True,'native depth/provenance differs')
    require(report['changes']['all_parameters_unchanged'] is True and report['changes']['unused_heads_unchanged'] is True,'native parameters changed')
    require(report['parameter_digests_before']==report['parameter_digests_after'],'native parameter identities differ')
    require(report['parameter_digests_before']['binder']['parameter_sha256']==BINDER_PARAMETERS,'selected binder identity differs')
    require(report['current_frame_input_bitwise_equal'] is True and report['current_frame_max_absolute_difference']==0,'current-frame input parity differs')
    require(type(report['mean_ce']) in (int,float) and math.isfinite(report['mean_ce']),'invalid native scalar loss')
    if identity is not None:require(report['parameter_digests_before']==identity,'cross-invocation checkpoint identities differ')
    return report['parameter_digests_before']

def invoke(spec,name,mode,batch,deadline,control='factual',identity=None):
    c=Path(spec['campaign']);verify_files(spec['frozen_files'])
    audit=mode=='audit'
    require(remaining(deadline,1200)>=(250 if audit else 370),'insufficient model/finalization/binding window')
    cfg=dict(schema='looped-native-binding-config-v1',source_revision=spec['source'],registration=str(R/'registration.md'),registration_sha256=digest(R/'registration.md'),mode=mode,output_dir=str(c/name),physical_batch=batch,updates=0,max_seconds=120,core_loops=4,binder_loops=4,control=control,panel_seed=20260923,panel_tag=0x4e415449564542,query_groups=32,history=read(R/'history.json')['history'] if audit else [],audit_root=None if audit else str(c/'audit'),audit_manifest_sha256=None if audit else root_manifest(c/'audit')[0],core_checkpoint=str(CORE),core_sha256=CHECKPOINTS[str(CORE)],head_checkpoint=str(HEAD),head_sha256=CHECKPOINTS[str(HEAD)],import_manifest=str(IMPORT),binder_checkpoint=str(BINDER),binder_sha256=CHECKPOINTS[str(BINDER)])
    p=c/'invocations'/f'{name}.json';save(p,cfg);frozen=dict(spec['frozen_files']);frozen[str(p)]=digest(p)
    if not audit:
        for q in (c/'audit').iterdir():
            if q.is_file():frozen[str(q)]=digest(q)
    expected=dict(status='complete_pending_analysis',optimizer_updates=0,input_rows=768 if audit or mode=='evaluate' else 4 if mode=='qualify' else batch)
    if audit:expected['model_forwards']=0
    else:expected.update(physical_batch=batch,core_loops=4,binder_loops=4,control=control,privileged_role_warm_start=True)
    a=c/'invocations'/f'{name}-authority.json'
    save(a,dict(schema='looped-grounded-policy-launch-v1',accepted=True,created_local=now(),config_sha256=digest(p),mode=mode,campaign=str(c),name=name,binary=spec['binary'],binary_sha256=spec['binary_sha256'],repository=spec['repository'],source=spec['source'],frozen_files=frozen,expected_report=expected))
    outer=tracked(spec,[PYTHON,str(R12/'supervise.py'),'--config',str(p),'--sha256',digest(p),'--authority',str(a),'--authority-sha256',digest(a)],name+'-model',remaining(deadline,250),allow_failure=mode=='batch_smoke')
    state=read(c/f'{name}.exit.json')
    if not state['accepted']:
        require(mode=='batch_smoke' and state['capacity_failure'] is True,'non-capacity native failure')
        require(outer['pid_gone'] and outer['group_gone'] and not outer['owned_survivors'] and not outer['group_survivors'] and outer['cleanup_error'] is None,'failed smoke not cleaned')
        print(f'Capacity limit at {name}',flush=True);return None
    require(outer['accepted'] is True and root_manifest(c/name)[0]==state['manifest_sha256'],'native invocation integrity differs')
    report=read(c/name/'report.json')
    if not audit:
        tracked(spec,[PYTHON,str(R12/'bind_nsight.py'),'--root',str(c/name)],name+'-bind',remaining(deadline,120));healthy_capture(c,name);report_integrity(report,mode,batch,identity)
    print(f'Completed {name}',flush=True);return report

def premise(spec,deadline):
    c=Path(spec['campaign']);files=dict(spec['frozen_files'])
    require(files.get(str(ORIGINAL_PANEL))==ORIGINAL_PANEL_SHA and digest(ORIGINAL_PANEL)==ORIGINAL_PANEL_SHA and digest(c/'audit/panel-rows.jsonl')==ORIGINAL_PANEL_SHA,'retry must preserve the exact original C19 panel')
    for p in (c/'audit').iterdir():
        if p.is_file():files[str(p)]=digest(p)
    outer=c/'audit.manifest.sha256';files[str(outer)]=digest(outer)
    cfg=dict(schema='looped-native-binding-premise-v1',audit=binding(c/'audit/panel-rows.jsonl'),audit_manifest=binding(c/'audit/manifest.json'),history=binding(R/'history.json'),registration=binding(R/'registration.md'),source_revision=spec['source'],binary_sha256=spec['binary_sha256'],frozen_files=files)
    path=c/'premise-config.json';save(path,cfg)
    for name,script in [('primary-premise','analysis.py'),('independent-premise','independent_review.py')]:
        out=c/f'{name}.json';tracked(spec,[PYTHON,str(R/script),'--premise','--config',str(path),'--output',str(out)],name,remaining(deadline,60));require(read(out)['accepted'] is True,'native panel premise rejected')

def receipt(spec,names,batch):
    c=Path(spec['campaign']);files=dict(spec['frozen_files']);identity=read(c/'qualify-b4/report.json')['parameter_digests_before']
    for name in names:
        state=read(c/f'{name}.exit.json');require(state['accepted'] is True and state['bindings_unchanged'] is True,'invocation receipt incomplete')
        require(root_manifest(c/name)[0]==state['manifest_sha256'],'root manifest changed')
        cfg=read(c/'invocations'/f'{name}.json')
        if name!='audit':
            healthy_capture(c,name);report_integrity(read(c/name/'report.json'),cfg['mode'],cfg['physical_batch'],identity)
        for op in ['model']+([] if name=='audit' else ['bind']):
            state=read(c/'operations'/f'{name}-{op}.exit.json');require(state['accepted'] and state['pid_gone'] and state['group_gone'] and not state['owned_survivors'] and not state['group_survivors'] and state['cleanup_error'] is None,'operation cleanup incomplete')
        for p in [c/f'{name}.exit.json',c/f'{name}.process.json',*[p for p in (c/name).rglob('*') if p.is_file()],*[p for suffix in ('.bound','.nsight') for p in (c/name).with_suffix(suffix).rglob('*') if p.is_file()]]:files[str(p)]=digest(p)
    for name in ['primary-premise','independent-premise']:
        state=read(c/'operations'/f'{name}.exit.json');require(state['accepted'] is True and state['pid_gone'] and state['group_gone'] and not state['owned_survivors'] and not state['group_survivors'] and state['cleanup_error'] is None and read(c/f'{name}.json')['accepted'] is True,'premise receipt rejected')
    # Bind external audit seal and every completed inner operation receipt.
    for p in [c/'audit.manifest.sha256', *[q for q in (c/'operations').iterdir() if q.is_file()]]:files[str(p)]=digest(p)
    for name in ['capacity-selection.json','launch-spec.json','clock.json','premise-config.json','primary-premise.json','independent-premise.json']:
        p=c/name;files[str(p)]=digest(p)
    streams={name:binding(c/name/'evaluation-rows.jsonl') for name in ('factual','uniform')}
    verify_files(files)
    result=dict(schema='looped-native-binding-integrity-v1',accepted=True,checks={key:True for key in ('source','build','checkpoints','zero_updates','unchanged_parameters','qualification','profiles','cleanup')},source_revision=spec['source'],binary_sha256=spec['binary_sha256'],dependency_revision=DEPENDENCY,vision_checkpoint_sha256=CHECKPOINTS[str(CORE)],selector_sha256=CHECKPOINTS[str(HEAD)],binder_checkpoint_sha256=CHECKPOINTS[str(BINDER)],vision_loops=4,binder_loops=4,physical_batch=batch,checkpoint_files=CHECKPOINTS,parameter_identity=identity,streams=streams,frozen_files=files,created_local=now())
    p=c/'integrity.json';save(p,result);files[str(p)]=digest(p)
    cfg=dict(schema='looped-native-binding-analysis-config-v1',registration=binding(R/'registration.md'),history=binding(R/'history.json'),audit=binding(c/'audit/panel-rows.jsonl'),checkpoint=binding(BINDER),integrity_receipt=binding(p),streams=streams,frozen_files=files)
    save(c/'analysis-config.json',cfg)

def execute(spec):
    c=Path(spec['campaign']);start=time.monotonic();save(c/'clock.json',dict(started_monotonic=start,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_local=now()))
    deadline=campaign_deadline(c)-240;names=[];failed=[]
    invoke(spec,'audit','audit',1,deadline);names.append('audit');premise(spec,deadline)
    qualified=invoke(spec,'qualify-b4','qualify',4,deadline);names.append('qualify-b4');identity=qualified['parameter_digests_before']
    selected=None;attempts=[];limit='configured_panel_ceiling'
    for batch in BATCHES:
        require(deadline-time.monotonic()>=370,'insufficient capacity/evaluation reserve')
        passed=True
        for prefix in ('smoke','confirm'):
            name=f'{prefix}-b{batch}';report=invoke(spec,name,'batch_smoke',batch,deadline,identity=identity)
            if report is None:failed.append(name);passed=False;break
            names.append(name)
        attempts.append(dict(physical_batch=batch,confirmed=passed))
        if not passed:limit='observed_capacity_failure';break
        selected=batch
    require(selected is not None,'no stable native batch')
    save(c/'capacity-selection.json',dict(accepted=True,physical_batch=selected,actual_scientific_batch=min(selected,768),scientific_rows=768,accumulation=1,optimizer_updates=0,candidates=attempts,failed_roots=failed,limit=limit,claim='Largest confirmed power-of-two candidate up to1024;768 scientific rows never duplicated. No universal hardware maximum.',created_local=now()))
    for name,control in [('factual','factual'),('uniform','uniform_attention')]:
        invoke(spec,name,'evaluate',selected,deadline,control,identity);names.append(name)
    receipt(spec,names,selected);remaining(deadline,960)
    save(c/'execution.json',dict(accepted=True,names=names,failed_capacity_roots=failed,physical_batch=selected,elapsed_seconds=time.monotonic()-start,finished_local=now()))

def analyze(spec):
    c=Path(spec['campaign']);deadline=campaign_deadline(c);require(read(c/'execution.json')['accepted'] is True,'execution incomplete')
    verify_files(spec['frozen_files']);require(clean_revision()==spec['source'],'source changed')
    cfg=c/'analysis-config.json'
    tracked(spec,[PYTHON,str(R/'analysis.py'),'--config',str(cfg),'--output',str(c/'analysis.json')],'analysis',remaining(deadline-120,60))
    tracked(spec,[PYTHON,str(R/'independent_review.py'),'--config',str(cfg),'--report',str(c/'analysis.json'),'--output',str(c/'independent-review.json')],'independent-review',remaining(deadline-60,60))
    report=read(c/'analysis.json');review=read(c/'independent-review.json')
    require(report['accepted'] is True and review['accepted'] is True and review['report_sha256']==digest(c/'analysis.json'),'native scoring/review rejected')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','execute','analyze'));p.add_argument('--campaign',type=Path,required=True);p.add_argument('--binary',type=Path);p.add_argument('--spec-sha256');a=p.parse_args()
    if a.stage=='prepare':
        require(a.binary is not None,'binary required');spec=prepare(a.campaign,a.binary);print(dict(source=spec['source'],binary_sha256=spec['binary_sha256'],spec_sha256=digest(a.campaign/'launch-spec.json')))
    else:
        require(digest(a.campaign/'launch-spec.json')==a.spec_sha256,'launch spec changed');spec=read(a.campaign/'launch-spec.json');(execute if a.stage=='execute' else analyze)(spec);print(f'C20 {a.stage} complete.')
if __name__=='__main__':main()
