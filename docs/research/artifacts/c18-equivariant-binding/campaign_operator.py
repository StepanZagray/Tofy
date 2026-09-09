"""Registered C18 paired sequencing; generic, hash-pinned C12 process supervision."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import importlib.util
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.dont_write_bytecode = True
R = Path(__file__).resolve().parent
R12 = R.parent / '2026-09-09T143629Z-tofy-looped-grounded-policy-learning'
R15 = R.parent / '2026-09-09T173927Z-tofy-looped-demonstration-grounding'
R17 = R.parent / '2026-09-09T192158Z-tofy-looped-binding-profile-amortization'
REPO = Path('/home/stepan/Projects/code/Tofy-equivariant-binding')
PYTHON = '/home/stepan/venvs/tensorboard/bin/python3'
DEPENDENCY = '1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a'
PINNED = {
    'supervise.py': '0c1eb08878efbb9f178828e04fcce74712d826cffc23d9aa2a2a363b51f151c9',
    'driver.py': '8de46c86aa3d2a95e97616cb0908026521ee536a75729934d9691c36d68c2078',
    'bind_nsight.py': '82870b0247b142533b3702d5332bca43aa10f9785c9b96a8acd3ec405ce8be5b',
}
for _name, _sha in PINNED.items():
    with (R12 / _name).open('rb') as _f:
        if hashlib.file_digest(_f, 'sha256').hexdigest() != _sha:
            raise ValueError(f'C12 lifecycle changed: {_name}')
sys.path.insert(0, str(R12))
from supervise import read, save, digest, verify_files, require, now, root_manifest
from driver import tracked
sys.path.remove(str(R12))

ARTIFACTS = ('registration.md', 'campaign_operator.py', 'operator_tests.py', 'data.py', 'data_tests.py', 'analysis.py', 'analysis_tests.py', 'independent_review.py', 'independent_review_tests.py', 'generate_datasets.py', 'seal_completed.py')
SOURCE_FILES = ('Cargo.toml', 'Cargo.lock', 'build.rs', 'src/p2/looped_agent/binding.rs', 'src/p2/looped_agent/model.rs', 'src/p2/looped_agent/mod.rs', 'src/p2/looped_agent/profile.rs', 'src/p2/optimizer.rs', 'examples/action_binding_probe.rs', 'examples/action_binding/engine.rs', 'examples/grounded_policy/evidence.rs')
COUNTS = {'fit': 1536, 'heldout': 768, 'cached_visual': 1024}

def clean_revision():
    revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    require(not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=all'], text=True).strip(), 'dirty checkout')
    subprocess.run(['git', '-C', str(REPO), 'merge-base', '--is-ancestor', 'HEAD', '@{upstream}'], check=True)
    return revision

def remaining(deadline, ceiling):
    seconds = min(ceiling, deadline - time.monotonic())
    require(seconds > 0, 'campaign budget exhausted')
    return seconds

def campaign_deadline(campaign):
    clock = read(campaign / 'clock.json')
    require(clock['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'campaign host rebooted')
    require(type(clock['started_monotonic']) in (int,float) and math.isfinite(clock['started_monotonic']) and 0 <= clock['started_monotonic'] <= time.monotonic(), 'campaign clock invalid')
    return clock['started_monotonic'] + 1800

def timing_components(report, records, supervisor):
    count = report['optimizer_updates']
    require(type(count) is int and count in (2,5) and len(records) == count, 'registered disposable count required')
    def duration(value):
        require(type(value) in (int,float) and math.isfinite(value) and value >= 0, 'invalid duration')
        return float(value)
    cumulative=[]
    for index,row in enumerate(records,1):
        require(type(row['update']) is int and row['update'] == index, 'timing update order differs')
        cumulative.append(duration(row['elapsed_seconds']))
    ordinary=[b-a for a,b in zip(cumulative,cumulative[1:])]
    require(all(x>0 for x in ordinary), 'nonmonotonic update times')
    whole=duration(report['updates_elapsed_seconds'])
    reported=duration(report['elapsed_seconds'])
    checkpoint=duration(report['checkpoint_seconds'])
    require(cumulative[-1]<=reported and checkpoint<=reported, 'timing records/checkpoint exceed model lifetime')
    selected_residual=whole-sum(ordinary)
    setup_residual=reported-whole
    wrapper_residual=duration(supervisor['model_phase_seconds'])-reported
    require(selected_residual>0 and setup_residual>=0 and wrapper_residual>=0, 'inconsistent timing boundaries')
    return dict(ordinary_seconds=max(ordinary), selected_update_residual_seconds=selected_residual, setup_and_postloop_residual_seconds=setup_residual, supervisory_residual_seconds=wrapper_residual, checkpoint_seconds=checkpoint, observed_ordinary_updates=len(ordinary))

def admission(trials, updates):
    require(len(trials)==2 and [t['report']['optimizer_updates'] for t in trials]==[2,5], 'two- and five-update trials required')
    components=[timing_components(t['report'],t['records'],t['supervisor']) for t in trials]
    maxima={key:max(row[key] for row in components) for key in components[0] if key!='observed_ordinary_updates'}
    require(type(updates) is int and updates>=3, 'invalid training count')
    ordinary_count=updates-3
    predicted=1.25*(ordinary_count*maxima['ordinary_seconds']+3*maxima['selected_update_residual_seconds']+maxima['setup_and_postloop_residual_seconds']+maxima['supervisory_residual_seconds'])+maxima['checkpoint_seconds']
    return dict(accepted=predicted<=600, reserved_training_seconds=predicted, model_limit_seconds=600, ordinary_updates=ordinary_count, selected_updates=3, reserve_multiplier=1.25, measured_components=components, componentwise_maxima=maxima, interpretation='Empirical stage/cadence forecast, not a runtime upper-bound theorem; selected residual includes cold work and finalization, not isolated kernel time.')

def validate_gradients(metrics):
    for key in ('input_gradient_norm','shared_core_gradient_norm','body_gradient_norm','head_gradient_norm','pre_clip_norm','clip_scale'):
        value=metrics[key]
        require(type(value) in (int,float) and math.isfinite(value) and value>0,'invalid gradient family/clip')
    require(metrics['clip_scale']<=1 and type(metrics['mean_ce']) in (int,float) and math.isfinite(metrics['mean_ce']),'invalid loss/clipping')

def timing_trial(campaign,name,report):
    import json
    with (campaign/name/'updates.jsonl').open() as source: records=[json.loads(line) for line in source]
    batch=report['physical_batch']
    require(batch==report['effective_batch'] and report['accumulation']==1 and report['input_rows']==report['optimizer_updates']*batch,'smoke must execute full candidate updates')
    require(len(records)==report['optimizer_updates'],'smoke update stream incomplete')
    for index,row in enumerate(records,1):
        m=row['metrics']
        require(row['update']==index and m['rows']==batch and m['requested_physical_batch']==batch and m['physical_batch']==batch and m['microbatches']==1 and m['tail_batch']==batch,'smoke actual batch differs')
        validate_gradients(m)
    return dict(report=report, records=records, supervisor=read(campaign/f'{name}.exit.json'))

KINDS = ('legacy','equivariant')
PARAMETERS = {'legacy':1580804,'equivariant':1579265}
BATCHES = (512,1024,2048,4096,8192,16384,32768)
LEGACY_INITIAL = '55d6d89a7e8a22049d074ae828cabb2f114093ec88886df67744a453c0cd364e'
PRESENTATIONS = 588800
SCHEDULE_SHA = 'a46eac4347bfe16d55d220240452cb67643ccd007a97d2e56da16be45cb6f51e'

def dataset_path(batch, mode, updates=0):
    suffix = f'smoke-{updates}' if mode=='batch_smoke' else 'training'
    return R/'datasets'/f'b{batch}-{suffix}.json'

def capture_updates(count):
    require(count>=4,'insufficient training budget for selected captures')
    return [2,count//2,count]

def prepare(campaign,binary):
    require(campaign.is_absolute() and not campaign.exists() and '.' not in campaign.name,'new absolute dotless campaign required')
    revision=clean_revision()
    build=read(R/'build.json')
    require(build['source_revision']==revision and build['binary_sha256']==digest(binary),'build source/binary differs')
    require(read(R/'operations/build-cuda.exit.json')['accepted'] is True and read(R/'operations/build-cuda.process.json')['command']==build['command'],'build execution not accepted')
    require(build['features']=='cudnn,profiling,serde_json/float_roundtrip' and build['features'] in build['command'],'build backend features differ')
    files=dict(read(R/'source-freeze.json')['files'])
    for name in ARTIFACTS:
        p=R/name
        require(digest(p)==digest(REPO/'docs/research/artifacts/c18-equivariant-binding'/name),'operator snapshot differs')
        files[str(p)]=digest(p)
    for name in SOURCE_FILES:
        p=REPO/name;files[str(p)]=digest(p)
    for p in [R/'source-freeze.json',R/'build.json',*[R/'operations'/f'build-cuda.{s}' for s in ('process.json','exit.json','stdout.log')]]:
        files[str(p)]=digest(p)
    require(digest(R17/'completed-campaign.manifest.json')=='61378b3e4ad96573aacbdbaa1fbac9619962b586895f5e49b1eae9d8431ab2a6','C17 seal changed')
    parent=read(R17/'completed-campaign.manifest.json')
    for rel,row in parent['files'].items():
        p=Path(parent['campaign'])/rel
        require(p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],'parent inventory changed')
    for batch in BATCHES:
        for mode,count in (('train',0),('batch_smoke',2),('batch_smoke',5)):
            p=dataset_path(batch,mode,count);require(str(p) in files and files[str(p)]==digest(p),'dataset not frozen')
    verify_files(files)
    campaign.mkdir();(campaign/'invocations').mkdir()
    shutil.copy2(binary,campaign/'action_binding_probe')
    files[str(campaign/'action_binding_probe')]=digest(campaign/'action_binding_probe')
    spec=dict(schema='looped-action-equivariance-launch-v1',campaign=str(campaign),repository=str(REPO),source=revision,binary=str(campaign/'action_binding_probe'),binary_sha256=digest(campaign/'action_binding_probe'),dependency_revision=DEPENDENCY,frozen_files=files,created_local=now())
    save(campaign/'launch-spec.json',spec)
    return spec

def healthy_capture(campaign,name,count):
    rows=read(campaign/f'{name}.bound/summary.json')
    require(len(rows)==count and all(x['health']['structurally_valid'] and x['health']['capture_complete'] and x['raw_application_labels_verified'] and x['gpu']['status']=='available' and x['gpu']['provenance_binding']=='bound' for x in rows),'incomplete bound capture')

def invoke(spec,name,kind,deadline,mode,batch,updates=0,cohort=None,loops=4,cleared=False,query_cleared=False):
    require(kind in KINDS,'unknown architecture')
    campaign=Path(spec['campaign']);verify_files(spec['frozen_files'])
    datafile=dataset_path(batch,mode,updates)
    data=read(datafile);schedule=data['schedule']
    if mode=='train': updates=len(data['updates'])
    physical=4 if mode=='qualify' else batch
    captures=capture_updates(updates) if mode=='train' else [1] if mode=='batch_smoke' else []
    cfg=dict(schema='looped-action-binding-config-v2',source_revision=spec['source'],registration=str(R/'registration.md'),registration_sha256=digest(R/'registration.md'),dataset=str(datafile),dataset_sha256=digest(datafile),mode=mode,model_kind=kind,output_dir=str(campaign/name),physical_batch=physical,effective_batch=batch,schedule_presentations=schedule['presentations'],schedule_sha256=schedule['indices_sha256'],profile_updates=captures,updates=updates,max_seconds=600 if mode=='train' else 120,checkpoint=None,checkpoint_sha256=None,cohort=cohort,loops=loops,cleared=cleared,query_cleared=query_cleared)
    require(remaining(deadline,1800)>=cfg['max_seconds']+130,'insufficient remaining invocation window')
    frozen=dict(spec['frozen_files'])
    if mode=='eval_final':
        checkpoint=campaign/f'{kind}-train-seed0/final.safetensors'
        require(read(campaign/f'{kind}-train-seed0.exit.json')['accepted'] is True,'training incomplete')
        cfg.update(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint))
        frozen[str(checkpoint)]=digest(checkpoint)
        frozen[str(checkpoint.parent/'manifest.json')]=root_manifest(checkpoint.parent)[0]
    cfgpath=campaign/'invocations'/f'{name}.json';save(cfgpath,cfg);frozen[str(cfgpath)]=digest(cfgpath)
    input_rows=schedule['presentations'] if mode in ('train','batch_smoke') else 4 if mode=='qualify' else COUNTS[cohort]
    expected=dict(status='complete_pending_analysis',optimizer_updates=updates,physical_batch=physical,input_rows=input_rows,loops=loops,cleared=cleared,query_cleared=query_cleared,executed_vision_core_forwards=0,parameter_count=PARAMETERS[kind],model_kind=kind)
    authpath=campaign/'invocations'/f'{name}-authority.json'
    save(authpath,dict(schema='looped-grounded-policy-launch-v1',accepted=True,created_local=now(),config_sha256=digest(cfgpath),mode=mode,campaign=str(campaign),name=name,binary=spec['binary'],binary_sha256=spec['binary_sha256'],repository=spec['repository'],source=spec['source'],frozen_files=frozen,expected_report=expected))
    operation=tracked(spec,[PYTHON,str(R12/'supervise.py'),'--config',str(cfgpath),'--sha256',digest(cfgpath),'--authority',str(authpath),'--authority-sha256',digest(authpath)],name+'-model',remaining(deadline,cfg['max_seconds']+130),allow_failure=mode=='batch_smoke')
    state=read(campaign/f'{name}.exit.json')
    if not state['accepted']:
        require(mode=='batch_smoke' and state['capacity_failure'] is True,'non-capacity failure stops campaign')
        require(operation['pid_gone'] and operation['group_gone'] and not operation['owned_survivors'] and not operation['group_survivors'] and operation['cleanup_error'] is None,'capacity trial cleanup incomplete')
        print(f'Capacity limit at {name}',flush=True)
        return None
    require(operation['accepted'] is True,'outer lifecycle rejected invocation')
    tracked(spec,[PYTHON,str(R12/'bind_nsight.py'),'--root',str(campaign/name)],name+'-bind',remaining(deadline,180))
    healthy_capture(campaign,name,len(captures) if mode in ('train','batch_smoke') else 1)
    require(root_manifest(campaign/name)[0]==state['manifest_sha256'],'model manifest differs')
    report=read(campaign/name/'report.json')
    if mode in ('qualify','eval_initial','eval_final'):
        require(report['changes']['all_parameters_unchanged'] is True and report['starting_parameter_sha256']==report['evaluated_parameter_sha256'],'evaluation changed weights')
    if mode=='batch_smoke':require(report['restored_changes']['all_parameters_unchanged'] is True,'smoke restoration failed')
    if kind=='legacy':require(report['initial_parameter_sha256']==LEGACY_INITIAL,'legacy initialization changed')
    print(f'Completed {name}',flush=True)
    return report

def training_integrity(campaign,kind,data,batch):
    root=campaign/f'{kind}-train-seed0';report=read(root/'report.json');schedule=data['schedule']
    require(report['optimizer_updates']==len(data['updates']) and report['input_rows']==PRESENTATIONS and report['effective_batch']==batch and report['physical_batch']==batch and report['accumulation']==1,'training schedule differs')
    require(schedule['indices_sha256']==SCHEDULE_SHA and schedule['presentations']==PRESENTATIONS,'original example stream differs')
    require(not report['changes']['all_parameters_unchanged'] and report['changes']['changed_body_names'] and report['changes']['changed_head_names'],'training did not update body/head')
    import json
    with (root/'updates.jsonl').open() as source:records=[json.loads(line) for line in source]
    require(len(records)==len(data['updates']),'update count differs')
    for i,(record,indices) in enumerate(zip(records,data['updates']),1):
        m=record['metrics'];actual=min(batch,len(indices))
        require(record['update']==i and m['rows']==len(indices) and m['requested_physical_batch']==batch and m['physical_batch']==actual and m['microbatches']==1,'actual update batch differs')
        validate_gradients(m)
    consumed=0
    with (root/'training-stream.jsonl').open() as source:
        for update,indices in enumerate(data['updates'],1):
            for slot,index in enumerate(indices):
                line=source.readline();require(bool(line),'training stream truncated')
                row=json.loads(line)
                require(row==dict(update=update,slot=slot,dataset_index=index,row=data['fit'][index]),'training presentation differs')
                consumed+=1
        require(not source.readline(),'excess training rows')
    require(consumed==PRESENTATIONS,'consumed example budget differs')
    return report

def receipt(spec,kind,names,streams,batch):
    campaign=Path(spec['campaign']);datafile=dataset_path(batch,'train');data=read(datafile)
    files=dict(spec['frozen_files']);train=training_integrity(campaign,kind,data,batch)
    checkpoints={stage:dict(sha256=digest(campaign/f'{kind}-train-seed0'/f'{stage}.safetensors'),parameter_sha256=train[f'{stage}_parameter_sha256']) for stage in ('initial','final')}
    for name in names:
        root=campaign/name;state=read(campaign/f'{name}.exit.json')
        require(state['accepted'] and state['bindings_unchanged'],'invocation integrity failed')
        require(root_manifest(root)[0]==state['manifest_sha256'],'manifest changed')
        healthy_capture(campaign,name,3 if name==f'{kind}-train-seed0' else 1)
        report=read(root/'report.json')
        require(report['initial_parameter_sha256']==train['initial_parameter_sha256'],'initializer differs')
        for suffix in ('model','bind'):
            op=read(campaign/'operations'/f'{name}-{suffix}.exit.json')
            require(op['accepted'] and op['pid_gone'] and op['group_gone'] and not op['owned_survivors'] and not op['group_survivors'] and op['cleanup_error'] is None,'operation cleanup incomplete')
        for p in [root.with_suffix('.exit.json'),root.with_suffix('.process.json'),root/'manifest.json',root/'report.json',*[p for suffix in ('.bound','.nsight') for p in root.with_suffix(suffix).rglob('*') if p.is_file()]]: files[str(p)]=digest(p)
        short=name.removeprefix(kind+'-')
        if short in streams:
            stream=streams[short]
            require(report['starting_parameter_sha256']==checkpoints[stream['stage']]['parameter_sha256'] and report['evaluated_parameter_sha256']==report['starting_parameter_sha256'] and report['changes']['all_parameters_unchanged'],'evaluated checkpoint differs')
            p=root/'evaluation-rows.jsonl'
            stream.update(rows=str(p),sha256=digest(p),checkpoint_sha256=checkpoints[stream['stage']]['sha256'],parameter_sha256=checkpoints[stream['stage']]['parameter_sha256']);files[str(p)]=digest(p)
    for name in ('initial.safetensors','final.safetensors','training-stream.jsonl','updates.jsonl'):
        p=campaign/f'{kind}-train-seed0'/name;files[str(p)]=digest(p)
    for name in ('capacity-selection.json','admission.json','clock.json','launch-spec.json','initial-equivariance.json'):
        p=campaign/name;files[str(p)]=digest(p)
    require(read(campaign/'admission.json')['accepted'],'training admission failed')
    verify_files(files)
    path=campaign/f'{kind}-integrity.json'
    completed_schedule=dict(updates=len(data['updates']),effective_batch=batch,presentations=PRESENTATIONS,indices_sha256=SCHEDULE_SHA,tail_rows=len(data['updates'][-1]) if len(data['updates'][-1])<batch else 0)
    save(path,dict(schema='looped-action-binding-integrity-v2',accepted=True,checks={k:True for k in ('data','source','build','initialization','completed_training','gradients','device','profiles','checkpoints','cleanup')},model_kind=kind,effective_batch=batch,completed_schedule=completed_schedule,source_revision=spec['source'],binary_sha256=spec['binary_sha256'],dependency_revision=DEPENDENCY,dataset_sha256=digest(datafile),registration_sha256=digest(R/'registration.md'),checkpoints=checkpoints,frozen_files=files,streams=streams,physical_batch=batch,accumulation=1,created_local=now()))
    files[str(path)]=digest(path)
    selected={name:{key:s[key] for key in ('rows','sha256','cohort','stage','loops','cleared','query_cleared')} for name,s in streams.items()}
    save(campaign/f'{kind}-analysis-config.json',dict(schema='looped-action-binding-analysis-config-v2',model_kind=kind,effective_batch=batch,dataset=str(datafile),dataset_sha256=digest(datafile),registration=dict(path=str(R/'registration.md'),sha256=digest(R/'registration.md')),streams=selected,integrity_receipt=str(path),integrity_receipt_sha256=digest(path),frozen_files=files))

def initial_check(spec,kind,batch):
    campaign=Path(spec['campaign']);datafile=dataset_path(batch,'train')
    selected={cohort:campaign/f'{kind}-initial-{cohort}-l4/evaluation-rows.jsonl' for cohort in ('fit','heldout')}
    reports={cohort:read(p.parent/'report.json') for cohort,p in selected.items()}
    qualification=read(campaign/f'{kind}-qualify-b4/report.json')
    files=dict(spec['frozen_files'])
    files[str(datafile)]=digest(datafile)
    for p in selected.values():
        files[str(p)]=digest(p)
        for q in (p.parent/'report.json',p.parent/'manifest.json',p.parent.with_suffix('.exit.json')):files[str(q)]=digest(q)
    for report in reports.values():
        require(report['initial_parameter_sha256']==qualification['initial_parameter_sha256'] and report['starting_parameter_sha256']==report['initial_parameter_sha256'] and report['changes']['all_parameters_unchanged'],'initial checkpoint identity differs')
    binding=dict(schema='looped-action-binding-initial-check-v2',dataset_sha256=digest(datafile),fit_sha256=digest(selected['fit']),heldout_sha256=digest(selected['heldout']),model_kind=kind,effective_batch=batch,source_revision=spec['source'],binary_sha256=spec['binary_sha256'],source_verified=True,initialization_verified=True,frozen_files=files)
    path=campaign/f'{kind}-initial-check-config.json';save(path,binding)
    module_spec=importlib.util.spec_from_file_location('c18_initial_scoring',R/'analysis.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    result=module.initial_check(datafile,selected['fit'],selected['heldout'],binding)
    require(result['accepted'] is True,'initial check integrity failed')
    if kind=='equivariant':require(result['passed'] is True,'numerical equivariance failed before training')
    return result

def execute(spec):
    start=time.monotonic();deadline=start+1500;campaign=Path(spec['campaign'])
    verify_files(spec['frozen_files']);require(clean_revision()==spec['source'],'source changed')
    save(campaign/'clock.json',dict(started_local=now(),started_monotonic=start,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),maximum_wall_seconds=1800,analysis_and_seal_reserve_seconds=300))
    names={kind:[] for kind in KINDS};streams={kind:{} for kind in KINDS};failed=[]
    def run(kind,short,mode,batch,**kwargs):
        name=kind+'-'+short
        result=invoke(spec,name,kind,deadline,mode,batch,**kwargs)
        if result is not None:
            names[kind].append(name)
            if mode.startswith('eval_'):
                streams[kind][short]=dict(cohort=kwargs['cohort'],stage='initial' if mode=='eval_initial' else 'final',loops=kwargs.get('loops',4),cleared=kwargs.get('cleared',False),query_cleared=kwargs.get('query_cleared',False))
        else: failed.append(name)
        return result
    qualified={kind:run(kind,'qualify-b4','qualify',512) for kind in KINDS}
    require(qualified['legacy']['initial_shared_core_parameter_sha256']==qualified['equivariant']['initial_shared_core_parameter_sha256'],'paired shared core initialization differs')
    selected=None;chosen_trials=None;attempted=[];limit='configured_candidate_ceiling'
    for batch in BATCHES:
        require(deadline-time.monotonic()>=1050,'qualification consumed reserved training/evaluation window')
        trials={};success=True
        for kind in KINDS:
            trials[kind]=[]
            for count,prefix in ((2,'smoke'),(5,'confirm')):
                report=run(kind,f'{prefix}-b{batch}','batch_smoke',batch,updates=count)
                if report is None:success=False;break
                require(report['initial_parameter_sha256']==qualified[kind]['initial_parameter_sha256'],'smoke initializer differs')
                trials[kind].append(timing_trial(campaign,f'{kind}-{prefix}-b{batch}',report))
            if not success:break
        attempted.append(dict(batch=batch,joint_stable=success))
        if not success:limit='observed_capacity_failure';break
        selected=batch;chosen_trials=trials
    require(selected is not None,'no jointly stable batch')
    save(campaign/'capacity-selection.json',dict(accepted=True,physical_batch=selected,effective_batch=selected,accumulation=1,candidates=attempted,failed_roots=failed,limit=limit,claim='Largest tested stable power-of-two physical batch for both paired architectures; not a universal allocator peak.',created_local=now()))
    data=read(dataset_path(selected,'train'));updates=len(data['updates'])
    admissions={kind:admission(chosen_trials[kind],updates) for kind in KINDS}
    save(campaign/'admission.json',dict(accepted=all(x['accepted'] for x in admissions.values()),arms=admissions,physical_batch=selected,effective_batch=selected,accumulation=1,optimizer_updates=updates,presentations=PRESENTATIONS,tail_rows=len(data['updates'][-1]) if len(data['updates'][-1])<selected else 0,profile_updates=capture_updates(updates),created_local=now()))
    require(all(x['accepted'] for x in admissions.values()),'registered training budget cannot be met')
    checks={}
    for kind in KINDS:
        for cohort in ('fit','heldout'):run(kind,f'initial-{cohort}-l4','eval_initial',selected,cohort=cohort)
        checks[kind]=initial_check(spec,kind,selected)
    save(campaign/'initial-equivariance.json',dict(accepted=True,arms=checks,created_local=now()))
    for kind in KINDS:run(kind,'train-seed0','train',selected)
    for kind in KINDS:
        for loops in (1,2,4,8):
            for cohort in ('fit','heldout'):run(kind,f'final-{cohort}-l{loops}','eval_final',selected,cohort=cohort,loops=loops)
        for cohort in ('fit','heldout'):
            run(kind,f'final-{cohort}-effects-zero','eval_final',selected,cohort=cohort,cleared=True)
            run(kind,f'final-{cohort}-query-zero','eval_final',selected,cohort=cohort,query_cleared=True)
        run(kind,'final-cached-visual-l4','eval_final',selected,cohort='cached_visual')
        receipt(spec,kind,names[kind],streams[kind],selected)
    require(read(campaign/'legacy-train-seed0/report.json')['initial_shared_core_parameter_sha256']==read(campaign/'equivariant-train-seed0/report.json')['initial_shared_core_parameter_sha256'],'paired train initialization differs')
    remaining(deadline,1800)
    save(campaign/'execution.json',dict(accepted=True,elapsed_seconds=time.monotonic()-start,names=names,failed_capacity_roots=failed,physical_batch=selected,accumulation=1,finished_local=now()))

def analyze(spec):
    campaign=Path(spec['campaign']);deadline=campaign_deadline(campaign)
    require(read(campaign/'execution.json')['accepted'] is True,'execution incomplete')
    verify_files(spec['frozen_files']);require(clean_revision()==spec['source'],'source changed')
    stages=[]
    for kind in KINDS:
        config=campaign/f'{kind}-analysis-config.json';report=campaign/f'{kind}-analysis.json'
        stages.append((kind+'-analysis',[PYTHON,str(R/'analysis.py'),'--config',str(config),'--output',str(report)]))
        stages.append((kind+'-independent-review',[PYTHON,str(R/'independent_review.py'),'--config',str(config),'--report',str(report),'--output',str(campaign/f'{kind}-independent-review.json')]))
    for index,(name,command) in enumerate(stages):
        tracked(spec,command,name,remaining(deadline-60*(4-index),60))
    reports={kind:read(campaign/f'{kind}-analysis.json') for kind in KINDS}
    verified={kind:read(campaign/f'{kind}-independent-review.json') for kind in KINDS}
    for kind in KINDS:
        require(reports[kind]['accepted'] is True and verified[kind]['accepted'] is True and verified[kind]['report_sha256']==digest(campaign/f'{kind}-analysis.json'),'analysis/review not accepted')
    require(reports['legacy']['dataset_sha256']==reports['equivariant']['dataset_sha256'],'paired scored data differs')
    differences={}
    for name in reports['legacy']['summaries']:
        left=reports['legacy']['summaries'][name]['canonical'];right=reports['equivariant']['summaries'][name]['canonical']
        require(left['rows']==right['rows'],'comparison populations differ')
        differences[name]=dict(rows=left['rows'],legacy_correct=left['correct'],equivariant_correct=right['correct'],correct_difference=right['correct']-left['correct'],accuracy_difference=right['accuracy']-left['accuracy'],ce_difference=right['ce']-left['ce'])
    save(campaign/'comparison.json',dict(schema='looped-action-equivariance-pair-v1',accepted=True,decisions={kind:r['decision'] for kind,r in reports.items()},differences=differences,analysis_sha256={kind:digest(campaign/f'{kind}-analysis.json') for kind in KINDS},limits='One fixed-seed matched representation comparison. Same larger joint batch and588800 presentations; fewer optimizer updates than historical C17. Action-renaming transfer is imposed symmetry coverage across16orbit types, not independent semantic/ARC generalization. No production timing claim.',created_local=now()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','execute','analyze'));parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--binary',type=Path);parser.add_argument('--spec-sha256');args=parser.parse_args()
    if args.stage=='prepare':
        require(args.binary is not None,'binary required');spec=prepare(args.campaign,args.binary)
        print(dict(source=spec['source'],binary_sha256=spec['binary_sha256'],spec_sha256=digest(args.campaign/'launch-spec.json')))
    else:
        require(digest(args.campaign/'launch-spec.json')==args.spec_sha256,'launch specification differs');spec=read(args.campaign/'launch-spec.json');(execute if args.stage=='execute' else analyze)(spec)
        print(f'C18 {args.stage} complete.')

if __name__=='__main__':main()
