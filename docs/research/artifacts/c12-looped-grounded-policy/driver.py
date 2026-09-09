#!/usr/bin/env python3
"""Closed C12 stage sequencing; no outcome-driven recipe/checkpoint selection."""
import argparse,datetime,importlib.util,json,math,os,signal,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
from supervise import require,read,save,digest,verify_files,now,root_manifest
R=Path(__file__).resolve().parent
PYTHON='/home/stepan/venvs/tensorboard/bin/python3'
CPU=Path('/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/supervise_cpu.py')
CPU_SHA='6796e12fb06a87f713f76677baf543758943716287e8b4a9f2ef68a2ec7a0cd7'
C8=Path('/home/stepan/Projects/code/.tofy-runs/looped-frozen-features-20260909T115455-IST')
C11=Path('/home/stepan/Projects/code/.tofy-runs/looped-cuda-readout-retry-20260909T151301-IST')

def duration(x):
    require(type(x) in (int,float) and math.isfinite(x) and x>=0,'invalid elapsed duration');return float(x)
def admission(report,audit):
    require(report['optimizer_updates']==5,'five-update confirmation required')
    prediction=1150*duration(report['updates_elapsed_seconds'])/5
    extra=duration(audit['training_generation_and_audit_seconds'])+duration(report['checkpoint_seconds'])
    accepted=1.15*prediction<=3600 and extra<=.15*prediction
    return dict(accepted=accepted,predicted_update_seconds=prediction,reserved_training_seconds=1.15*prediction,generator_and_checkpoint_allowance_seconds=extra,reserve_seconds=.15*prediction)
def clean_operation(state):
    return state['returncode']==0 and state['error'] is None and state['cleanup_error'] is None and state['pid_gone'] and state['group_gone'] and not state['owned_survivors'] and not state['group_survivors']
def tracked(spec,command,name,timeout,allow_failure=False):
    require(timeout>0,'operation budget exhausted');verify_files({str(CPU):CPU_SHA})
    module=importlib.util.spec_from_file_location('c12_outer',CPU);life=importlib.util.module_from_spec(module);module.loader.exec_module(life)
    signal.signal(signal.SIGINT,life.stop_signal);signal.signal(signal.SIGTERM,life.stop_signal)
    root=Path(spec['campaign'])/'operations'/name;root.parent.mkdir(exist_ok=True)
    require(not list(root.parent.glob(name+'.*')),'operation root reused')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',NVIDIA_TF32_OVERRIDE='0')
    def record(pid,start):
        save(root.with_suffix('.process.json'),dict(pid=pid,supervisor_pid=os.getpid(),process_start_ticks=start,command=command,cwd=spec['repository'],timeout_seconds=timeout,started_local=now()))
        print(json.dumps(dict(operation=name,pid=pid,supervisor_pid=os.getpid(),started_local=now())),flush=True)
    with root.with_suffix('.stdout.log').open('x') as output:state=life.supervise(command,Path(spec['repository']),env,output,record,timeout=timeout)
    state.update(accepted=clean_operation(state),finished_local=now());save(root.with_suffix('.exit.json'),state)
    if not allow_failure:require(state['accepted'],f'operation failed: {root}')
    return state

def budgets(spec):
    c=Path(spec['campaign']);clock=c/'clock.json'
    wall=0 if not clock.exists() else (datetime.datetime.now().astimezone()-datetime.datetime.fromisoformat(read(clock)['started_local'])).total_seconds()
    totals=dict(qualification=0.,training=0.,evaluation=0.,wall=wall)
    for path in c.glob('*.exit.json'):
        state=read(path);config=read(c/'invocations'/f'{path.name[:-10]}.json')
        mode=config['mode'];key='qualification' if mode in ('qualify','batch_smoke') else 'training' if mode=='train' else 'evaluation' if mode.startswith('eval_') else None
        if key:totals[key]+=duration(state['model_phase_seconds'])
    require(0<=wall<=5400 and totals['qualification']<=600 and totals['training']<=3600 and totals['evaluation']<=600,'campaign phase budget exhausted')
    return totals

def completed_stage(spec,stage):
    c=Path(spec['campaign']);path=c/f'{stage}-stage.json'
    require(digest(path)==(c/f'{stage}-stage.sha256').read_text().strip(),'predecessor stage pin differs')
    value=read(path);require(value['accepted'] is True and value['source']==spec['source'],'predecessor stage not accepted')
    verify_files(value['frozen_files'])
    for name in value['names']:
        state=read(c/f'{name}.exit.json')
        require(state['accepted'] is True and root_manifest(c/name)[0]==state['manifest_sha256'],'predecessor root differs')
        if name!='audit':
            operation=read(c/'operations'/f'{name}-bind.exit.json');require(operation['accepted'] is True,'predecessor binder not accepted')
            bundles=read(c/f'{name}.bound/summary.json')
            require(len(bundles)==state['captures_expected'] and all(x['health']['structurally_valid'] and x['health']['capture_complete'] and x['raw_application_labels_verified'] for x in bundles),'predecessor profile incomplete')
    return value

def seal_stage(spec,stage,names):
    c=Path(spec['campaign']);files={}
    for name in names:
        state=read(c/f'{name}.exit.json');require(state['accepted'] is True,'stage contains failed invocation')
        require(root_manifest(c/name)[0]==state['manifest_sha256'],'stage root differs')
        files.update({str(c/f'{name}.exit.json'):digest(c/f'{name}.exit.json'),str(c/name/'manifest.json'):state['manifest_sha256']})
        if name!='audit':
            operation=c/'operations'/f'{name}-bind.exit.json';require(read(operation)['accepted'] is True,'stage binder failed')
            files[str(operation)]=digest(operation)
            for artifact in (c/f'{name}.bound').rglob('*'):
                if artifact.is_file():files[str(artifact)]=digest(artifact)
    if stage=='qualification':
        for name in ('qualification-analysis.json','parity-analysis.json'):files[str(c/name)]=digest(c/name)
    path=c/f'{stage}-stage.json';save(path,dict(accepted=True,created_local=now(),source=spec['source'],names=names,budgets=budgets(spec),frozen_files=files))
    with (c/f'{stage}-stage.sha256').open('x') as out:out.write(digest(path)+'\n')

def config_for(spec,name,mode,batch=1,updates=0,cohort=None,cleared=False):
    c=Path(spec['campaign']);limits=budgets(spec)
    key='qualification' if mode in ('qualify','batch_smoke') else 'training' if mode=='train' else 'evaluation' if mode.startswith('eval_') else None
    ceiling=120 if mode in ('audit','qualify') else 600 if mode=='batch_smoke' or mode.startswith('eval_') else 3600
    remaining=ceiling if key is None else min(ceiling,dict(qualification=600,training=3600,evaluation=600)[key]-limits[key])
    remaining=min(remaining,5400-limits['wall']-150);require(remaining>=1,'insufficient stage budget')
    cfg=dict(schema='looped-grounded-policy-config-v1',source_revision=spec['source'],registration=str(R/'registration.md'),registration_sha256=spec['registration_sha256'],mode=mode,output_dir=str(c/name),physical_batch=batch,updates=updates,max_seconds=math.floor(remaining),core_checkpoint=str(C8/'features-initial'/'initial.safetensors'),core_sha256=spec['initial_core_sha256'],head_checkpoint=str(C11/'imports/initial/c10_true/parameters.f32'),head_sha256=spec['initial_head_sha256'],import_manifest=str(C11/'imports/initial/c10_true/manifest.json'),audit_root=None,audit_manifest_sha256=None,history=spec['history'] if mode=='audit' else [],cohort=cohort,cleared=cleared)
    if mode=='train' or mode.startswith('eval_'):
        audit=read(c/'audit.exit.json');require(audit['accepted'] is True,'audit not accepted');cfg.update(audit_root=str(c/'audit'),audit_manifest_sha256=audit['manifest_sha256'])
    if mode=='eval_final':
        state=read(c/'train-seed0.exit.json');require(state['accepted'] is True,'training not accepted')
        report=read(c/'train-seed0/report.json');require(report['optimizer_updates']==1150,'terminal checkpoint required')
        cfg.update(core_checkpoint=str(c/'train-seed0/final-core.safetensors'),core_sha256=report['final_core_sha256'],head_checkpoint=str(c/'train-seed0/final-head.safetensors'),head_sha256=report['final_head_sha256'])
    return cfg

def launch(spec,name,mode,batch=1,updates=0,cohort=None,cleared=False):
    c=Path(spec['campaign']);verify_files(spec['frozen_files']);cfg=config_for(spec,name,mode,batch,updates,cohort,cleared)
    if mode=='train':require(read(c/'qualification-analysis.json')['reserved_training_seconds']<=cfg['max_seconds'],'available training window below registered admitted projection')
    path=c/'invocations'/f'{name}.json';path.parent.mkdir(exist_ok=True);save(path,cfg)
    frozen=dict(spec['frozen_files']);frozen.update({str(path):digest(path),cfg['core_checkpoint']:cfg['core_sha256'],cfg['head_checkpoint']:cfg['head_sha256']})
    if cfg['audit_root']:frozen[str(c/'audit/manifest.json')]=cfg['audit_manifest_sha256']
    if mode=='eval_final':
        frozen[str(c/'train-seed0/manifest.json')]=read(c/'train-seed0.exit.json')['manifest_sha256']
        frozen[str(c/'train-stage.json')]=digest(c/'train-stage.json')
    expected=dict(status='complete_pending_analysis',optimizer_updates=updates)
    if mode!='audit':expected['physical_batch']=batch
    authority=dict(schema='looped-grounded-policy-launch-v1',accepted=True,created_local=now(),config_sha256=digest(path),mode=mode,campaign=str(c),name=name,binary=str(c/'grounded_policy_probe'),binary_sha256=spec['binary_sha256'],repository=spec['repository'],source=spec['source'],frozen_files=frozen,expected_report=expected)
    auth=c/'invocations'/f'{name}-authority.json';save(auth,authority)
    operation=tracked(spec,[PYTHON,'-B',str(R/'supervise.py'),'--config',str(path),'--sha256',digest(path),'--authority',str(auth),'--authority-sha256',digest(auth)],name+'-supervisor',min(cfg['max_seconds']+130,5400-budgets(spec)['wall']),allow_failure=mode=='batch_smoke')
    state=read(c/f'{name}.exit.json');budgets(spec)
    if state['accepted']:
        require(operation['accepted'],'outer lifecycle rejected successful invocation')
        if mode!='audit':
            tracked(spec,[PYTHON,'-B',str(R/'bind_nsight.py'),'--root',str(c/name)],name+'-bind',min(120,5400-budgets(spec)['wall']))
            bundles=read(c/f'{name}.bound/summary.json');require(len(bundles)==(3 if mode=='train' else 1),'bundle count differs')
            require(all(x['health']['structurally_valid'] and x['health']['capture_complete'] and x['raw_application_labels_verified'] for x in bundles),'unhealthy profile bundle')
    else:
        require(mode=='batch_smoke' and state['capacity_failure'] is True,'non-capacity failure stops campaign')
        require(operation['pid_gone'] and operation['group_gone'] and not operation['owned_survivors'] and operation['cleanup_error'] is None,'failed trial outer cleanup incomplete')
    print(json.dumps(dict(completed=name,accepted=state['accepted'],budgets=budgets(spec))),flush=True)
    return state

def verify_reference_bindings(spec):
    references_required=[C11/'qual-initial-c10_true'/name for name in ('logits.f32','attention.f32','pooled.f32')]+[C8/'features-initial'/name for name in ('known-features-current.f32','known-features-cls.f32')]
    require(all(str(p) in spec['frozen_files'] for p in references_required),'mandatory numerical references missing from source freeze')
    verify_files({str(p):spec['frozen_files'][str(p)] for p in references_required})

def parity(spec):
    verify_reference_bindings(spec)
    import numpy as np
    c=Path(spec['campaign']);references={
        'logits':np.fromfile(C11/'qual-initial-c10_true/logits.f32',dtype='<f4').reshape(512,4)[:4],
        'attention':np.fromfile(C11/'qual-initial-c10_true/attention.f32',dtype='<f4').reshape(512,2,64)[:4],
        'pooled':np.fromfile(C11/'qual-initial-c10_true/pooled.f32',dtype='<f4').reshape(512,256)[:4],
        'current':np.fromfile(C8/'features-initial/known-features-current.f32',dtype='<f4').reshape(768,64,128)[:4],
        'cls':np.fromfile(C8/'features-initial/known-features-cls.f32',dtype='<f4').reshape(768,128)[:4]}
    checks={}
    for batch in (1,4):
        rows=[readline for readline in map(json.loads,(c/f'parity-b{batch}/qualification-rows.jsonl').read_text().splitlines())]
        require([r['index'] for r in rows]==list(range(4)),'qualification indices differ')
        for key,ref in references.items():
            actual=np.array([r[key] for r in rows]);require(actual.shape==ref.shape and np.isfinite(actual).all(),'invalid qualification tensor')
            error=np.abs(actual-ref);atol=1e-5 if key=='attention' else 1e-4
            require(np.all(error<=atol+1e-5*np.abs(ref)),f'CUDA translation differs {batch}/{key}')
            if key in ('logits','attention'):require(np.array_equal(actual.argmax(axis=-1),ref.argmax(axis=-1)),'argmax translation differs')
            checks[f'{batch}/{key}']=dict(max_absolute_error=float(error.max()),atol=atol,rtol=1e-5)
    save(c/'parity-analysis.json',dict(accepted=True,checks=checks,created_local=now()))

def qualify(spec):
    c=Path(spec['campaign']);completed_stage(spec,'audit')
    save(c/'clock.json',dict(started_local=now(),pid=os.getpid(),source=spec['source']))
    for batch in (1,4):launch(spec,f'parity-b{batch}','qualify',batch)
    def deadline_signal(_number,_frame):raise TimeoutError('registered parity CPU budget exceeded')
    previous=signal.signal(signal.SIGALRM,deadline_signal)
    parity_seconds=min(120,5400-budgets(spec)['wall']);require(parity_seconds>0,'parity wall deadline exhausted')
    signal.setitimer(signal.ITIMER_REAL,parity_seconds)
    try:parity(spec)
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    tested={};low,high=0,65
    batch=64
    while True:
        state=launch(spec,f'batch-{batch}-u2','batch_smoke',batch,2);tested[batch]=state['accepted']
        if state['accepted']:low=max(low,batch)
        else:high=min(high,batch)
        require(not any(a>b and passed and tested[b] is False for a,passed in tested.items() for b in tested),'capacity observations contradict monotonicity')
        if high-low<=1:break
        batch=(low+high)//2
    require(low>0,'no stable physical batch')
    state=launch(spec,f'confirm-{low}-u5','batch_smoke',low,5);require(state['accepted'],'selected batch confirmation failed')
    report=read(c/f'confirm-{low}-u5/report.json');audit=read(c/'audit/report.json')['audit'];gate=admission(report,audit)
    changes=report['changes'];require(changes['unused_heads_unchanged'] and changes['changed_body_names'] and changes['changed_head_names'] and report['restored_changes']['all_parameters_unchanged'],'gradient/update/restore qualification failed')
    gate.update(physical_batch=low,accumulation=math.ceil(64/low),effective_batch=64,tests={str(k):v for k,v in tested.items()},upper_excluded=high if high<=64 else None,qualification_model_seconds=budgets(spec)['qualification'],created_local=now())
    save(c/'qualification-analysis.json',gate);require(gate['accepted'],'registered runtime admission failed; do not train')
    return ['parity-b1','parity-b4']+[f'batch-{b}-u2' for b,passed in tested.items() if passed]+[f'confirm-{low}-u5']

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--stage',choices=('audit','qualification','frozen','train','final','analysis'),required=True);p.add_argument('--analysis-config',type=Path);p.add_argument('--analysis-sha256');args=p.parse_args()
    require(digest(args.spec)==args.sha256,'campaign specification differs');spec=read(args.spec);verify_files(spec['frozen_files']);verify_reference_bindings(spec)
    c=Path(spec['campaign']);require(c.is_dir(),'campaign absent')
    if args.stage=='analysis':
        completed_stage(spec,'final')
        require(args.analysis_config is not None and digest(args.analysis_config)==args.analysis_sha256,'analysis configuration missing or changed')
        tracked(spec,[PYTHON,'-B',str(R/'analyze.py'),'--config',str(args.analysis_config),'--config-sha256',args.analysis_sha256,'--output',str(c/'analysis.json')],'analysis',min(120,5400-budgets(spec)['wall']))
        save(c/'analysis-stage.json',dict(accepted=True,created_local=now(),source=spec['source'],analysis_sha256=digest(c/'analysis.json'),budgets=budgets(spec)))
        return
    require(args.analysis_config is None and args.analysis_sha256 is None,'analysis flags forbidden outside analysis stage')
    if args.stage=='audit':
        launch(spec,'audit','audit');names=['audit']
    elif args.stage=='qualification':names=qualify(spec)
    else:
        completed_stage(spec,'qualification')
        qualification=read(c/'qualification-analysis.json');require(qualification['accepted'] is True,'qualification not accepted');batch=qualification['physical_batch']
        if args.stage=='frozen':
            names=[f'frozen-{cohort}' for cohort in ('seen','familiar','heldout')]
            for cohort in ('seen','familiar','heldout'):launch(spec,f'frozen-{cohort}','eval_initial',batch,cohort=cohort)
        elif args.stage=='train':
            completed_stage(spec,'frozen')
            launch(spec,'train-seed0','train',batch,1150);names=['train-seed0']
        else:
            completed_stage(spec,'train');names=[f'final-{cohort}-{condition}' for cohort in ('seen','familiar','heldout') for condition in ('factual','cleared')]
            for cohort in ('seen','familiar','heldout'):
                for cleared in (False,True):launch(spec,f'final-{cohort}-'+('cleared' if cleared else 'factual'),'eval_final',batch,cohort=cohort,cleared=cleared)
    seal_stage(spec,args.stage,names)
if __name__=='__main__':main()
