#!/usr/bin/env python3
"""One registered C12 invocation with pinned source and inherited tested cleanup."""
import argparse,datetime,hashlib,importlib.util,json,math,os,re,signal,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
LIFECYCLE=Path('/home/stepan/Projects/code/.tofy-runs/looped-frozen-features-20260909T115455-IST/supervise.py')
LIFECYCLE_SHA='e047b88eed16fb9d8459d82f26a5d5f3ae2dbfc6765ccfa8325c4e9f8ee7de41'
DEPENDENCY='1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a'
NSYS='/home/stepan/Projects/code/.tofy-tools/nsight-2026.4.1/opt/nvidia/nsight-systems-cli/2026.4.1/bin/nsys'

def require(ok,message):
    if not ok: raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as out:json.dump(value,out,indent=2);out.write('\n')
def digest(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),f'nonregular binding {path}')
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify_files(files):
    require(isinstance(files,dict) and files,'empty source binding')
    for path,sha in files.items():require(Path(path).is_absolute() and digest(path)==sha,f'binding mismatch {path}')
def now():return datetime.datetime.now().astimezone().isoformat()
def root_manifest(root):
    manifest=root/'manifest.json';sha=digest(manifest)
    require(sha==root.with_suffix('.manifest.sha256').read_text().strip(),'root manifest pin differs')
    value=read(manifest)
    require(value['schema']=='looped-grounded-policy-artifacts-v1','root schema differs')
    files=value['files'];require(set(files)=={p.name for p in root.iterdir() if p.name!='manifest.json'},'root population differs')
    for name,expected in files.items():
        require(Path(name).name==name and digest(root/name)==expected,'root artifact differs')
    profiles=read(root/'profiles.json')
    require(Path(profiles['root'])==root.with_suffix('.profiles'),'profile root escapes invocation')
    actual={str(p.relative_to(Path(profiles['root']))) for p in Path(profiles['root']).rglob('*') if p.is_file()}
    require(actual==set(profiles['files']),'profile artifact inventory differs')
    for name,expected in profiles['files'].items():require(digest(Path(profiles['root'])/name)==expected,'profile differs')
    host=profiles['host_trace'];require(Path(host['path'])==root.with_suffix('.host.json'),'host trace escapes invocation');require(digest(host['path'])==host['sha256'],'host trace differs')
    return sha,profiles,read(host['path'])
def cleanup_valid(state):
    require(state['failure'] is None and state['returncode']==0,'backend/lifecycle failure')
    require(state['pid_gone'] and state['model_pid_gone'] and state['group_gone'] and not state['owned_survivors'] and not state['group_survivors'] and state['cleanup_error'] is None,'owned process cleanup incomplete')
def capacity_only(state,stdout):
    """Only explicit allocation/reserve failures can shrink the batch bracket."""
    text=(str(state.get('failure',''))+' '+stdout).lower()
    forbidden=('nonfinite','non-finite','ac is offline','temperature','illegal memory','device-side assert','source mismatch','binding mismatch')
    if any(s in text for s in forbidden):return False
    failure=state.get('failure')
    diagnosed=(failure=="RuntimeError('registered 512 MiB GPU reserve breached')" or isinstance(failure,str) and re.fullmatch(r"RuntimeError\('model/profiler exited [1-9][0-9]*'\)",failure) is not None and ('cuda_error_out_of_memory' in stdout.lower() or 'cuda out of memory' in stdout.lower()))
    return diagnosed and state.get('group_gone') is True and state.get('pid_gone') is True and state.get('model_pid_gone') is True and not state.get('owned_survivors') and not state.get('group_survivors') and state.get('cleanup_error') is None and state.get('bindings_unchanged') is True and state.get('postcheck_kind')=='backend_failure'

def main():
    require(__debug__,'Python optimization forbidden')
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--authority',type=Path,required=True);p.add_argument('--authority-sha256',required=True);args=p.parse_args()
    verify_files({str(args.config):args.sha256,str(args.authority):args.authority_sha256,str(LIFECYCLE):LIFECYCLE_SHA})
    config=read(args.config);authority=read(args.authority)
    require(authority['schema']=='looped-grounded-policy-launch-v1' and authority['accepted'] is True,'launch not registered')
    require(authority['config_sha256']==args.sha256 and authority['mode']==config['mode'],'invocation authority differs')
    verify_files(authority['frozen_files'])
    root=Path(config['output_dir']);campaign=Path(authority['campaign']);binary=Path(authority['binary'])
    require(root.parent==campaign and root.name==authority['name'] and not list(campaign.glob(root.name+'*')),'never reuse run/sibling roots')
    require(digest(binary)==authority['binary_sha256'] and config['source_revision']==authority['source'],'source binary/config differs')
    for path,revision in [(Path(authority['repository']),authority['source']),(Path(authority['repository']).parent/'candle_graph',DEPENDENCY)]:
        require(subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()==revision,'source HEAD differs')
        require(not subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=all'],text=True).strip(),'dirty source')
        subprocess.run(['git','-C',str(path),'merge-base','--is-ancestor','HEAD','@{upstream}'],check=True)
    spec=importlib.util.spec_from_file_location('c12_lifecycle',LIFECYCLE);life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
    signal.signal(signal.SIGTERM,life.terminated);signal.signal(signal.SIGINT,life.terminated)
    audit=config['mode']=='audit';command=[str(binary),'--config',str(args.config),'--config-sha256',args.sha256]
    captures=0 if audit else 3 if config['mode']=='train' else 1
    if not audit:
        nsdir=root.with_suffix('.nsight');nsdir.mkdir()
        command=[NSYS,'profile','--trace=cuda,nvtx,osrt,cudnn,cublas','--sample=process-tree','--cpuctxsw=process-tree','--backtrace=lbr','--capture-range=nvtx','--nvtx-capture=tofy.looped/capture',f'--capture-range-end=repeat:{captures}:defer','--wait=all','--kill=none','--output',str(nsdir/'capture'),*command]
    env=os.environ.copy();env.update(TOFY_PERF_TRACE=str(root.with_suffix('.host.json')),NSYS_NVTX_PROFILER_REGISTER_ONLY='0',NVIDIA_TF32_OVERRIDE='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def sample(timeout):
        if audit:return 0,8151,0
        require(Path('/sys/class/power_supply/ACAD/online').read_text().strip()=='1','AC is offline')
        result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits','--id=0'],capture_output=True,text=True,check=True,timeout=timeout)
        used,total,temp=map(int,result.stdout.strip().split(','));require(temp<85,'temperature limit breached')
        return used,total,temp
    sample(5)
    def record(pid,pgid):
        value=dict(pid=pid,pgid=pgid,supervisor_pid=os.getpid(),command=command,config=str(args.config),config_sha256=args.sha256,authority_sha256=args.authority_sha256,started_local=now(),binary_sha256=digest(binary),environment={key:env[key] for key in ('TOFY_PERF_TRACE','NSYS_NVTX_PROFILER_REGISTER_ONLY','NVIDIA_TF32_OVERRIDE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
        save(root.with_suffix('.process.json'),value);print(json.dumps(value),flush=True)
    with root.with_suffix('.stdout.log').open('x') as out,root.with_suffix('.telemetry.jsonl').open('x') as telemetry:
        state=life.supervise_child(command,root,env,out,telemetry,record,sample,model_seconds=config['max_seconds'],finalization_seconds=120)
    state['finished_local']=now();state['captures_expected']=captures
    state['bindings_unchanged']=False
    try:
        verify_files(authority['frozen_files']);verify_files({str(args.authority):args.authority_sha256,str(args.config):args.sha256,str(binary):authority['binary_sha256']})
        state['bindings_unchanged']=True
    except Exception as error:state['binding_error']=repr(error)
    state['postcheck_kind']='backend_failure' if state['failure'] is not None or state['returncode']!=0 else 'integrity_failure'
    try:
        require(state['bindings_unchanged'],'bindings changed during invocation')
        cleanup_valid(state)
        sha,profiles,host=root_manifest(root)
        state['manifest_sha256']=sha
        require((host==[] and profiles['files']=={}) if audit else bool(host) and bool(profiles['files']),'host/profile capability mismatch')
        require(len(list(Path(profiles['root']).glob('*/trace.jsonl')))==captures,'capture count differs')
        if audit:require(not root.with_suffix('.nsight').exists() and not root.with_suffix('.bound').exists(),'CPU audit contains CUDA artifacts')
        report=read(root/'report.json');metadata=read(root/'metadata.json')
        require(report['status']=='complete_pending_analysis','model report failed')
        require(metadata['config']==config and metadata['provenance']['binary_sha256']==digest(binary) and metadata['provenance']['source_revision']==authority['source'],'metadata differs')
        for key,value in authority['expected_report'].items():require(type(report.get(key)) is type(value) and report.get(key)==value,f'registered report field differs {key}')
        state['reported_model_elapsed_seconds']=life.reported_model_elapsed(report,config['max_seconds'])
        verify_files(authority['frozen_files']);state['accepted']=True
    except Exception as error:
        state.update(accepted=False,integrity_error=repr(error))
    state['capacity_failure']=False
    if not state['accepted'] and config['mode']=='batch_smoke':
        state['capacity_failure']=capacity_only(state,root.with_suffix('.stdout.log').read_text())
        if state['capacity_failure']:
            try:
                used,total,temp=sample(5);require(total-used>=512,'device reserve not recovered')
            except Exception as error:state.update(capacity_failure=False,recovery_error=repr(error))
    state['classification']=('implementation_smoke' if config['mode'] in ('qualify','batch_smoke') else 'complete_pending_analysis') if state['accepted'] else 'failed_infrastructure_or_integrity'
    save(root.with_suffix('.exit.json'),state);print(json.dumps(state),flush=True)
    if not state['accepted']:raise SystemExit(1)
if __name__=='__main__':main()
