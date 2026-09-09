"""CPU-only F32 identity repair over a point-in-time sealed native runtime."""
import argparse
import hashlib
import signal
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode = True
R = Path(__file__).resolve().parent
PARENT = R.parent/'2026-09-09T212336Z-tofy-looped-native-profile-recovery'
REPO = Path('/home/stepan/Projects/code/Tofy-native-binding-results')
PARENT_SHA = 'a8bd0584802dc3f3652f9f83ba0551cd64634a4f523df4844a65e9e5250e3e3f'
PYTHON = '/home/stepan/venvs/tensorboard/bin/python3'
ARTIFACTS = ('registration.md','rescore_operator.py','analysis.py','analysis_tests.py','numpy_binding.py','numpy_binding_tests.py','independent_review.py','independent_review_tests.py')
# Reuse immutable, already tested manifest/process checks; no old main is run.
for _name,_expected in {'campaign_operator.py':'9e96a07727cbce30ae2e9d9bb4bb3b1851d935bbad27cb6adaecea8b436d957f','seal_completed.py':'3481e4b43e9335fde592bf39f4db415761706f4fd50684c55bda07f008a42272'}.items():
    with (PARENT/_name).open('rb') as _file:
        if hashlib.file_digest(_file,'sha256').hexdigest()!=_expected:raise ValueError('pinned parent protocol changed')
sys.path.insert(0, str(PARENT))
import campaign_operator as op
import seal_completed as parent_seal
sys.path.remove(str(PARENT))

def revision():
    value = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    op.require(not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=all'],text=True).strip(),'dirty analyzer source')
    subprocess.run(['git','-C',str(REPO),'merge-base','--is-ancestor','HEAD','@{upstream}'],check=True)
    return value

def bind(path):return {'path':str(path),'sha256':op.digest(path)}

def parent():
    path = PARENT/'runtime-campaign.manifest.json'
    op.require(op.digest(path)==PARENT_SHA,'parent runtime inventory changed')
    value = op.read(path);c=Path(value['campaign'])
    op.require(value['classification']=='completed_frozen_runtime_scoring_failed' and value['cuda_bundles']==11,'parent classification/captures differ')
    op.require({str(p.relative_to(c)) for p in c.rglob('*') if p.is_file()}==set(value['files']),'parent inventory membership changed')
    for name,item in value['files'].items():
        p=c/name;op.require(p.stat().st_size==item['bytes'] and op.digest(p)==item['sha256'],'parent runtime bytes changed')
    op.verify_files(value['external_bindings'])
    parent_seal.verify_processes_gone(set(value['pids_verified_gone']),set(value['groups_verified_gone']))
    receipt=op.read(c/'integrity.json');op.verify_files(receipt['frozen_files'])
    op.require(receipt['accepted'] is True and set(receipt['checks'])==parent_seal.CHECKS and all(x is True for x in receipt['checks'].values()),'original runtime rejected')
    op.require(value['source']==receipt['source_revision'] and value['binary_sha256']==receipt['binary_sha256'],'parent source/binary differs')
    return value,receipt

def prepare(c):
    op.require(c.is_absolute() and not c.exists(),'new absolute result root required')
    rev=revision();manifest,old=parent();files=dict(op.read(R/'source-freeze.json')['files'])
    for name in ARTIFACTS:
        op.require(op.digest(R/name)==op.digest(REPO/'docs/research/artifacts/c21-native-f32-rescore'/name),'archive differs')
    op.verify_files(files)
    c.mkdir();files[str(R/'source-freeze.json')]=op.digest(R/'source-freeze.json')
    files.update(old['frozen_files']);files[str(PARENT/'runtime-campaign.manifest.json')]=PARENT_SHA
    runtime=Path(manifest['campaign']);files[str(runtime/'integrity.json')]=op.digest(runtime/'integrity.json')
    # The old receipt remains untouched; this extension binds corrected analyzers.
    receipt=dict(old,analyzer_source_revision=rev,original_integrity_receipt=bind(runtime/'integrity.json'),runtime_inventory=bind(PARENT/'runtime-campaign.manifest.json'),frozen_files=files,created_local=op.now())
    op.save(c/'integrity.json',receipt);files=dict(files);files[str(c/'integrity.json')]=op.digest(c/'integrity.json')
    cfg=dict(schema='looped-native-binding-analysis-config-v1',registration=bind(R/'registration.md'),history=bind(R/'history.json'),audit=bind(runtime/'audit/panel-rows.jsonl'),checkpoint=bind(op.BINDER),integrity_receipt=bind(c/'integrity.json'),streams=old['streams'],frozen_files=files)
    op.save(c/'analysis-config.json',cfg)
    op.save(c/'launch.json',dict(schema='looped-native-f32-rescore-launch-v1',analyzer_source_revision=rev,model_source_revision=old['source_revision'],model_binary_sha256=old['binary_sha256'],parent_inventory_sha256=PARENT_SHA,config_sha256=op.digest(c/'analysis-config.json'),created_local=op.now(),gpu_invocations=0,optimizer_updates=0))
    return rev

def deadline(c):
    clock=op.read(c/'clock.json')
    op.require(clock['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'host rebooted')
    start=clock['started_monotonic'];op.require(type(start) in (int,float) and 0<=start<=time.monotonic(),'invalid monotonic clock')
    return start+300

def analyze(c):
    op.require(revision()==op.read(c/'launch.json')['analyzer_source_revision'],'analyzer source changed')
    op.save(c/'clock.json',dict(started_monotonic=time.monotonic(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_local=op.now()))
    cfg=c/'analysis-config.json';op.require(op.digest(cfg)==op.read(c/'launch.json')['config_sha256'],'config changed')
    spec={'campaign':str(c),'repository':str(REPO)}
    op.tracked(spec,[PYTHON,str(R/'analysis.py'),'--config',str(cfg),'--output',str(c/'analysis.json')],'analysis',op.remaining(deadline(c)-120,60))
    op.tracked(spec,[PYTHON,str(R/'independent_review.py'),'--config',str(cfg),'--report',str(c/'analysis.json'),'--output',str(c/'independent-review.json')],'independent-review',op.remaining(deadline(c)-60,60))
    a=op.read(c/'analysis.json');v=op.read(c/'independent-review.json')
    op.require(a['accepted'] is True and v['accepted'] is True and v['report_sha256']==op.digest(c/'analysis.json'),'corrected scoring rejected')
    op.save(c/'execution.json',dict(accepted=True,decision=a['decision'],analyzer_source_revision=revision(),gpu_invocations=0,optimizer_updates=0,created_local=op.now()))

def seal(c):
    op.remaining(deadline(c),60);manifest,old=parent()
    cfg=op.read(c/'analysis-config.json');receipt=op.read(c/'integrity.json');a=op.read(c/'analysis.json');v=op.read(c/'independent-review.json')
    parent_seal.result_contract(a,v,cfg,receipt,op.digest(c/'analysis-config.json'),op.digest(c/'analysis.json'))
    op.require(op.read(c/'execution.json')['accepted'] is True and revision()==receipt['analyzer_source_revision'],'rescoring execution/source differs')
    op.require(cfg['frozen_files']==receipt['frozen_files']|{str(c/'integrity.json'):op.digest(c/'integrity.json')},'frozen closure differs')
    files=dict(cfg['frozen_files']);pids=set();groups=set()
    for base in (c,R/'operations'):
        for p in base.rglob('*.json'):
            if p.name.startswith('outer-seal.'):
                op.require(base!=c,'sealer wrapper must remain external');continue
            if p.name.endswith(('.process.json','.exit.json')):
                parent_seal.scan_process_record(op.read(p),pids,groups)
                if base!=c:files[str(p)]=op.digest(p)
    for p in (R/'operations').glob('*.stdout.log'):
        if not p.name.startswith('outer-seal.'):files[str(p)]=op.digest(p)
    for name in ('analysis','independent-review'):
        op.read(c/'operations'/f'{name}.process.json')
        state=op.read(c/'operations'/f'{name}.exit.json');op.require(state['accepted'] is True,'failed scoring process');parent_seal.cleanup_contract(state)
    parent_seal.verify_processes_gone(pids,groups);op.verify_files(files)
    op.save(c/'lifecycle.json',dict(state='complete',classification='completed_frozen_native_rescore',decision=a['decision'],created_local=op.now(),all_owned_processes_gone=True,gpu_invocations=0,optimizer_updates=0))
    inventory={str(p.relative_to(c)):dict(bytes=p.stat().st_size,sha256=op.digest(p)) for p in sorted(c.rglob('*')) if p.is_file()}
    op.require(all(not p.is_symlink() for p in c.rglob('*')),'result symlink')
    result=dict(schema='tofy-native-f32-rescore-evidence-v1',campaign=str(c),parent_inventory_sha256=PARENT_SHA,model_source_revision=a['source_revision'],model_binary_sha256=a['binary_sha256'],analyzer_source_revision=receipt['analyzer_source_revision'],decision=a['decision'],files=inventory,external_bindings=files,pids_verified_gone=sorted(pids),groups_verified_gone=sorted(groups),created_local=op.now(),scope='Corrected F32 identity scoring of unchanged retained runtime; no new GPU work or fresh independent replication. Point-in-time verification; own external seal wrapper closes afterward.')
    op.save(R/'completed-campaign.manifest.json',result)
    for n,row in inventory.items():op.require(op.digest(c/n)==row['sha256'],'result changed during sealing')
    op.require({str(p.relative_to(c)) for p in c.rglob('*') if p.is_file()}==set(inventory),'final inventory membership changed')
    op.verify_files(files);op.remaining(deadline(c),60)
    digest=op.digest(R/'completed-campaign.manifest.json')
    with (R/'completed-campaign.manifest.sha256').open('x') as f:f.write(digest+'\n')
    verification=dict(accepted=True,manifest_sha256=digest,files=len(inventory),bytes=sum(v['bytes'] for v in inventory.values()),bindings=len(files),pids_gone=len(pids),groups_gone=len(groups),decision=a['decision'],created_local=op.now())
    op.save(R/'completed-campaign-verification.json',verification);print(json.dumps(verification))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','analyze','seal'));parser.add_argument('--campaign',type=Path,required=True);args=parser.parse_args()
    if args.stage=='seal':
        def expired(*_):raise TimeoutError('sealing exceeds60seconds')
        signal.signal(signal.SIGALRM,expired);signal.alarm(60)
    try:{'prepare':prepare,'analyze':analyze,'seal':seal}[args.stage](args.campaign)
    finally:
        if args.stage=='seal':signal.alarm(0)
if __name__=='__main__':main()
