"""Seal C16 only after numerical execution and both scoring processes exit."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import time

R=Path(__file__).resolve().parent
C=Path((R/'campaign-path.txt').read_text().strip())
_spec=importlib.util.spec_from_file_location('c16_seal_operator',R/'campaign_operator.py')
op=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(op)
deadline=op.campaign_deadline(C)
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
    with Path(p).open('x') as f: json.dump(v,f,indent=2);f.write('\n')
def now(): return datetime.datetime.now().astimezone().isoformat()
def require(ok,message):
    if not ok: raise ValueError(message)

a=read(C/'analysis.json')
v=read(C/'independent-review.json')
execution=read(C/'execution.json')
require(a['accepted'] is True and v['accepted'] is True and v['report_sha256']==sha(C/'analysis.json') and execution['accepted'] is True,'analysis/execution not accepted')
require(a['decision'] in ('supported_single_seed_binding_prerequisite','registered_binding_screen_not_supported'),'unknown scientific decision')
require(time.monotonic()<deadline,'campaign deadline expired before sealing')
config=read(C/'analysis-config.json')
require(a['config_sha256']==sha(C/'analysis-config.json') and a['integrity_receipt_sha256']==sha(config['integrity_receipt']),'scored config/receipt changed')
op.verify_files(config['frozen_files'])
integrity=read(config['integrity_receipt'])
op.verify_files(integrity['frozen_files'])
bindings=dict(read(C/'launch-spec.json')['frozen_files'])
bindings.update(config['frozen_files'])
bindings[str(C/'analysis-config.json')]=sha(C/'analysis-config.json')
bindings.update(read(R/'source-freeze.json')['files'])
bindings[str(R/'source-freeze.json')]=sha(R/'source-freeze.json')
bindings[str(R/'seal_completed.py')]=sha(R/'seal_completed.py')
pids=set()
def scan(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('pid','supervisor_pid','model_pid','pgid') and isinstance(item,int) and item>0: pids.add(item)
            if key=='owned_pids': pids.update(item)
            if key in ('owned_survivors','group_survivors'): require(not item,'owned survivors recorded')
            if key=='cleanup_error': require(item is None,'cleanup error recorded')
            scan(item)
    elif isinstance(value,list):
        for item in value: scan(item)
for base in (C,R/'operations'):
    for path in base.rglob('*.json'):
        if path.name.startswith('outer-seal.'): continue
        if path.name.endswith(('.process.json','.exit.json')):
            scan(read(path))
            if base!=C: bindings[str(path)]=sha(path)
for path in (R/'operations').glob('*.stdout.log'):
    if not path.name.startswith('outer-seal.'): bindings[str(path)]=sha(path)
for pid in pids: require(not Path(f'/proc/{pid}').exists(),f'process still exists: {pid}')
for path,digest in bindings.items(): require(sha(path)==digest,f'binding changed: {path}')
captures=0
for name in execution['names']:
    require(op.root_manifest(C/name)[0]==read(C/f'{name}.exit.json')['manifest_sha256'],'invocation manifest changed after scoring')
    rows=read(C/f'{name}.bound/summary.json')
    require(len(rows)==(3 if name=='train-seed0' else 1) and all(x['health']['structurally_valid'] and x['health']['capture_complete'] and x['raw_application_labels_verified'] and x['gpu']['status']=='available' and x['gpu']['provenance_binding']=='bound' for x in rows),'incomplete wired capture')
    captures+=len(rows)
save(C/'lifecycle.json',dict(state='complete',classification='completed_single_seed_screen',decision=a['decision'],created_local=now(),optimizer_updates=1150,all_owned_processes_gone=True,analysis_sha256=sha(C/'analysis.json'),independent_review_sha256=sha(C/'independent-review.json')))
files={}
for path in sorted(C.rglob('*')):
    require(not path.is_symlink(),'artifact symlink')
    if path.is_file(): files[str(path.relative_to(C))]=dict(bytes=path.stat().st_size,sha256=sha(path))
manifest=dict(schema='tofy-action-binding-evidence-v1',campaign=str(C),created_local=now(),classification='completed_single_seed_screen',source=a['source_revision'],binary_sha256=a['binary_sha256'],decision=a['decision'],files=files,external_bindings=bindings,pids_verified_gone=sorted(pids),cuda_bundles=captures,integrity_scope='Point-in-time verification, not immutable storage.')
path=R/'completed-campaign.manifest.json'
save(path,manifest)
digest=sha(path)
require({str(p.relative_to(C)) for p in C.rglob('*') if p.is_file()}==set(files),'inventory changed')
for rel,row in files.items(): require(sha(C/rel)==row['sha256'] and (C/rel).stat().st_size==row['bytes'],'artifact changed')
for path,expected in bindings.items(): require(sha(path)==expected,'external binding changed')
require(time.monotonic()<deadline,'campaign deadline expired during sealing')
(R/'completed-campaign.manifest.sha256').write_text(digest+'\n')
verification=dict(accepted=True,created_local=now(),manifest_sha256=digest,files=len(files),bytes=sum(x['bytes'] for x in files.values()),bindings=len(bindings),pids_gone=len(pids),cuda_bundles=captures)
save(R/'completed-campaign-verification.json',verification)
state=read(R/'operator-state.json')
state.update(state='complete',decision=a['decision'],manifest_sha256=digest,active_processes=[],updated_local=now())
(R/'operator-state.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(verification))
