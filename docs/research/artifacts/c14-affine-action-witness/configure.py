"""Generate a fresh C14 statistics campaign from reviewed pushed source snapshots."""
import argparse,datetime,hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
P=Path('/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST')
R12=Path('/home/stepan/Research/_runs/2026-09-09T143629Z-tofy-looped-grounded-policy-learning')
R13=Path('/home/stepan/Research/_runs/2026-09-09T165412Z-tofy-looped-query-sharpening')
OP=Path('/home/stepan/Projects/code/Tofy-grounded-diagnostics')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--operator-revision',required=True);a=p.parse_args();c=a.campaign
assert c.is_absolute() and not c.exists(),'new campaign required'
assert subprocess.check_output(['git','-C',str(OP),'rev-parse','HEAD'],text=True).strip()==a.operator_revision
assert not subprocess.check_output(['git','-C',str(OP),'status','--porcelain','--untracked-files=all'],text=True).strip()
subprocess.run(['git','-C',str(OP),'merge-base','--is-ancestor','HEAD','@{upstream}'],check=True)
files=dict(read(P/'campaign-spec.json')['frozen_files'])
for name in ('registration.md','ridge.py','ridge_tests.py','analyze.py','analyze_tests.py','independent_review.py','independent_review_tests.py','configure.py'):
 assert sha(R/name)==sha(OP/'docs/research/artifacts/c14-affine-action-witness'/name),'operator snapshot differs';files[str(R/name)]=sha(R/name)
streams={};audits={}
for cohort in ('seen','familiar','heldout'):
 audits[cohort]=bound(P/'audit'/f'{cohort}-factual-audit.jsonl')
 for name,root in (('initial',f'frozen-{cohort}'),('final',f'final-{cohort}-factual')):streams[f'{name}/{cohort}']=bound(P/root/'evaluation-rows.jsonl')
for x in list(streams.values())+list(audits.values()):files[x['path']]=x['sha256']
parents=[bound(R12/'completed-campaign.manifest.json'),bound(R13/'completed-campaign.manifest.json')]
assert [x['sha256'] for x in parents]==['0a4ac9a8d3f46ebfca141360384a18761ea4c9fd175c43176788dd422ec7ab6f','fbaef96bc4e520301edfb2b79e811467d226e45509a7aee078ebdfdebabcac6f']
for x in parents:files[x['path']]=x['sha256']
source=read(P/'analysis.json')['source'];source['checkpoints']=read(P/'analysis.json')['checkpoints']
c.mkdir(exist_ok=False)
config=dict(schema='looped-affine-action-witness-v1',operator_repository=str(OP),operator_revision=a.operator_revision,registration=bound(R/'registration.md'),source=source,output_root=str(c),streams=streams,audits=audits,parent_seals=parents,frozen_files=files,environment=dict(python='3.14.7',numpy='2.4.2',blas_threads=1,device='CPU F64 statistics',model_forwards=0,neural_optimizer_updates=0,feature_producer_physical_batch=34,profiling='No new Candle/model workload; prior C12 producer captures retained with their declared gaps.'),created_local=datetime.datetime.now().astimezone().isoformat())
save(c/'config.json',config);save(R/'launch-binding.json',dict(campaign=str(c),config_sha256=sha(c/'config.json'),operator_revision=a.operator_revision,created_local=datetime.datetime.now().astimezone().isoformat()));(R/'campaign-path.txt').write_text(str(c)+'\n');print(json.dumps(dict(campaign=str(c),config_sha256=sha(c/'config.json'))))
