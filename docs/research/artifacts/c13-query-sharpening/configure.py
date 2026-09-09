"""Bind the preselected C13 invocation after its registered CPU premise passes."""
import argparse,shutil,subprocess
from pathlib import Path
from diagnostic import R,R12,PARENT,REPO
from supervise import read,save,digest,verify_files,require,now
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--operator-revision',required=True);a=p.parse_args();c=a.campaign
require(read(c/'premise.json')['accepted'] is True,'CPU premise failed')
operator=Path('/home/stepan/Projects/code/Tofy-grounded-diagnostics')
require(subprocess.check_output(['git','-C',str(operator),'rev-parse','HEAD'],text=True).strip()==a.operator_revision,'operator revision')
require(not subprocess.check_output(['git','-C',str(operator),'status','--porcelain','--untracked-files=all'],text=True).strip(),'operator dirty')
subprocess.run(['git','-C',str(operator),'merge-base','--is-ancestor','HEAD','@{upstream}'],check=True)
files=dict(read(PARENT/'campaign-spec.json')['frozen_files'])
for name in ('diagnostic.py','diagnostic_tests.py','independent_review.py','independent_review_tests.py','configure.py','registration.md'):
 require(digest(R/name)==digest(operator/'docs/research/artifacts/c13-query-sharpening'/name),'operator snapshot mismatch');files[str(R/name)]=digest(R/name)
for path in (R12/'completed-campaign.manifest.json',R12/'completed-campaign-verification.json',PARENT/'final-seen-factual/evaluation-rows.jsonl',PARENT/'audit/seen-factual-audit.jsonl',PARENT/'audit/manifest.json',PARENT/'train-seed0/final-core.safetensors',PARENT/'train-seed0/final-head.safetensors',c/'queries-x16.safetensors',c/'transformation.json',c/'premise.json'):
 files[str(path)]=digest(path)
require(not (c/'grounded_policy_probe').exists(),'binary root reused');shutil.copy2(PARENT/'grounded_policy_probe',c/'grounded_policy_probe');files[str(c/'grounded_policy_probe')]=digest(c/'grounded_policy_probe')
cfg=read(PARENT/'invocations/final-seen-factual.json');cfg.update(registration=str(R/'registration.md'),registration_sha256=digest(R/'registration.md'),output_dir=str(c/'sharpened-seen'),head_checkpoint=str(c/'queries-x16.safetensors'),head_sha256=digest(c/'queries-x16.safetensors'),max_seconds=120)
save(c/'config.json',cfg);files[str(c/'config.json')]=digest(c/'config.json')
expected=dict(status='complete_pending_analysis',optimizer_updates=0,cohort='seen',cleared=False,input_rows=1024,physical_batch=34)
auth=dict(schema='looped-grounded-policy-launch-v1',accepted=True,created_local=now(),config_sha256=digest(c/'config.json'),mode='eval_final',campaign=str(c),name='sharpened-seen',binary=str(c/'grounded_policy_probe'),binary_sha256=digest(c/'grounded_policy_probe'),repository=REPO,source=cfg['source_revision'],frozen_files=files,expected_report=expected)
save(c/'authority.json',auth);files[str(c/'authority.json')]=digest(c/'authority.json')
verify_files(files);save(c/'launch-spec.json',dict(frozen_files=files,operator_revision=a.operator_revision,source=cfg['source_revision'],created_local=now()))
print('C13 fixed invocation bound; zero optimizer updates; seen factual only.')
