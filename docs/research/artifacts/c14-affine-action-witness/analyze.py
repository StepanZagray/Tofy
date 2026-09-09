"""C14 fixed affine witness; retained arrays only, no neural/model execution."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,ctypes,datetime,hashlib,importlib.util,json,math,platform,subprocess,time
from pathlib import Path
import numpy as np
from ridge import fit,predict
R=Path(__file__).resolve().parent
A_PATH=Path('/home/stepan/Research/_runs/2026-09-09T143629Z-tofy-looped-grounded-policy-learning/analyze.py')
assert hashlib.sha256(A_PATH.read_bytes()).hexdigest()=='1159f62d0cd260193907f45cb32eaf547388973ca5257f30318a59327a5b6a81'
spec=importlib.util.spec_from_file_location('c12_scoring',A_PATH);A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
COHORTS=('seen','familiar','heldout');ARMS=('final_true','initial_true','final_permuted')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return A.decode(Path(p).read_bytes())
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def require(v,m):A.require(v,m)
def verified(binding):
 p=Path(binding['path']);require(p.is_absolute() and p.is_file() and not p.is_symlink() and sha(p)==binding['sha256'],'bound input differs');return p

def summary(scores,audit):
 labels=np.asarray([r['correct_action'] for r in audit]);omitted=np.asarray([r['omitted_action'] for r in audit]);pred=scores.argmax(1);hit=pred==labels;m=len(labels)//64
 require(scores.shape==(len(labels),4) and np.isfinite(scores).all(),'score matrix')
 subsets={}
 for name,mask in (('omitted',labels==omitted),('demonstrated',labels!=omitted)):
  subsets[name]=dict(rows=int(mask.sum()),correct=int(hit[mask].sum()),accuracy=float(hit[mask].mean()))
 value=dict(rows=len(labels),groups=64,maps_per_group=m,correct=int(hit.sum()),accuracy=float(hit.mean()),label_histogram=np.bincount(labels,minlength=4).tolist(),prediction_histogram=np.bincount(pred,minlength=4).tolist(),all_maps_correct_groups=int(hit.reshape(64,m).all(1).sum()),subsets=subsets)
 return value,hit.reshape(64,m).mean(1)

def interval(values,indices):
 samples=values[indices].mean(1);return dict(estimate=float(values.mean()),ci95=np.quantile(samples,[.025,.975],method='linear').tolist())

def decide(summaries,contrasts,group_controls):
 controls=all(x['correct']==x['rows']//4 and x['identical_group_predictions'] for x in group_controls.values()) and all(summaries[f'final_permuted/{c}']['accuracy']<=.5 for c in ('familiar','heldout'))
 arms={}
 for arm in ('initial_true','final_true'):
  arms[arm]=controls and all(summaries[f'{arm}/{c}']['accuracy']>=.9 for c in COHORTS) and all(contrasts[f'{arm}/{c}/native']['ci95'][0]>0 and contrasts[f'{arm}/{c}/constant']['ci95'][0]>.25 for c in ('familiar','heldout'))
 decision='inconclusive_failed_control' if not controls else 'affine_witness_already_present_initially' if all(arms.values()) else 'final_affine_witness_supported_exploratorily' if arms['final_true'] else 'initial_only_affine_witness' if arms['initial_true'] else 'registered_affine_witness_not_supported'
 return dict(controls_valid=bool(controls),true_arms=arms),decision

def runtime(cfg):
 require(platform.python_version()=='3.14.7' and np.__version__=='2.4.2','numeric runtime version differs')
 paths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'openblas' in line.lower()}
 require(len(paths)==1,'expected one NumPy OpenBLAS runtime');path=next(iter(paths));require(path in cfg['frozen_files'] and sha(path)==cfg['frozen_files'][path],'unbound BLAS runtime')
 library=ctypes.CDLL(path);get=None
 for prefix in ('scipy_',''):
  for suffix in ('64_','_64_','','_'):
   try:get=getattr(library,prefix+'openblas_get_num_threads'+suffix);break
   except AttributeError:continue
  if get is not None:break
 require(get is not None,'OpenBLAS thread introspection unavailable');get.restype=ctypes.c_int;require(get()==1,'OpenBLAS threads differ')
 return dict(python=platform.python_version(),numpy=np.__version__,blas_library=path,blas_sha256=sha(path),blas_threads=get(),model_forwards=0,neural_optimizer_updates=0)

def run(cfg_path,cfg_sha):
 start=time.monotonic();require(sha(cfg_path)==cfg_sha,'config pin differs');cfg=read(cfg_path)
 for p,h in cfg['frozen_files'].items():verified(dict(path=p,sha256=h))
 verified(cfg['registration']);environment=runtime(cfg);repo=cfg['operator_repository']
 require(subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()==cfg['operator_revision'],'operator source HEAD')
 require(not subprocess.check_output(['git','-C',repo,'status','--porcelain','--untracked-files=all'],text=True).strip(),'operator source dirty');subprocess.run(['git','-C',repo,'merge-base','--is-ancestor','HEAD','@{upstream}'],check=True)
 for outer in cfg['parent_seals']:
  manifest=read(verified(outer));root=Path(manifest['campaign'])
  require({str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==set(manifest['files']),'parent inventory differs')
  for rel,row in manifest['files'].items():require(sha(root/rel)==row['sha256'],'parent artifact differs')
  for p,h in manifest['external_bindings'].items():verified(dict(path=p,sha256=h))
 audits={};streams={};features={};native={};native_groups={}
 for c in COHORTS:
  n=512 if c=='heldout' else 1024;audits[c]=A.rows(verified(cfg['audits'][c]).read_bytes(),n)
  for i,row in enumerate(audits[c]):A.validate_identity(row,c,'factual',i)
  m=n//64
  for g in range(64):require(np.bincount([r['correct_action'] for r in audits[c][g*m:(g+1)*m]],minlength=4).tolist()==[m//4]*4,'balanced groups')
  for source in ('initial','final'):
   key=f'{source}/{c}';rows=A.rows(verified(cfg['streams'][key]).read_bytes(),n);streams[key]=rows;A.measure(rows,audits[c],'frozen' if source=='initial' else 'final')
   features[key]=np.asarray([r['pooled'] for r in rows],dtype=np.float64);native[key],native_groups[key]=summary(np.asarray([r['logits'] for r in rows]),audits[c])
 y=np.asarray([r['correct_action'] for r in audits['seen']]);rng=np.random.Generator(np.random.PCG64(1941));permutation=np.concatenate([16*g+rng.permutation(16) for g in range(64)]);shuffled=y[permutation]
 require(all(np.bincount(shuffled[g*16:g*16+16],minlength=4).tolist()==[4]*4 for g in range(64)),'within-group shuffle balance')
 draws_seen=np.random.Generator(np.random.PCG64(1942)).integers(0,64,(10000,64));draws_fresh=np.random.Generator(np.random.PCG64(1943)).integers(0,64,(10000,64))
 params={};summaries={};groups={};contrasts={};controls={};dispersion={};fitcounts={}
 for arm in ARMS:
  source='initial' if arm=='initial_true' else 'final';train=features[f'{source}/seen'];targets=shuffled if arm=='final_permuted' else y
  params[arm]=fit(train,targets)
  fitpred=predict(train,params[arm]).argmax(1);fitcounts[arm]=dict(fitting_label_correct=int((fitpred==targets).sum()),true_label_correct=int((fitpred==y).sum()),rows=1024)
  for c in COHORTS:
   key=f'{arm}/{c}';x=features[f'{source}/{c}'];scores=predict(x,params[arm]);summaries[key],groups[key]=summary(scores,audits[c]);idx=draws_seen if c=='seen' else draws_fresh
   contrasts[key+'/native']=interval(groups[key]-native_groups[f'{source}/{c}'],idx);contrasts[key+'/constant']=interval(groups[key]-.25,idx)
   m=len(x)//64;block=x.reshape(64,m,256);means=block.mean(1);repeated=np.repeat(means,m,axis=0);nullscores=predict(repeated,params[arm]);nullsummary,_=summary(nullscores,audits[c]);pred=nullscores.argmax(1).reshape(64,m)
   controls[key]=dict(correct=nullsummary['correct'],rows=len(x),identical_group_predictions=bool((pred==pred[:,:1]).all()))
   dispersion[f'{source}/{c}']=dict(within_group_rms=float(np.sqrt(np.mean((block-means[:,None,:])**2))),between_group_rms=float(np.sqrt(np.mean((means-means.mean(0))**2))))
 for c in COHORTS:contrasts[f'final_minus_initial/{c}']=interval(groups[f'final_true/{c}']-groups[f'initial_true/{c}'],draws_seen if c=='seen' else draws_fresh)
 gates,decision=decide(summaries,contrasts,controls)
 def serial(v):
  if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
  if isinstance(v,np.ndarray):return v.tolist()
  if isinstance(v,np.generic):return v.item()
  return v
 out=Path(cfg['output_root']);parameter_path=out/'parameters.json';save(parameter_path,serial(params))
 require(time.monotonic()-start<=120,'analysis CPU cap exceeded')
 result=dict(accepted=True,classification='exploratory_frozen_affine_witness',decision=decision,gates=gates,summaries=summaries,contrasts=contrasts,native=native,fit_counts=fitcounts,group_mean_controls=controls,dispersion=dispersion,parameters=dict(path=str(parameter_path),sha256=sha(parameter_path)),config_sha256=cfg_sha,registration_sha256=cfg['registration']['sha256'],permutation_sha256=hashlib.sha256(permutation.astype('<u8').tobytes()).hexdigest(),bootstrap_seeds=dict(seen=1942,fresh=1943),environment=environment,data_access='Only seen features/targets determine fits; no transfer-dependent preprocessing or selection.',elapsed_seconds=time.monotonic()-start,created_local=datetime.datetime.now().astimezone().isoformat())
 for p,h in cfg['frozen_files'].items():verified(dict(path=p,sha256=h))
 return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();save(a.output,run(a.config,a.config_sha256));print(a.output)
