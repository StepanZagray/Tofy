"""C13: one fixed frozen intervention. No fitting or temperature selection."""
import argparse,datetime,json,os,struct,sys
from pathlib import Path
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
 os.environ[name]="1"
import numpy as np
R=Path(__file__).resolve().parent
R12=Path('/home/stepan/Research/_runs/2026-09-09T143629Z-tofy-looped-grounded-policy-learning')
sys.path.insert(0,str(R12))
import analyze as A
from driver import tracked,PYTHON
from supervise import digest,read,save,verify_files,root_manifest,now,require
PARENT=Path('/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST')
REPO='/home/stepan/Projects/code/Tofy-grounded-training'
SHAPES={'queries':[2,128],'output.weight':[4,256],'output.bias':[4]}

def head(path):
 raw=Path(path).read_bytes();require(len(raw)>=8,'short tensor header');n=struct.unpack('<Q',raw[:8])[0]
 require(0<n<=len(raw)-8,'bad header length');h=A.decode(raw[8:8+n]);require(set(h)==set(SHAPES),'tensor names')
 arr={};spans=[]
 for k,shape in SHAPES.items():
  v=h[k];require(set(v)=={'dtype','shape','data_offsets'} and v['dtype']=='F32' and v['shape']==shape,'tensor descriptor')
  start,end=v['data_offsets'];require(type(start) is int and type(end) is int and 0<=start<end<=len(raw)-8-n and end-start==4*np.prod(shape),'tensor range')
  spans.append((start,end));arr[k]=np.frombuffer(raw,offset=8+n+start,count=(end-start)//4,dtype='<f4').reshape(shape).copy();require(np.isfinite(arr[k]).all(),'nonfinite head')
 cursor=0
 for start,end in sorted(spans):require(start==cursor,'gapped/overlap head');cursor=end
 require(cursor==len(raw)-8-n,'trailing tensor data');return raw,h,arr,8+n

def sharpen(p):
 require(p.shape==(1024,2,64) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all(),'attention shape/range')
 require(np.allclose(p.sum(2),1,atol=1e-5,rtol=1e-5),'attention sum')
 with np.errstate(divide='ignore'):z=16*np.log(p)
 z-=z.max(2,keepdims=True);q=np.exp(z);q/=q.sum(2,keepdims=True);require(np.isfinite(q).all(),'power nonfinite');return q

def loadrows(path):return A.rows(Path(path).read_bytes(),1024)
def identities(rows):
 audits=[{k:r[k] for k in A.AUDIT_FIELDS} for r in rows]
 for i,row in enumerate(audits):A.validate_identity(row,'seen','factual',i)
 return audits

def prepare(c):
 require(not (c/'premise.json').exists(),'premise already exists');save(c/'clock.json',dict(started_local=now()))
 outer=R12/'completed-campaign.manifest.json';require(digest(outer)=='0a4ac9a8d3f46ebfca141360384a18761ea4c9fd175c43176788dd422ec7ab6f','parent seal')
 seal=read(outer);verify_files(seal['external_bindings'])
 require({str(p.relative_to(PARENT)) for p in PARENT.rglob('*') if p.is_file()}==set(seal['files']),'parent inventory')
 for rel,row in seal['files'].items():require(digest(PARENT/rel)==row['sha256'],'parent file differs')
 baseline=loadrows(PARENT/'final-seen-factual/evaluation-rows.jsonl');audit=identities(baseline)
 p=np.asarray([r['attention'] for r in baseline],dtype=np.float64);q=sharpen(p)
 roles=np.asarray([[r['agent_patch'],r['goal_patch']] for r in audit]);winner=p.argmax(2)
 require((winner==roles).all() and (p==p.max(2,keepdims=True)).sum(2).max()==1,'role winner/tie')
 mass=np.take_along_axis(q,roles[:,:,None],2).squeeze(2)
 premise=dict(accepted=bool((mass>=.99).all()),classification='exploratory',multiplier=16,predicted_min_target_mass=mass.min(0).tolist(),predicted_mean_target_mass=mass.mean(0).tolist(),source_parent_seal=digest(outer),created_local=now())
 save(c/'premise.json',premise)
 if not premise['accepted']:print(json.dumps(premise));return
 original=PARENT/'train-seed0/final-head.safetensors';raw,h,arr,offset=head(original)
 start,end=h['queries']['data_offsets'];scaled=np.ldexp(arr['queries'],4).astype('<f4');require(np.isfinite(scaled).all(),'query overflow')
 new=raw[:offset+start]+scaled.tobytes()+raw[offset+end:];path=c/'queries-x16.safetensors';
 with path.open('xb') as output:output.write(new)
 raw2,h2,arr2,off2=head(path);require(h==h2 and offset==off2 and np.array_equal(arr2['queries'].view('u4'),scaled.view('u4')),'scaled identity')
 require(raw2[:offset+start]==raw[:offset+start] and raw2[offset+end:]==raw[offset+end:],'foreign bytes changed')
 for name in ('output.weight','output.bias'):require(arr[name].tobytes()==arr2[name].tobytes(),'affine tensor changed')
 save(c/'transformation.json',dict(accepted=True,original_sha256=digest(original),scaled_sha256=digest(path),query_multiplier=16,unchanged_affine_and_foreign_bytes=True,header_unchanged=True))
 print(json.dumps(premise))

def launch(c,spec_sha):
 require(spec_sha is not None and digest(c/'launch-spec.json')==spec_sha,'launch specification pin differs')
 require(read(c/'premise.json')['accepted'],'premise failed');spec=read(c/'launch-spec.json');verify_files(spec['frozen_files'])
 cfg=read(c/'config.json');auth=read(c/'authority.json')
 tracked(dict(campaign=str(c),repository=REPO),[PYTHON,'-B',str(R12/'supervise.py'),'--config',str(c/'config.json'),'--sha256',digest(c/'config.json'),'--authority',str(c/'authority.json'),'--authority-sha256',digest(c/'authority.json')],'evaluate',260)
 state=read(c/'sharpened-seen.exit.json');require(state['accepted'] and state['bindings_unchanged'],'eval failed')
 tracked(dict(campaign=str(c),repository=REPO),[PYTHON,'-B',str(R12/'bind_nsight.py'),'--root',str(c/'sharpened-seen')],'bind',120)
 profiles=read(c/'sharpened-seen.bound/summary.json');require(len(profiles)==1 and all(p['health']['structurally_valid'] and p['health']['capture_complete'] and p['raw_application_labels_verified'] for p in profiles),'profile failed')
 report=read(c/'sharpened-seen/report.json')
 for key,value in auth['expected_report'].items():require(report[key]==value,'report differs')
 root_manifest(c/'sharpened-seen');verify_files(spec['frozen_files'])
 files=dict(spec['frozen_files']);files[str(c/'sharpened-seen/evaluation-rows.jsonl')]=digest(c/'sharpened-seen/evaluation-rows.jsonl')
 receipt=c/'integrity-receipt.json';save(receipt,dict(accepted=True,zero_updates=True,source_verified=True,profile_verified=True,files=files))
 def bound(p):return dict(path=str(p),sha256=digest(p))
 save(c/'analysis-config.json',dict(baseline=bound(PARENT/'final-seen-factual/evaluation-rows.jsonl'),treatment=bound(c/'sharpened-seen/evaluation-rows.jsonl'),original_head=bound(PARENT/'train-seed0/final-head.safetensors'),scaled_head=bound(c/'queries-x16.safetensors'),audit=bound(PARENT/'audit/seen-factual-audit.jsonl'),integrity_receipt=bound(receipt),registration=str(R/'registration.md'),registration_sha256=digest(R/'registration.md')))
 print(json.dumps(dict(evaluation_accepted=True,model_phase_seconds=state['model_phase_seconds'])))

def analyze(c):
 cfg=read(c/'analysis-config.json');verify_files({x['path']:x['sha256'] for x in cfg.values() if isinstance(x,dict)})
 receipt=read(cfg['integrity_receipt']['path']);require(all(receipt[k] is True for k in ('accepted','zero_updates','source_verified','profile_verified')),'receipt');verify_files(receipt['files'])
 orig,oh,oa,offset=head(cfg['original_head']['path']);new,nh,na,noffset=head(cfg['scaled_head']['path']);require(oh==nh and offset==noffset,'head header changed')
 start,end=oh['queries']['data_offsets'];require(new[:offset+start]==orig[:offset+start] and new[offset+end:]==orig[offset+end:],'foreign head bytes changed')
 require(np.array_equal(na['queries'].view('u4'),np.ldexp(oa['queries'],4).astype('<f4').view('u4')),'query scaling differs')
 baseline=loadrows(cfg['baseline']['path']);treatment=loadrows(cfg['treatment']['path']);audit=loadrows(cfg['audit']['path']);require(identities(baseline)==audit and identities(treatment)==audit,'audit identity')
 b,bg=A.measure(baseline,audit,'final');t,tg=A.measure(treatment,audit,'final')
 p=np.asarray([r['attention'] for r in baseline]);q=sharpen(p);actual=np.asarray([r['attention'] for r in treatment]);roles=np.asarray([[r['agent_patch'],r['goal_patch']] for r in audit]);mass=np.take_along_axis(actual,roles[:,:,None],2).squeeze(2);predmass=np.take_along_axis(q,roles[:,:,None],2).squeeze(2)
 require((actual.argmax(2)==roles).all() and (mass>=.99).all(),'actual concentration/winners');require(np.allclose(actual,q,atol=1e-4,rtol=1e-4),'attention power mismatch')
 pooled=np.asarray([r['pooled'] for r in treatment]);logits=np.asarray([r['logits'] for r in treatment]);reconstructed=pooled@na['output.weight'].astype(np.float64).T+na['output.bias'].astype(np.float64)
 require(np.allclose(logits,reconstructed,atol=2e-5,rtol=2e-5),'affine reconstruction mismatch')
 indices=np.random.Generator(np.random.PCG64(1940)).integers(0,64,size=(10000,64));contrasts=A.bootstrap(dict(accuracy_delta=tg['accuracy']-bg['accuracy'],ce_delta=tg['ce']-bg['ce'],accuracy_minus_constant=tg['accuracy']-.25),indices)
 gain=contrasts['accuracy_delta']['ci95'][0]>0
 decision='frozen_sharpening_recovers_registered_fit' if t['accuracy']>=.90 and gain else 'partial_accuracy_gain_only' if gain else 'concentration_insufficient_for_accuracy_recovery'
 result=dict(accepted=True,classification='exploratory_reused_seen_cohort',decision=decision,baseline=b,treatment=t,contrasts=contrasts,controls=dict(power_max_absolute_error=float(abs(actual-q).max()),affine_max_absolute_error=float(abs(logits-reconstructed).max()),role_mass_min=mass.min(0).tolist(),role_mass_mean=mass.mean(0).tolist(),predicted_role_mass_min=predmass.min(0).tolist(),predicted_role_mass_mean=predmass.mean(0).tolist()),config_sha256=digest(c/'analysis-config.json'),registration_sha256=digest(R/'registration.md'),created_local=now())
 elapsed=(datetime.datetime.now().astimezone()-datetime.datetime.fromisoformat(read(c/'clock.json')['started_local'])).total_seconds();require(0<=elapsed<=600,'execution/analysis budget')
 result['execution_wall_seconds']=elapsed;save(c/'analysis.json',result);print(json.dumps(result))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','launch','analyze'));p.add_argument('--campaign',type=Path,required=True);p.add_argument('--spec-sha256');args=p.parse_args();launch(args.campaign,args.spec_sha256) if args.stage=='launch' else globals()[args.stage](args.campaign)
