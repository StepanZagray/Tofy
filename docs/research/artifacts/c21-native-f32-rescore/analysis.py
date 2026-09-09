"""C19 finite native composition scorer. Geometry and binder math are independent of Rust."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import signal
import time
import numpy as np
import numpy_binding as binder

SCHEMA = 'looped-native-binding-analysis-config-v1'
ROW_KEYS = {'input_index','query_index','episode_id','data_seed','permutation_id','policy_label','input_sha256','query_sha256','metadata_sha256','public_cells','public_metadata'}
EXTRA_KEYS = {'evaluation_index','learned_attention','adapter_records','logits','prediction','control'}
CHECKS = {'source','build','checkpoints','zero_updates','unchanged_parameters','qualification','profiles','cleanup'}
DIRECTIONS = [(0,-1),(0,1),(-1,0),(1,0)]
FIT_MAPS = [0,2,4,5,7,8,9,10,13,14,15,16,18,19,21,23]
MAPS = list(itertools.permutations(range(4)))
SEED, TAG, GROUPS = 20260923, 0x4e415449564542, 32
DEPENDENCY = '1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a'
VISION = '4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802'
SELECTOR = 'a43ad78fe7a6b1594d23c205fb191c8ac81610deaaff25198cb421d423edc678'
require = binder.require


def sha(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and path.resolve() == path, 'absolute regular nonsymlink file required')
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def verified(spec):
    require(isinstance(spec,dict) and set(spec)=={'path','sha256'}, 'file binding schema')
    require(isinstance(spec['sha256'],str) and re.fullmatch('[0-9a-f]{64}',spec['sha256']), 'SHA syntax')
    require(sha(spec['path'])==spec['sha256'], 'file hash mismatch: '+str(spec['path']))
    return Path(spec['path'])


def parse(raw):
    return json.loads(raw,object_pairs_hook=binder.unique_object,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def document(spec):
    path=verified(spec);require(path.stat().st_size <= 128*1024**2,'JSON size cap')
    return parse(path.read_bytes())


def jsonl(spec):
    path=verified(spec);require(path.stat().st_size <= 128*1024**2,'JSONL size cap')
    with path.open() as handle:
        return [parse(line) for line in handle if line.strip()]


def numeric(value,shape,integer=False):
    objects=np.asarray(value,dtype=object)
    require(objects.shape==shape,'numeric array shape')
    allowed=(int,) if integer else (int,float)
    require(all(type(v) in allowed for v in objects.flat),'numeric array scalar type')
    result=np.asarray(value,dtype=np.int64 if integer else np.float64)
    require(np.isfinite(result).all(),'nonfinite array')
    if not integer:
        with np.errstate(over='ignore'): result=result.astype(np.float32).astype(np.float64)
        require(np.isfinite(result).all(),'nonfinite F32 array')
    return result


def public_metadata(actions):
    result=np.zeros((7,64,10),dtype='<f4')
    for frame in range(7):
        result[frame,:,frame%2 if frame<6 else 2]=1
        if frame<6:result[frame,:,3+actions[frame//2]]=1
        result[frame,:,7]=np.arange(64,dtype=np.float32)%8/np.float32(7)
        result[frame,:,8]=np.arange(64,dtype=np.float32)//8/np.float32(7)
        result[frame,:,9]=np.float32(frame//2)/np.float32(3) if frame<6 else 1
    return result


def geometry(row,index,groups=GROUPS):
    require(set(row)==ROW_KEYS,'audit row keys')
    for key in ['input_index','query_index','episode_id','data_seed','permutation_id','policy_label']:
        require(type(row[key]) is int,'integer identity')
    q,m=divmod(index,24)
    require(index<groups*24 and row['input_index']==index and row['query_index']==q and row['permutation_id']==m,'panel order/membership')
    require(row['episode_id']==TAG+q and row['data_seed']==SEED,'panel seed/episode ID')
    cells=numeric(row['public_cells'],(7,64),True)
    require(((cells>=0)&(cells<=3)).all(),'cell palette')
    role=np.empty((7,2),dtype=np.int64)
    for frame in range(7):
        for slot,color in enumerate((2,3)):
            locations=np.flatnonzero(cells[frame]==color)
            require(len(locations)==1,'visible unique agent/goal')
            role[frame,slot]=locations[0]
    require(not (cells[:6]==1).any() and (role[:6,1]==63).all(),'support wall/goal contract')
    require(2<=role[0,0]//8<=5 and 2<=role[0,0]%8<=5,'support initial range')
    require(np.array_equal(cells[1],cells[2]) and np.array_equal(cells[3],cells[4]),'support chronology')
    boundaries=[i for i in range(64) if i%8 in (0,7) or i//8 in (0,7)]
    require((cells[6,boundaries]==1).all(),'query boundary walls')
    xy=np.stack((role%8,role//8),axis=-1)
    deltas=np.array([xy[2*s+1,0]-xy[2*s,0] for s in range(3)]+[xy[6,1]-xy[6,0]])
    require(all(tuple(d) in DIRECTIONS for d in deltas),'noncardinal support/query displacement')
    meta=numeric(row['public_metadata'],(4480,)).astype('<f4').reshape(7,64,10)
    actions=meta[[0,2,4],0,3:7].argmax(axis=1).tolist()
    require(len(set(actions))==3 and meta.tobytes()==public_metadata(actions).tobytes(),'public metadata/order/actions')
    require(all(tuple(deltas[s])==DIRECTIONS[MAPS[m][action]] for s,action in enumerate(actions)),'visible support/control mapping')
    direction=DIRECTIONS.index(tuple(deltas[3]));label=MAPS[m].index(direction)
    require(row['policy_label']==label,'visible policy label')
    pixels=np.repeat(cells.astype('<u4'),64,axis=1)
    hashes={'input_sha256':hashlib.sha256(pixels.tobytes()+meta.tobytes()).hexdigest(), 'query_sha256':hashlib.sha256(pixels[6].tobytes()).hexdigest(), 'metadata_sha256':hashlib.sha256(meta.tobytes()).hexdigest()}
    require(all(row[k]==v for k,v in hashes.items()),'reconstructed input/query/metadata hash')
    return {'roles':role,'xy':xy,'deltas':deltas,'actions':actions,'label':label,'demonstrated':label in actions,'direction':direction,'metadata':meta}


def panel(rows,groups=GROUPS):
    require(len(rows)==groups*24,'panel row count')
    truth=[geometry(row,i,groups) for i,row in enumerate(rows)]
    queries=set();inputs=set()
    for group in range(groups):
        first=rows[group*24];base=truth[group*24]
        require(first['query_sha256'] not in queries,'duplicate query group');queries.add(first['query_sha256'])
        for i in range(group*24,(group+1)*24):
            require(rows[i]['query_sha256']==first['query_sha256'] and rows[i]['public_cells'][6]==first['public_cells'][6],'all-map query pairing')
            require(truth[i]['actions']==base['actions'] and np.array_equal(truth[i]['roles'][0],base['roles'][0]),'all-map calibration pairing')
            require(rows[i]['input_sha256'] not in inputs,'duplicate full input');inputs.add(rows[i]['input_sha256'])
        labels=[x['label'] for x in truth[group*24:(group+1)*24]]
        require(np.array_equal(np.bincount(labels,minlength=4),[6]*4),'per-query label balance')
        require(sum(x['demonstrated'] for x in truth[group*24:(group+1)*24])==18,'per-query demonstrated balance')
    directions=np.bincount([truth[i*24]['direction'] for i in range(groups)],minlength=4)
    require((directions>0).all(),'all four desired directions required without panel repair')
    summary={'rows':len(rows),'groups':groups,'label_histogram':np.bincount([x['label'] for x in truth],minlength=4).tolist(),'direction_histogram':directions.tolist(),'demonstrated':sum(x['demonstrated'] for x in truth),'omitted':sum(not x['demonstrated'] for x in truth)}
    return truth,summary


def historical(history_spec,queries):
    history=document(history_spec)
    expected={
        '09559eb4dd1b933de724da81aef3bb80547e010a00e38e39de31c9f72fab1c6d':768,
        'bd72fd9707759dd19074a22b9d62899419b28711722b3617c21cc277a6a25c0d':256,
        '09a7aa423b17f32bbae14ae9178a15d5fff8428936afdf23ebc4f09878a6ddca':256,
        '61ae87a267cc545b05884dc412160c0313a0c2bc1015cfa2323bde256f91eaeb':256,
        '55f200b3a136476896d57f284a7d5db4e8f8fc2b7b4fa54ef309d1837f3c5910':73600,
        'fbb137383a16a27d72141ef66028c1c9cb470afccc2b03864ec0a961a94d73f2':1024}
    require(set(history)>= {'history'} and len(history['history'])==6 and {x['sha256'] for x in history['history']}==set(expected),'exact six historical files required')
    known=set();files=[];seen_paths=set()
    for spec in history['history']:
        require(spec['path'] not in seen_paths,'duplicate history source');seen_paths.add(spec['path'])
        rows=jsonl(spec);require(len(rows)==expected[spec['sha256']],'historical row count');current=set()
        for row in rows:
            if 'id' in row and 'episode_id' in row:require(row['id']==row['episode_id'] and type(row['id']) is int,'historical identity alias')
            if 'id' in row:require(type(row['id']) is int and row['id']>=0,'historical id')
            if 'episode_id' in row:require(type(row['episode_id']) is int and row['episode_id']>=0,'historical episode ID')
            value=row.get('query_sha256');require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value),'historical query SHA')
            current.add(value)
        known.update(current);files.append(spec|{'rows':len(rows),'hash_occurrences':len(rows),'unique_queries':len(current)})
    overlap=known & set(queries);require(not overlap,'historical query collision')
    return {'files':files,'source_rows':sum(x['rows'] for x in files),'unique_queries':len(known),'overlap_queries':len(overlap)}


def positions(attention,truth):
    metadata=np.stack([x['metadata'] for x in truth])
    coordinates=(metadata[:,:,:,7:9]*np.float32(7)).astype(np.float64)
    return np.einsum('nfrp,nfpd->nfrd',attention,coordinates)


def records_from_attention(attention,truth):
    points=positions(attention,truth);records=np.zeros((len(truth),4,7),dtype=np.float64)
    for step in range(3):
        records[:,step,:2]=points[:,2*step+1,0]-points[:,2*step,0]
        for i,item in enumerate(truth):records[i,step,2+item['actions'][step]]=1
    records[:,3,:2]=points[:,6,1]-points[:,6,0];records[:,3,6]=1
    return records


def attention_array(value):
    a=numeric(value,(7,2,64))
    require(((a>=0)&(a<=1)).all() and (np.abs(a.sum(axis=-1)-1)<=1e-5).all(),'attention normalization/range')
    return a


def ground(attention,truth):
    roles=np.stack([x['roles'] for x in truth]);xy=np.stack([x['xy'] for x in truth])
    points=positions(attention,truth);prediction=attention.argmax(axis=-1)
    correct=prediction==roles;mass=np.take_along_axis(attention,roles[:,:,:,None],axis=3)[:,:,:,0]
    error=np.abs(points-xy);frames=[]
    for frame in range(7):
        entry={'frame_index':frame}
        for slot,name in enumerate(('agent','goal')):
            entry[name]={'correct':int(correct[:,frame,slot].sum()),'accuracy':float(correct[:,frame,slot].mean()),'mean_mass':float(mass[:,frame,slot].mean()),'minimum_mass':float(mass[:,frame,slot].min()),'mean_position_l1_error':float(error[:,frame,slot].sum(axis=-1).mean()),'maximum_position_linf_error':float(error[:,frame,slot].max())}
        frames.append(entry)
    predicted_xy=np.stack((prediction%8,prediction//8),axis=-1)
    deltas=records_from_attention(attention,truth)[:,:,:2];true_deltas=np.stack([x['deltas'] for x in truth])
    displacements=[]
    for record in range(4):
        a,b=((2*record,0),(2*record+1,0)) if record<3 else ((6,0),(6,1))
        delta=predicted_xy[:,b[0],b[1]]-predicted_xy[:,a[0],a[1]]
        e=np.abs(deltas[:,record]-true_deltas[:,record])
        displacements.append({'record_index':record,'correct_argmax_delta':int(np.all(delta==true_deltas[:,record],axis=-1).sum()),'correct_argmax_locations':int((correct[:,a[0],a[1]]&correct[:,b[0],b[1]]).sum()),'mean_absolute_error':float(e.mean()),'maximum_absolute_error':float(e.max())})
    return {'rows':len(truth),'all_roles_correct':int(correct.all(axis=(1,2)).sum()),'consumed_roles_correct':int((correct[:,:6,0].all(axis=1)&correct[:,6,:].all(axis=1)).sum()),'frames':frames,'displacements':displacements}


def action_summary(logits,truth):
    z=np.asarray(logits,dtype=np.float64);labels=np.array([x['label'] for x in truth]);pred=z.argmax(axis=1)
    true=z[np.arange(len(z)),labels];shift=z.max(axis=1)
    losses=np.log(np.exp(z-shift[:,None]).sum(axis=1))+shift-true
    wrong=z.copy();wrong[np.arange(len(z)),labels]=-np.inf;margins=true-wrong.max(axis=1)
    correct=pred==labels;dem=np.array([x['demonstrated'] for x in truth]);subsets={}
    for name,mask in [('demonstrated',dem),('omitted',~dem)]:
        subsets[name]={'rows':int(mask.sum()),'correct':int(correct[mask].sum()),'accuracy':float(correct[mask].mean()),'ce':float(losses[mask].mean()),'minimum_true_margin':float(margins[mask].min())}
    def subset(mask):
        return {'rows':int(mask.sum()),'correct':int(correct[mask].sum()),'accuracy':float(correct[mask].mean()),'ce':float(losses[mask].mean()),'minimum_true_margin':float(margins[mask].min())}
    indices=np.arange(len(z));directions=np.array([x['direction'] for x in truth])
    permap=[{'map_id':i}|subset(indices%24==i) for i in range(24)]
    pergroup=[{'query_index':i}|subset(indices//24==i) for i in range(len(z)//24)]
    perdirection=[{'direction':i}|subset(directions==i) for i in range(4)]
    familiar=np.isin(indices%24,FIT_MAPS)
    split={'familiar':subset(familiar),'heldout':subset(~familiar)}
    missing=np.array([next(a for a in range(4) if a not in t['actions']) for t in truth])
    return {'rows':len(z),'groups':len(z)//24,'correct':int(correct.sum()),'accuracy':float(correct.mean()),'ce':float(losses.mean()),'minimum_true_margin':float(margins.min()),'label_histogram':np.bincount(labels,minlength=4).tolist(),'prediction_histogram':np.bincount(pred,minlength=4).tolist(),'all_maps_correct_groups':int(correct.reshape(-1,24).all(axis=1).sum()),'missing_id_predictions':int((pred==missing).sum()),'subsets':subsets,'per_map':permap,'per_group':pergroup,'per_direction':perdirection,'map_splits':split,'group_accuracy_range':[min(x['accuracy'] for x in pergroup),max(x['accuracy'] for x in pergroup)]}


def validate_audit_identity(row,identity):
    keys=ROW_KEYS-{'public_metadata'}
    require(json.dumps({k:row[k] for k in keys},sort_keys=True)==json.dumps({k:identity[k] for k in keys},sort_keys=True),'evaluation audit identity')
    actual=numeric(row['public_metadata'],(4480,)).astype('<f4').tobytes()
    expected=numeric(identity['public_metadata'],(4480,)).astype('<f4').tobytes()
    require(actual==expected,'evaluation public metadata F32 identity')


def score_stream(output,audit,truth,name,weights):
    require(len(output)==len(audit),'output population count')
    learned=[];actual_records=[];logits=[]
    for index,(row,identity) in enumerate(zip(output,audit)):
        require(set(row)==ROW_KEYS|EXTRA_KEYS,'evaluation row keys')
        validate_audit_identity(row,identity)
        require(type(row['evaluation_index']) is int and row['evaluation_index']==index,'evaluation order')
        require(row['control']==('factual' if name=='factual' else 'uniform_attention'),'stream control')
        a=attention_array(row['learned_attention']);z=numeric(row['logits'],(4,))
        require(type(row['prediction']) is int and row['prediction']==int(z.argmax()),'stored prediction')
        learned.append(a);actual_records.append(numeric(row['adapter_records'],(4,7)));logits.append(z)
    learned=np.stack(learned);effective=learned if name=='factual' else np.full_like(learned,1/64)
    reconstructed=records_from_attention(effective,truth);actual_records=np.stack(actual_records);logits=np.stack(logits)
    require(np.array_equal(actual_records[:,:,2:],reconstructed[:,:,2:]),'exact public routing/record-kind columns')
    record_parity=binder.parity(actual_records,reconstructed)
    # Replay uses independently reconstructed records, never stored Rust records.
    replayed=binder.replay(reconstructed.astype(np.float32),weights,4)
    replay={'records':record_parity,'logits':binder.logit_parity(logits,replayed)}
    return action_summary(logits,truth),{'learned':ground(learned,truth),'effective':ground(effective,truth)},replay,logits,actual_records,learned


def controls(truth,uniform_logits,uniform_records):
    expected=records_from_attention(np.full((len(truth),7,2,64),1/64),truth)
    require(np.array_equal(expected[:,:,:2],np.zeros((len(truth),4,2))),'uniform analytic displacement')
    parity=binder.parity(uniform_records,expected)
    z=uniform_logits.reshape(-1,24,4);r=uniform_records.reshape(-1,24,4,7)
    equal_winners=bool((z.argmax(axis=2)==z[:,0:1].argmax(axis=2)).all())
    equal_records=bool((r==r[:,0:1]).all());require(equal_winners and equal_records,'uniform equal-input determinism')
    within=binder.logit_parity(z.reshape(-1,4),np.broadcast_to(z[:,0:1],z.shape).reshape(-1,4))
    balanced=bool(((z.argmax(axis=2)==np.array([x['label'] for x in truth]).reshape(-1,24)).sum(axis=1)==6).all())
    require(balanced,'uniform per-query exact quarter')
    labels=np.array([x['label'] for x in truth]);missing=np.array([next(a for a in range(4) if a not in t['actions']) for t in truth])
    analytic_correct=0
    for item,unobserved in zip(truth,missing):
        effects=np.zeros((4,2))
        for step,action in enumerate(item['actions']):effects[action]=item['deltas'][step]
        effects[unobserved]=-effects.sum(axis=0)
        analytic_correct+=int((effects@item['deltas'][3]).argmax()==item['label'])
    require(analytic_correct==len(truth),'external analytic geometry control')
    return {'constant_action_0_correct':int((labels==0).sum()),'always_missing_correct':int((labels==missing).sum()),'ideal_geometry_correct':analytic_correct,'uniform_six_correct_per_query':balanced,'uniform_identical_records_within_query':equal_records,'uniform_identical_winners_within_query':equal_winners,'uniform_records':parity,'uniform_logits_within_query':within}


def decision(summaries,control):
    f,u=summaries['factual'],summaries['uniform']
    gates={'factual_all_actions':f['correct']==768 and f['all_maps_correct_groups']==32 and f['subsets']['demonstrated']['correct']==576 and f['subsets']['omitted']['correct']==192,'factual_minimum_true_margin':f['minimum_true_margin']>=.001,'uniform_control':u['correct']==192 and control['uniform_identical_records_within_query'] and control['uniform_identical_winners_within_query'] and control['uniform_six_correct_per_query']}
    return gates,('supported_frozen_native_composition' if all(gates.values()) else 'frozen_native_composition_not_supported')


def validate_config(config):
    required={'schema','registration','history','audit','checkpoint','integrity_receipt','streams','frozen_files'}
    require(set(config)==required and config['schema']==SCHEMA,'analysis config schema')
    require(set(config['streams'])=={'factual','uniform'},'two scientific streams')
    require(Path(binder.__file__).resolve()==Path(__file__).with_name('numpy_binding.py').resolve(),'unexpected replay module')
    receipt=document(config['integrity_receipt'])
    require(receipt['schema']=='looped-native-binding-integrity-v1' and receipt['accepted'] is True,'accepted integrity receipt')
    require(set(receipt['checks'])==CHECKS and all(v is True for v in receipt['checks'].values()),'runtime integrity checks')
    require(re.fullmatch('[0-9a-f]{40}',receipt['source_revision']) and re.fullmatch('[0-9a-f]{64}',receipt['binary_sha256']),'source/binary identity syntax')
    require(receipt['dependency_revision']==DEPENDENCY and receipt['vision_checkpoint_sha256']==VISION and receipt['selector_sha256']==SELECTOR and receipt['binder_checkpoint_sha256']==binder.CHECKPOINT_SHA256,'fixed dependencies/checkpoints')
    require(type(receipt['vision_loops']) is int and receipt['vision_loops']==4 and type(receipt['binder_loops']) is int and receipt['binder_loops']==4,'four-loop frozen inference')
    require(config['checkpoint']['sha256']==binder.CHECKPOINT_SHA256,'binder checkpoint binding')
    files=config['frozen_files'];expected=receipt['frozen_files']|{config['integrity_receipt']['path']:config['integrity_receipt']['sha256']}
    require(files==expected,'receipt/config closed frozen bindings')
    for path,digest in files.items():verified({'path':path,'sha256':digest})
    for spec in [config[k] for k in ['registration','history','audit','checkpoint','integrity_receipt']]+list(config['streams'].values()):
        require(files.get(spec['path'])==spec['sha256'],'unbound analysis input')
    for script in ['analysis.py','analysis_tests.py','numpy_binding.py','numpy_binding_tests.py']:
        path=str(Path(__file__).with_name(script).resolve());require(files.get(path)==sha(path),'unbound analyzer source/tests')
    require(VISION in files.values() and SELECTOR in files.values(),'unbound vision/selector files')
    return receipt


def analyze(config):
    receipt=validate_config(config)
    audit=jsonl(config['audit']);truth,panel_summary=panel(audit)
    history=historical(config['history'],[x['query_sha256'] for x in audit])
    weights=binder.load_checkpoint(verified(config['checkpoint']))
    summaries={};grounding={};replay={};arrays={}
    for name in ['factual','uniform']:
        summaries[name],grounding[name],replay[name],*arrays[name]=score_stream(jsonl(config['streams'][name]),audit,truth,name,weights)
    control=controls(truth,arrays['uniform'][0],arrays['uniform'][1])
    control['paired_learned_attention']=binder.parity(arrays['factual'][2],arrays['uniform'][2])
    gates,verdict=decision(summaries,control)
    require(validate_config(config)==receipt,'runtime receipt drift during scoring')
    return {'schema':'looped-native-binding-analysis-v1','accepted':True,'registration_sha256':config['registration']['sha256'],'source_revision':receipt['source_revision'],'binary_sha256':receipt['binary_sha256'],'binder_checkpoint_sha256':binder.CHECKPOINT_SHA256,'history':history,'panel':panel_summary,'summaries':summaries,'grounding':grounding,'replay':replay,'controls':control,'gates':gates,'decision':verdict,'limits':['Finite 32-query panel, dependent 24 mappings; no population confidence or method promotion.','Frozen privileged visual selector; source/CUDA/lifecycle observations share the integrity receipt.','NumPy replays only the binder; no independent vision-forward replay.','No cross-map native-logit symmetry or useful-recurrence claim.']}



def premise(config):
    require(set(config)=={'schema','audit','audit_manifest','history','registration','source_revision','binary_sha256','frozen_files'} and config['schema']=='looped-native-binding-premise-v1','premise config schema')
    require(re.fullmatch('[0-9a-f]{40}',config['source_revision']) and re.fullmatch('[0-9a-f]{64}',config['binary_sha256']),'premise provenance syntax')
    files=config['frozen_files']
    for path,digest in files.items():verified({'path':path,'sha256':digest})
    for name in ['audit','audit_manifest','history','registration']:
        require(files.get(config[name]['path'])==config[name]['sha256'],'unbound premise input')
        verified(config[name])
    for name in ['analysis.py','analysis_tests.py','numpy_binding.py','numpy_binding_tests.py']:
        path=str(Path(__file__).with_name(name).resolve());require(files.get(path)==sha(path),'unbound premise scorer')
    manifest=document(config['audit_manifest']);root=Path(config['audit_manifest']['path']).parent
    require(manifest['schema']=='looped-grounded-policy-artifacts-v1' and set(manifest)=={'schema','files'},'CPU audit manifest schema')
    names={'launch.json','metadata.json','panel-rows.jsonl','profiles.json','report.json'}
    require(set(manifest['files'])==names and {p.name for p in root.iterdir()}==names|{'manifest.json'},'CPU audit inventory')
    require(Path(config['audit']['path'])==root/'panel-rows.jsonl' and manifest['files']['panel-rows.jsonl']==config['audit']['sha256'],'CPU audit row manifest binding')
    for name,digest in manifest['files'].items():
        path=str(root/name);require(files.get(path)==digest and sha(path)==digest,'CPU audit artifact binding')
    outer=root.with_suffix('.manifest.sha256');require(outer.read_text().strip()==config['audit_manifest']['sha256'] and files.get(str(outer))==sha(outer),'CPU audit outer manifest binding')
    report=document({'path':str(root/'report.json'),'sha256':manifest['files']['report.json']})
    require(report['status']=='complete_pending_analysis' and report['classification']=='input_audit','CPU audit report status')
    for key,value in {'model_forwards':0,'optimizer_updates':0,'input_rows':768,'query_groups':32,'panel_seed':SEED,'panel_tag':TAG}.items():
        require(type(report[key]) is int and report[key]==value,'CPU audit population/work: '+key)
    metadata=document({'path':str(root/'metadata.json'),'sha256':manifest['files']['metadata.json']})
    provenance=metadata['provenance']
    require(provenance['source_revision']==config['source_revision'] and provenance['binary_sha256']==config['binary_sha256'] and provenance['candle_graph_revision']==DEPENDENCY,'CPU audit source/binary/dependency binding')
    require(metadata['schema']=='looped-native-binding-v1' and metadata['device']=='cpu' and metadata['objective']=='public_input_audit','CPU audit metadata mode')
    for key in ['optimizer_updates','model_forwards']:require(type(metadata[key]) is int and metadata[key]==0,'CPU audit metadata work')
    rows=jsonl(config['audit']);_,summary=panel(rows)
    history=historical(config['history'],[x['query_sha256'] for x in rows])
    require(report['label_counts']==[192]*4 and report['excluded_unique_queries']==history['unique_queries'],'CPU report population parity')
    return {'schema':'looped-native-binding-premise-report-v1','accepted':True,'source_revision':config['source_revision'],'binary_sha256':config['binary_sha256'],'registration_sha256':config['registration']['sha256'],'audit_sha256':config['audit']['sha256'],'audit_manifest_sha256':config['audit_manifest']['sha256'],'panel':summary,'history':history,'classification':'input_audit_no_model_outputs'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);parser.add_argument('--premise',action='store_true');args=parser.parse_args()
    started=time.monotonic();signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('60-second CPU bound')));signal.alarm(60)
    out=Path(args.output);require(out.is_absolute() and not out.exists() and out.parent.resolve()==out.parent,'new absolute output')
    config_path=Path(args.config);config_hash=sha(config_path);config=document({'path':str(config_path),'sha256':config_hash})
    result=premise(config) if args.premise else analyze(config)
    require(sha(config_path)==config_hash,'analysis config drift')
    if args.premise:
        for path,digest in config['frozen_files'].items():verified({'path':path,'sha256':digest})
    result|={'config_sha256':config_hash,'elapsed_seconds':time.monotonic()-started,'pid':os.getpid()}
    with out.open('x') as handle:json.dump(result,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
    signal.alarm(0);print(json.dumps({'accepted':True,'decision':result.get('decision','input_premise_accepted'),'output':str(out),'sha256':sha(out)}))


if __name__=='__main__':main()
