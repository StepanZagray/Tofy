"""Hand-built uniform-cell fixtures, never the C19 random generator or panel."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import analysis as a
import numpy_binding as b


def fixture(groups=4):
    rows=[];directions=[(0,-1),(0,1),(-1,0),(1,0)]
    for query in range(groups):
        for mapping,p in enumerate(itertools.permutations(range(4))):
            actions=[0,1,2];frames=[];agent=27
            for action in actions:
                before=[0]*64;before[63]=3;before[agent]=2;frames.append(before)
                dx,dy=directions[p[action]];agent+=dx+8*dy
                after=[0]*64;after[63]=3;after[agent]=2;frames.append(after)
            current=[int(i%8 in (0,7) or i//8 in (0,7)) for i in range(64)]
            dx,dy=directions[query%4];current[27]=2;current[27+dx+8*dy]=3;frames.append(current)
            meta=[]
            for frame in range(7):
                for cell in range(64):
                    fields=[0.]*10;fields[frame%2 if frame<6 else 2]=1
                    if frame<6:fields[3+actions[frame//2]]=1
                    fields[7]=float(np.float32(cell%8)/np.float32(7));fields[8]=float(np.float32(cell//8)/np.float32(7))
                    fields[9]=float(np.float32(frame//2)/np.float32(3)) if frame<6 else 1.
                    meta.extend(fields)
            row={'input_index':query*24+mapping,'query_index':query,'episode_id':a.TAG+query,'data_seed':a.SEED,'permutation_id':mapping,'policy_label':p.index(query%4),'public_cells':frames,'public_metadata':meta}
            rehash(row);rows.append(row)
    return rows


def rehash(row):
    pixels=b''.join(int(cell).to_bytes(4,'little')*64 for frame in row['public_cells'] for cell in frame)
    meta=np.array(row['public_metadata'],dtype='<f4').tobytes()
    row.update(input_sha256=hashlib.sha256(pixels+meta).hexdigest(),query_sha256=hashlib.sha256(pixels[-4096*4:]).hexdigest(),metadata_sha256=hashlib.sha256(meta).hexdigest())


def oracle_attention(rows):
    weights=np.zeros((len(rows),7,2,64))
    for n,row in enumerate(rows):
        for frame,cells in enumerate(row['public_cells']):
            for role,color in enumerate((2,3)):weights[n,frame,role,cells.index(color)]=1
    return weights


def output_rows(rows,attention,control='factual'):
    result=[]
    for i,(row,att) in enumerate(zip(rows,attention)):
        # Independent scalar expectations from public metadata, no producer adapter.
        positions=[]
        for frame in range(7):
            roles=[]
            for role in range(2):
                roles.append([sum(float(att[frame,role,k] if control=='factual' else 1/64)*float(np.float32(row['public_metadata'][(frame*64+k)*10+7+d])*np.float32(7)) for k in range(64)) for d in range(2)])
            positions.append(roles)
        rec=[]
        for step in range(3):
            rec.append([positions[2*step+1][0][d]-positions[2*step][0][d] for d in range(2)]+[float(k==step) for k in range(4)]+[0.])
        rec.append([positions[6][1][d]-positions[6][0][d] for d in range(2)]+[0.,0.,0.,0.,1.])
        result.append(row|{'evaluation_index':i,'learned_attention':att.tolist(),'adapter_records':np.array(rec,dtype=np.float32).tolist(),'logits':[0.,0.,0.,0.],'prediction':0,'control':control})
    return result


class ScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=fixture();cls.truth,cls.summary=a.panel(cls.rows,4);cls.attention=oracle_attention(cls.rows)

    def test_independent_uniform_cell_hashes_labels_balance(self):
        self.assertEqual(self.summary['label_histogram'],[24]*4)
        self.assertEqual(self.summary['direction_histogram'],[1]*4)
        self.assertEqual(self.summary['demonstrated'],72);self.assertEqual(self.summary['omitted'],24)
        ground=a.ground(self.attention,self.truth)
        self.assertEqual(ground['all_roles_correct'],96);self.assertEqual(ground['consumed_roles_correct'],96)
        for item in ground['displacements']:self.assertEqual(item['correct_argmax_delta'],96);self.assertLess(item['maximum_absolute_error'],1e-6)

    def test_label_hash_geometry_metadata_and_bool_corruptions(self):
        mutations=[lambda r:r.update(policy_label=3),lambda r:r.update(input_sha256='0'*64),lambda r:r['public_metadata'].__setitem__(3,False),lambda r:r['public_metadata'].__setitem__(9,.125),lambda r:r.update(input_index=False),lambda r:r['public_cells'][0].__setitem__(1,2)]
        for mutate in mutations:
            row=copy.deepcopy(self.rows[0]);mutate(row)
            with self.assertRaises(ValueError):a.geometry(row,0,4)
        row=copy.deepcopy(self.rows[0]);row['public_cells'][1][19]=0;row['public_cells'][1][28]=2;rehash(row)
        with self.assertRaises(ValueError):a.geometry(row,0,4)

    def test_panel_order_duplicates_and_missing_direction(self):
        rows=copy.deepcopy(self.rows);rows[0],rows[1]=rows[1],rows[0]
        with self.assertRaises(ValueError):a.panel(rows,4)
        with self.assertRaises(ValueError):a.panel(self.rows[:72],3)
        rows=copy.deepcopy(self.rows);rows[24]['public_cells'][6]=rows[0]['public_cells'][6];rehash(rows[24])
        with self.assertRaises(ValueError):a.panel(rows,4)

    def test_correct_delta_does_not_imply_correct_locations(self):
        shifted=self.attention.copy()
        for n,row in enumerate(self.rows):
            for f in range(6):
                loc=row['public_cells'][f].index(2);shifted[n,f,0]=0;shifted[n,f,0,loc+8]=1
        g=a.ground(shifted,self.truth)
        self.assertEqual(g['consumed_roles_correct'],0)
        for d in g['displacements'][:3]:self.assertEqual(d['correct_argmax_delta'],96);self.assertEqual(d['correct_argmax_locations'],0)

    def test_soft_mass_can_preserve_winners_and_move_expectations(self):
        att=.51*self.attention+.49/64
        g=a.ground(att,self.truth)
        self.assertEqual(g['all_roles_correct'],96)
        self.assertGreater(g['frames'][0]['agent']['mean_position_l1_error'],.1)
        self.assertLess(g['frames'][0]['agent']['mean_mass'],.52)

    def test_uniform_control_and_missing_id_baseline(self):
        records=a.records_from_attention(np.full_like(self.attention,1/64),self.truth)
        np.testing.assert_array_equal(records[:,:,:2],0)
        logits=np.tile([0.,0.,0.,1.],(96,1));control=a.controls(self.truth,logits,records)
        self.assertEqual(control['always_missing_correct'],24);self.assertTrue(control['uniform_six_correct_per_query'])
        s=a.action_summary(logits,self.truth)
        self.assertEqual(s['correct'],24);self.assertEqual(s['subsets']['omitted']['correct'],24);self.assertEqual(s['subsets']['demonstrated']['correct'],0)
        self.assertEqual(s['map_splits']['familiar']['rows'],64);self.assertEqual(s['map_splits']['heldout']['rows'],32)
        logits[0]=[0,0,2,1]
        with self.assertRaises(ValueError):a.controls(self.truth,logits,records)

    def test_actual_records_replayed_from_attention_and_mutation_fails(self):
        rows=output_rows(self.rows,self.attention)
        weights={name:np.zeros(shape,dtype='<f4') for name,shape in b.expected_shapes().items()}
        scored=a.score_stream(rows,self.rows,self.truth,'factual',weights)
        self.assertEqual(scored[0]['correct'],24);self.assertEqual(scored[2]['logits']['ineligible_winners'],96)
        rows[0]['adapter_records'][0][0]+=.01
        with self.assertRaises(ValueError):a.score_stream(rows,self.rows,self.truth,'factual',weights)

    def test_bad_attention_nonfinite_prediction_and_identity_reject(self):
        weights={name:np.zeros(shape,dtype='<f4') for name,shape in b.expected_shapes().items()}
        mutations=[lambda r:r['learned_attention'][0][0].__setitem__(0,.1),lambda r:r['logits'].__setitem__(0,float('nan')),lambda r:r.update(prediction=1),lambda r:r.update(control='uniform_attention'),lambda r:r.update(policy_label=1),lambda r:r.update(evaluation_index=True)]
        for mutate in mutations:
            rows=output_rows(self.rows,self.attention);mutate(rows[0])
            with self.assertRaises(ValueError):a.score_stream(rows,self.rows,self.truth,'factual',weights)

    def test_ce_uses_stable_logits_and_includes_every_registered_stratum(self):
        z=np.full((96,4),-1000.);labels=np.array([x['label'] for x in self.truth]);z[np.arange(96),labels]=1000
        s=a.action_summary(z,self.truth)
        self.assertEqual(s['correct'],96);self.assertEqual(s['ce'],0);self.assertEqual(s['minimum_true_margin'],2000)
        self.assertEqual(len(s['per_group']),4);self.assertEqual(len(s['per_map']),24);self.assertEqual(len(s['per_direction']),4)
        self.assertEqual(s['group_accuracy_range'],[1.,1.])

    def test_gate_conjunction_and_no_grounding_threshold(self):
        factual={'correct':768,'all_maps_correct_groups':32,'minimum_true_margin':.001,'subsets':{'demonstrated':{'correct':576},'omitted':{'correct':192}}}
        control={'uniform_identical_records_within_query':True,'uniform_identical_winners_within_query':True,'uniform_six_correct_per_query':True}
        summaries={'factual':factual,'uniform':{'correct':192}}
        self.assertEqual(a.decision(summaries,control)[1],'supported_frozen_native_composition')
        for key in control:
            modified=control|{key:False};self.assertFalse(all(a.decision(summaries,modified)[0].values()))
        for key,value in [('correct',767),('minimum_true_margin',.0009999),('all_maps_correct_groups',31)]:
            altered=copy.deepcopy(summaries);altered['factual'][key]=value
            self.assertFalse(all(a.decision(altered,control)[0].values()))

    def test_hash_duplicate_json_symlink_and_nonfinite_parser(self):
        with tempfile.TemporaryDirectory(prefix='c19-scorer-') as temp:
            path=Path(temp)/'fixture.json';path.write_text('{}')
            digest=hashlib.sha256(path.read_bytes()).hexdigest();self.assertEqual(a.document({'path':str(path),'sha256':digest}),{})
            with self.assertRaises(ValueError):a.verified({'path':str(path),'sha256':'0'*64})
            link=Path(temp)/'link';link.symlink_to(path)
            with self.assertRaises(ValueError):a.sha(link)
        for raw in ['{"x":1,"x":2}','{"x":NaN}']:
            with self.assertRaises(ValueError):a.parse(raw)

    def test_invalid_receipt_never_becomes_scientific_negative(self):
        config={'schema':a.SCHEMA,'registration':{},'history':{},'audit':{},'checkpoint':{},'integrity_receipt':{},'streams':{'factual':{},'uniform':{}},'frozen_files':{}}
        with patch.object(a,'document',return_value={'schema':'looped-native-binding-integrity-v1','accepted':False}):
            with self.assertRaises(ValueError):a.validate_config(config)

    def test_premise_sealed_manifest_work_and_source_guards(self):
        with tempfile.TemporaryDirectory(prefix='c19-premise-') as temp:
            root=Path(temp)/'audit';root.mkdir()
            report={'status':'complete_pending_analysis','classification':'input_audit','model_forwards':0,'optimizer_updates':0,'input_rows':768,'query_groups':32,'panel_seed':a.SEED,'panel_tag':a.TAG,'label_counts':[192]*4,'excluded_unique_queries':0}
            meta={'schema':'looped-native-binding-v1','device':'cpu','objective':'public_input_audit','model_forwards':0,'optimizer_updates':0,'provenance':{'source_revision':'1'*40,'binary_sha256':'2'*64,'candle_graph_revision':a.DEPENDENCY}}
            config={'schema':'looped-native-binding-premise-v1','source_revision':'1'*40,'binary_sha256':'2'*64,'frozen_files':{}}
            def seal():
                content={'report.json':report,'metadata.json':meta,'profiles.json':{'files':{}},'launch.json':{}}
                for name,value in content.items():(root/name).write_text(json.dumps(value))
                (root/'panel-rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
                manifest={'schema':'looped-grounded-policy-artifacts-v1','files':{name:a.sha(root/name) for name in [*content,'panel-rows.jsonl']}}
                (root/'manifest.json').write_text(json.dumps(manifest))
                outer=root.with_suffix('.manifest.sha256');outer.write_text(a.sha(root/'manifest.json')+'\n')
                config['frozen_files']={str(path):a.sha(path) for path in [*root.iterdir(),outer]}
                for key,name in [('audit','panel-rows.jsonl'),('audit_manifest','manifest.json')]:config[key]={'path':str(root/name),'sha256':a.sha(root/name)}
                for name in ['registration','history']:
                    path=Path(temp)/(name+'.json');path.write_text('{}');config[name]={'path':str(path),'sha256':a.sha(path)};config['frozen_files'][str(path)]=a.sha(path)
                for name in ['analysis.py','analysis_tests.py','numpy_binding.py','numpy_binding_tests.py']:
                    path=Path(a.__file__).with_name(name).resolve();config['frozen_files'][str(path)]=a.sha(path)
            seal()
            # Four hand-built query groups exercise arithmetic; this mock alone
            # excludes real registered population generation from the fixture.
            with patch.object(a,'panel',return_value=(self.truth,self.summary)),patch.object(a,'historical',return_value={'unique_queries':0}):
                self.assertTrue(a.premise(config)['accepted'])
                report['model_forwards']=1;seal()
                with self.assertRaises(ValueError):a.premise(config)
                report['model_forwards']=0;meta['provenance']['source_revision']='3'*40;seal()
                with self.assertRaises(ValueError):a.premise(config)
                meta['provenance']['source_revision']='1'*40;seal();(root/'report.json').write_text('{}')
                with self.assertRaises(ValueError):a.premise(config)

    def test_runtime_closure_rejects_unbound_receipt_or_source(self):
        config={'schema':a.SCHEMA,'streams':{},'frozen_files':{}}
        for key in ['registration','history','audit','checkpoint','integrity_receipt']:
            config[key]={'path':'/synthetic/'+key,'sha256':b.CHECKPOINT_SHA256 if key=='checkpoint' else '4'*64}
        for name in ['factual','uniform']:config['streams'][name]={'path':'/synthetic/'+name,'sha256':'5'*64}
        bindings={spec['path']:spec['sha256'] for spec in [config[k] for k in ['registration','history','audit','checkpoint']]+list(config['streams'].values())}
        bindings|={'/synthetic/vision':a.VISION,'/synthetic/selector':a.SELECTOR}
        for name in ['analysis.py','analysis_tests.py','numpy_binding.py','numpy_binding_tests.py']:bindings[str(Path(a.__file__).with_name(name).resolve())]='6'*64
        receipt={'schema':'looped-native-binding-integrity-v1','accepted':True,'checks':dict.fromkeys(a.CHECKS,True),'source_revision':'1'*40,'binary_sha256':'2'*64,'dependency_revision':a.DEPENDENCY,'vision_checkpoint_sha256':a.VISION,'selector_sha256':a.SELECTOR,'binder_checkpoint_sha256':b.CHECKPOINT_SHA256,'vision_loops':4,'binder_loops':4,'frozen_files':bindings}
        config['frozen_files']=bindings|{config['integrity_receipt']['path']:config['integrity_receipt']['sha256']}
        with patch.object(a,'document',return_value=receipt),patch.object(a,'sha',side_effect=lambda p:config['frozen_files'][str(p)]):
            self.assertIs(a.validate_config(config),receipt)
            receipt['checks']['zero_updates']=False
            with self.assertRaises(ValueError):a.validate_config(config)
            receipt['checks']['zero_updates']=True
            config['frozen_files']['/synthetic/unbound']='7'*64
            with self.assertRaises(ValueError):a.validate_config(config)


class MetadataIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity=fixture()[0]

    def test_equivalent_decimal_json_is_exact_f32_identity(self):
        row=copy.deepcopy(self.identity)
        self.assertEqual(row['public_metadata'][17],0.1428571492433548)
        row['public_metadata'][17]=0.14285715
        self.assertNotEqual(json.dumps(row,sort_keys=True),json.dumps(self.identity,sort_keys=True))
        a.validate_audit_identity(row,self.identity)
        a.validate_audit_identity(self.identity,row)
        self.assertEqual(a.geometry(row,0,4)['label'],self.identity['policy_label'])

    def test_one_ulp_and_signed_zero_are_distinct(self):
        row=copy.deepcopy(self.identity)
        value=np.float32(row['public_metadata'][17])
        for direction in [np.float32(np.inf),np.float32(-np.inf)]:
            row['public_metadata'][17]=float(np.nextafter(value,direction))
            with self.assertRaises(ValueError):a.validate_audit_identity(row,self.identity)
        row=copy.deepcopy(self.identity);row['public_metadata'][1]=-0.0
        with self.assertRaises(ValueError):a.validate_audit_identity(row,self.identity)

    def test_boolean_nonfinite_overflow_shape_and_string_reject(self):
        for value in [False,True,float('nan'),float('inf'),float('-inf'),1e40,'0.14285715']:
            row=copy.deepcopy(self.identity);row['public_metadata'][17]=value
            with self.assertRaises(ValueError):a.validate_audit_identity(row,self.identity)
            with self.assertRaises(ValueError):a.validate_audit_identity(self.identity,row)
        row=copy.deepcopy(self.identity);row['public_metadata'].pop()
        with self.assertRaises(ValueError):a.validate_audit_identity(row,self.identity)

    def test_all_nonmetadata_fields_keep_exact_typed_identity(self):
        mutations=[lambda r:r.update(policy_label=False),lambda r:r.update(input_index=0.0),lambda r:r['public_cells'][0].__setitem__(0,0.0),lambda r:r.update(metadata_sha256='0'*64),lambda r:r.update(query_sha256='0'*64),lambda r:r.update(input_sha256='0'*64)]
        for mutate in mutations:
            row=copy.deepcopy(self.identity);row['public_metadata'][17]=0.14285715;mutate(row)
            with self.assertRaises(ValueError):a.validate_audit_identity(row,self.identity)


if __name__=='__main__':unittest.main()
