#!/usr/bin/env python3
"""Synthetic C14 checks; independent analytic/normal-equation comparators only."""
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('independent',Path(__file__).with_name('independent_review.py'))
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
np=R.np

def normal_fit(x,y):
    mean=x.mean(0);std=x.std(0);scale=np.maximum(std,1e-6);z=(x-mean)/scale
    target=np.eye(4)[y];intercept=target.mean(0);rhs=z.T@(target-intercept)/len(x)
    normal=z.T@z/len(x)+.01*np.eye(256)
    b=np.linalg.solve(normal,rhs) if np.any(rhs) else np.zeros((256,4))
    return dict(mean=mean.tolist(),std=std.tolist(),scale=scale.tolist(),intercept=intercept.tolist(),coefficients=b.tolist(),
                objective=float(np.sum((z@b+intercept-target)**2)/len(x)+.01*np.sum(b*b)),
                condition_number=float(np.linalg.cond(normal)),clamped_dimensions=int((std<1e-6).sum()),
                relative_normal_equation_residual=float(np.linalg.norm(normal@b-rhs)/max(np.linalg.norm(rhs),np.finfo(float).tiny)),
                quantiles=dict(probabilities=[0,.01,.1,.5,.9,.99,1],std=np.quantile(std,[0,.01,.1,.5,.9,.99,1]).tolist(),
                               scale=np.quantile(scale,[0,.01,.1,.5,.9,.99,1]).tolist()))

def populations(noise=True):
    rng=np.random.Generator(np.random.PCG64(771));out={}
    for stage in ('initial','final'):
        for c,maps in R.MAPS.items():
            n=64*len(maps);y=np.tile(np.arange(4),n//4);x=np.zeros((n,256))
            if noise:
                x[:,:4]=2*np.eye(4)[y];x[:,4:12]=rng.normal(size=(n,8))*.1
            else:x[:,0]=np.repeat(np.tile([-1.,1.],32),len(maps))
            out[stage+'/'+c]=(x,y,np.full(n,3),np.zeros(n,dtype=int))
    return out

class Controls(unittest.TestCase):
    def test_analytic_linear_solution_scale_floor_and_unseen_feature_values(self):
        y=np.tile(np.arange(4),256);x=np.zeros((1024,256));x[:,:4]=np.eye(4)[y]
        x[:,4]=np.repeat(np.arange(64),16)
        fitted,d=R.fit(x,y)
        covariance=np.eye(4)*.25-np.ones((4,4))/16
        expected=covariance/math.sqrt(3/16)/(4/3+.01)
        np.testing.assert_allclose(fitted['coefficients'][:4],expected,atol=1e-14,rtol=0)
        self.assertEqual(d['clamped_dimensions'],251)
        self.assertEqual(fitted['mean'][4],31.5)
        self.assertAlmostEqual(fitted['std'][4],math.sqrt((64**2-1)/12))
        transfer=x.copy();transfer[:,4]+=1000
        self.assertTrue(np.array_equal(R.scores(fitted,transfer).argmax(1),y))
        np.testing.assert_allclose(R.scores(fitted,transfer),transfer@(fitted['coefficients']/fitted['scale'][:,None])+
                                   fitted['intercept']-(fitted['mean']/fitted['scale'])@fitted['coefficients'],atol=1e-12,rtol=0)
        self.assertEqual(fitted['mean'][4],31.5)  # Evaluation never refits preprocessing.

    def test_exact_zero_rhs_geometry_only_and_argmax_disagreement(self):
        p=populations(False);x,y,_,_=p['final/seen'];fit,_=R.fit(x,y)
        self.assertTrue(np.all(fit['coefficients']==0))
        self.assertEqual(int(np.sum(R.scores(fit,x).argmax(1)==y)),256)
        self.assertTrue(np.all(R.scores(fit,x)==.25))
        parameters={a:normal_fit(x,R.permute(y) if a=='final_permuted' else y) for a in R.ARMS}
        parameters['final_true']['intercept'][1]+=1e-10
        with self.assertRaisesRegex(ValueError,'argmax differs'):
            R.recompute(p,parameters)

    def test_all_arms_normal_equation_agreement_group_controls_and_paired_draws(self):
        p=populations();parameters={}
        for arm in R.ARMS:
            stage='initial' if arm=='initial_true' else 'final';x,y,_,_=p[stage+'/seen']
            parameters[arm]=normal_fit(x,R.permute(y) if arm=='final_permuted' else y)
        result,extra=R.recompute(p,parameters)
        self.assertEqual(len(result['summaries']),9)
        self.assertEqual(result['summaries']['final_true/heldout']['correct'],512)
        self.assertEqual(result['summaries']['initial_true/familiar']['correct'],1024)
        self.assertTrue(all(v['identical_group_predictions'] and v['correct']*4==v['rows'] for v in result['group_mean_controls'].values()))
        self.assertEqual(extra['draws_u64le_sha256']['familiar'],extra['draws_u64le_sha256']['heldout'])
        values=np.asarray([-.25]*16+[0.]*16+[.25]*16+[.75]*16)
        draws=np.random.Generator(np.random.PCG64(1943)).integers(0,64,(10000,64))
        weights=np.stack([np.bincount(row,minlength=64) for row in draws])
        manual=[math.fsum(float(values[i]) for i in row)/64 for row in draws]
        self.assertEqual(R.interval(values,weights)['ci95'],np.quantile(manual,[.025,.975],method='linear').tolist())
        self.assertTrue(all(np.bincount(R.permute(p['final/seen'][1])[i:i+16],minlength=4).tolist()==[4]*4 for i in range(0,1024,16)))
        bad=copy.deepcopy(parameters);bad['final_true']['scale'][0]*=2
        with self.assertRaisesRegex(ValueError,'least-squares agreement'):R.recompute(p,bad)
        bad=copy.deepcopy(parameters);bad['final_true']['objective']+=.01
        with self.assertRaisesRegex(ValueError,'least-squares agreement'):R.recompute(p,bad)

    def test_label_hash_and_decision_controls(self):
        cells=[0]*64;cells[27]=2;cells[19]=3
        row=dict(row_index=0,group_index=0,permutation_id=0,schema='looped-grounded-policy-data-v1',cohort='seen',
                 condition='factual',support_cleared=False,query_cells=cells,agent_patch=27,goal_patch=19,correct_action=0,
                 inferred_controls=[0,1,2,3],observed_support_action_ids=[0,1,2],omitted_action=3)
        self.assertEqual(R.geometry(row,'seen',0),(0,3))
        row['correct_action']=1
        with self.assertRaisesRegex(ValueError,'geometry/label'):R.geometry(row,'seen',0)
        with tempfile.TemporaryDirectory(prefix='c14-independent-synthetic-') as temp:
            path=Path(temp)/'fixture';path.write_bytes(b'synthetic')
            binding=dict(path=str(path),sha256=hashlib.sha256(b'synthetic').hexdigest())
            self.assertEqual(R.pinned(binding),b'synthetic');path.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):R.pinned(binding)
            with self.assertRaisesRegex(ValueError,'frozen file hash'):R.verify_frozen(str(path),binding['sha256'])
            large=Path(temp)/'opaque-tool-fixture'
            with large.open('wb') as handle:handle.truncate(129*1024**2)
            with large.open('rb') as handle:sha=hashlib.file_digest(handle,'sha256').hexdigest()
            R.verify_frozen(str(large),sha)
            with self.assertRaisesRegex(ValueError,'bound file'):R.pinned(dict(path=str(large),sha256=sha))
        summaries={a+'/'+c:{'accuracy':.9} for a in ('initial_true','final_true') for c in R.MAPS}
        contrasts={a+'/'+c+'/'+kind:dict(estimate=.5,ci95=[.1 if kind=='native' else .3,.8])
                   for a in ('initial_true','final_true') for c in R.MAPS for kind in ('native','constant')}
        self.assertEqual(R.gates_and_decision(summaries,contrasts,True)[1],'affine_witness_already_present_initially')
        contrasts['initial_true/heldout/native']['ci95'][0]=0
        self.assertEqual(R.gates_and_decision(summaries,contrasts,True)[1],'final_affine_witness_supported_exploratorily')
        contrasts['final_true/familiar/constant']['ci95'][0]=.25
        self.assertEqual(R.gates_and_decision(summaries,contrasts,True)[1],'registered_affine_witness_not_supported')
        self.assertEqual(R.gates_and_decision(summaries,contrasts,False)[1],'inconclusive_failed_control')

if __name__=='__main__':unittest.main()
