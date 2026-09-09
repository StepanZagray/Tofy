import unittest
import numpy as np
import analyze as A
class Controls(unittest.TestCase):
 def test_group_bootstrap_preserves_balanced_control(self):
  audit=[dict(correct_action=i%4,omitted_action=3) for i in range(1024)]
  scores=np.zeros((1024,4));scores[:,2]=1;s,g=A.summary(scores,audit)
  self.assertEqual(s['correct'],256);self.assertEqual(s['all_maps_correct_groups'],0)
  draws=np.random.Generator(np.random.PCG64(1942)).integers(0,64,(10000,64))
  self.assertEqual(A.interval(g-.25,draws),dict(estimate=0.,ci95=[0.,0.]))
 def test_strict_gate_and_permutation_control(self):
  summaries={f'{a}/{c}':dict(accuracy=.95 if a!='final_permuted' else .25) for a in A.ARMS for c in A.COHORTS}
  contrasts={f'{a}/{c}/{b}':dict(ci95=[.7,.71]) for a in A.ARMS for c in A.COHORTS for b in ('native','constant')}
  controls={f'{a}/{c}':dict(correct=256,rows=1024,identical_group_predictions=True) for a in A.ARMS for c in A.COHORTS}
  gates,d=A.decide(summaries,contrasts,controls);self.assertEqual(d,'affine_witness_already_present_initially')
  contrasts['final_true/familiar/native']['ci95'][0]=0
  gates,d=A.decide(summaries,contrasts,controls);self.assertEqual(d,'initial_only_affine_witness')
  summaries['final_permuted/heldout']['accuracy']=.501
  gates,d=A.decide(summaries,contrasts,controls);self.assertEqual(d,'inconclusive_failed_control');self.assertFalse(gates['controls_valid'])
 def test_group_average_eliminates_map_information(self):
  y=np.tile(np.arange(4),256);x=np.eye(4)[y];f=A.fit(x,y)
  true=A.predict(x,f).argmax(1);self.assertTrue((true==y).all())
  repeated=np.repeat(x.reshape(64,16,4).mean(1),16,axis=0)
  pred=A.predict(repeated,f).argmax(1);self.assertEqual(int((pred==y).sum()),256)
if __name__=='__main__':unittest.main()
