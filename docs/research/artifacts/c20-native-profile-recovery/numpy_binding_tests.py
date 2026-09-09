import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy_binding as b
import numpy as np


def weights():
    return {name: np.zeros(shape, dtype='<f4') for name, shape in b.expected_shapes().items()}


def records():
    return np.array([[[0,-1,1,0,0,0,0],[0,1,0,1,0,0,0],[-1,0,0,0,1,0,0],[1,0,0,0,0,0,1]]],dtype=np.float32)


def safe_bytes(w, mutate=None):
    header, payload = {}, bytearray()
    for name, value in sorted(w.items()):
        start = len(payload); payload.extend(value.tobytes())
        header[name] = {'dtype':'F32','shape':list(value.shape),'data_offsets':[start,len(payload)]}
    if mutate:
        mutate(header, payload)
    h = json.dumps(header, separators=(',',':')).encode()
    return len(h).to_bytes(8,'little') + h + payload


class ReplayTests(unittest.TestCase):
    def test_rms_clamps_instead_of_adding_epsilon(self):
        np.testing.assert_array_equal(b.rms(np.zeros((1,256),np.float32)),0)
        np.testing.assert_allclose(b.rms(np.full((1,256),2e-6,np.float32)),.2,rtol=1e-6)
        np.testing.assert_allclose(b.rms(np.full((1,256),2,np.float32)),1,rtol=0,atol=0)

    def test_zero_blocks_reduce_to_independent_closed_form(self):
        w=weights();w['input_projection.weight'][:5]=np.eye(5,dtype=np.float32)
        w['policy_head.weight'][0,:5]=[.3,-.2,.7,.1,-.4]
        expected=np.array([[0,-1,1,1,0],[0,1,1,1,0],[-1,0,1,1,0],[0,0,0,1,0]],np.float64)
        expected=expected@np.array([.3,-.2,.7,.1,-.4])/np.sqrt((expected**2).sum(axis=1)/256)
        for loops in [1,2,4,8]:
            np.testing.assert_allclose(b.replay(records(),w,loops)[0],expected,atol=2e-6)

    def test_nonzero_body_depth_and_action_symmetry(self):
        rng=np.random.default_rng(92)
        w={n:rng.normal(0,.04,s).astype('<f4') for n,s in b.expected_shapes().items()}
        r=records();a=b.replay(r,w,1);z=b.replay(r,w,4)
        self.assertGreater(float(np.max(abs(a-z))),1e-3)
        order=[2,0,3,1];changed=r.copy();changed[:,:3,2:6]=r[:,:3,2:6][:,:,order]
        np.testing.assert_allclose(b.replay(changed,w),z[:,order],rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(b.replay(r[:,[2,0,1,3]],w),z,atol=1e-5)

    def test_checkpoint_payload_roundtrip_and_bad_layout(self):
        w=weights();w['policy_head.bias'][0]=.125
        with tempfile.TemporaryDirectory(prefix='c19-replay-') as d:
            path=Path(d)/'weights'
            raw=safe_bytes(w);path.write_bytes(raw)
            loaded=b.load_checkpoint(path,hashlib.sha256(raw).hexdigest())
            self.assertEqual(b.parameter_digest(loaded),b.parameter_digest(w))
            for mut in [lambda h,p:h['policy_head.bias'].update(dtype='F64'),lambda h,p:h['policy_head.bias']['data_offsets'].__setitem__(0,0),lambda h,p:p.extend(b'bad')]:
                raw=safe_bytes(w,mut);path.write_bytes(raw)
                with self.assertRaises(ValueError): b.load_checkpoint(path,hashlib.sha256(raw).hexdigest())
            raw=safe_bytes(w);path.write_bytes(raw)
            with self.assertRaises(ValueError):b.load_checkpoint(path,'0'*64)

    def test_nonfinite_and_shape_corruptions(self):
        w=weights();w['policy_head.bias'][0]=np.nan
        with self.assertRaises(ValueError):b.replay(records(),w)
        for x in [np.zeros((1,4,6)),np.full((1,4,7),np.inf)]:
            with self.assertRaises(ValueError):b.replay(x,weights())
        for loops in [0,9,True]:
            with self.assertRaises(ValueError):b.replay(records(),weights(),loops)

    def test_tolerance_and_near_tie_reporting(self):
        z=np.array([[1.,0.,0.,0.],[0.,1e-6,0.,0.]])
        actual=z.copy();actual[1,0]=2e-6
        p=b.logit_parity(actual,z)
        self.assertEqual(p['eligible_winners'],1);self.assertEqual(p['ineligible_winners'],1)
        actual[0,0]+=.01
        with self.assertRaises(ValueError):b.logit_parity(actual,z)


if __name__=='__main__':unittest.main()
