import json,struct,tempfile,unittest
from pathlib import Path
import numpy as np
from diagnostic import head,sharpen,SHAPES
class Controls(unittest.TestCase):
 def tensor_file(self,root,alter=None):
  header={};chunks=[];cursor=0
  for name,shape in SHAPES.items():
   data=np.arange(np.prod(shape),dtype='<f4').tobytes();header[name]=dict(dtype='F32',shape=shape,data_offsets=[cursor,cursor+len(data)]);chunks.append(data);cursor+=len(data)
  if alter:alter(header)
  h=json.dumps(header).encode();p=Path(root)/'head.safetensors';p.write_bytes(struct.pack('<Q',len(h))+h+b''.join(chunks));return p
 def test_tensor_roundtrip_and_overlap_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.tensor_file(d);_,_,a,_=head(p);self.assertEqual(a['queries'][0,1],1)
   self.tensor_file(d,lambda h:h['output.bias'].update(data_offsets=[0,16]))
   with self.assertRaises(ValueError):head(p)
 def test_power_matches_scaled_logits_and_concentrates(self):
  z=np.zeros((1024,2,64));z[:,:,4]=3;p=np.exp(z);p/=p.sum(2,keepdims=True);q=sharpen(p)
  scaled=np.exp(16*(z-z.max(2,keepdims=True)));scaled/=scaled.sum(2,keepdims=True)
  np.testing.assert_allclose(q,scaled,rtol=1e-12,atol=1e-15);self.assertTrue((q[:,:,4]>=.99).all());self.assertTrue((q.argmax(2)==4).all())
 def test_uniform_is_not_a_concentrating_positive(self):
  p=np.full((1024,2,64),1/64);q=sharpen(p);np.testing.assert_array_equal(q,p);self.assertTrue((q.max(2)<.99).all())
 def test_nonfinite_and_unnormalized_rejected(self):
  p=np.full((1024,2,64),1/64);p[0,0,0]=np.nan
  with self.assertRaises(ValueError):sharpen(p)
  with self.assertRaises(ValueError):sharpen(np.ones((1024,2,64)))
if __name__=='__main__':unittest.main()
