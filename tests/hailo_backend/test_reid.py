import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from src.framework.hailo.hailo_reid import HailoReID, ReIDJob, crop_bounds, l2_normalize
from src.framework.hailo.hybridsort_adapter import HybridSORTAdapter

class Binding:
    def __init__(self,output):self.result=output;self.image=None
    def input(self):return self
    def set_buffer(self,image):self.image=image

class Configured:
    def __init__(self,fail=False):self.threads=[];self.fail=fail;self.reused=False;self.stopped=False;self.calls=0
    def __enter__(self):return self
    def __exit__(self,*args):
        for t in self.threads:t.join()
    def get_async_queue_size(self):return 16
    def create_bindings(self,output_buffers):return Binding(next(iter(output_buffers.values())))
    def wait_for_async_ready(self,**kwargs):pass
    def shutdown(self):self.stopped=True
    def run_async(self,bindings,callback):
        snapshots=[b.image.copy() for b in bindings];done=threading.Event()
        self.calls+=1
        delay=.005 if self.calls%2 else .001
        def work():
            time.sleep(delay)
            for b,old in zip(bindings,snapshots):
                self.reused |= not np.array_equal(b.image,old)
                b.result[:]=old.mean(axis=(0,1))+1
            callback(completion_info=SimpleNamespace(exception=RuntimeError('SDK failure') if self.fail else None))
            done.set()
        t=threading.Thread(target=work);t.start();self.threads.append(t)
        def finish(timeout):
            if not done.wait(timeout/1000):raise TimeoutError()
        return SimpleNamespace(wait=finish)

class Model:
    def __init__(self,configured):
        self.i=SimpleNamespace(shape=(2,2,3),set_format_type=lambda _:None,set_format_order=lambda _:None)
        self.o=SimpleNamespace(shape=(3,),name='features',is_nms=False,set_format_type=lambda _:None)
        self.inputs=[self.i];self.outputs=[self.o];self.c=configured
    def input(self):return self.i
    def output(self):return self.o
    def set_batch_size(self,b):pass
    def configure(self):return self.c

class ReID(unittest.TestCase):
    def test_callback_accepts_sdk_keyword(self):
        job = ReIDJob(slot=None, start=0, count=1)
        error = RuntimeError('SDK failure')
        job.callback(completion_info=SimpleNamespace(exception=error))
        self.assertIs(job.error, error)
        job.callback(completion_info=SimpleNamespace(exception=None))
        self.assertIsNone(job.error)

    def exercise(self,fail=False,bad_hash=False,batch=2):
        c=Configured(fail);model=Model(c)
        device=SimpleNamespace(create_infer_model=lambda _:model)
        cv=SimpleNamespace(INTER_LINEAR=1,COLOR_BGR2RGB=1,
            resize=lambda img,size,**kw:np.tile(img[0,0],(size[1],size[0],1)),
            cvtColor=lambda src,code,dst:np.copyto(dst,src[:,:,::-1]))
        hp=SimpleNamespace(FormatType=SimpleNamespace(UINT8=1,FLOAT32=2))
        order=SimpleNamespace(FormatOrder=SimpleNamespace(NHWC=1))
        mods={'cv2':cv,'hailo_platform':hp,'hailo_platform.pyhailort.pyhailort':order}
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules,mods), ExitStack() as stack:
            root=Path(folder);hef=root/'model.hef';hef.write_bytes(b'fake')
            meta=dict(input_contract='RGB_UINT8_0_255_NORMALIZATION_IN_HEF',height=2,width=2,
                      embedding_dim=3,hef_sha256='bad' if bad_hash else hashlib.sha256(b'fake').hexdigest())
            manifest=root/'model.json';manifest.write_text(json.dumps(meta))
            if bad_hash:
                with self.assertRaisesRegex(ValueError,'SHA256'):HailoReID(device,stack,hef,manifest,2,2)
                return
            reid=HailoReID(device,stack,hef,manifest,batch,2)
            image=np.zeros((2,10,3),np.uint8)
            boxes=[];expected=[]
            for i in range(5):
                bgr=np.array([i+1,10,20],np.uint8)
                image[:,i*2:i*2+2]=bgr
                boxes.append([i*2,0,i*2+2,2])
                expected.append(bgr[::-1].astype(float)+1)
            if fail:
                with self.assertRaisesRegex(RuntimeError,'OSNet async'):reid.get_features(np.array(boxes),image)
                self.assertTrue(c.stopped)
            else:
                out=reid.get_features(np.array(boxes),image)
                np.testing.assert_allclose(out,l2_normalize(expected),atol=1e-6)
                self.assertEqual(out.shape,(5,3))
                self.assertFalse(c.reused)
                old_calls=c.calls
                self.assertEqual(reid.get_features(np.empty((0,4)),image).shape,(0,3))
                self.assertEqual(c.calls,old_calls)
                # Reuse on next request cannot modify returned features.
                saved=out.copy()
                reid.get_features(np.array(boxes),image)
                np.testing.assert_array_equal(saved,out)
    def test_order_padding_rgb_empty_and_ownership(self):self.exercise()
    def test_batch4_padding_order_and_ownership(self):self.exercise(batch=4)
    def test_callback_error_aborts(self):self.exercise(fail=True)
    def test_manifest_rejects_wrong_hef(self):self.exercise(bad_hash=True)
    def test_invalid_features(self):
        for a in (np.zeros((1,3)),np.array([[np.nan,0,1]])):
            with self.assertRaises(ValueError):l2_normalize(a)
    def test_crop_clipping_and_invalid(self):
        self.assertEqual(crop_bounds([-5,-1,12,8],10,6),(0,0,10,6))
        with self.assertRaises(ValueError):crop_bounds([20,0,21,1],10,6)

class Adapter(unittest.TestCase):
    def test_external_embeddings_no_cpu_reid_order_and_gap(self):
        class Tracker:
            def __init__(self,**kwargs):
                assert kwargs['reid_weights'] is None and kwargs['with_reid']
                self.model=None;self.calls=[]
            # Emulate 16.0.8's decorator, which hides the embs signature.
            def update(self,*args,**kwargs):
                dets,image=args;embs=kwargs['embs']
                self.calls.append((dets.copy(),image,embs.copy()))
                return np.array([[1,1,3,3,42,.8,0,0]],float)
        mods={'torch':SimpleNamespace(set_num_threads=lambda _:None,device=lambda x:x),
              'boxmot':SimpleNamespace(HybridSort=Tracker,__version__='16.0.8',__file__='mock')}
        received=[]
        def features(boxes,image,*_):received.append(boxes.copy());return np.ones((len(boxes),3),np.float32)
        metrics=SimpleNamespace(values={},add=lambda name,value=1:metrics.values.__setitem__(name,metrics.values.get(name,0)+value))
        with patch.dict(sys.modules,mods):
            adapter=HybridSORTAdapter(SimpleNamespace(get_features=features),{})
            state=SimpleNamespace(index=0,image=np.zeros((4,4,3),np.uint8))
            dets=np.array([[1,1,3,3,.8,0]],np.float32)
            out=adapter.update(state,dets,metrics)
            state.index=3
            adapter.update(state,dets,metrics)
            self.assertEqual(metrics.values['mot_source_gap'],2)
            self.assertEqual(out[0,4],42)
            np.testing.assert_array_equal(received[0],dets[:,:4])
            with self.assertRaises(ValueError):adapter.update(state,dets,metrics)

if __name__=='__main__':unittest.main()
