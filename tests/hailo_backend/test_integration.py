import sys
import threading
import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from src.gimb.ctrls.det_loop_hailo import HailoDetectionLoop
from src.framework.hailo.geometry import decode, tile_origins, letterbox_geometry
from src.framework.hailo.yolo import HailoYOLO
import config.config as config


class Buffer:
    def __init__(self): self.items = []
    def enqueue(self, obj, timestamp_ns): self.items.append(NS(obj=obj, timestamp_ns=timestamp_ns))
    def latest(self): return self.items[-1] if self.items else None


def wait_until(check):
    end = time.monotonic()+2
    while not check():
        if time.monotonic() > end: raise AssertionError('Timed out waiting for worker')
        time.sleep(.002)


class Backend:
    def __init__(self, settings, sahi):
        self.calls = []; self.closed = False; self.fail = False
        self.entered = threading.Event(); self.release = threading.Event(); self.release.set()
    def detect(self, image, stop):
        return np.array([[1, 2, 8, 9, .9, 3]], np.float32)
    def track(self, packet, stop):
        self.entered.set(); self.release.wait(2)
        if self.fail: raise ValueError('association failed')
        self.calls.append(packet)
        return np.array([[1,2,8,9,7,.9,3,0]], np.float64)
    def close(self): self.closed = True


class LoopTests(unittest.TestCase):
    def make_loop(self):
        frame = Buffer()
        state = NS(cameras=[NS(frames=frame)], detections=[Buffer()],
                   det_durations=[Buffer()], mot_durations=[Buffer()])
        results = []
        controller = NS(on_mot_update=lambda cam, tracks, ts: results.append((cam, tracks, ts)))
        settings = NS(mot_queue_size=2, mot_max_wait_ms=0, max_reid_objects=16)
        with patch.object(config.Detector, "use_sahi", True):
            loop = HailoDetectionLoop(state, controller, settings, Backend, cam_id=0)
        loop.on_start(); loop.is_active = True
        self.addCleanup(loop.on_stop)
        return loop, frame, state, results

    def test_exact_timestamp_image_and_track_contract(self):
        loop, frame, state, results = self.make_loop()
        image = np.zeros((10,10,3), np.uint8)
        frame.enqueue(image, 123456)
        loop.update(); wait_until(lambda: len(results)==1)
        self.assertIs(loop.backend.calls[0].image, image)
        self.assertEqual(results[0][2], 123456)
        self.assertEqual(results[0][1][0].track_id, 7)
        self.assertEqual(results[0][1][0].cls, 3)
        self.assertEqual(state.detections[0].latest().timestamp_ns, 123456)
        loop.update()
        self.assertEqual(len(state.detections[0].items), 1)
        frame.enqueue(image, 123450); loop.update()
        self.assertEqual(len(state.detections[0].items), 1)

    def test_slow_mot_does_not_block_yolo_and_uses_latest_pair(self):
        loop, frame, state, results = self.make_loop()
        backend = loop.backend; backend.release.clear()
        image = np.zeros((10,10,3), np.uint8)
        frame.enqueue(image, 1); loop.update(); self.assertTrue(backend.entered.wait(1))
        for ts in range(2, 10): frame.enqueue(image, ts); loop.update()
        self.assertEqual(len(state.detections[0].items), 9)
        self.assertLessEqual(loop.queue.qsize(), 2)
        backend.release.set(); wait_until(lambda: len(results) == 3)
        self.assertEqual([row[2] for row in results], [1,8,9])
        self.assertGreater(loop.mot_dropped, 0)

    def test_worker_failure_propagates_and_closes_device(self):
        loop, frame, state, results = self.make_loop(); backend = loop.backend
        backend.fail = True
        frame.enqueue(np.zeros((10,10,3), np.uint8), 1); loop.update()
        wait_until(lambda: not loop.errors.empty())
        with self.assertRaisesRegex(RuntimeError, 'MOT worker'): loop.update()
        self.assertTrue(backend.closed); self.assertTrue(loop.done.is_set())
        self.assertFalse(results)

    def test_close_waits_for_worker_before_releasing_resources(self):
        loop, frame, state, results = self.make_loop(); backend = loop.backend
        backend.release.clear()
        frame.enqueue(np.zeros((10,10,3), np.uint8), 1); loop.update()
        self.assertTrue(backend.entered.wait(1))
        closer = threading.Thread(target=loop.on_stop); closer.start()
        time.sleep(.01); self.assertFalse(backend.closed)
        backend.release.set(); closer.join(1)
        self.assertFalse(closer.is_alive()); self.assertTrue(backend.closed)
        self.assertEqual(results, [])

    def test_disabled_detector_does_not_consume_source(self):
        loop, frame, state, results = self.make_loop()
        frame.enqueue(np.zeros((10,10,3), np.uint8), 1)
        with patch.object(config.Detector, 'enable_local_detector', False): loop.update()
        self.assertEqual(state.detections[0].items, [])


class FullFrameTests(unittest.TestCase):
    make_loop = LoopTests.make_loop
    def test_async_full_frames_have_original_timestamps_and_fifo_order(self):
        loop, frames, state, results = self.make_loop()
        loop.use_sahi = False; loop.async_full_frame = True
        loop.settings.full_frame_batch = 2
        loop.settings.full_frame_max_wait_ms = 100
        loop.settings.timeout_ms = 1000
        jobs = []
        def submit(images, stop):
            job = NS(done=threading.Event(), images=images)
            jobs.append(job)
            return job
        loop.backend.submit_frames = submit
        loop.backend.can_submit_frames = lambda: len(loop.pending_frames) < 2
        loop.backend.collect_frames = lambda job: [loop.backend.detect(image, loop.stop_event) for image in job.images]
        images = [np.full((10,10,3), i, np.uint8) for i in range(4)]
        for i, image in enumerate(images):
            frames.enqueue(image, 100+i); loop.update()
        self.assertEqual(len(jobs), 2)
        self.assertEqual(len(state.detections[0].items), 0)
        self.assertIs(jobs[0].images[0], images[0])
        jobs[1].done.set(); loop.update()
        self.assertEqual(len(state.detections[0].items), 0)
        jobs[0].done.set(); loop.update()
        self.assertEqual([s.timestamp_ns for s in state.detections[0].items], [100,101,102,103])

    def test_partial_batch_flushes_and_stalled_callback_times_out(self):
        loop, frames, state, results = self.make_loop(); loop.use_sahi = False; loop.async_full_frame = True
        loop.settings.full_frame_batch = 4
        loop.settings.full_frame_max_wait_ms = 0
        loop.settings.timeout_ms = 1
        loop.backend.can_submit_frames = lambda: True
        loop.backend.submit_frames = lambda images, stop: NS(done=threading.Event(), images=images)
        frames.enqueue(np.zeros((10,10,3),np.uint8), 1); loop.update()
        self.assertEqual(len(loop.pending_frames),1)
        backend = loop.backend
        time.sleep(.003)
        with self.assertRaisesRegex(TimeoutError, 'callback timeout'): loop.update()
        self.assertTrue(backend.closed)


class Binding:
    def __init__(self): self.image = None; self.result = None; self.busy = False
    def input(self): return self
    def output(self): return self
    def set_buffer(self, image): self.image = image
    def get_buffer(self): return self.result


class Configured:
    def __init__(self, error=False): self.jobs=[]; self.error=error; self.stopped=False; self.reused=False
    def __enter__(self): return self
    def __exit__(self, *exc):
        for job in self.jobs: job.join()
    def get_async_queue_size(self): return 16
    def create_bindings(self, **kw): return Binding()
    def wait_for_async_ready(self, **kw): pass
    def shutdown(self):
        self.stopped=True
        for job in self.jobs: job.join()
    def run_async(self, bindings, callback):
        done=threading.Event(); saved=[b.image.copy() for b in bindings]
        for b in bindings:
            self.reused |= b.busy; b.busy=True
        def work():
            time.sleep(.006 if len(self.jobs)%2 == 0 else .001)
            for b, original in zip(bindings, saved):
                self.reused |= not np.array_equal(b.image, original)
                b.result=[np.array([[0,0,1,1,.9]], np.float32)]; b.busy=False
            callback(completion_info=NS(exception=ValueError('fail') if self.error else None))
            done.set()
        t=threading.Thread(target=work); self.jobs.append(t); t.start()
        def wait(timeout):
            if not done.wait(timeout/1000): raise TimeoutError()
        return NS(wait=wait)


class YoloTests(unittest.TestCase):
    def exercise(self, fail=False):
        c=Configured(fail)
        inp=NS(shape=(4,4,3),set_format_type=lambda x: None,set_format_order=lambda x: None)
        out=NS(shape=(1,10,5),name='nms',is_nms=True,format=NS(order=2),set_format_type=lambda x:None)
        model=NS(inputs=[inp],outputs=[out],input=lambda:inp,output=lambda:out,
                 set_batch_size=lambda b:None,configure=lambda:c)
        hp=NS(FormatType=NS(UINT8=1,FLOAT32=2)); order=NS(FormatOrder=NS(NHWC=1,HAILO_NMS_BY_CLASS=2))
        cv=NS(COLOR_BGR2RGB=1, cvtColor=lambda src, code, dst: np.copyto(dst, src[:,:,::-1]))
        settings=NS(sahi_batch=2, full_frame_batch=1,inflight=2,timeout_ms=1000,
                    confidence=.5,iou=.45,overlap=0.,yolo_hef='mock.hef')
        modules={'hailo_platform':hp,'hailo_platform.pyhailort.pyhailort':order,'cv2':cv}
        with patch.dict(sys.modules,modules), ExitStack() as stack:
            detector=HailoYOLO(NS(create_infer_model=lambda path:model),stack,settings,True)
            # Five tiles, three batches, last tile padded. Padding must never be returned.
            with patch('src.framework.hailo.yolo.merge',lambda chunks,*args:np.concatenate(chunks)):
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'YOLO inference'): detector.infer(np.zeros((4,20,3),np.uint8),threading.Event())
                    self.assertTrue(c.stopped)
                else:
                    result=detector.infer(np.zeros((4,20,3),np.uint8),threading.Event())
                    self.assertEqual(len(result),5)
                    np.testing.assert_allclose(result[:,0],[0,4,8,12,16])
                    self.assertEqual(len(c.jobs),3)
                    self.assertFalse(c.reused)
    def test_padding_async_ownership_and_tile_offsets(self): self.exercise()
    def test_callback_error_cancels_before_reuse(self): self.exercise(True)

    def test_full_frame_batch_padding_and_slot_lifetime(self):
        c=Configured()
        inp=NS(shape=(4,4,3),set_format_type=lambda x:None,set_format_order=lambda x:None)
        out=NS(shape=(1,10,5),name='nms',is_nms=True,format=NS(order=2),set_format_type=lambda x:None)
        model=NS(inputs=[inp],outputs=[out],input=lambda:inp,output=lambda:out,
                 set_batch_size=lambda b:None,configure=lambda:c)
        cv=NS(COLOR_BGR2RGB=1,INTER_LINEAR=1,
              resize=lambda img,size,**kw:np.tile(img[0,0],(size[1],size[0],1)),
              cvtColor=lambda src,code,dst:np.copyto(dst,src[:,:,::-1]))
        modules={'cv2':cv, 'hailo_platform':NS(FormatType=NS(UINT8=1,FLOAT32=2)),
                 'hailo_platform.pyhailort.pyhailort':NS(FormatOrder=NS(NHWC=1,HAILO_NMS_BY_CLASS=2))}
        settings=NS(sahi_batch=2,full_frame_batch=4,inflight=1,timeout_ms=1000,
                    confidence=.5,iou=.45,overlap=0.,yolo_hef='mock.hef')
        with patch.dict(sys.modules,modules), ExitStack() as stack:
            detector=HailoYOLO(NS(create_infer_model=lambda path:model),stack,settings,False)
            images=[np.full((8,8,3),v,np.uint8) for v in [10,20,30]]
            job=detector.submit_frames(images,threading.Event())
            self.assertFalse(detector.can_submit_frames())
            self.assertTrue(job.done.wait(1))
            # Even a completed job owns the slot until decoding/copying has finished.
            self.assertFalse(detector.can_submit_frames())
            with patch('src.framework.hailo.yolo.merge',lambda chunks,*args:np.concatenate(chunks)):
                results=detector.collect_frames(job)
            self.assertEqual(len(results),3)
            self.assertTrue(detector.can_submit_frames())
            self.assertFalse(c.reused)
            np.testing.assert_allclose(results[0][0,:4],[0,0,8,8])

    def test_letterbox_inverse_and_non_square_frame(self):
        fw,fh,iw,ih=1920,1200,640,640
        nw,nh,left,top=letterbox_geometry(fw,fh,iw,ih)
        original=np.array([300,210,900,600],np.float32)
        transformed=(original*[nw/fw,nh/fh,nw/fw,nh/fh]+[left,top,left,top])/[iw,ih,iw,ih]
        raw=[np.array([[*transformed[[1,0,3,2]],.8]],np.float32)]
        decoded=decode(raw,0,0,iw,ih,fw,fh,.5,(nw/fw,nh/fh,left,top))
        np.testing.assert_allclose(decoded[0][0,:4],original,atol=.001)

    def test_small_frame_and_cached_slices(self):
        self.assertEqual(tile_origins(100,200,640,640,0,True),((0,0),))
        a=tile_origins(1920,1200,640,640,0,True)
        self.assertEqual(len(a),6)
        self.assertIs(a,tile_origins(1920,1200,640,640,0,True))


if __name__ == '__main__': unittest.main()
