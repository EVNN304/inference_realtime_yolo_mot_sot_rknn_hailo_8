import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from src.framework.hailo.performance import PipelineStats

class Clock:
    def __init__(self):self.now=0.
    def monotonic(self):return self.now
    def monotonic_ns(self):return round(self.now*1e9)
    def sleep(self, seconds):self.now+=seconds

class PerformanceTests(unittest.TestCase):
    def test_counts_frames_not_batch_latency_and_expires(self):
        clock=Clock(); stats=PipelineStats(clock=clock.monotonic)
        for t in range(1,31):
            clock.now=t/10
            stats.tick('yolo',3)
        self.assertAlmostEqual(stats.rate('yolo'),30)
        clock.now=7
        self.assertEqual(stats.rate('yolo'),0)

    def test_video_decode_included_in_30fps_deadline(self):
        clock=Clock(); timestamps=[]
        class Capture:
            index=0
            def read(self):
                clock.now+=.020; self.index+=1
                return True,NS(shape=(1080,1920,3),index=self.index)
            def grab(self):self.index+=1;return True
            def retrieve(self):return True,NS(shape=(1080,1920,3),index=self.index)
        mods={
            'cv2':NS(VideoCapture=object),
            'src.gimb.domain.cameras':NS(TrOsCamHolder=object,TrOsCamId=NS(TV_N=0)),
            'src.gimb.state':NS(TrOsState=object),
        }
        path=Path('src/gimb/ctrls/video_images_loop.py')
        with patch.dict(sys.modules,mods):
            spec=importlib.util.spec_from_file_location('paced_video_test',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module.time=clock
            state=NS(cameras=[NS(frames=NS(enqueue=lambda img,ts:timestamps.append(ts)),resolution=None)])
            loop=module.TrOsVideoImagesLoop(state)
            loop._realtime=True;loop._caps=NS(tv_n=Capture())
            loop._stats=PipelineStats(clock=clock.monotonic)
            for _ in range(31):loop.update()
        self.assertAlmostEqual((timestamps[-1]-timestamps[0])/1e9,1.,places=6)
        self.assertEqual(len(timestamps),31)

if __name__=='__main__':unittest.main()
