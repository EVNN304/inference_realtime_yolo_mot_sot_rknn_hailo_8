import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import config.config as config
from src.gimb.ctrls.detection_backend import create_detection_loop, selected_backends
from src.framework.hailo.backend import HailoBackend


class SelectionTests(unittest.TestCase):
    def test_independent_selection_and_legacy_route(self):
        original = lambda *a: 'legacy'
        accelerated = lambda *a: 'accelerated'
        modules = {
            'src.gimb.ctrls.det_loop_two': NS(TrOsDetectionTwoChannelLoop=original),
            'src.gimb.ctrls.det_loop_hailo': NS(HailoDetectionLoop=accelerated),
        }
        with patch.dict(sys.modules, modules):
            for detector in ('rknn','hailo'):
                for mot in ('rknn','hailo'):
                    with patch.object(config,'detector_backend',detector), patch.object(config,'mot_backend',mot):
                        expected = 'legacy' if detector == mot == 'rknn' else 'accelerated'
                        self.assertEqual(create_detection_loop(None,None), expected)
        with self.assertRaisesRegex(ValueError,'mot_backend'):
            selected_backends(NS(detector_backend='hailo',mot_backend='typo'))

    def test_only_selected_models_created_and_one_device(self):
        calls=[]
        class Device:
            @staticmethod
            def create_params(): return NS()
            def __init__(self, params): calls.append('device')
            def __enter__(self): return self
            def __exit__(self,*args): calls.append('device closed')
        class Model:
            def __init__(self,name): self.name=name; calls.append(name)
            def shutdown(self): calls.append(self.name+' closed')
            close=shutdown
        modules={
            'hailo_platform': NS(VDevice=Device,HailoSchedulingAlgorithm=NS(ROUND_ROBIN=1)),
            'src.gimb.domain.cameras': NS(TrOsCamId=NS(TV_N=0)),
            'src.framework.hailo.yolo': NS(HailoYOLO=lambda *a: Model('hailo yolo')),
            'src.framework.hailo.hailo_reid': NS(HailoReID=lambda *a: Model('hailo reid')),
            'src.framework.hailo.hybridsort_adapter': NS(HybridSORTAdapter=lambda *a: Model('association')),
            'src.framework.hailo.rknn_bridge': NS(RKNNDetectorBridge=lambda *a: Model('rknn yolo'),
                                                 RKNNMOTBridge=lambda *a: Model('rknn mot')),
        }
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules,modules):
            path=Path(folder)/'model';path.write_bytes(b'test')
            settings=NS(yolo_hef=str(path),reid_hef=str(path),reid_manifest=str(path),
                        reid_batch=4,reid_inflight=2,timeout_ms=100,tracker_options={},torch_threads=1)
            for detector,mot in [('hailo','hailo'),('hailo','rknn'),('rknn','hailo')]:
                calls.clear()
                with patch.object(config,'detector_backend',detector), patch.object(config,'mot_backend',mot):
                    backend=HailoBackend(settings,False)
                    backend.close()
                self.assertEqual(calls.count('device'),1)
                self.assertIn(detector+' yolo',calls)
                self.assertIn('hailo reid' if mot=='hailo' else 'rknn mot',calls)
                self.assertNotIn('rknn yolo' if detector=='hailo' else 'hailo yolo',calls)
                self.assertEqual(calls[-1],'device closed')


if __name__ == '__main__': unittest.main()
