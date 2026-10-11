"""Adapter only: reuse the project's RKNN factories and their existing algorithms."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from src.framework.mot.boxmot_wrapper import BoxMotWrapperBase
from src.framework.core.box import BoxTwoCorners
from src.framework.det.detection import Detection


class RKNNDetectorBridge:
    def __init__(self, config, cam_id, use_sahi):
        from src.framework.det.detector_factory import create_yolo_sahi_model
        path = config.Detector.model_path[0]
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        self.model = create_yolo_sahi_model(path, config.Detector.confidence[0], 640, cam_id)
        self.ratio = config.Detector.downscale_ratio[0]
        self.sahi = use_sahi

    def infer(self, image, stop):
        import cv2
        from src.framework.core.slice import slice_image
        if stop.is_set():
            raise RuntimeError('RKNN detection cancelled')
        frame = image
        if self.ratio is not None:
            frame = cv2.resize(frame, None, fx=1/self.ratio, fy=1/self.ratio,
                               interpolation=cv2.INTER_AREA)
        if self.sahi:
            slices = slice_image(frame, window_wh=(640, 640), overlay=(.05, .05))
        else:
            slices = [SimpleNamespace(image=frame, coords=SimpleNamespace(x_offset=0, y_offset=0))]
        rows = BoxMotWrapperBase._to_boxmot_dets(self.model.infer(slices))
        if self.ratio is not None:
            rows[:, :4] *= self.ratio
        return rows

    def close(self):
        # The original RKNN models release themselves in __del__.
        self.model._model._executor.shutdown(wait=True)
        self.model = None


class RKNNMOTBridge:
    def __init__(self, config, cam_id):
        from src.framework.mot.mot_factory import create_mot_tracker
        path = config.mot_settings.get('reid_model_path', '')
        if not path.endswith('.rknn') or not Path(path).is_file():
            raise ValueError(f'mot_backend=rknn requires an existing .rknn ReID model: {path}')
        self.tracker = create_mot_tracker(cam_id)

    def update(self, packet, stop):
        if stop.is_set():
            raise RuntimeError('RKNN MOT cancelled')
        dets = [Detection(BoxTwoCorners(*map(float, r[:4])), int(r[5]), float(r[4]))
                for r in packet.detections]
        tracks = self.tracker.track(packet.image, dets)
        return np.asarray([[*t.box.as_xyxy().as_tuple(), t.track_id, t.conf, t.cls, -1]
                           for t in tracks], dtype=np.float64).reshape(-1, 8)

    def close(self):
        self.tracker = None
