"""Select YOLO and OSNet independently; share one VDevice when both use Hailo."""
from contextlib import ExitStack
from pathlib import Path
from .geometry import Metrics


class HailoBackend:
    def __init__(self, settings, use_sahi):
        import config.config as config
        from src.gimb.ctrls.detection_backend import selected_backends
        from src.gimb.domain.cameras import TrOsCamId
        detector, mot = selected_backends(config)
        self.stack = ExitStack()
        self.metrics = Metrics()
        self.yolo = self.reid = self.mot = None
        self.hailo_detector = detector == 'hailo'
        self.hailo_mot = mot == 'hailo'
        paths = []
        if self.hailo_detector:
            paths.append(settings.yolo_hef)
        if self.hailo_mot:
            paths.extend((settings.reid_hef, settings.reid_manifest))
        for value in paths:
            if not Path(value).is_file():
                raise FileNotFoundError(value)
        try:
            from hailo_platform import VDevice, HailoSchedulingAlgorithm
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            self.device = self.stack.enter_context(VDevice(params))
            if self.hailo_detector:
                from .yolo import HailoYOLO
                self.yolo = HailoYOLO(self.device, self.stack, settings, use_sahi)
                self.stack.callback(self.yolo.shutdown)
            else:
                from .rknn_bridge import RKNNDetectorBridge
                self.yolo = RKNNDetectorBridge(config, TrOsCamId.TV_N, use_sahi)
                self.stack.callback(self.yolo.close)
            if self.hailo_mot:
                from .hailo_reid import HailoReID
                from .hybridsort_adapter import HybridSORTAdapter
                self.reid = HailoReID(self.device, self.stack, settings.reid_hef,
                                     settings.reid_manifest, settings.reid_batch,
                                     settings.reid_inflight, settings.timeout_ms)
                self.stack.callback(self.reid.shutdown)
                self.mot = HybridSORTAdapter(self.reid, settings.tracker_options, settings.torch_threads)
            else:
                from .rknn_bridge import RKNNMOTBridge
                self.mot = RKNNMOTBridge(config, TrOsCamId.TV_N)
                self.stack.callback(self.mot.close)
        except BaseException:
            self.stack.close()
            raise

    def can_submit_frames(self):
        return self.yolo.can_submit_frames()

    def submit_frames(self, frames, stop):
        return self.yolo.submit_frames(frames, stop)

    def collect_frames(self, job):
        return self.yolo.collect_frames(job)

    def detect(self, image, stop):
        return self.yolo.infer(image, stop)

    def track(self, packet, stop):
        if self.hailo_mot:
            return self.mot.update(packet, packet.detections, self.metrics, stop)
        return self.mot.update(packet, stop)

    def close(self):
        # Owners have stopped before ExitStack shuts down jobs and releases models/device.
        self.stack.close()
