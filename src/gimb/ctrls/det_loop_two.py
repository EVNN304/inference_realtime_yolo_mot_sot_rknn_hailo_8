import logging
import time

import cv2
import numpy as np

import config.config as config
from src.gimb.ctrls.tracking_controller import TrOsTrackingController
from src.gimb.domain.cameras import TrOsCamId
from src.gimb.state import TrOsState
from src.framework.core.box import BoxTwoCorners
from src.framework.core.slice import slice_image, ImageSlice
from src.framework.det.detection import Detection
from src.framework.det.yolo_sahi import YoloSAHI
from src.framework.det.detector_factory import create_yolo_sahi_model
from src.framework.mot.mot_factory import create_mot_tracker
from src.framework.loop.loop import ILoop
from src.framework.mot.boxmot_wrapper import BoxMotWrapperBase


class _FullFrameSlice:
    def __init__(self, image: np.ndarray):
        self.image = image

        class _Coords:
            x_offset = 0
            y_offset = 0

        self.coords = _Coords()


class TrOsDetectionTwoChannelLoop(ILoop):
    def __init__(
        self,
        state: TrOsState,
        tracking_controller: TrOsTrackingController,
    ) -> None:
        super().__init__()
        self._state = state
        self._cam_id = TrOsCamId.TV_N
        self._logger = logging.getLogger("tv_det")
        self._tracking_controller = tracking_controller
        self._img_sz = 640
        self._window_wh = (self._img_sz, self._img_sz)
        self._overlay = (0.05, 0.05)
        cam_idx = 0
        self._confidence = config.Detector.confidence[cam_idx]
        self._model_path = config.Detector.model_path[cam_idx]
        self._sampling_ratio = config.Detector.downscale_ratio[cam_idx]
        self._model: YoloSAHI = None
        self._mot: BoxMotWrapperBase = None
        self._frames_processed = 0
        self._latest_processed_time = -1

    def on_start(self) -> None:
        if self._model_path is None:
            self.is_active = False
            self._logger.warning("No model provided for detection")
            return
        self._model = create_yolo_sahi_model(
            model_path=self._model_path,
            confidence=self._confidence,
            imgsz=self._img_sz,
            cam_id=self._cam_id,
        )
        self._mot = create_mot_tracker(self._cam_id)

    def update(self) -> None:
        if not config.Detector.enable_local_detector:
            time.sleep(50e-3)
            return

        cam_id = self._cam_id
        frame_sample = self._state.cameras[cam_id].frames.latest()
        if frame_sample is None:
            self._logger.error("Detector can't detect because no frames given")
            self._frames_processed += 1
            time.sleep(20e-3)
            return

        frame = frame_sample.obj
        frame_timestamp = frame_sample.timestamp_ns
        orig_frame = frame

        if self._sampling_ratio is not None:
            frame = cv2.resize(
                frame,
                None,
                fx=1 / self._sampling_ratio,
                fy=1 / self._sampling_ratio,
                interpolation=cv2.INTER_AREA,
            )

        if frame_timestamp == self._latest_processed_time:
            time.sleep(20e-3)
            self._frames_processed += 1
            return

        if config.Detector.use_sahi:
            slices = self._slice_frame(frame)
            mode_str = f"SAHI ({len(slices)} slices)"
        else:
            slices = [_FullFrameSlice(frame)]
            mode_str = "Full Frame"

        start = time.perf_counter_ns()
        detections = self._model.infer(slices)

        if self._sampling_ratio is not None:
            detections = self._upscale_detections(detections, self._sampling_ratio)

        self._state.detections[cam_id].enqueue(detections, frame_timestamp)
        self._logger.info(
            f"Detected '{cam_id}' detected %d objects [%s]", len(detections), mode_str
        )
        self._state.det_durations[cam_id].enqueue(
            time.perf_counter_ns() - start, frame_timestamp
        )

        start_mot = time.perf_counter_ns()
        MAX_REID_OBJECTS = 16
        detections_for_mot = (
            detections[:MAX_REID_OBJECTS]
            if len(detections) > MAX_REID_OBJECTS
            else detections
        )
        mot_result = self._mot.track(orig_frame, detections_for_mot)
        self._tracking_controller.on_mot_update(cam_id, mot_result, frame_timestamp)
        self._state.mot_durations[cam_id].enqueue(
            time.perf_counter_ns() - start_mot, frame_sample.timestamp_ns
        )



        self._frames_processed += 1
        self._latest_processed_time = frame_sample.timestamp_ns

    def on_stop(self) -> None:
        pass

    def _slice_frame(self, frame: np.ndarray) -> list[ImageSlice]:
        return slice_image(frame, window_wh=self._window_wh, overlay=self._overlay)

    @staticmethod
    def _upscale_detections(
        detections: list[Detection], scale_factor: float
    ) -> list[Detection]:
        new_boxes = [
            BoxTwoCorners(
                b.x1 * scale_factor, b.y1 * scale_factor, b.x2 * scale_factor, b.y2 * scale_factor
            )
            for b in [d.box.as_xyxy() for d in detections]
        ]
        return [d.copy_with(box=b) for d, b in zip(detections, new_boxes)]
