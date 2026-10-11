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
    """
    Псевдо-слайс для обработки полного кадра без SAHI.
    Имитирует интерфейс ImageSlice с нулевыми оффсетами.
    """
    def __init__(self, image: np.ndarray):
        self.image = image
        class _Coords:
            x_offset = 0
            y_offset = 0
        self.coords = _Coords()


class TrOsDetectionSingleLoop(ILoop):
    def __init__(
        self,
        state: TrOsState,
        tracking_controller: TrOsTrackingController,
        cam_id: TrOsCamId,
    ) -> None:
        super().__init__()
        self._state = state
        self._tracking_controller = tracking_controller
        self._cam_id = cam_id

        # Маппинг для доступа к спискам в конфиге (как в detector_factory)
        self._cam_index_map = {
            TrOsCamId.IR_W: 0,
            TrOsCamId.TV_W: 1,
            TrOsCamId.IR_N: 2,
            TrOsCamId.TV_N: 3
        }
        self._cam_idx = self._cam_index_map.get(self._cam_id, 0)

        self._img_sz = 640  # Приведено к стандарту YOLO11s (было 288)
        self._window_wh = (self._img_sz, self._img_sz)
        self._overlay = (0.05, 0.05)
        self._half_float = True

        self._logger = logging.getLogger("det")
        self._confidence = config.Detector.confidence[self._cam_idx]
        self._model_path = config.Detector.model_path[self._cam_idx]
        self._downscale_ratio = config.Detector.downscale_ratio[self._cam_idx]

        self._model: YoloSAHI = None
        self._mot: BoxMotWrapperBase = None
        
        self._frames_processed = 0
        self._latest_processed_time = -1

    def on_start(self) -> None:
        if not self._model_path:
            self._logger.error("Can't init detector for '%s' because no model provided", self._cam_id)
            self.is_active = False
            return


        self._model = create_yolo_sahi_model(
            model_path=self._model_path,
            confidence=self._confidence,
            imgsz=self._img_sz,
            cam_id=self._cam_id,
        )
        
        self._mot = create_mot_tracker(self._cam_id)

    def update(self) -> None:
        # Проверка активного канала
        if self._state.detector_desired_state.channel != self._cam_id.channel:
            time.sleep(100e-3)
            return


        frame_sample = self._state.cameras[self._cam_id].frames.latest()
        if frame_sample is None:
            self._logger.error("Detector '%s' can't detect because no frames given", self._cam_id)
            self._frames_processed += 1
            time.sleep(20e-3)
            return

        frame = frame_sample.obj
        frame_timestamp = frame_sample.timestamp_ns
        orig_frame = frame

        if self._downscale_ratio is not None:
            frame = cv2.resize(
                frame,
                None,
                fx=1 / self._downscale_ratio,
                fy=1 / self._downscale_ratio,
                interpolation=cv2.INTER_AREA,
            )

        self._logger.debug(
            "Received frame latency %.1f ms",
            1e-6 * (self._state.time_ns() - frame_timestamp),
        )

        if frame_timestamp == self._latest_processed_time:
            time.sleep(20e-3)
            self._logger.debug("Skipping already processed frame")
            self._frames_processed += 1
            return

        # 🔥 ПЕРЕКЛЮЧАТЕЛЬ SAHI / FULL FRAME
        if config.Detector.use_sahi:
            slices = self._slice_frame(frame)
            mode_str = f"SAHI ({len(slices)} slices)"
        else:
            slices = [_FullFrameSlice(frame)]
            mode_str = "Full Frame"

        start = time.perf_counter_ns()
        detections = self._model.infer(slices)

        if self._downscale_ratio is not None:
            detections = self._upscale_detections(detections, self._downscale_ratio)

        self._state.detections[self._cam_id].enqueue(detections, frame_timestamp)
        self._logger.info(f"Detected '{self._cam_id}' detected %d objects [%s]", len(detections), mode_str)
        
        duration = time.perf_counter_ns() - start
        self._logger.debug("Detection duration %.1f ms", 1e-6 * duration)
        self._state.det_durations[self._cam_id].enqueue(duration, frame_timestamp)

        # =====================================================================
        # ЗАПУСК МОТ С ОГРАНИЧЕНИЕМ ДО 16 ОБЪЕКТОВ
        # =====================================================================
        start_mot = time.perf_counter_ns()
        
        MAX_REID_OBJECTS = 16
        detections_for_mot = detections[:MAX_REID_OBJECTS] if len(detections) > MAX_REID_OBJECTS else detections
        
        mot_result = self._mot.track(orig_frame, detections_for_mot)
        duration_mot = time.perf_counter_ns() - start_mot
        
        self._tracking_controller.on_mot_update(self._cam_id, mot_result, frame_timestamp)
        self._logger.info(f"MOT '{self._cam_id}' tracking now %d objects", len(mot_result))
        
        self._logger.debug("MOT duration %.1f ms", 1e-6 * duration_mot)
        self._logger.debug(
            "MOT total latency is %.1f ms",
            1e-6 * (self._state.time_ns() - frame_timestamp),
        )
        self._state.mot_durations[self._cam_id].enqueue(duration_mot, frame_timestamp)

        # 🔥 ОТЛАДОЧНАЯ ОТРИСОВКА
        debug_frame = orig_frame.copy()
        for track in mot_result:
            x1, y1 = int(track.box.x1), int(track.box.y1)
            x2, y2 = int(track.box.x2), int(track.box.y2)
            
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"ID:{track.track_id} {track.conf:.2f}"
            cv2.putText(debug_frame, label, (x1, max(0, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
        cv2.imshow(f"MOT Tracking {self._cam_id}", debug_frame)
        cv2.waitKey(1)

        self._frames_processed += 1
        self._latest_processed_time = frame_timestamp

    def on_stop(self) -> None:
        pass

    def _slice_frame(self, frame: np.ndarray) -> list[ImageSlice]:
        return slice_image(frame, window_wh=self._window_wh, overlay=self._overlay)

    @staticmethod
    def _upscale_detections(
        detections: list[Detection],
        scale_factor: float,
    ) -> list[Detection]:
        new_boxes = [d.box.as_xyxy() for d in detections]
        new_boxes = [
            BoxTwoCorners(
                b.x1 * scale_factor,
                b.y1 * scale_factor,
                b.x2 * scale_factor,
                b.y2 * scale_factor,
            )
            for b in new_boxes
        ]
        return [d.copy_with(box=b) for d, b in zip(detections, new_boxes)]

    def time_ns(self) -> float:
        return time.time_ns()
