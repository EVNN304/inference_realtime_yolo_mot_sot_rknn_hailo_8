import logging
import time
import traceback
from enum import Enum

import cv2
import config.config as config
import numpy as np

from src.gimb.ctrls.tracking_controller import TrOsTrackingController
from src.gimb.domain.cameras import TrOsCamId
from src.gimb.domain.tracking import TrOsTrackingCommand
from src.gimb.state import TrOsState
from src.framework.core.box import BoxTopLeft
from src.framework.loop.loop import ILoop
from src.framework.sot.single_object_tracker import ISingleObjectTracker
from src.framework.sot.nanotrack_tracker import NanoTrackTracker
from src.gimb.ctrls.camera_motion import OrbCameraMotionEstimator

class LoopStatus(Enum):
    DISABLED = 0
    TRACKING = 1
    LOST = 2


class TrackingLoop(ILoop):
    def __init__(
        self,
        state: TrOsState,
        tracking_controller: TrOsTrackingController,
    ) -> None:
        super().__init__()
        self._state = state
        self._controller = tracking_controller
        self._tracker: ISingleObjectTracker = None
        self._cam_id = TrOsCamId.TV_N
        self._cmd_listener = self._state.tracker_cmd[self._cam_id].subscribe()
        self._status: LoopStatus = LoopStatus.DISABLED
        self._camera = self._state.cameras[self._cam_id]
        self._logger = logging.getLogger("sot")
        self._latest_update_timestamp = -1
        self._init_timestamp = -1
        self._skip_frames_to_go_next = 2
        self.tracker_choice = config.Tracking.tracker
        
        self._motion_estimator = OrbCameraMotionEstimator()
        self._prev_frame = None
        self._last_bbox = None
        
    def on_start(self) -> None:
        if self.tracker_choice == "stark":
            from src.framework.sot.stark_tracker import StarkTracker
            self._tracker = StarkTracker(
                "stark_lightning_X_trt_onnx",
                StarkTracker.Config(
                    yaml_path="models/stark/baseline_rephead_4_lite_search5.yaml",
                    model_path="models/stark/STARKLightningXtrt_ep0500.pth.tar",
                    onnx_template_path="models/stark/backbone_bottleneck_pe.onnx",
                    onnx_complete_path="models/stark/complete.onnx",
                ),
            )
        elif self.tracker_choice == "nanotrack":
            self._tracker = NanoTrackTracker(
                rknn_backbone_path=config.Tracking.nanotrack_backbone,
                onnx_head_path=config.Tracking.nanotrack_head,
                core_mask=4,
            )

        print("TRACK_INIT", self.tracker_choice)
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        box = BoxTopLeft(20, 20, 10, 10)
        flg = self._tracker.initialize(frame, box)
        self._tracker.track(frame)
        
                # Сброс CMC
        self._motion_estimator.reset()
        self._prev_frame = None
        self._last_bbox = None

    def update(self) -> None:
        cam_id = self._cam_id

        if self._status != LoopStatus.TRACKING:
            latest_cmd_sample = self._state.tracker_cmd[cam_id].latest()
            if latest_cmd_sample is not None and (
                latest_cmd_sample.obj.enabled or latest_cmd_sample.obj.needs_init
            ):
                self._try_init_tracking(latest_cmd_sample.obj)
                return
            time.sleep(10e-3)
            return
        else:
            latest_cmd_sample = self._state.tracker_cmd[cam_id].latest()
            if latest_cmd_sample is not None:
                latest_cmd = latest_cmd_sample.obj
                if not latest_cmd.enabled:
                    self._status = LoopStatus.DISABLED
                    self._motion_estimator.reset()
                    self._prev_frame = None
                    self._last_bbox = None
                    self._controller.on_sot_disabled(cam_id)
                    return
                if latest_cmd.timestamp_ns != self._init_timestamp:
                    self._status = LoopStatus.DISABLED
                    self._try_init_tracking(latest_cmd)
                    return

            frame_sample = self._camera.frames.latest()
            if (
                frame_sample is None
                or frame_sample.timestamp_ns == self._latest_update_timestamp
            ):
                time.sleep(10e-3)
                return
	                # --- Camera Motion Compensation ---
            if self._prev_frame is not None and self._last_bbox is not None:
                M, num_inliers = self._motion_estimator.estimate(
                    prev_frame=self._prev_frame,
                    curr_frame=frame_sample.obj,
                    exclude_bbox=self._last_bbox,
                )
                if num_inliers >= 4:
                    if hasattr(self._tracker, 'center_pos'):
                        comp_center = OrbCameraMotionEstimator.warp_point(
                            self._tracker.center_pos, M
                        )
                        h, w = frame_sample.obj.shape[:2]
                        comp_center[0] = np.clip(comp_center[0], 0, w - 1)
                        comp_center[1] = np.clip(comp_center[1], 0, h - 1)
                        self._tracker.center_pos = comp_center

                    if hasattr(self._tracker, 'size'):
                        scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
                        if 0.70 <= scale <= 1.40:
                            self._tracker.size *= scale
            # --------------------------------



            start = time.perf_counter_ns()
            result = self._tracker.track(frame_sample.obj)
            stop = time.perf_counter_ns()
            self._prev_frame = frame_sample.obj.copy()
            bbox = getattr(result, 'bbox', None) or getattr(result, 'box', None)
            print("BBBG", bbox)
            if bbox is not None and len([bbox.x, bbox.y, bbox.w, bbox.h]) == 4:
               self._last_bbox = np.array([bbox.x, bbox.y, bbox.w, bbox.h], dtype=np.float32)
            
            
            self._latest_update_timestamp = frame_sample.timestamp_ns


            self._controller.on_sot_update(cam_id, result, self._latest_update_timestamp)
            self._logger.info("Tracking '%s' update: %s", cam_id, result)
            self._logger.debug(
                "Tracking '%s' duration: %.1f ms", cam_id, 1e-6 * (stop - start)
            )
            self._state.sot_durations[cam_id].enqueue(stop - start)

    def _try_init_tracking(self, tracking_cmd: TrOsTrackingCommand) -> None:
        if not (tracking_cmd.enabled or tracking_cmd.needs_init):
            return

        frame_sample = (
            self._camera.frames.latest()
            if tracking_cmd.timestamp_ns is None
            else self._camera.frames.get_by_timestamp(
                tracking_cmd.timestamp_ns, mode="nearest"
            )
        )
        if frame_sample is None:
            self._logger.error("[%s]: Can't receive frame needed", self._cam_id)
            return

        init_success = self._tracker.initialize(frame_sample.obj, tracking_cmd.box)
        if init_success:
            self._logger.info("Successfully initialized tracking in '%s'", self._cam_id)
            self._status = LoopStatus.TRACKING
            self._latest_update_timestamp = self._init_timestamp = (
                tracking_cmd.timestamp_ns
            )

            # Инициализация CMC
            self._motion_estimator.reset()
            self._prev_frame = frame_sample.obj.copy()
            if hasattr(tracking_cmd.box, 'x1'):
                self._last_bbox = np.array(
                    [tracking_cmd.box.x1, tracking_cmd.box.y1,
                     tracking_cmd.box.x2, tracking_cmd.box.y2],
                    dtype=np.float32,
                )
            elif hasattr(tracking_cmd.box, 'x'):
                self._last_bbox = np.array(
                    [tracking_cmd.box.x, tracking_cmd.box.y,
                     tracking_cmd.box.x + tracking_cmd.box.w,
                     tracking_cmd.box.y + tracking_cmd.box.h],
                    dtype=np.float32,
                )
            else:
                try:
                    self._last_bbox = np.array(tracking_cmd.box, dtype=np.float32)
                except Exception:
                    self._last_bbox = None

            self._controller.on_sot_initialized(
                self._cam_id, tracking_cmd.box, frame_sample.timestamp_ns
            )
        else:
            self._logger.warning("[%s]: Tracker initialization failed", self._cam_id)

    def on_stop(self) -> None:
        pass

    def time_ns(self) -> float:
        return time.time_ns()
