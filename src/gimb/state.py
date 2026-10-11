import statistics
import time
from typing import List, Optional

import numpy as np

from src.gimb.domain.cameras import (
    TrOsCamHolder,
    TrOsCamData,
    TROS_TV_N,
    TROS_ALIGNMENT_PATH,
    TrOsChannel,
    TrOsCamId,
)
from src.gimb.domain.detector import TrOsDetectorDesiredState
from src.gimb.domain.tracking import (
    TrOsTrackingDesiredState,
    TrOsTrackInitMode,
    TrOsTrackingCommand,
    TrOsTrackedTarget,
    TrOsTrackingStatus,
)
from src.framework.core.box import Box
from src.framework.core.thread_circular_buffer import ThreadCircularBuffer
from src.framework.det.detection import Detection
from src.framework.mot.track import Track
from src.framework.sot.single_track import SingleTrack

import config.config as config


class TrOsState:
    FRAME_CAPACITY = 60
    STAT_CAPACITY = 60
    DNN_CAPACITY = 20
    CMD_CAPACITY = 10

    def __init__(self) -> None:

        self.cameras = TrOsCamHolder(
            tv_n=TrOsCamData(
                frames=ThreadCircularBuffer[np.ndarray](capacity=self.FRAME_CAPACITY),
                camera=TROS_TV_N,
                alignment_path=TROS_ALIGNMENT_PATH / "tv_n.ini",
            )
        )

        self.detector_desired_state = TrOsDetectorDesiredState(enabled=True)

        self.tracking_desired_state = TrOsTrackingDesiredState(
            enabled=True,
            channel=TrOsChannel.TV,
            init_mode=TrOsTrackInitMode.DETECTION,
            init_data_rad=(0.0, 0.0),
            recv_time_ns=time.monotonic_ns(),
        )

        self.tracked_target = TrOsTrackedTarget.disabled()


        self.detections = TrOsCamHolder(
            tv_n=ThreadCircularBuffer[List[Detection]](self.DNN_CAPACITY)
        )
        self.det_durations = TrOsCamHolder(
            tv_n=ThreadCircularBuffer[int](self.STAT_CAPACITY)
        )

        self.interesting_classes = {0, 1, 2, 3, 4, 5}

        self.mot_tracks = TrOsCamHolder(
            tv_n=ThreadCircularBuffer[List[Track]](self.DNN_CAPACITY)
        )
        self.mot_durations = TrOsCamHolder(
            tv_n=ThreadCircularBuffer[int](self.STAT_CAPACITY)
        )

        self.tracker_cmd = TrOsCamHolder(
            tv_n=ThreadCircularBuffer[TrOsTrackingCommand](self.CMD_CAPACITY)
        )

        self.single_tracks = TrOsCamHolder(
            tv_n=ThreadCircularBuffer[SingleTrack](self.DNN_CAPACITY)
        )
        self.sot_durations = TrOsCamHolder(
            tv_n=ThreadCircularBuffer[int](self.STAT_CAPACITY)
        )

        self.class_names = ["car", "truck", "heavy", "moto", "boat", "people"]

    def get_det_fps(self, cam_id: TrOsCamId) -> float:
        stats = getattr(self, "hailo_performance", None)
        if stats is not None:
            return stats.rate("yolo")
        buf = self.det_durations[cam_id]
        if buf is None:
            return 0.0
        return self._get_fps([1e-9 * d.obj for d in buf if d.obj is not None])

    def get_mot_fps(self, cam_id: TrOsCamId) -> float:
        stats = getattr(self, "hailo_performance", None)
        if stats is not None:
            return stats.rate("mot")
        buf = self.mot_durations[cam_id]
        if buf is None:
            return 0.0
        return self._get_fps([1e-9 * d.obj for d in buf if d.obj is not None])

    def get_sot_fps(self, cam_id: TrOsCamId) -> float:
        buf = self.sot_durations[cam_id]
        if buf is None:
            return 0.0
        return self._get_fps([1e-9 * d.obj for d in buf if d.obj is not None])

    @staticmethod
    def _get_fps(durations: list[float]) -> float:
        if len(durations) == 0:
            return 0.0
        return 1 / statistics.mean(durations)

    def recalculate_box(self, src_id: TrOsCamId, dst_id: TrOsCamId, box: Box) -> Box:
        angular_box = self.cameras[src_id].alignment.px2ang_box(box)
        return self.cameras[dst_id].alignment.ang2px_box(angular_box)

    def time_ns(self) -> int:
        return time.monotonic_ns()
