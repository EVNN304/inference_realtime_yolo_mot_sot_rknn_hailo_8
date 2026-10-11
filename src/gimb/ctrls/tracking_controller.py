import logging
import time
from typing import Optional, Sequence, Tuple

from src.gimb.domain.cameras import TrOsCamId, TrOsChannel
from src.gimb.domain.tracking import (
    TrOsTrackingDesiredState,
    TrOsTrackInitMode,
    TrOsTrackingCommand,
    TrOsTrackedTarget,
    TrOsTrackingStatus,
)
from src.gimb.state import TrOsState
from src.framework.cameras.alignment import AngularPoint, AngularSize, AngularBox, Point
import config.config as config
from src.framework.core.box import Box
from src.framework.core.box_probability_evaluator import (
    BoxProbabilityEvaluator,
    BoxMetricConfig,
)
from src.framework.mot.track import Track
from src.framework.sot.single_track import SingleTrack


class TrOsTrackingController:
    _MOT_IOU_THRESH = 0.35
    _DET_TIMEOUT_NS = 2_000_000_000

    def __init__(self, state: TrOsState) -> None:
        self._state = state
        self._logger = logging.getLogger("track")
        self._sot_box_validator = BoxProbabilityEvaluator(
            BoxMetricConfig(
                weight_center=0.5,
                weight_scale=3.5,
                weight_aspect=2.0,
                weight_iou=0.3,
                weight_size=0.0,
            )
        )
        self._sot_box_thresh = config.Tracking.sot_box_sanity_thresh
        self._sot_frames_processed = 0
        self._sot_fail_count = 0
        self._max_sot_fail_count = 2
        self._sot_init_frames = 10
        
        # Состояние ассоциации ТОЛЬКО для одной камеры
        self._latest_mot_track: Optional[Track] = None
        self._latest_mot_ts: int = -1

    @property
    def _is_enabled(self) -> bool:
        return self._state.tracked_target.status != TrOsTrackingStatus.DISABLED

    def on_sot_initialized(self, cam_id: TrOsCamId, box: Box, timestamp: int) -> None:
        self._sot_frames_processed = 0
        self._state.single_tracks.tv_n.enqueue(SingleTrack(box), timestamp)

    def on_new_cmd(self, new_state: TrOsTrackingDesiredState) -> None:
        self._state.tracking_desired_state = new_state
        if not new_state.enabled:
            self._logger.info("Tracking disabled by command")
            self._on_disable_tracking()
            return
        if new_state.init_mode == TrOsTrackInitMode.DETECTION:
            self._handle_detection_init(new_state)
        elif new_state.init_mode == TrOsTrackInitMode.CENTER:
            self._handle_center_init(new_state)

    def _handle_detection_init(self, new_state: TrOsTrackingDesiredState) -> None:
        search_point = AngularPoint(*new_state.init_data_rad)
        cmd, mot_track, timestamp_ns = self._try_create_track_init_command_from_detection(search_point)
        
        if cmd is None:
            self._logger.info("Init tracking in detection mode failed because no detections")
            return

        self._latest_mot_track = mot_track
        self._latest_mot_ts = timestamp_ns
        self._logger.info(f"Init tracking in detection mode using target with id {mot_track.track_id}. Box: {cmd.box}")
        
        camera = self._state.cameras.tv_n
        angular_box = camera.alignment.px2ang_box(cmd.box)
        self._update_angular_target(angular_box, timestamp_ns, mot_track.conf)
        self._state.tracker_cmd.tv_n.enqueue(cmd, timestamp_ns)

    def _handle_center_init(self, new_state: TrOsTrackingDesiredState) -> None:
        self._logger.info("Init tracking with center mode")
        now_ns = time.monotonic_ns()
        target_size = AngularSize(*new_state.init_data_rad)
        camera = self._state.cameras.tv_n
        ang_box = AngularBox.from_size_and_center(center=AngularPoint(0.0, 0.0), size=target_size)
        cmd = TrOsTrackingCommand(enabled=True, needs_init=True, box=camera.alignment.ang2px_box(ang_box), timestamp_ns=now_ns)
        
        self._state.tracker_cmd.tv_n.enqueue(cmd, now_ns)
        self._update_angular_target(ang_box, now_ns, 1.0)

    def _try_create_track_init_command_from_detection(self, search_point: AngularPoint) -> Tuple[Optional[TrOsTrackingCommand], Optional[Track], int]:
        mot_sample = self._state.mot_tracks.tv_n.latest()
        if mot_sample is None or time.monotonic_ns() > mot_sample.timestamp_ns + self._DET_TIMEOUT_NS:
            return None, None, -1
            
        tracks = mot_sample.obj
        camera = self._state.cameras.tv_n
        search_point_px = camera.alignment.angular_to_pixel(search_point)
        min_distance = 10000
        min_track = None
        
        for track in tracks:
            if track.cls not in self._state.interesting_classes:
                continue
            centered_box = track.box.as_centered()
            dist = Point.distance(search_point_px, Point(centered_box.cx, centered_box.cy))
            if dist < min_distance:
                min_distance = dist
                min_track = track
                
        if min_track is None:
            return None, None, -1
            
        return (
            TrOsTrackingCommand(enabled=True, needs_init=True, box=min_track.box, timestamp_ns=mot_sample.timestamp_ns),
            min_track,
            mot_sample.timestamp_ns,
        )

    def _select_best_mot_track(self, tracks: Sequence[Track]) -> Optional[Track]:
        if not tracks:
            return None
        camera = self._state.cameras.tv_n
        im_h, im_w = camera.resolution if camera.resolution else (1080, 1920)
        center_x, center_y = im_w / 2.0, im_h / 2.0
        max_dist = (center_x**2 + center_y**2)**0.5

        best_track = None
        best_score = -1.0
        for track in tracks:
            if track.cls not in self._state.interesting_classes:
                continue
            box = track.box.as_centered()
            dist = ((box.cx - center_x)**2 + (box.cy - center_y)**2)**0.5
            norm_dist = dist / max_dist if max_dist > 0 else 0.0
            score = track.conf * (1.0 - 0.3 * norm_dist)
            if score > best_score:
                best_score = score
                best_track = track
        return best_track

    def on_sot_update(self, cam_id: TrOsCamId, track: SingleTrack, sot_ts: int) -> None:
        if not self._is_enabled:
            return
        
        self._state.single_tracks.tv_n.enqueue(track, sot_ts)
        camera = self._state.cameras.tv_n
        angular_box = camera.alignment.px2ang_box(track.box)
        box_metric = self._evaluate_sot_box(track.box)
        self._sot_frames_processed += 1

        if box_metric < self._sot_box_thresh:
            if self._latest_mot_track is not None and (sot_ts - self._latest_mot_ts) <= self._DET_TIMEOUT_NS:
                ang_box = camera.alignment.px2ang_box(self._latest_mot_track.box)
                self._logger.warning("SOT box is invalid. Reenabling from MOT.")
                self._state.tracker_cmd.tv_n.enqueue(
                    TrOsTrackingCommand(enabled=True, needs_init=True, box=camera.alignment.ang2px_box(ang_box), timestamp_ns=self._latest_mot_ts)
                )
                self._update_angular_target(ang_box, self._latest_mot_ts, self._latest_mot_track.conf)
                self._sot_fail_count = 0
            else:
                ang_box = self._state.tracked_target.angular_box
                if self._sot_frames_processed <= self._sot_init_frames:
                    self._update_angular_target(ang_box, sot_ts, box_metric)
                else:
                    self._sot_fail_count += 1
                    if self._sot_fail_count > self._max_sot_fail_count:
                        self._logger.warning("SOT box is invalid. Going to extrapolated state.")
                        self._go_to_extrapolation_state(sot_ts, box_metric)
            return
        else:
            self._logger.info("Tracking updated with %s", track)
            self._update_angular_target(angular_box, sot_ts, box_metric)
            self._sot_fail_count = 0

    def on_mot_update(self, cam_id: TrOsCamId, tracks: Sequence[Track], mot_ts: int) -> None:
        camera = self._state.cameras.tv_n
        self._state.mot_tracks.tv_n.enqueue(tracks, mot_ts)
        
        # 1. Авто-захват, если трекинг включен, но цель еще не захвачена
        if self._state.tracking_desired_state.enabled and not self._is_enabled:
            if self._state.tracking_desired_state.init_mode == TrOsTrackInitMode.DETECTION:
                best_track = self._select_best_mot_track(tracks)
                if best_track is not None:
                    self._logger.info(f"AUTO-INIT: Capturing best MOT track ID={best_track.track_id}, conf={best_track.conf:.2f}")
                    self._latest_mot_track = best_track
                    self._latest_mot_ts = mot_ts
                    ang_box = camera.alignment.px2ang_box(best_track.box)
                    self._update_angular_target(ang_box, mot_ts, best_track.conf)
                    self._state.tracker_cmd.tv_n.enqueue(
                        TrOsTrackingCommand(enabled=True, needs_init=True, box=best_track.box, timestamp_ns=mot_ts),
                        mot_ts,
                    )
                    return

        if not self._is_enabled:
            return


        single_track_sample = self._state.single_tracks.tv_n.get_by_timestamp(mot_ts, mode="nearest")
        if single_track_sample is not None and (mot_ts - single_track_sample.timestamp_ns) <= self._DET_TIMEOUT_NS:
            single_track = single_track_sample.obj
            track_ang_box = camera.alignment.px2ang_box(single_track.box)
            

            target_track = None
            if self._latest_mot_track is not None:
                for t in tracks:
                    if t.track_id == self._latest_mot_track.track_id:
                        target_track = t
                        break
            
            # Если трек с тем же ID исчез (например, из-за окклюзии), берем лучший доступный
            if target_track is None:
                target_track = self._select_best_mot_track(tracks)
            
            if target_track is not None:
                target_ang_box = camera.alignment.px2ang_box(target_track.box)
                iou = AngularBox.iou(track_ang_box, target_ang_box)
                
                # Если IoU низкий, перезахватываем цель
                if iou < self._MOT_IOU_THRESH:
                    self._logger.warning(f"Reset SOT: low IoU ({iou:.2f} < {self._MOT_IOU_THRESH}) with MOT track {target_track.track_id}")
                    self._latest_mot_track = target_track
                    self._latest_mot_ts = mot_ts
                    
                    self._state.tracker_cmd.tv_n.enqueue(
                        TrOsTrackingCommand(enabled=True, needs_init=True, box=target_track.box, timestamp_ns=mot_ts),
                        mot_ts,
                    )
                    self._update_angular_target(target_ang_box, mot_ts, target_track.conf)

    def _update_angular_target(self, angular_box: AngularBox, timestamp_ns: int, conf: float = 1.0) -> None:
        self._state.tracked_target = TrOsTrackedTarget(
            status=TrOsTrackingStatus.ENABLED,
            angular_box=angular_box,
            timestamp_ns=timestamp_ns,
            confidence=conf,
        )

    def _evaluate_sot_box(self, box: Box) -> float:
        camera = self._state.cameras.tv_n
        im_width, im_height = camera.resolution[1], camera.resolution[0]
        frame_area = im_width * im_height
        
        if self._state.tracked_target.status == TrOsTrackingStatus.DISABLED:
            return 1.0
        if box.width > 0.6 * im_width or box.height > 0.6 * im_height or box.area() > 0.55 * frame_area:
            return 0.0
            
        target_box = camera.alignment.ang2px_box(self._state.tracked_target.angular_box)
        score = self._sot_box_validator.evaluate_detailed(box, target_box)
        return score.total_score

    def _on_disable_tracking(self) -> None:
        self._state.tracked_target = TrOsTrackedTarget.disabled()
        self._latest_mot_track = None
        self._latest_mot_ts = -1
        self._state.tracker_cmd.tv_n.enqueue(TrOsTrackingCommand.disable())

    def on_sot_disabled(self, cam_id: TrOsCamId) -> None:
        self._state.single_tracks.tv_n.clear()

    def _go_to_extrapolation_state(self, timestamp_ns: int, conf: float = 0.0) -> None:
        self._state.tracked_target = self._state.tracked_target.copy_with(
            status=TrOsTrackingStatus.EXTRAPOLATING,
            confidence=conf,
            timestamp_ns=timestamp_ns,
        )
        self._logger.info("Tracker goes extrapolation first time")

    @property
    def _is_extrapolating(self) -> bool:
        return self._state.tracked_target.status == TrOsTrackingStatus.EXTRAPOLATING
