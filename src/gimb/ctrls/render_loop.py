import time
from collections import deque
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from src.gimb.domain.cameras import TrOsCamId, TrOsCamHolder
from src.gimb.domain.tracking import TrOsTrackingStatus
from src.gimb.state import TrOsState
from src.framework.core.box import Box
from src.framework.det.detection import IDetection
from src.framework.loop.loop import ILoop
from src.framework.mot.track import Track


def _clip_box_to_image(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.as_xyxy().as_tuple()
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


class TrOsRenderLoop(ILoop):
    def __init__(self, state: TrOsState) -> None:
        super().__init__()
        self._state = state
        

        self._enabled_cameras = {
            TrOsCamId.TV_N,
        }
        
        self._show_det = True
        self._show_mot = True
        self._show_sot = True
        self._show_turret_cmd = True
        self._framerate = 30


        self._upscale = {TrOsCamId.TV_N: 1.0}
        self._rotate = {TrOsCamId.TV_N: None}

        self._frames_storage = 90
        self._rendered_previous_frames = TrOsCamHolder(
            tv_n=deque(maxlen=self._frames_storage),
        )

        self._detections_timeout_ns = 600_000_000
        self._mot_timeout_ns = 700_000_000
        self._sot_timeout_ns = 500_000_000

        self._det_color = (7, 193, 255)      # Amber  (#FFC107)
        self._mot_color = (255, 121, 41)     # Blue   (#2979FF)
        self._sot_color = (83, 200, 0)       # Green  (#00C853)
        self._tracked_color = (0, 0, 213)    # Red    (#D50000)
        self._extrapolated_color = (255, 0, 170) # Violet (#AA00FF)

    def on_start(self) -> None:
        pass

    def update(self) -> None:
        render_start = time.perf_counter()
        

        for cam_id in self._enabled_cameras:
            frame_sample = self._state.cameras[cam_id].frames.latest()
            if frame_sample is None:
                continue

            img = frame_sample.obj.copy()
            fps_lines = []

            if self._show_det:
                img = self._render_detections(img, cam_id, frame_sample.timestamp_ns)
                fps_lines.append(f"det fps: {self._state.get_det_fps(cam_id):.1f}")

            if self._show_mot:
                img = self._render_tracks(img, cam_id, frame_sample.timestamp_ns)
                fps_lines.append(f"mot fps: {self._state.get_mot_fps(cam_id):.1f}")

            if self._show_sot:
                img = self._render_sot(img, cam_id, frame_sample.timestamp_ns)
                fps_lines.append(f"sot fps: {self._state.get_sot_fps(cam_id):.1f}")

            if self._show_turret_cmd:
                img = self._render_turret_cmd(img, cam_id, frame_sample.timestamp_ns)

            upscale_coeff = self._upscale.get(cam_id, 1.0)
            img = cv2.resize(img, None, fx=upscale_coeff, fy=upscale_coeff)
            
            rotate_op = self._rotate.get(cam_id)
            if rotate_op is not None:
                img = cv2.rotate(img, rotate_op)

            img = self._draw_fps_block(img, fps_lines)
            self._rendered_previous_frames[cam_id].append(img)
            cv2.imshow(str(cam_id), img)

        render_stop = time.perf_counter()
        render_dur_ms = int(1e3 * (render_stop - render_start))
        wait_time = 1000 // self._framerate - render_dur_ms
        wait_time = max(wait_time, 1)
        key = cv2.waitKey(wait_time) & 0xFF

        if key == 32:
            previous_frame_count = 0
            while True:
                key = cv2.waitKey(0)
                if key == 32:
                    break
                if key in (81, 2424832):
                    previous_frame_count += 1
                    previous_frame_count = min(previous_frame_count, self._frames_storage)
                elif key in (83, 2555904):
                    previous_frame_count -= 1
                    previous_frame_count = max(previous_frame_count, 0)

                self._render_previous_frames(previous_frame_count)

    def _render_previous_frames(self, count=0):
        for cam_id in self._enabled_cameras:
            rendered_frames = self._rendered_previous_frames[cam_id]
            if len(rendered_frames) == 0 or count > len(rendered_frames):
                continue
            index = len(rendered_frames) - count
            index = max(min(index, len(rendered_frames) - 1), 0)
            img = rendered_frames[index]
            cv2.imshow(str(cam_id), img)

    def on_stop(self) -> None:
        pass

    def _render_detections(
        self,
        img: np.ndarray,
        cam_id: TrOsCamId,
        frame_timestamp_ns: int,
    ) -> np.ndarray:
        det_sample = self._state.detections[cam_id].latest()
        if det_sample is None:
            return img

        age_ns = frame_timestamp_ns - det_sample.timestamp_ns
        if age_ns < 0 or age_ns > self._detections_timeout_ns:
            return img

        return self._draw_detection_boxes(
            img,
            det_sample.obj,
            class_names=self._state.class_names,
            color=self._det_color,
            thickness=1,
            font_scale=0.5,
            draw_label=True,
            draw_conf=True,
            alpha=0.25,
        )

    @staticmethod
    def _draw_detection_boxes(
        img: np.ndarray,
        detections: Iterable[IDetection],
        *,
        color=(0, 191, 255),
        class_names: Optional[Sequence[str]] = None,
        conf_fmt: str = "{:.2f}",
        thickness: int = 1,
        font_scale: float = 0.5,
        draw_label: bool = True,
        draw_conf: bool = True,
        alpha: float = 0.0,
    ) -> np.ndarray:
        overlay = None
        if alpha > 0.0:
            overlay = img.copy()

        h, w = img.shape[:2]

        for det in detections:
            cls = int(det.cls)
            conf = float(det.conf)

            x1, y1, x2, y2 = _clip_box_to_image(det.box, width=w, height=h)
            if x2 <= x1 or y2 <= y1:
                continue

            target = overlay if overlay is not None else img

            if overlay is not None:
                cv2.rectangle(target, (x1, y1), (x2, y2), color, thickness=-1)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)

            if draw_label:
                if class_names is not None and 0 <= cls < len(class_names):
                    cls_text = class_names[cls]
                else:
                    cls_text = str(cls)

                label = cls_text
                if draw_conf:
                    label = f"{label} {conf_fmt.format(conf)}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
                baseline = max(baseline, 2)

                y_text = y1 - 5
                y_bg_top = y1 - th - baseline - 6
                if y_bg_top < 0:
                    y_bg_top = y1
                    y_text = y1 + th + baseline + 1

                x_bg_left = x1
                x_bg_right = min(w - 1, x1 + tw + 6)
                y_bg_bottom = min(h - 1, y_bg_top + th + baseline + 6)

                cv2.rectangle(
                    img,
                    (x_bg_left, y_bg_top),
                    (x_bg_right, y_bg_bottom),
                    color,
                    thickness=-1,
                )

                cv2.putText(
                    img,
                    label,
                    (x1 + 3, y_text),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        if overlay is not None:
            a = max(0.0, min(1.0, float(alpha)))
            return cv2.addWeighted(overlay, a, img, 1.0 - a, 0.0)

        return img

    def _render_tracks(
        self,
        img: np.ndarray,
        cam_id: TrOsCamId,
        frame_timestamp_ns: int,
    ) -> np.ndarray:
        track_sample = self._state.mot_tracks[cam_id].latest()
        if track_sample is None:
            return img

        age_ns = frame_timestamp_ns - track_sample.timestamp_ns
        if age_ns < 0 or age_ns > self._mot_timeout_ns:
            return img

        return self._draw_track_boxes(
            img,
            track_sample.obj,
            color=self._mot_color,
            thickness=2,
            font_scale=0.5,
            draw_label=True,
            alpha=0.0,
            label_prefix="id:",
        )

    @staticmethod
    def _draw_fps_block(
        img: np.ndarray,
        lines: list[str],
        *,
        origin: tuple[int, int] = (10, 20),
        font_scale: float = 0.5,
        color: tuple[int, int, int] = (255, 255, 255),
        bg_color: tuple[int, int, int] = (0, 0, 0),
        alpha: float = 0.4,
    ) -> np.ndarray:
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX

        sizes = [cv2.getTextSize(line, font, font_scale, 1) for line in lines]
        max_w = max(w for (w, h), _ in sizes)
        total_h = sum(h + b + 4 for (_, h), b in sizes)

        overlay = img.copy()

        cv2.rectangle(
            overlay,
            (x - 5, y - 15),
            (x + max_w + 10, y + total_h),
            bg_color,
            thickness=-1,
        )

        img = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)

        y_cursor = y
        for line, ((w, h), baseline) in zip(lines, sizes):
            cv2.putText(
                img,
                line,
                (x, y_cursor),
                font,
                font_scale,
                color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            y_cursor += h + baseline + 4

        return img

    @staticmethod
    def _draw_track_boxes(
        img: np.ndarray,
        tracks: Iterable[Track],
        *,
        color: tuple[int, int, int] = (254, 79, 48),
        thickness: int = 2,
        font_scale: float = 0.5,
        draw_label: bool = True,
        alpha: float = 0.0,
        label_prefix: str = "id:",
    ) -> np.ndarray:
        overlay = None
        if alpha > 0.0:
            overlay = img.copy()

        h, w = img.shape[:2]

        for track in tracks:
            x1, y1, x2, y2 = _clip_box_to_image(track.box, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            target = overlay if overlay is not None else img

            if overlay is not None:
                cv2.rectangle(target, (x1, y1), (x2, y2), color, thickness=-1)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)

            if draw_label:
                label = f"{label_prefix} {track.track_id}" if label_prefix else str(track.track_id)

                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
                baseline = max(baseline, 2)

                y_bg_top = y2 + 1
                y_bg_bottom = y2 + th + baseline + 7
                y_text = y2 + th + baseline + 1

                if y_bg_bottom >= h:
                    y_bg_bottom = y2
                    y_bg_top = max(0, y2 - th - baseline - 6)
                    y_text = max(th + 1, y2 - 5)

                x_bg_left = x1
                x_bg_right = min(w - 1, x1 + tw + 6)

                cv2.rectangle(
                    img,
                    (x_bg_left, y_bg_top),
                    (x_bg_right, y_bg_bottom),
                    color,
                    thickness=-1,
                )

                cv2.putText(
                    img,
                    label,
                    (x1 + 3, y_text),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        if overlay is not None:
            a = max(0.0, min(1.0, float(alpha)))
            return cv2.addWeighted(overlay, a, img, 1.0 - a, 0.0)

        return img

    def _render_sot(
        self,
        img: np.ndarray,
        cam_id: TrOsCamId,
        frame_timestamp_ns: int,
    ) -> np.ndarray:
        track_sample = self._state.single_tracks[cam_id].latest()
        if track_sample is None:
            return img

        age_ns = frame_timestamp_ns - track_sample.timestamp_ns
        if age_ns < 0 or age_ns > self._sot_timeout_ns:
            return img

        return self._draw_single_box(
            img,
            track_sample.obj.box,
            color=self._sot_color,
            thickness=1,
            font_scale=0.5,
            draw_label=False,
            alpha=0.3,
            label="sot",
        )

    def _render_turret_cmd(self, img, cam_id, timestamp_ns) -> np.ndarray:
        tracked_target = self._state.tracked_target
        if tracked_target.status == TrOsTrackingStatus.DISABLED:
            return img

        if tracked_target.status == TrOsTrackingStatus.ENABLED:
            color = self._tracked_color
        else:
            color = self._extrapolated_color

        ang_box = tracked_target.angular_box
        pix_box = self._state.cameras[cam_id].alignment.ang2px_box(ang_box)

        return self._draw_single_box(
            img,
            pix_box,
            color=color,
            thickness=2,
            font_scale=0.5,
            draw_label=False,
            alpha=0.0,
            label=f"{tracked_target.confidence:.2f}",
        )

    @staticmethod
    def _draw_single_box(
        img: np.ndarray,
        box: Box,
        *,
        color: tuple[int, int, int],
        thickness: int = 2,
        font_scale: float = 0.5,
        draw_label: bool = True,
        alpha: float = 0.0,
        label: str = "",
    ) -> np.ndarray:
        overlay = None
        if alpha > 0.0:
            overlay = img.copy()

        h, w = img.shape[:2]

        x1, y1, x2, y2 = _clip_box_to_image(box, w, h)

        if x2 <= x1 or y2 <= y1:
            return img

        target = overlay if overlay is not None else img

        if overlay is not None:
            cv2.rectangle(target, (x1, y1), (x2, y2), color, thickness=-1)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)

        if draw_label and label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
            baseline = max(baseline, 2)

            y_bg_top = y2 + 1
            y_bg_bottom = y2 + th + baseline + 7
            y_text = y2 + th + baseline + 1

            if y_bg_bottom >= h:
                y_bg_bottom = y2
                y_bg_top = max(0, y2 - th - baseline - 6)
                y_text = max(th + 1, y2 - 5)

            x_bg_left = x1
            x_bg_right = min(w - 1, x1 + tw + 6)

            cv2.rectangle(
                img,
                (x_bg_left, y_bg_top),
                (x_bg_right, y_bg_bottom),
                color,
                thickness=-1,
            )

            cv2.putText(
                img,
                label,
                (x1 + 3, y_text),
                font,
                font_scale,
                (0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        if overlay is not None:
            a = max(0.0, min(1.0, float(alpha)))
            return cv2.addWeighted(overlay, a, img, 1.0 - a, 0.0)

        return img
