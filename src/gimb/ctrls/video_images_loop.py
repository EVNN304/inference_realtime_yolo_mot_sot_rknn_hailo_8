import math
import logging
import time
from typing import Optional
import cv2

from src.gimb.domain.cameras import TrOsCamHolder, TrOsCamId
from src.gimb.state import TrOsState
from src.framework.loop.loop import ILoop
import config.config as config


class TrOsVideoImagesLoop(ILoop):
    def __init__(self, state: TrOsState) -> None:
        super().__init__()
        self._state = state
        self._caps: Optional[TrOsCamHolder[cv2.VideoCapture]] = None
        self._logger = logging.getLogger("video_images")
        self._realtime = "hailo" in (
            getattr(config, "detector_backend", "rknn"),
            getattr(config, "mot_backend", "rknn"),
        )
        self._due = None
        self._period = 1 / 30.0
        self._stats = None

    def on_start(self) -> None:

        self._caps = TrOsCamHolder(
            tv_n=self._try_create_capture(path=config.DatasetVideoPath.tv_n)
        )

        if self._realtime:
            from src.framework.hailo.performance import get_stats
            self._stats = get_stats(self._state)
            if self._caps.tv_n is None:
                raise RuntimeError("Cannot open video file")
            fps = float(getattr(config.Hailo, "video_fps_override", 0) or
                        self._caps.tv_n.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                fps = 30.0
                self._logger.warning("Missing file FPS, using 30")
            self._period = 1.0 / fps
            self._logger.info("Realtime video: %.3f FPS; decode time included in pacing", fps)

    def _try_create_capture(self, path: str) -> Optional[cv2.VideoCapture]:
        if not path:
            return None
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._logger.warning(f"Failed to open video source: {path}")
            return None
        return cap

    def update(self) -> None:
        if not self._realtime:
            return self._legacy_update()
        cap = self._caps.tv_n
        started = time.monotonic()
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            self._due = None
            if not ok:
                raise RuntimeError("Cannot read video after rewind")
        now = time.monotonic()
        self._stats.observe("decode_ms", (now-started)*1000)
        if self._due is None:
            self._due = now
        # Stay on the file's clock after a slow decode/scheduling stall. Advance
        # past expired source frames rather than showing a rapid catch-up burst.
        late_frames = max(0, int((now-self._due)/self._period))
        skipped = 0
        for _ in range(min(late_frames, 120)):
            if not cap.grab():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Cannot rewind video")
                self._due = time.monotonic()
                skipped = 0
                break
            skipped += 1
            self._due += self._period
        if skipped:
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError("Cannot retrieve video frame")
            self._stats.tick("source_skipped", skipped)
        remaining = self._due-time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        timestamp = time.monotonic_ns()
        cam = self._state.cameras[TrOsCamId.TV_N]
        cam.frames.enqueue(frame, timestamp)
        if cam.resolution is None:
            cam.resolution = frame.shape[:2]
        self._stats.tick("source")
        self._due += self._period

    def _legacy_update(self) -> None:

        if self._caps is not None and self._caps.tv_n is not None:
            cap = self._caps.tv_n
            ret, frame = cap.read()
            if not ret:

                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    return
            
            self._state.cameras[TrOsCamId.TV_N].frames.enqueue(frame, time.monotonic_ns())
            if self._state.cameras[TrOsCamId.TV_N].resolution is None:
                self._state.cameras[TrOsCamId.TV_N].resolution = frame.shape[:2]
        
        time.sleep(1 / 30.0)  # ~30 FPS

    def on_stop(self) -> None:
        if self._caps is not None and self._caps.tv_n is not None:
            self._caps.tv_n.release()
