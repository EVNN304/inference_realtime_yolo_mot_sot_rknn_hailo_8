from __future__ import annotations

from enum import Enum
from typing import Optional, Protocol

import cv2
import numpy as np


class VideoCaptureLike(Protocol):
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        ...

    def isOpened(self) -> bool:
        ...

    def release(self) -> None:
        ...

    def get(self, prop_id: int) -> float:
        ...

    def set(self, prop_id: int, value: float) -> bool:
        ...


class LoopMode(str, Enum):
    NONE = "none"
    FORWARD = "forward"
    PING_PONG = "ping_pong"


class LoopingVideoCapture:
    """
    Wrapper around a VideoCapture-like object that adds looping.

    Supported modes:
    - NONE:
        Normal playback, no loop.
    - FORWARD:
        After the last frame, jump to frame 0 and continue.
    - PING_PONG:
        Playback goes forward, then backward, then forward...
        Requires caching frames that were already read.

    Notes:
    - FORWARD mode works best when underlying capture supports seeking:
          cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    - PING_PONG mode does not require reverse seek support, because frames
      are cached during forward reading.
    """

    def __init__(
        self,
        cap: VideoCaptureLike,
        *,
        loop_mode: LoopMode = LoopMode.NONE,
        copy_frame: bool = True,
        max_cache_frames: Optional[int] = None,
    ) -> None:
        self._cap = cap
        self._loop_mode = loop_mode
        self._copy_frame = copy_frame
        self._released = False

        self._frame_count = self._try_get_frame_count()
        self._supports_seek = self._detect_seek_support()

        # Cache for ping-pong playback.
        # We store frames in playback order as they are first seen.
        self._cache: list[np.ndarray] = []
        self._max_cache_frames = max_cache_frames

        # Direction/state for ping-pong
        self._playing_backward = False
        self._backward_index = -1

        # Current logical position of last returned frame
        self._pos_frames = 0

    def _try_get_frame_count(self) -> Optional[int]:
        try:
            value = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if value > 0:
                return value
        except Exception:
            pass
        return None

    def _detect_seek_support(self) -> bool:
        try:
            cur = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            ok = self._cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
            return bool(ok)
        except Exception:
            return False

    def isOpened(self) -> bool:
        return (not self._released) and self._cap.isOpened()

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._cap.release()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self.isOpened():
            return False, None

        if self._loop_mode == LoopMode.NONE:
            return self._read_normal()

        if self._loop_mode == LoopMode.FORWARD:
            return self._read_forward_loop()

        if self._loop_mode == LoopMode.PING_PONG:
            return self._read_ping_pong()

        return False, None

    def _maybe_copy(self, frame: np.ndarray) -> np.ndarray:
        return frame.copy() if self._copy_frame else frame

    def _append_cache(self, frame: np.ndarray) -> None:
        if self._max_cache_frames is None:
            self._cache.append(frame.copy())
            return

        if len(self._cache) < self._max_cache_frames:
            self._cache.append(frame.copy())

    def _read_normal(self) -> tuple[bool, Optional[np.ndarray]]:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None

        self._pos_frames += 1
        return True, self._maybe_copy(frame)

    def _restart_forward(self) -> bool:
        if self._supports_seek:
            ok = self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if ok:
                self._pos_frames = 0
                return True
        return False

    def _read_forward_loop(self) -> tuple[bool, Optional[np.ndarray]]:
        ret, frame = self._cap.read()
        if ret and frame is not None:
            self._pos_frames += 1
            return True, self._maybe_copy(frame)

        if not self._restart_forward():
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None

        self._pos_frames += 1
        return True, self._maybe_copy(frame)

    def _read_ping_pong(self) -> tuple[bool, Optional[np.ndarray]]:
        # If we are currently replaying cached frames backward
        if self._playing_backward:
            if 0 <= self._backward_index < len(self._cache):
                frame = self._cache[self._backward_index]
                self._backward_index -= 1
                self._pos_frames += 1

                # Once we reached the first frame, switch back to forward
                if self._backward_index < 0:
                    self._playing_backward = False

                return True, self._maybe_copy(frame)

            self._playing_backward = False

        # Normal forward read from source
        ret, frame = self._cap.read()
        if ret and frame is not None:
            self._append_cache(frame)
            self._pos_frames += 1
            return True, self._maybe_copy(frame)

        # End reached. Try to reverse using cache.
        if len(self._cache) <= 1:
            # Nothing meaningful to ping-pong
            return False, None

        # Start backward playback from the frame before the last one,
        # so we do not duplicate the edge frame:
        # 0,1,2,3 -> 2,1,0 -> 1,2,3 ...
        self._playing_backward = True
        self._backward_index = len(self._cache) - 2

        if self._backward_index < 0:
            return False, None

        frame = self._cache[self._backward_index]
        self._backward_index -= 1
        self._pos_frames += 1

        if self._backward_index < 0:
            self._playing_backward = False
            # Restart forward source if possible
            self._restart_forward()

        return True, self._maybe_copy(frame)

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self._pos_frames)

        return self._cap.get(prop_id)

    def set(self, prop_id: int, value: float) -> bool:
        # Reset wrapper state when caller seeks manually
        ok = self._cap.set(prop_id, value)

        if ok and prop_id == cv2.CAP_PROP_POS_FRAMES:
            self._playing_backward = False
            self._backward_index = -1
            self._pos_frames = int(value)

            # Cache becomes invalid after arbitrary seek
            self._cache.clear()

        return ok
