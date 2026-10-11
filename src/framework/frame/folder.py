from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class _State:
    index: int
    direction: int  # +1 or -1


class FolderVideoCapture:
    """
    VideoCapture-like reader for image sequences from a folder.

    Supports:
    - forward playback
    - ping-pong playback (forward -> backward -> ...)
    - looping
    """

    DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

    def __init__(
        self,
        folder: str | Path,
        *,
        extensions: Sequence[str] = DEFAULT_EXTS,
        recursive: bool = False,
        ping_pong: bool = False,
        loop: bool = False,
        copy_frame: bool = True,
        imread_flag: int = cv2.IMREAD_COLOR,
    ) -> None:
        self._folder = Path(folder)
        if not self._folder.exists():
            raise ValueError(f"Folder does not exist: {folder}")

        self._extensions = tuple(e.lower() for e in extensions)
        self._recursive = recursive
        self._ping_pong = ping_pong
        self._loop = loop
        self._copy_frame = copy_frame
        self._imread_flag = imread_flag

        self._files = self._scan_files()
        if not self._files:
            raise ValueError("No images found in folder")

        self._released = False
        self._state = _State(index=0, direction=1)
        self._virtual_pos = 0

        # Lazy load: only load first frame for metadata
        first = self._read_image(self._files[0])
        self._height, self._width = first.shape[:2]

    def _scan_files(self) -> list[Path]:
        if self._recursive:
            files = self._folder.rglob("*")
        else:
            files = self._folder.glob("*")

        files = [
            f for f in files
            if f.is_file() and f.suffix.lower() in self._extensions
        ]

        # deterministic order
        return sorted(files)

    def _read_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), self._imread_flag)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        return img

    # --- OpenCV-like API ---

    def isOpened(self) -> bool:
        return not self._released

    def release(self) -> None:
        self._released = True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._released:
            return False, None

        idx = self._state.index
        if idx < 0 or idx >= len(self._files):
            return False, None

        frame = self._read_image(self._files[idx])
        if self._copy_frame:
            frame = frame.copy()

        next_state = self._next_state(self._state)

        if next_state is None:
            self._released = not self._loop
        else:
            self._state = next_state

        self._virtual_pos += 1
        return True, frame

    def _next_state(self, state: _State) -> Optional[_State]:
        n = len(self._files)
        i, d = state.index, state.direction

        if n == 1:
            return _State(0, 1) if self._loop else None

        ni = i + d

        if not self._ping_pong:
            if ni >= n:
                if self._loop:
                    return _State(0, 1)
                return None
            return _State(ni, 1)

        # ping-pong logic
        if ni >= n:
            return _State(n - 2, -1)
        if ni < 0:
            if self._loop:
                return _State(1, 1)
            return None

        return _State(ni, d)

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._files))
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self._virtual_pos)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            idx = int(value)
            if 0 <= idx < len(self._files):
                self._state = _State(index=idx, direction=1)
                self._virtual_pos = idx
                return True
        return False
