import mmap
import os
import struct
import time
from collections import deque

from dataclasses import dataclass
from typing import Tuple, Optional, Deque

import numpy as np

from src.framework.frame.shm.format_manager import FormatManager, ImageFormat


@dataclass
class FrameMeta:
    last_update: int
    last_update_time: int
    type: int
    period: int
    width: int
    height: int
    image_type: ImageFormat

@dataclass
class ShmemFramedTimestamp:
    update_time : int
    offset : int

class SharedMemoryCapReader:
    META_HEADER_SIZE = 1024
    STORAGE_HEADER_SIZE = 1024
    CAMERA_HEADER_SIZE = 2048
    HEADER_SIZE = META_HEADER_SIZE + STORAGE_HEADER_SIZE + CAMERA_HEADER_SIZE

    META_HEADER_OFFSET = 0
    STORAGE_HEADER_OFFSET = META_HEADER_SIZE
    CAMERA_HEADER_OFFSET = STORAGE_HEADER_OFFSET + STORAGE_HEADER_SIZE

    CAMERA_META_INFO_SIZE = 44
    STORAGE_META_INFO_SIZE = 16

    def __init__(self, path: str, img_format: str = 'rgb'):
        self.path = path

        self._fd = None
        self._mm = None
        self._last_update = 0
        self._last_update_time = 0
        self._last_shmem_framed_timestamp = ShmemFramedTimestamp(0, 0)
        img_format = getattr(ImageFormat, img_format.upper(),
                             None) or ImageFormat.RGB
        self._timestamp_history: Deque[ShmemFramedTimestamp] = deque(maxlen=120)
        self._format_manager = FormatManager(img_format)
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self, new_only: bool = False) -> bool:
        try:
            self._is_running = self._start(new_only)
        except (IOError, ValueError, struct.error):
            self._stop()
        return self._is_running

    @property
    def last_update(self) -> int:
        return self._last_update

    @property
    def last_update_time(self) -> int:
        return self._last_update_time

    @property
    def last_shmem_framed_timestamp(self) -> ShmemFramedTimestamp:
        return self._last_shmem_framed_timestamp

    def read(self, timeout: float = 0.2, new_only=False) -> Tuple[bool, Optional[np.ndarray]]:
        start = time.time()
        ret, frame = False, None
        while not ret and start + timeout > time.time():
            ret, frame = self.read_no_wait(new_only=new_only)
            if not ret:
                time.sleep(5e-3)
        return ret, frame

    def read_no_wait(self, new_only: bool = False) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            ret, frame = self._read(new_only)
        except (RuntimeError, ValueError):
            ret, frame = False, None
        return ret, frame

    def read_by_timestamp(self, frame_timestamp: ShmemFramedTimestamp) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            if not self._is_running:
                raise RuntimeError(
                    "The streamer must be running to read the frame. "
                    "Please call 'start' before reading."
                )
            try:
                meta = self._read_meta()
            except ValueError as ext:
                raise RuntimeError(
                    'Error occurred while reading the meta.') from ext

            if meta.image_type is ImageFormat.I420:
                meta.height = int(1.5 * meta.height)
            channels = self._format_manager.get_channels_count(meta.image_type)

            image = np.empty((meta.height, meta.width, channels), dtype=np.uint8)

            raw = np.memmap(self.path,
                            mode="r",
                            shape=(meta.height, meta.width, channels),
                            offset=frame_timestamp.offset)
            np.copyto(image, raw)

            h, w, _ = image.shape
            data = np.ndarray(shape=(h, w, channels), dtype=np.uint8, buffer=image)
            ret, frame = True, data

        except (RuntimeError, ValueError):
            ret, frame = False, None
        return ret, frame

    def find_by_nearest_timestamp_ns(self, timestamp_ns: int) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._timestamp_history:
            return False, None

        timestamp_us = 1e3 * timestamp_ns
        nearest = min(
            self._timestamp_history,
            key=lambda x: abs(x.update_time - timestamp_us),
        )

        return self.read_by_timestamp(frame_timestamp=nearest)

    def _read(self, new_only: bool) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._is_running:
            raise RuntimeError(
                "The streamer must be running to read the frame. "
                "Please call 'start' before reading."
            )

        try:
            meta = self._read_meta()
        except ValueError as ext:
            raise RuntimeError(
                'Error occurred while reading the meta.') from ext

        if new_only and not self._is_frame_new(meta):
            return False, None

        if meta.image_type is ImageFormat.I420:
            meta.height = int(1.5 * meta.height)
        channels = self._format_manager.get_channels_count(meta.image_type)

        image = np.empty((meta.height, meta.width, channels), dtype=np.uint8)

        try:
            self._mm.seek(self.STORAGE_HEADER_OFFSET)
            storage_bytes = self._mm.read(self.STORAGE_META_INFO_SIZE)
            storage = struct.unpack_from("=QQ", storage_bytes, 0)
        except (ValueError, struct.error) as ext:
            raise RuntimeError(
                'Error while reading storage meta info.') from ext

        offset = self.HEADER_SIZE + storage[1]
        raw = np.memmap(self.path,
                        mode="r",
                        shape=(meta.height, meta.width, channels),
                        offset=offset)
        np.copyto(image, raw)

        h, w, _ = image.shape
        data = np.ndarray(shape=(h, w, channels), dtype=np.uint8, buffer=image)
        self._last_update = meta.last_update
        self._last_update_time = meta.last_update_time
        self._last_shmem_framed_timestamp = ShmemFramedTimestamp(meta.last_update_time, offset)
        self._timestamp_history.append(self._last_shmem_framed_timestamp)
        return True, data

    def release(self) -> None:
        self._stop()

    def _read_meta(self) -> FrameMeta:
        self._mm.seek(self.META_HEADER_OFFSET)
        meta_bytes = self._mm.read(self.CAMERA_META_INFO_SIZE)
        meta = list(struct.unpack_from("=QQQQIII", meta_bytes, 0))
        meta = FrameMeta(*meta)
        meta.image_type = ImageFormat(meta.image_type)
        return meta

    def _start(self, new_only: bool) -> bool:
        self._stop()
        self._fd = os.open(self.path, os.O_RDONLY)
        self._mm = mmap.mmap(self._fd, 0, access=mmap.PROT_READ)
        is_running = True
        if new_only:
            meta = self._read_meta()
            is_running = self._is_frame_new(meta)
        return is_running

    def _is_frame_new(self, meta: FrameMeta):
        is_frame_new = meta.last_update != self._last_update
        return is_frame_new

    def _stop(self) -> None:
        if self._is_open():
            os.close(self._fd)
            self._fd = None
        if self._is_readable():
            self._mm.close()
            self._mm = None
        self._is_running = False

    def _is_open(self) -> bool:
        return self._fd is not None

    def _is_readable(self) -> bool:
        return self._mm is not None


class MockSharedMemoryCapReader:
    def __init__(self, width, height, channel = 3):
        self.width = width
        self.height = height
        self.channel = channel
        self._last_update = 0

    @property
    def last_update(self) -> int:
        return self._last_update
        
    def read(self) -> Tuple[bool, np.ndarray]:
        data = np.ndarray(shape=(self.height, self.width, self.channel), dtype=np.uint8)
        self._last_update = int(time.time() * 1000)
        return True, data
    
    def start(self) -> bool:
        return True
