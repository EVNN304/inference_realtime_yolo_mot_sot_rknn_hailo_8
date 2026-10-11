from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Generic, Iterator, TypeVar, Callable, Optional, Tuple

import numpy as np

import config.config as config
from src.framework.cameras.alignment import CameraIntrinsics
from src.framework.cameras.alignment_config import AlignmentConfig
from src.framework.cameras.camera import Camera
from src.framework.cameras.fov import FieldOfView
from src.framework.cameras.resolution import Resolution
from src.framework.cameras.sensor import Sensor
from src.framework.cameras.sensor_lib import IMX250
from src.framework.core.thread_circular_buffer import ThreadCircularBuffer

T = TypeVar("T")
Frame = np.ndarray


class TrOsChannel(IntEnum):
    TV = 0


class TrOsCamId(IntEnum):
    TV_N = 0


TROS_TV_N = Camera(sensor=IMX250, focal_length_mm=50)
TROS_ALIGNMENT_PATH = Path(config.alignment_path)


@dataclass(slots=True)
class TrOsCamHolder(Generic[T]):
    _values: list[T] = field(init=False, repr=False)

    def __init__(self, tv_n: T) -> None:
        self._values = [tv_n]

    @property
    def tv_n(self) -> T:
        return self._values[TrOsCamId.TV_N]

    @tv_n.setter
    def tv_n(self, value: T) -> None:
        self._values[TrOsCamId.TV_N] = value

    def __getitem__(self, idx: int) -> T:
        return self._values[idx]

    def __setitem__(self, idx: int, value: T) -> None:
        self._values[idx] = value

    def __iter__(self) -> Iterator[T]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def items(self) -> Iterator[tuple[TrOsCamId, T]]:
        for cam_id in TrOsCamId:
            yield cam_id, self._values[cam_id]

    def for_all(self, func: Callable[[TrOsCamId, T], None]) -> None:
        for cam_id in TrOsCamId:
            func(cam_id, self._values[cam_id])


class TrOsCamData:
    __slots__ = (
        "frames",
        "info",
        "alignment",
        "cropped_fov",
        "resolution",
    )

    def __init__(
        self, frames: ThreadCircularBuffer[Frame], camera: Camera, alignment_path: str
    ) -> None:
        self.frames = frames
        self.info = camera
        alignment_config = AlignmentConfig.load_from_file(alignment_path)
        self.alignment = alignment_config.to_pipeline(
            CameraIntrinsics.from_camera(self.info)
        )
        self.cropped_fov = FieldOfView(
            hor_rad=alignment_config.fov.fov_h_rad,
            ver_rad=alignment_config.fov.fov_v_rad,
        )
        self.resolution: Optional[Tuple[int, int]] = None
