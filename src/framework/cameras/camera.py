import math

from dataclasses import dataclass

from src.framework.cameras.fov import FieldOfView
from src.framework.cameras.sensor import Sensor


@dataclass
class Camera:
    sensor: Sensor
    focal_length_mm: float

    @property
    def fov(self) -> FieldOfView:
        hor = 0.5 * self.sensor.sensor_width_mm / self.focal_length_mm
        ver = 0.5 * self.sensor.sensor_height_mm / self.focal_length_mm
        hor_fov = 2 * math.atan(hor)
        ver_fov = 2 * math.atan(ver)
        return FieldOfView(hor_fov, ver_fov)


class MotorizedCamera:
    def __init__(
        self,
        sensor: Sensor,
        min_focal_length_mm: float,
        max_focal_length_mm: float,
    ) -> None:
        self.sensor = sensor
        self._min_focal_length_mm = min_focal_length_mm
        self._max_focal_length_mm = max_focal_length_mm
        self._current_focal_length_mm = self._min_focal_length_mm

    @property
    def min_focal_length_mm(self) -> float:
        return self._min_focal_length_mm

    @property
    def max_focal_length_mm(self) -> float:
        return self._max_focal_length_mm

    @property
    def current_focal_length_mm(self) -> float:
        return self._current_focal_length_mm

    @property
    def current_camera(self) -> Camera:
        return Camera(self.sensor, self._current_focal_length_mm)

    @property
    def current_zoom(self) -> float:
        return self._current_focal_length_mm / self._min_focal_length_mm

    def update_focal_length_mm(self, new_focal_length_mm: float) -> None:
        self._current_focal_length_mm = new_focal_length_mm

    def update_current_zoom(self, new_zoom: float) -> None:
        self.update_focal_length_mm(self._min_focal_length_mm * new_zoom)

    @property
    def wide(self) -> Camera:
        return Camera(self.sensor, self._min_focal_length_mm)

    @property
    def tele(self) -> Camera:
        return Camera(self.sensor, self._max_focal_length_mm)
