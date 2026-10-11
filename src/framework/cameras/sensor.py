from dataclasses import dataclass

from src.framework.cameras.resolution import Resolution


@dataclass
class Sensor:
    resolution: Resolution
    pixel_size_um: float

    @property
    def sensor_width_mm(self) -> float:
        return self.resolution.width * self.pixel_size_um * 1e-3

    @property
    def sensor_height_mm(self) -> float:
        return self.resolution.height * self.pixel_size_um * 1e-3
