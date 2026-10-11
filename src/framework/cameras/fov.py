import math
from typing import NamedTuple


class FieldOfView(NamedTuple):
    """
    Filed of view in radians
    """
    hor_rad: float
    ver_rad: float

    @property
    def hor_deg(self) -> float:
        return math.degrees(self.hor_rad)

    @property
    def ver_deg(self) -> float:
        return math.degrees(self.ver_rad)
