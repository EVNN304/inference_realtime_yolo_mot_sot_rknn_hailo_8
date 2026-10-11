from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class Spherical:
    azimuth: float
    elevation: float
    distance: Optional[float]
