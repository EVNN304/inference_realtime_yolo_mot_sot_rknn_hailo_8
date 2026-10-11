from dataclasses import dataclass
from typing import Generic, TypeVar

from src.framework.core.spherical import Spherical

T = TypeVar('T')

@dataclass(frozen=True)
class Angular(Generic[T]):
    """
    coords : Spherical
        Angular coordinates of the measurement in spherical representation.

    data : T
        Associated payload stored together with the coordinates.
        Can be detection, frame, feature, tracker state, etc.

    timestamp : int
        Time of the measurement in nanoseconds.
        Expected to be monotonic time or system time depending on the pipeline.
    """
    coords: Spherical
    data: T
    timestamp_ns: int
