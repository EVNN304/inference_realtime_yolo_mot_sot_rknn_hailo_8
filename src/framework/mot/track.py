from abc import abstractmethod
from dataclasses import dataclass

from src.framework.core.box import Box
from src.framework.det.detection import IDetection


class ITrack(IDetection):
    __slots__ = ()

    @property
    @abstractmethod
    def track_id(self) -> int:
        ...


@dataclass(frozen=True, slots=True)
class Track(ITrack):
    box: Box
    cls: int
    conf: float
    track_id: int
