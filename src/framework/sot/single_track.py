from abc import abstractmethod, ABC
from dataclasses import dataclass

from src.framework.core.box import Box


class ISingleTrack(ABC):
    __slots__ = ()

    @property
    @abstractmethod
    def box(self) -> "Box":
        """The detection bounding box"""
        ...

    @property
    @abstractmethod
    def conf(self) -> float:
        """Confidence score between 0 and 1"""
        ...


@dataclass(frozen=True, slots=True)
class SingleTrack(ISingleTrack):
    box: Box
    conf: float = 1.0
