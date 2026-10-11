from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.framework.core.box import Box


_UNSET = object()


class IDetection(ABC):
    __slots__ = ()

    @property
    @abstractmethod
    def box(self) -> "Box":
        """The detection bounding box"""
        ...

    @property
    @abstractmethod
    def cls(self) -> int:
        """Class id"""
        ...

    @property
    @abstractmethod
    def conf(self) -> float:
        """Confidence score between 0 and 1"""
        ...


@dataclass(frozen=True, slots=True)
class Detection(IDetection):
    box: Box
    cls: int
    conf: float

    def copy_with(
        self,
        *,
        box: Box | Any = _UNSET,
        cls: int | Any = _UNSET,
        conf: float | Any = _UNSET,
    ) -> "Detection":
        return Detection(
            box=self.box if box is _UNSET else box,
            cls=self.cls if cls is _UNSET else cls,
            conf=self.conf if conf is _UNSET else conf,
        )


class ScoresDetection(IDetection):
    """Implementation which stores full score around all detections"""

    __slots__ = ("_box", "_scores", "_t_ns")

    def __init__(self, box: "Box", scores: np.ndarray) -> None:
        s = np.asarray(scores, dtype=np.float32)
        if s.ndim != 1:
            raise ValueError(f"scores must be 1D (num_classes,), got shape={s.shape}")

        s = s.copy()
        s.setflags(write=False)

        self._box = box
        self._scores = s
        # self._t_ns = t_ns

    @property
    def box(self) -> "Box":
        return self._box

    @property
    def scores(self) -> np.ndarray:
        return self._scores

    @property
    def cls(self) -> int:
        return int(np.argmax(self._scores))

    @property
    def conf(self) -> float:
        # compute from argmax; no extra method
        idx = self.cls
        return float(self._scores[idx])

    # @property
    # def t_ns(self) -> int:
    #     return self._t_ns
