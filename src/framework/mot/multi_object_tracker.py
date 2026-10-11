from abc import abstractmethod, ABC
from typing import Sequence

import numpy as np

from src.framework.det.detection import IDetection
from src.framework.mot.track import Track


class IMultiObjectTracker(ABC):
    @abstractmethod
    def track(self, img: np.ndarray, dets: Sequence[IDetection]) -> Sequence[Track]:
        ...

    def __call__(self, img: np.ndarray, dets: Sequence[IDetection]) -> Sequence[Track]:
        return self.track(img, dets)
