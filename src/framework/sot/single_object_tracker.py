from abc import abstractmethod, ABC
from typing import Optional

import numpy as np

from src.framework.core.box import Box
from src.framework.sot.single_track import SingleTrack


class ISingleObjectTracker(ABC):
    @abstractmethod
    def initialize(self, frame: np.ndarray, init_box: Box) -> bool: ...

    @abstractmethod
    def track(
        self, frame: np.ndarray, search_area: Optional[Box] = None
    ) -> SingleTrack: ...

    def __call__(
        self, frame: np.ndarray, search_area: Optional[Box] = None
    ) -> SingleTrack:
        return self.track(frame, search_area)
