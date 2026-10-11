from typing import Sequence, Optional, List

import numpy as np

from src.framework.core.box import BoxTwoCorners
from src.framework.det.detection import IDetection
from src.framework.mot.multi_object_tracker import IMultiObjectTracker
from src.framework.mot.track import Track


class BoxMotWrapperBase(IMultiObjectTracker):
    def __init__(
        self,
        tracker: "BaseTracker",
    ) -> None:
        self._tracker = tracker
        # self._last_t_ns: int = 0

    def track(self, img: np.ndarray, dets: Sequence[IDetection]) -> List[Track]:
        det_arr = self._to_boxmot_dets(dets)

        outputs = self._tracker.update(det_arr, img)

        return self._from_boxmot_outputs(outputs)

    @staticmethod
    def _to_boxmot_dets(dets: Sequence[IDetection]) -> np.ndarray:
        if not dets:
            return np.empty((0, 6), dtype=np.float32)

        arr = np.empty((len(dets), 6), dtype=np.float32)
        for i, d in enumerate(dets):
            b = d.box.as_xyxy()
            arr[i, 0] = b.x1
            arr[i, 1] = b.y1
            arr[i, 2] = b.x2
            arr[i, 3] = b.y2
            arr[i, 4] = float(d.conf)
            arr[i, 5] = float(d.cls)
        return arr

    @staticmethod
    def _from_boxmot_outputs(outputs: Optional[np.ndarray]) -> List[Track]:
        if outputs is None:
            return []
        if not isinstance(outputs, np.ndarray) or outputs.size == 0:
            return []

        # Defensive: allow both shape (N, 8) and (N, >=7) depending on boxmot version.
        if outputs.ndim != 2 or outputs.shape[1] < 7:
            raise ValueError(
                f"Unexpected ByteTrack output shape: {getattr(outputs, 'shape', None)}"
            )

        tracks: List[Track] = []
        for row in outputs:
            x1 = float(row[0])
            y1 = float(row[1])
            x2 = float(row[2])
            y2 = float(row[3])

            track_id = int(row[4])
            conf = float(row[5])
            cls = int(row[6])

            tracks.append(
                Track(
                    box=BoxTwoCorners(x1, y1, x2, y2),
                    cls=cls,
                    conf=conf,
                    track_id=track_id,
                )
            )

        return tracks
