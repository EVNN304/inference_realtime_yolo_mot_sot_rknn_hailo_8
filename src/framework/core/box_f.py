from typing import Tuple

import numpy as np


def get_bbox_center(bbox: np.ndarray) -> Tuple[float, float]:
    return (bbox[0] + bbox[2] / 2), (bbox[1] + bbox[3] / 2)


def get_low_bottom(bbox: np.ndarray) -> Tuple[float, float]:
    return bbox[0] + bbox[2], bbox[1] + bbox[3]
