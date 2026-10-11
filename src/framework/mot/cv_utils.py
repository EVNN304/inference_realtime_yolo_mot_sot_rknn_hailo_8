from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.framework.core.box import Box
from src.framework.mot.track import ITrack


def _class_color_bgr(cls: int) -> Tuple[int, int, int]:
    """
    Deterministic, high-contrast-ish BGR color per class id.
    Uses HSV -> BGR to spread hues by class.
    """
    if cls < 0:
        cls = -cls

    hue = int((cls * 0.61803398875 % 1.0) * 179)  # OpenCV HSV hue range: [0,179]
    sat = 220
    val = 255
    hsv = np.uint8([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _track_color_bgr(track_id: int) -> Tuple[int, int, int]:
    """
    Deterministic, high-contrast-ish BGR color per track id (instance color).
    Uses HSV -> BGR to spread hues by id.
    """
    if track_id < 0:
        track_id = -track_id

    hue = int((track_id * 0.61803398875 % 1.0) * 179)  # OpenCV HSV hue range: [0,179]
    sat = 220
    val = 255
    hsv = np.uint8([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _to_xyxy_int(box: Box) -> Tuple[int, int, int, int]:
    b = box.as_xyxy()
    x1 = int(round(min(b.x1, b.x2)))
    y1 = int(round(min(b.y1, b.y2)))
    x2 = int(round(max(b.x1, b.x2)))
    y2 = int(round(max(b.y1, b.y2)))
    return x1, y1, x2, y2


def draw_box(
    img: np.ndarray, box: Box, color: Tuple[int, int, int] = (0, 255, 0), thickness=2
) -> np.ndarray:
    x1, y1, x2, y2 = _to_xyxy_int(box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)
    return img


def draw_tracks(
    img: np.ndarray,
    tracks: Iterable[ITrack],
    *,
    class_names: Optional[Sequence[str]] = None,
    conf_fmt: str = "{:.2f}",
    thickness: int = 2,
    font_scale: float = 0.5,
    draw_label: bool = True,
    draw_conf: bool = True,
    draw_id: bool = True,
    id_prefix: str = "#",
    alpha: float = 0.0,
) -> np.ndarray:
    """
    Draws track boxes + labels on `img`.

    Track color is derived from `track_id` (instance-stable coloring across frames).

    Args:
        img: HxWxC BGR image (OpenCV).
        tracks: iterable of Track-like objects (box/cls/conf/track_id).
        class_names: optional list mapping class id -> name.
        conf_fmt: formatting for confidence value.
        thickness: rectangle line thickness.
        font_scale: label font scale.
        draw_label: draw class name/id text.
        draw_conf: include confidence in label.
        draw_id: include track id in label.
        id_prefix: prefix before track id (e.g. "#", "id=", "T").
        alpha: if > 0, draws filled translucent rectangles (overlay).

    Returns:
        The image (same object if alpha==0, otherwise a new blended array).
    """
    img = img.copy()
    overlay = None
    if alpha > 0.0:
        overlay = img.copy()

    h, w = img.shape[:2]

    for trk in tracks:
        cls = int(trk.cls)
        conf = float(trk.conf)
        track_id = int(trk.track_id)

        color = _track_color_bgr(track_id)

        x1, y1, x2, y2 = _to_xyxy_int(trk.box)

        # Clip to image bounds (safe for drawing)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        target = overlay if overlay is not None else img

        if overlay is not None:
            cv2.rectangle(target, (x1, y1), (x2, y2), color, thickness=-1)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)

        if not draw_label:
            continue

        if class_names is not None and 0 <= cls < len(class_names):
            cls_text = class_names[cls]
        else:
            cls_text = str(cls)

        parts = []
        if draw_id:
            parts.append(f"{id_prefix}{track_id}")
        parts.append(cls_text)
        if draw_conf:
            parts.append(conf_fmt.format(conf))
        label = " ".join(parts)

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        baseline = max(baseline, 2)

        # Place label above box if possible, otherwise inside
        y_text = y1 - 5
        y_bg_top = y1 - th - baseline - 6
        if y_bg_top < 0:
            y_bg_top = y1
            y_text = y1 + th + baseline + 1

        x_bg_left = x1
        x_bg_right = min(w - 1, x1 + tw + 6)
        y_bg_bottom = min(h - 1, y_bg_top + th + baseline + 6)

        # Background rectangle (solid)
        cv2.rectangle(
            img, (x_bg_left, y_bg_top), (x_bg_right, y_bg_bottom), color, thickness=-1
        )
        cv2.putText(
            img,
            label,
            (x1 + 3, y_text),
            font,
            font_scale,
            (0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    if overlay is not None:
        a = float(alpha)
        a = 0.0 if a < 0.0 else 1.0 if a > 1.0 else a
        return cv2.addWeighted(overlay, a, img, 1.0 - a, 0.0)

    return img
