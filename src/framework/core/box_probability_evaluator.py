from math import exp, log
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.framework.core.box import Box


@dataclass(slots=True)
class BoxMetricConfig:
    # ---- weights ----
    weight_center: float = 4.0
    weight_scale: float = 1.5
    weight_aspect: float = 1.0
    weight_iou: float = 2.5
    weight_size: float = 0.25

    # ---- sigmas (tolerance) ----
    center_sigma: float = 0.75
    scale_sigma: float = 0.50
    aspect_sigma: float = 0.35

    # ---- IoU shaping ----
    iou_gamma: float = 1.0

    # ---- sanity thresholds ----
    min_width: float = 1.0
    min_height: float = 1.0
    min_area: float = 4.0
    max_width: float | None = None
    max_height: float | None = None
    max_area: float | None = None

    # ---- optional gating ----
    max_normalized_center_distance: Optional[float] = 2.5

    # ---- numerical stability ----
    eps: float = 1e-9


@dataclass(slots=True, frozen=True)
class BoxMetricResult:
    center_score: float
    scale_score: float
    aspect_score: float
    iou_score: float
    size_score: float
    total_score: float


class BoxProbabilityEvaluator:
    def __init__(self, config: BoxMetricConfig = BoxMetricConfig()) -> None:
        self.cfg = config

    def evaluate(self, previous_box: Box, new_box: Box) -> float:
        return self.evaluate_detailed(previous_box, new_box).total_score

    def evaluate_detailed(self, previous_box: Box, new_box: Box) -> BoxMetricResult:
        cfg = self.cfg

        prev = previous_box.as_centered()
        new = new_box.as_centered()

        prev_w = max(float(prev.w), cfg.eps)
        prev_h = max(float(prev.h), cfg.eps)
        new_w = max(float(new.w), cfg.eps)
        new_h = max(float(new.h), cfg.eps)

        prev_area = max(previous_box.area(), cfg.eps)
        new_area = max(new_box.area(), cfg.eps)

        center_score = self._center_score(
            prev.cx,
            prev.cy,
            prev_w,
            prev_h,
            new.cx,
            new.cy,
        )

        scale_score = self._scale_score(prev_area, new_area)
        aspect_score = self._aspect_score(prev_w, prev_h, new_w, new_h)

        iou = float(Box.iou(previous_box, new_box))
        iou_score = self._iou_score(iou)

        size_score = self._size_score(new_w, new_h, new_area)

        final_score = self._combine(
            center_score,
            scale_score,
            aspect_score,
            iou_score,
            size_score,
        )

        return BoxMetricResult(
            center_score=center_score,
            scale_score=scale_score,
            aspect_score=aspect_score,
            iou_score=iou_score,
            size_score=size_score,
            total_score=final_score,
        )

    def _center_score(
        self,
        prev_cx: float,
        prev_cy: float,
        prev_w: float,
        prev_h: float,
        new_cx: float,
        new_cy: float,
    ) -> float:
        cfg = self.cfg

        dx = new_cx - prev_cx
        dy = new_cy - prev_cy

        nx = dx / max(prev_w, cfg.eps)
        ny = dy / max(prev_h, cfg.eps)

        dist = float(np.hypot(nx, ny))

        if (
            cfg.max_normalized_center_distance is not None
            and dist > cfg.max_normalized_center_distance
        ):
            return 0.0

        return exp(-(dist * dist) / (2.0 * cfg.center_sigma * cfg.center_sigma))

    def _scale_score(self, prev_area: float, new_area: float) -> float:
        cfg = self.cfg

        ratio = new_area / max(prev_area, cfg.eps)
        delta = abs(log(max(ratio, cfg.eps)))

        return exp(-(delta * delta) / (2.0 * cfg.scale_sigma * cfg.scale_sigma))

    def _aspect_score(
        self,
        prev_w: float,
        prev_h: float,
        new_w: float,
        new_h: float,
    ) -> float:
        cfg = self.cfg

        prev_ar = prev_w / max(prev_h, cfg.eps)
        new_ar = new_w / max(new_h, cfg.eps)

        delta = abs(log(max(new_ar / max(prev_ar, cfg.eps), cfg.eps)))

        return exp(-(delta * delta) / (2.0 * cfg.aspect_sigma * cfg.aspect_sigma))

    def _iou_score(self, iou: float) -> float:
        cfg = self.cfg
        iou = float(np.clip(iou, 0.0, 1.0))
        return iou**cfg.iou_gamma

    def _size_score(self, w: float, h: float, area: float) -> float:
        cfg = self.cfg

        if w < cfg.min_width or h < cfg.min_height or area < cfg.min_area:
            return 0.0

        if (
            (cfg.max_width is not None and w > cfg.max_width)
            or (cfg.max_height is not None and h > cfg.max_height)
            or (cfg.max_area is not None and area > cfg.max_area)
        ):
            return 0.0

        return 1.0

    def _combine(
        self,
        center: float,
        scale: float,
        aspect: float,
        iou: float,
        size: float,
    ) -> float:
        cfg = self.cfg

        weights = np.asarray(
            [
                cfg.weight_center,
                cfg.weight_scale,
                cfg.weight_aspect,
                cfg.weight_iou,
                cfg.weight_size,
            ],
            dtype=np.float64,
        )

        scores = np.asarray(
            [
                center,
                scale,
                aspect,
                iou,
                size,
            ],
            dtype=np.float64,
        )

        wsum = float(weights.sum())
        if wsum <= 0.0:
            return 0.0

        return float(np.clip(np.dot(weights, scores) / wsum, 0.0, 1.0))
