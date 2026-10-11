from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union, Literal

import numpy as np

DTypeLike = Union[np.dtype, type, str]


class Box(ABC):
    __slots__ = ()

    @abstractmethod
    def as_top_left(self) -> "BoxTopLeft":
        ...

    @abstractmethod
    def as_centered(self) -> "BoxCentered":
        ...

    @abstractmethod
    def as_xyxy(self) -> "BoxTwoCorners":
        ...

    @abstractmethod
    def as_tuple(self) -> Tuple[float, float, float, float]:
        ...

    @abstractmethod
    def area(self) -> float:
        ...

    @property
    @abstractmethod
    def width(self) -> float:
        ...

    @property
    @abstractmethod
    def height(self) -> float:
        ...

    @abstractmethod
    def scaled(
        self,
        scale_x: float,
        scale_y: float | None = None,
    ) -> "Box":
        """
        Scale box size around its center.

        scale_x:
            Horizontal scale factor.
        scale_y:
            Vertical scale factor. If None, scale_x is used for both axes.
        """
        ...

    @abstractmethod
    def interpolate(
        self,
        target: "Box",
        factor: float,
    ) -> "Box":
        """
        Interpolate this box toward target.

        factor=0.0 -> return self
        factor=1.0 -> return target converted to the same representation
        """
        ...

    def as_numpy(self, *, dtype: DTypeLike = np.float64) -> np.ndarray:
        arr = np.asarray(self.as_tuple(), dtype=dtype)
        return arr

    def convert(self, kind: Literal["top_left", "centered", "two_corners"]) -> "Box":
        if kind == "top_left":
            return self.as_top_left()
        if kind == "centered":
            return self.as_centered()
        if kind == "two_corners":
            return self.as_xyxy()
        raise ValueError(kind)

    @classmethod
    def iou(cls, box1: "Box", box2: "Box") -> float:
        a = box1.as_xyxy().normalized()
        b = box2.as_xyxy().normalized()

        inter_x1 = max(a.x1, b.x1)
        inter_y1 = max(a.y1, b.y1)
        inter_x2 = min(a.x2, b.x2)
        inter_y2 = min(a.y2, b.y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
        area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)

        union_area = area_a + area_b - inter_area

        if union_area <= 0.0:
            return 0.0

        return inter_area / union_area

    def center_distance_to(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Distance from box center to point (x, y)
        """

        c = self.as_centered()

        dx = c.cx - x
        dy = c.cy - y

        return float(np.hypot(dx, dy))


@dataclass(frozen=True, slots=True)
class BoxTopLeft(Box):
    x: float
    y: float
    w: float
    h: float

    def as_top_left(self) -> "BoxTopLeft":
        return self

    def as_centered(self) -> "BoxCentered":
        return BoxCentered(
            cx=self.x + self.w * 0.5,
            cy=self.y + self.h * 0.5,
            w=self.w,
            h=self.h,
        )

    def as_xyxy(self) -> "BoxTwoCorners":
        return BoxTwoCorners(
            x1=self.x,
            y1=self.y,
            x2=self.x + self.w,
            y2=self.y + self.h,
        )

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.x, self.y, self.w, self.h

    def area(self) -> float:
        return self.w * self.h

    @property
    def width(self) -> float:
        return self.w

    @property
    def height(self) -> float:
        return self.h

    def scaled(
        self,
        scale_x: float,
        scale_y: float | None = None,
    ) -> "BoxTopLeft":
        return self.as_centered().scaled(scale_x, scale_y).as_top_left()

    def interpolate(
        self,
        target: "Box",
        factor: float,
    ) -> "BoxTopLeft":
        if not 0.0 <= factor <= 1.0:
            raise ValueError(f"factor must be in [0, 1], got {factor}")

        other = target.as_top_left()

        return BoxTopLeft(
            x=self.x + (other.x - self.x) * factor,
            y=self.y + (other.y - self.y) * factor,
            w=self.w + (other.w - self.w) * factor,
            h=self.h + (other.h - self.h) * factor,
        )


@dataclass(frozen=True, slots=True)
class BoxCentered(Box):
    cx: float
    cy: float
    w: float
    h: float

    def as_top_left(self) -> BoxTopLeft:
        return BoxTopLeft(
            x=self.cx - self.w * 0.5,
            y=self.cy - self.h * 0.5,
            w=self.w,
            h=self.h,
        )

    def as_centered(self) -> "BoxCentered":
        return self

    def as_xyxy(self) -> "BoxTwoCorners":
        return self.as_top_left().as_xyxy()

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.cx, self.cy, self.w, self.h

    def area(self) -> float:
        return self.w * self.h

    @property
    def width(self) -> float:
        return self.w

    @property
    def height(self) -> float:
        return self.h

    def scaled(
        self,
        scale_x: float,
        scale_y: float | None = None,
    ) -> "BoxCentered":
        if scale_y is None:
            scale_y = scale_x

        return BoxCentered(
            cx=self.cx,
            cy=self.cy,
            w=self.w * scale_x,
            h=self.h * scale_y,
        )

    def interpolate(
        self,
        target: "Box",
        factor: float,
    ) -> "BoxCentered":
        if not 0.0 <= factor <= 1.0:
            raise ValueError(f"factor must be in [0, 1], got {factor}")

        other = target.as_centered()

        return BoxCentered(
            cx=self.cx + (other.cx - self.cx) * factor,
            cy=self.cy + (other.cy - self.cy) * factor,
            w=self.w + (other.w - self.w) * factor,
            h=self.h + (other.h - self.h) * factor,
        )


@dataclass(frozen=True, slots=True)
class BoxTwoCorners(Box):
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized(self) -> "BoxTwoCorners":
        nx1 = min(self.x1, self.x2)
        nx2 = max(self.x1, self.x2)
        ny1 = min(self.y1, self.y2)
        ny2 = max(self.y1, self.y2)
        return BoxTwoCorners(nx1, ny1, nx2, ny2)

    def as_top_left(self) -> BoxTopLeft:
        b = self.normalized()
        return BoxTopLeft(
            x=b.x1,
            y=b.y1,
            w=b.x2 - b.x1,
            h=b.y2 - b.y1,
        )

    def as_centered(self) -> "BoxCentered":
        return self.as_top_left().as_centered()

    def as_xyxy(self) -> "BoxTwoCorners":
        return self

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def width(self) -> float:
        return abs(self.x2 - self.x1)

    def height(self) -> float:
        return abs(self.y2 - self.y1)

    def scaled(
        self,
        scale_x: float,
        scale_y: float | None = None,
    ) -> "BoxTwoCorners":
        return self.as_centered().scaled(scale_x, scale_y).as_xyxy()

    def interpolate(
        self,
        target: "Box",
        factor: float,
    ) -> "BoxTwoCorners":
        if not 0.0 <= factor <= 1.0:
            raise ValueError(f"factor must be in [0, 1], got {factor}")

        a = self.normalized()
        b = target.as_xyxy().normalized()

        return BoxTwoCorners(
            x1=a.x1 + (b.x1 - a.x1) * factor,
            y1=a.y1 + (b.y1 - a.y1) * factor,
            x2=a.x2 + (b.x2 - a.x2) * factor,
            y2=a.y2 + (b.y2 - a.y2) * factor,
        )
