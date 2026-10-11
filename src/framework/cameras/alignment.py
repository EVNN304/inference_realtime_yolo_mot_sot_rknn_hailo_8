import math
from dataclasses import dataclass
from math import atan2, tan, sqrt, cos, sin, atan
from typing import TypeVar, Generic, NamedTuple

from src.framework.cameras.camera import Camera
from src.framework.core.box import Box, BoxTwoCorners

T = TypeVar("T", float, int)


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    @classmethod
    def distance(cls, p1: "Point", p2: "Point") -> float:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


@dataclass(frozen=True, slots=True)
class NormalizedPoint:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class AngularPoint:
    """Angular point in radians"""

    x: float
    y: float

    @property
    def azimuth(self) -> float:
        return self.x

    @property
    def elevation(self) -> float:
        return self.y

    @classmethod
    def distance(cls, p1: "AngularPoint", p2: "AngularPoint") -> float:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


@dataclass(frozen=True, slots=True)
class AngularSize:
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class AngularBox:
    tl: AngularPoint
    br: AngularPoint

    @classmethod
    def from_size_and_center(
        cls, center: AngularPoint, size: AngularSize
    ) -> "AngularBox":
        half_w = size.width / 2.0
        half_h = size.height / 2.0

        tl = AngularPoint(
            x=center.azimuth - half_w,
            y=center.elevation + half_h,
        )

        br = AngularPoint(
            x=center.azimuth + half_w,
            y=center.elevation - half_h,
        )

        return cls(tl=tl, br=br)

    @property
    def area(self) -> float:
        w = max(0.0, self.br.x - self.tl.x)
        h = max(0.0, self.tl.y - self.br.y)  # note: inverted Y
        return w * h

    @classmethod
    def iou(cls, a: "AngularBox", b: "AngularBox") -> float:
        # Intersection box
        inter_left = max(a.tl.x, b.tl.x)
        inter_right = min(a.br.x, b.br.x)

        inter_top = min(a.tl.y, b.tl.y)  # smaller top (since higher is larger y)
        inter_bottom = max(a.br.y, b.br.y)  # larger bottom

        inter_w = max(0.0, inter_right - inter_left)
        inter_h = max(0.0, inter_top - inter_bottom)

        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            return 0.0

        a_area = a.area
        b_area = b.area

        union = a_area + b_area - inter_area
        if union <= 0.0:
            return 0.0

        return inter_area / union

    @property
    def center(self) -> AngularPoint:
        return AngularPoint(
            x=(self.tl.x + self.br.x) * 0.5,
            y=(self.tl.y + self.br.y) * 0.5,
        )

    @property
    def azimuth(self) -> float:
        """Azimuth of the box center"""
        return (self.tl.x + self.br.x) * 0.5

    @property
    def elevation(self) -> float:
        """Elevation of the box center"""
        return (self.tl.y + self.br.y) * 0.5

    @property
    def width(self) -> float:
        return abs(self.br.x - self.tl.x)

    @property
    def height(self) -> float:
        return abs(self.tl.y - self.br.y)

    def fits_inside_fov(
        self,
        fov_hor_rad: float,
        fov_ver_rad: float,
        margin_rad: float,
    ) -> bool:
        half_hor = fov_hor_rad * 0.5
        half_ver = fov_ver_rad * 0.5

        return (
            self.tl.x >= -half_hor + margin_rad
            and self.br.x <= half_hor - margin_rad
            and self.tl.y <= half_ver - margin_rad
            and self.br.y >= -half_ver + margin_rad
        )

    def max_fill_ratio(
        self,
        fov_hor_rad: float,
        fov_ver_rad: float,
    ) -> float:
        hor_ratio = self.width / fov_hor_rad
        ver_ratio = self.height / fov_ver_rad
        return max(hor_ratio, ver_ratio)


class NormalizedToAngular:
    @staticmethod
    def apply(norm: NormalizedPoint) -> AngularPoint:
        azimuth = atan(norm.x)
        elevation = -atan2(norm.y, sqrt((norm.x * norm.x) + 1.0))
        return AngularPoint(x=azimuth, y=elevation)

    @staticmethod
    def inverse(angles: AngularPoint) -> NormalizedPoint:
        x_norm = tan(angles.x)
        y_norm = tan(-angles.y) * sqrt((x_norm * x_norm) + 1.0)
        return NormalizedPoint(x=x_norm, y=y_norm)


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    # Focal length in pixels along x-axis
    f_x: float
    # Focal length in pixels along y-axis
    f_y: float
    # Principal point x-coordinate
    c_x: float
    # Principal point y-coordinate
    c_y: float

    @classmethod
    def from_camera(cls, camera: Camera) -> "CameraIntrinsics":
        pixel_size_mm = 1e-3 * camera.sensor.pixel_size_um
        focal_length_mm = camera.focal_length_mm
        pixel_focal_length = focal_length_mm / pixel_size_mm
        return CameraIntrinsics(
            f_x=pixel_focal_length,
            f_y=pixel_focal_length,
            # Assuming that center of the camera equals center of the frame
            c_x=0.5 * camera.sensor.resolution.width,
            c_y=0.5 * camera.sensor.resolution.height,
        )

    def apply(self, pixel_coords: Point) -> NormalizedPoint:
        """Convert pixel coordinates to normalized coordinates."""
        return NormalizedPoint(
            x=(pixel_coords.x - self.c_x) / self.f_x,
            y=(pixel_coords.y - self.c_y) / self.f_y,
        )

    def inverse(self, norm: NormalizedPoint) -> Point:
        """Convert normalized coordinates back to pixel coordinates."""
        return Point(x=(self.f_x * norm.x) + self.c_x, y=(self.f_y * norm.y) + self.c_y)

    def to_angle(self, pixel_coords: Point) -> AngularPoint:
        """Convert pixel coordinates directly to angular coordinates."""
        norm = self.apply(pixel_coords)
        return NormalizedToAngular.apply(norm)

    def to_pixel(self, angles: AngularPoint) -> Point:
        """Convert angular coordinates back to pixel coordinates."""
        norm = NormalizedToAngular.inverse(angles)
        return self.inverse(norm)


@dataclass(frozen=True, slots=True)
class RollCorrection:
    """
    Corrects roll misalignment around the optical axis.

    Positive roll is clockwise in the image.
    """

    roll_angle_rad: float = 0.0

    def apply(self, norm: NormalizedPoint) -> NormalizedPoint:
        """
        Apply roll correction to normalized camera coordinates.

        This rotates the point in the normalized image plane.
        """
        c = cos(self.roll_angle_rad)
        s = sin(self.roll_angle_rad)

        return NormalizedPoint(
            x=(norm.x * c) - (norm.y * s),
            y=(norm.x * s) + (norm.y * c),
        )

    def inverse(self, norm: NormalizedPoint) -> NormalizedPoint:
        """
        Undo roll correction from normalized camera coordinates.
        """
        c = cos(self.roll_angle_rad)
        s = sin(self.roll_angle_rad)

        return NormalizedPoint(
            x=(norm.x * c) + (norm.y * s),
            y=(-norm.x * s) + (norm.y * c),
        )


@dataclass(slots=True)
class AlignmentPipeline:
    intrinsics: CameraIntrinsics
    roll_correction: RollCorrection

    def pixel_to_angular(self, pixel_coords: Point) -> AngularPoint:
        norm = self.intrinsics.apply(pixel_coords)
        corrected_norm = self.roll_correction.apply(norm)
        return NormalizedToAngular.apply(corrected_norm)

    def angular_to_pixel(self, angles: AngularPoint) -> Point:
        norm = NormalizedToAngular.inverse(angles)
        uncorrected_norm = self.roll_correction.inverse(norm)
        return self.intrinsics.inverse(uncorrected_norm)

    def px2ang_box(self, box: Box) -> AngularBox:
        xyxy_px = box.as_xyxy()
        tl_ang = self.pixel_to_angular(Point(xyxy_px.x1, xyxy_px.y1))
        br_ang = self.pixel_to_angular(Point(xyxy_px.x2, xyxy_px.y2))
        return AngularBox(tl=tl_ang, br=br_ang)

    def ang2px_box(self, box: AngularBox) -> Box:
        tl_px = self.angular_to_pixel(box.tl)
        br_px = self.angular_to_pixel(box.br)
        return BoxTwoCorners(
            min(tl_px.x, br_px.x),
            min(tl_px.y, br_px.y),
            max(tl_px.x, br_px.x),
            max(tl_px.y, br_px.y),
        )
