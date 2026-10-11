from dataclasses import dataclass, field
from math import degrees, radians
from pathlib import Path
from typing import TextIO

from src.framework.cameras.alignment import (
    CameraIntrinsics,
    AlignmentPipeline,
    RollCorrection,
)


@dataclass(slots=True, frozen=True)
class IntrinsicsConfig:
    hor_offset_px: float = 0.0
    ver_offset_px: float = 0.0

    k_hor_offset_path: str = field(default="hor_px", init=False, repr=False)
    k_ver_offset_path: str = field(default="ver_px", init=False, repr=False)

    @classmethod
    def load(cls, values: dict[str, str]) -> "IntrinsicsConfig":
        return cls(
            hor_offset_px=_get_float(values, "hor_px", 0.0),
            ver_offset_px=_get_float(values, "ver_px", 0.0),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            self.k_hor_offset_path: self.hor_offset_px,
            self.k_ver_offset_path: self.ver_offset_px,
        }


@dataclass(slots=True, frozen=True)
class RollConfig:
    roll_rad: float = 0.0

    k_roll_path: str = field(default="roll_deg", init=False, repr=False)

    @classmethod
    def load(cls, values: dict[str, str]) -> "RollConfig":
        return cls(
            roll_rad=radians(_get_float(values, "roll_deg", 0.0)),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            self.k_roll_path: degrees(self.roll_rad),
        }


@dataclass(slots=True)
class FieldOfViewConfig:
    fov_h_rad: float = 0.0
    fov_v_rad: float = 0.0

    k_fov_h_path: str = field(default="fov_h_deg", init=False, repr=False)
    k_fov_v_path: str = field(default="fov_v_deg", init=False, repr=False)

    @classmethod
    def load(cls, values: dict[str, str]) -> "FieldOfViewConfig":
        return cls(
            fov_h_rad=radians(_get_float(values, "fov_h_deg", 0.0)),
            fov_v_rad=radians(_get_float(values, "fov_v_deg", 0.0)),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            self.k_fov_h_path: degrees(self.fov_h_rad),
            self.k_fov_v_path: degrees(self.fov_v_rad),
        }


@dataclass(slots=True)
class AlignmentConfig:
    intrinsics: IntrinsicsConfig = field(default_factory=IntrinsicsConfig)
    roll: RollConfig = field(default_factory=RollConfig)
    fov: FieldOfViewConfig = field(default_factory=FieldOfViewConfig)

    @classmethod
    def load(cls, io: TextIO) -> "AlignmentConfig":
        values = _read_flat_ini(io)

        return cls(
            intrinsics=IntrinsicsConfig.load(values),
            roll=RollConfig.load(values),
            fov=FieldOfViewConfig.load(values)
        )

    @classmethod
    def load_from_file(cls, file_path: str | Path) -> "AlignmentConfig":
        path = Path(file_path)

        if not path.exists():
            return cls()

        with path.open("r", encoding="utf-8") as io:
            res = cls.load(io)
            return res

    @classmethod
    def try_load_from_file(cls, file_path: str | Path) -> "AlignmentConfig | None":
        path = Path(file_path)

        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as io:
            return cls.load(io)

    def dump(self, io: TextIO) -> None:
        values: dict[str, float] = {}
        values.update(self.intrinsics.as_dict())
        values.update(self.roll.as_dict())
        values.update(self.fov.as_dict())
        _write_flat_ini(io, values)

    def dump_to_file(self, file_path: str | Path) -> bool:
        path = Path(file_path)

        try:
            with path.open("w", encoding="utf-8") as io:
                self.dump(io)
            return True
        except OSError:
            return False

    def apply_to_intrinsics(self, intrinsics: CameraIntrinsics) -> CameraIntrinsics:
        return CameraIntrinsics(
            f_x=intrinsics.f_x,
            f_y=intrinsics.f_y,
            c_x=intrinsics.c_x + self.intrinsics.hor_offset_px,
            c_y=intrinsics.c_y + self.intrinsics.ver_offset_px,
        )

    def to_pipeline(self, intrinsics: CameraIntrinsics) -> AlignmentPipeline:
        return AlignmentPipeline(
            intrinsics=self.apply_to_intrinsics(intrinsics),
            roll_correction=RollCorrection(
                roll_angle_rad=self.roll.roll_rad,
            ),
        )


def _get_float(values: dict[str, str], key: str, default: float) -> float:
    raw = values.get(key)

    if raw is None:
        return default

    try:
        return float(raw)
    except ValueError:
        return default


def _read_flat_ini(io: TextIO) -> dict[str, str]:
    result: dict[str, str] = {}

    for raw_line in io:
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#") or line.startswith(";"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()

    return result


def _write_flat_ini(io: TextIO, values: dict[str, float]) -> None:
    ordered_keys = (
        "hor_px",
        "ver_px",
        "roll_deg",
        "fov_h_deg",
        "fov_v_deg",
    )

    written_keys: set[str] = set()

    for key in ordered_keys:
        if key in values:
            io.write(f"{key}={values[key]}\n")
            written_keys.add(key)

    for key, value in values.items():
        if key not in written_keys:
            io.write(f"{key}={value}\n")
