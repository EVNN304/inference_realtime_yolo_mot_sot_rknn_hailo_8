from dataclasses import dataclass, replace, field
from enum import IntEnum
from typing import Optional, Tuple, Any

from src.gimb.domain.cameras import TrOsChannel
from src.framework.cameras.alignment import AngularBox, AngularPoint
from src.framework.core.box import Box

_UNSET = object()


class TrOsTrackInitMode(IntEnum):
    DETECTION = 0
    CENTER = 1
    SEGMENTATION = 2


@dataclass(slots=True, frozen=True)
class TrOsTrackingDesiredState:
    enabled: bool
    channel: TrOsChannel
    init_mode: TrOsTrackInitMode
    init_data_rad: Tuple[float, float]
    recv_time_ns: int


@dataclass(slots=True, frozen=True)
class TrOsTrackingCommand:
    enabled: bool
    needs_init: bool
    box: Box
    timestamp_ns: Optional[int] = None

    @classmethod
    def disable(cls) -> "TrOsTrackingCommand":
        return TrOsTrackingCommand(enabled=False, needs_init=False, box=None)

    def copy_with(
        self,
        *,
        enabled: bool = _UNSET,
        needs_init: bool = _UNSET,
        box: Box = _UNSET,
        timestamp_ns: Optional[int] = _UNSET,
    ) -> "TrOsTrackingCommand":
        return TrOsTrackingCommand(
            enabled=self.enabled if enabled is _UNSET else enabled,
            needs_init=self.needs_init if needs_init is _UNSET else needs_init,
            box=self.box if box is _UNSET else box,
            timestamp_ns=self.timestamp_ns if timestamp_ns is _UNSET else timestamp_ns,
        )


class TrOsTrackingStatus(IntEnum):
    DISABLED = 0
    ENABLED = 1
    EXTRAPOLATING = 2


@dataclass(slots=True, frozen=True)
class TrOsTrackedTarget:
    status: TrOsTrackingStatus
    angular_box: AngularBox
    timestamp_ns: int = -1
    confidence: float = 1.0

    @classmethod
    def disabled(cls, timeout_ns: int = -1) -> "TrOsTrackedTarget":
        return TrOsTrackedTarget(
            status=TrOsTrackingStatus.DISABLED,
            angular_box=AngularBox(
                tl=AngularPoint(-0.1, 0.1),
                br=AngularPoint(0.1, -0.1),
            ),
            timestamp_ns=timeout_ns,
        )

    def copy_with(
        self,
        *,
        status=_UNSET,
        angular_box=_UNSET,
        timestamp_ns=_UNSET,
        confidence=_UNSET,
    ) -> "TrOsTrackedTarget":
        return replace(
            self,
            status=self.status if status is _UNSET else status,
            angular_box=self.angular_box if angular_box is _UNSET else angular_box,
            timestamp_ns=self.timestamp_ns if timestamp_ns is _UNSET else timestamp_ns,
            confidence=self.confidence if confidence is _UNSET else confidence,
        )
