from dataclasses import dataclass
from enum import Enum

from src.gimb.domain.cameras import TrOsChannel


@dataclass(slots=True, frozen=True)
class TrOsDetectorDesiredState:
    enabled: bool
    channel: TrOsChannel = TrOsChannel.TV


class TrOsDetectorChannel(Enum):
    WIDE = 0
    NARROW = 1
    BOTH = 2
