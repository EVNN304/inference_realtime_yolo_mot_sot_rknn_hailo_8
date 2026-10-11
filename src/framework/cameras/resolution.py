from typing import NamedTuple

class Resolution(NamedTuple):
    width: int
    height: int

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height
