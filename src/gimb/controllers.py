from src.gimb.ctrls.tracking_controller import TrOsTrackingController
from src.gimb.state import TrOsState


class TrOsControllers:
    def __init__(self, state: TrOsState) -> None:
        self.tracking_controller = TrOsTrackingController(state)
