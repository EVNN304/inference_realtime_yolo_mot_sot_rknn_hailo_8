import logging
import time
from typing import List

from src.gimb.controllers import TrOsControllers
from src.gimb.ctrls.network_handlers.cmd_detector_ctrl_handler import TrOsCmdDetectorCtrlHandler
from src.gimb.ctrls.network_handlers.cmd_tracker_ctrl_handler import (
    TrOsCmdTrackerCtrlHandler,
)
from src.gimb.ctrls.senders.base_sender import TracktorDataBaseSender
from src.gimb.ctrls.senders.detector_sender import TrOsDetectorSender
from src.gimb.ctrls.senders.tracker_sender import TrOsTrackerSender
from src.gimb.state import TrOsState
from src.framework.loop.loop import ILoop
from src.framework.protocols.tracktor.network.dispatcher import Dispatcher
from src.framework.protocols.tracktor.network.services_commands_protocol.scp_node import (
    ServicesCommandsProtocolNode,
)
from src.framework.protocols.tracktor.network.tracktor_data_protocol.tdp_node import (
    TracktorDataProtocolNode,
)


class TrOsNetworkLoop(ILoop):
    def __init__(self, state: TrOsState, controllers: TrOsControllers) -> None:
        super().__init__()
        self._logger = logging.getLogger("net")
        self._state = state
        self._controllers = controllers
        self._scp_node: ServicesCommandsProtocolNode = None
        self._tdp_node: TracktorDataProtocolNode = None
        self._senders: List[TracktorDataBaseSender] = [
            TrOsDetectorSender(state),
            TrOsTrackerSender(state),
        ]

    def on_start(self) -> None:
        scp_handlers = [
            TrOsCmdTrackerCtrlHandler(
                self._controllers.tracking_controller,
            ),
            TrOsCmdDetectorCtrlHandler(
                self._state
            ),
        ]
        self._scp_node = ServicesCommandsProtocolNode(Dispatcher(scp_handlers))
        self._tdp_node = TracktorDataProtocolNode(Dispatcher())

        self._scp_node.open()
        self._tdp_node.open()

        self._logger.info("Network initialized")

    def update(self) -> None:
        self._send_data()
        try:
            self._scp_node.tick()
            self._tdp_node.tick()
        except Exception as err:
            self._logger.exception(err)

    def on_stop(self) -> None:
        pass

    def _send_data(self) -> None:
        for sender in self._senders:
            maybe_packet = sender.try_prepare_pckt()
            if maybe_packet:
                self._tdp_node.send(maybe_packet)
