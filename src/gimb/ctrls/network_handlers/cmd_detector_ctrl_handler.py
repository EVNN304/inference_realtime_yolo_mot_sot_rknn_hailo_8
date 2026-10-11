import logging

from src.gimb.domain.cameras import TrOsChannel
from src.gimb.domain.detector import TrOsDetectorDesiredState
from src.gimb.state import TrOsState
from src.framework.protocols.tracktor.network.handlers import Handler, TMessage
from src.framework.protocols.tracktor.protocols.services_commands_protocol.protocol import (
    DetectorCtrlCmdPacket,
)


class TrOsCmdDetectorCtrlHandler(Handler[DetectorCtrlCmdPacket]):
    def __init__(self, state: TrOsState) -> None:
        super().__init__(DetectorCtrlCmdPacket)
        self._state = state
        self._logger = logging.getLogger("state")

    def handle(self, msg: DetectorCtrlCmdPacket) -> None:
         new_state = self._pckt_to_desired_state(msg)
         if self._state.detector_desired_state.channel != new_state.channel:
             self._logger.info("Changed detector channel to '%s'", new_state.channel)
         self._state.detector_desired_state = new_state

    @staticmethod
    def _pckt_to_desired_state(pckt: DetectorCtrlCmdPacket) -> TrOsDetectorDesiredState:
        flags = pckt.payload.commands.flags
        return TrOsDetectorDesiredState(
            enabled=bool(flags.enable),
            channel=TrOsChannel(flags.channel),
        )
