import ctypes
from ctypes import Union, c_int16, c_int32, c_uint8, c_uint16
from enum import IntEnum
from typing import Dict, Optional, Type

from ..base import BaseStructure
from .base import (
    SERVICES_COMMANDS_ID_IDX,
    SERVICES_COMMANDS_ID_SIZE,
    ServicesCommandsHeader,
    ServicesCommandsPacket,
)


# Command IDs
class CommandIDs(IntEnum):
    DIST_MNG_SRV_CMD = 0xCD2A
    BAL_COR_MNG_SRV_CMD = 0x5692
    BCM_MAN_SETS_CMD = 0xA73E
    WORKMODE_MNG_SRV_CMD_C = 0xD27D
    WORKMODE_MNG_SRV_CMD_G = 0x8B36
    WORKMODE_MNG_SRV_CMD_D = 0x1BC4
    LRF_CTRL_CMD = 0x6A41
    OS_CAP_CTRL_CMD = 0xBFE1
    SECTOR_SEARCH_CTRL_CMD = 0x5F35
    TOPOREF_CMD = 0x51F9
    TARGET_SECTOR_CMD = 0x9C8B
    TRACKER_CTRL_CMD = 0x7F1A
    OS_SET_CTRL_CMD = 0xB733
    DETECTOR_CTRL_CMD = 0xD82A


# --- Distance Manager Service Command ---
class DistMngSrvCmdFlagsBits(BaseStructure):
    """Distance Manager Command flags bits"""

    _pack_ = 1
    _fields_ = [
        (
            "source",
            c_uint8,
            3,
        ),  # Distance source (0 - Manual, 1 - LRF, 2 - Auto, 3 - Target Designation)
        ("reserved", c_uint8, 5),
    ]


class DistMngSrvCmdFlags(Union):
    """Distance Manager Command flags union"""

    _pack_ = 1
    _fields_ = [
        ("flags", DistMngSrvCmdFlagsBits),
        ("bits", c_uint8),
    ]


class DistMngSrvCmdPayload(BaseStructure):
    """Distance Manager Service Command payload"""

    _pack_ = 1
    _fields_ = [
        ("flags", DistMngSrvCmdFlags),
        ("manual_distance", c_uint16),
    ]


class DistMngSrvCmdPacket(ServicesCommandsPacket):
    """Distance Manager Service Command packet"""

    _pack_ = 1
    _fields_ = [
        ("payload", DistMngSrvCmdPayload),
    ]
    CMD_ID = CommandIDs.DIST_MNG_SRV_CMD
    _payload_class = DistMngSrvCmdPayload


# --- Ballistics Corrections Manager Service Command ---
class BalCorMngSrvCmdPayload(BaseStructure):
    """Ballistics Corrections Manager Service Command payload"""

    _pack_ = 1
    _fields_ = [("correction_azimuth", c_int16), ("correction_elevation", c_int16)]


class BalCorMngSrvCmdPacket(ServicesCommandsPacket):
    """Ballistics Corrections Manager Service Command packet"""

    _pack_ = 1
    _fields_ = [
        ("payload", BalCorMngSrvCmdPayload),
    ]
    CMD_ID = CommandIDs.BAL_COR_MNG_SRV_CMD
    _payload_class = BalCorMngSrvCmdPayload


# --- Ballistics Corrections Manager Manual Settings Command ---
class BCMManualSetsCmdBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("manual_temperature", c_uint8, 1),
        ("manual_gun_shoots", c_uint8, 1),
        ("reserved", c_uint8, 6),
    ]


class BCMManualSetsCmdFlags(Union):
    _pack_ = 1
    _fields_ = [("flags", BCMManualSetsCmdBits), ("bits", c_uint8)]


class BCMManualSetsCmdPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("cmd", BCMManualSetsCmdFlags),
        ("air_density_deviation", c_int16),
        ("ammo_temperature", c_int16),
        ("guns_shoot_counter", c_uint16 * 4),
    ]


class BCMManualSetsCmdPacket(ServicesCommandsPacket):
    _pack_ = 1
    _fields_ = [("payload", BCMManualSetsCmdPayload)]
    CMD_ID = CommandIDs.BCM_MAN_SETS_CMD
    _payload_class = BCMManualSetsCmdPayload


# --- Workmode Manager Service Command from Commander ---
class WorkmodeMngSrvCmdCFlagsBits(BaseStructure):
    """Workmode Manager Command from Commander flags bits"""

    _pack_ = 1
    _fields_ = [
        ("ground_mode", c_uint8, 1),
        ("mirror_mode", c_uint8, 1),
        ("angles_measurement_mode", c_uint8, 1),
        ("reserved", c_uint8, 5),
    ]


class WorkmodeMngSrvCmdCFlags(Union):
    """Workmode Manager Command from Commander flags union"""

    _pack_ = 1
    _fields_ = [
        ("flags", WorkmodeMngSrvCmdCFlagsBits),
        ("bits", c_uint8),
    ]


class WorkmodeMngSrvCmdCPayload(BaseStructure):
    """Workmode Manager Service Command from Commander payload"""

    _pack_ = 1
    _fields_ = [("flags", WorkmodeMngSrvCmdCFlags), ("burst_mode", c_uint8)]


class WorkmodeMngSrvCmdCPacket(ServicesCommandsPacket):
    """Workmode Manager Service Command from Commander packet"""

    _pack_ = 1
    _fields_ = [
        ("payload", WorkmodeMngSrvCmdCPayload),
    ]
    CMD_ID = CommandIDs.WORKMODE_MNG_SRV_CMD_C
    _payload_class = WorkmodeMngSrvCmdCPayload


# --- Workmode Manager Service Command from Gunner ---
class WorkmodeMngSrvCmdGFlagsBits(BaseStructure):
    """Workmode Manager Command from Gunner flags bits"""

    _pack_ = 1
    _fields_ = [
        ("tracking_init_mode", c_uint8, 1),
        ("drives_mode", c_uint8, 1),
        ("ballistics_aiming", c_uint8, 1),  # SRP
        ("treadle_fire", c_uint8, 1),
        ("reserved", c_uint8, 4),
    ]


class WorkmodeMngSrvCmdGFlags(Union):
    """Workmode Manager Command from Gunner flags union"""

    _pack_ = 1
    _fields_ = [
        ("flags", WorkmodeMngSrvCmdGFlagsBits),
        ("bits", c_uint8),
    ]


class WorkmodeMngSrvCmdGPayload(BaseStructure):
    """Workmode Manager Service Command from Gunner payload"""

    _pack_ = 1
    _fields_ = [("flags", WorkmodeMngSrvCmdGFlags)]


class WorkmodeMngSrvCmdGPacket(ServicesCommandsPacket):
    """Workmode Manager Service Command from Gunner packet"""

    _pack_ = 1
    _fields_ = [
        ("payload", WorkmodeMngSrvCmdGPayload),
    ]
    CMD_ID = CommandIDs.WORKMODE_MNG_SRV_CMD_G
    _payload_class = WorkmodeMngSrvCmdGPayload


class WorkmodeMngSrvCmdDPayload(BaseStructure):
    """Workmode Manager Service Command from Distance operator payload"""

    _pack_ = 1
    _fields_ = [("target_angle", c_uint8)]


class WorkmodeMngSrvCmdDPacket(ServicesCommandsPacket):
    """Workmode Manager Service Command from Distance operator packet"""

    _pack_ = 1
    _fields_ = [("payload", WorkmodeMngSrvCmdDPayload)]
    CMD_ID = CommandIDs.WORKMODE_MNG_SRV_CMD_D
    _payload_class = WorkmodeMngSrvCmdDPayload


# --- Laser Rangefinder Control ---
class LrfCtrlCommandsBits(BaseStructure):
    """LRF Commands bits"""

    _pack_ = 1
    _fields_ = [
        (
            "lrf_measure",
            c_uint16,
            1,
        ),  # Initiate laser rangefinder measurement (0 - No, 1 - Measure)
        ("single", c_uint16, 1),  # Single measurement mode
        ("pointer_mode", c_uint16, 1),  # Pointer mode
        ("continious_freq", c_uint16, 5),  # Continuous measurement mode frequency code
        ("reserved", c_uint16, 8),
    ]


class LrfCtrlCommands(Union):
    """LRF Commands union"""

    _pack_ = 1
    _fields_ = [
        ("flags", LrfCtrlCommandsBits),
        ("bits", c_uint16),
    ]


class LrfCtrlCmdPayload(BaseStructure):
    """LRF Control Command payload"""

    _pack_ = 1
    _fields_ = [
        ("commands", LrfCtrlCommands),
    ]


class LrfCtrlCmdPacket(ServicesCommandsPacket):
    """LRF Control Command packet"""

    _pack_ = 1
    _fields_ = [
        ("payload", LrfCtrlCmdPayload),
    ]
    CMD_ID = CommandIDs.LRF_CTRL_CMD
    _payload_class = LrfCtrlCmdPayload


# --- Optical Station Cap Control ---
class OsCapCtrlCommandsBits(BaseStructure):
    """Optical Station Cap Command bits"""

    _pack_ = 1
    _fields_ = [
        ("open", c_uint8, 1),
        ("reserved", c_uint8, 7),
    ]


class OsCapCtrlCommands(Union):
    """Optical Station Cap Commands union"""

    _pack_ = 1
    _fields_ = [
        ("flags", OsCapCtrlCommandsBits),
        ("bits", c_uint8),
    ]


class OsCapCtrlCmdPayload(BaseStructure):
    """Optical Station Cap Control Command payload"""

    _pack_ = 1
    _fields_ = [
        ("commands", OsCapCtrlCommands),
    ]


class OsCapCtrlCmdPacket(ServicesCommandsPacket):
    """Optical Station Cap Control Command packet"""

    _pack_ = 1
    _fields_ = [
        ("payload", OsCapCtrlCmdPayload),
    ]
    CMD_ID = CommandIDs.OS_CAP_CTRL_CMD
    _payload_class = OsCapCtrlCmdPayload


class SectorSearchCmdCommandsBits(BaseStructure):
    """Sector Search control command bits"""

    _pack_ = 1
    _fields_ = [
        ("status", c_uint8, 2),
        ("is_circle", c_uint8, 1),
        ("channel_type", c_uint8, 1),
        ("channel_size", c_uint8, 1),
        ("reserved", c_uint8, 3),
    ]


class SectorSearchCmdCommands(Union):
    """Sector Search control commands union"""

    _pack_ = 1
    _fields_ = [
        ("flags", SectorSearchCmdCommandsBits),
        ("bits", c_uint8),
    ]


class SectorSearchCmdPayload(BaseStructure):
    """Sector Search Control payload"""

    _pack_ = 1
    _fields_ = [
        ("commands", SectorSearchCmdCommands),
        ("search_bisector", c_uint16),
        ("search_sector_width", c_uint16),
        ("elevation_begin", c_uint16),
        ("elevation_end", c_uint16),
    ]


class SectorSearchCmdPacket(ServicesCommandsPacket):
    """Sector Search control command"""

    _pack_ = 1
    _fields_ = [
        ("payload", SectorSearchCmdPayload),
    ]
    CMD_ID = CommandIDs.SECTOR_SEARCH_CTRL_CMD
    _payload_class = SectorSearchCmdPayload


class ToporefStatus(IntEnum):
    GNSS = 0
    INS = 1
    MANUAL_SET = 2


class ToporefCmdCommandsBits(BaseStructure):
    """Topological Reference command bits"""

    _pack_ = 1
    _fields_ = [
        ("status", c_uint8, 2),
        ("reserved", c_uint8, 6),
    ]


class ToporefCmdCommands(Union):
    """Topological reference command commands union"""

    _pack_ = 1
    _fields_ = [
        ("flags", ToporefCmdCommandsBits),
        ("bits", c_uint8),
    ]


class ToporefCmdPayload(BaseStructure):
    """Topological reference payload"""

    _pack_ = 1
    _fields_ = [
        ("commands", ToporefCmdCommands),
        ("longitude", c_int32),
        ("latitude", c_int32),
        ("altitude", c_int16),
        ("heading", c_uint16),
    ]


class ToporefCmdPacket(ServicesCommandsPacket):
    """Sector Search control command"""

    _pack_ = 1
    _fields_ = [
        ("payload", ToporefCmdPayload),
    ]
    CMD_ID = CommandIDs.TOPOREF_CMD
    _payload_class = ToporefCmdPayload


class TargetSectorCmdPayload(BaseStructure):
    """Target sector command payload"""

    _pack_ = 1
    _fields_ = [("bisector", c_uint16), ("sector_width", c_uint16)]


class TargetSectorCmdPacket(ServicesCommandsPacket):
    """Target sector command packet"""

    _pack_ = 1
    _fields_ = [("payload", TargetSectorCmdPayload)]
    CMD_ID = CommandIDs.TARGET_SECTOR_CMD
    _payload_class = TargetSectorCmdPayload


# --- Tracker Control Command ---
class TrackerCtrlCommandsBits(BaseStructure):
    """Tracker Control Command bits"""

    _pack_ = 1
    _fields_ = [
        ("enable", c_uint8, 1),
        ("init_mode", c_uint8, 3),
        ("channel", c_uint8, 1),
        ("reserved", c_uint8, 3),
    ]


class TrackerCtrlCommands(Union):
    """Tracker Control Command union"""

    _pack_ = 1
    _fields_ = [
        ("flags", TrackerCtrlCommandsBits),
        ("bits", c_uint8),
    ]


class DetNearModeSettings(BaseStructure):
    """Detector Near Mode Settings"""

    _pack_ = 1
    _fields_ = [
        ("reserved01", c_int16),
        ("reserved02", c_int16),
    ]


class CenterModeSettings(BaseStructure):
    """Center Mode Settings"""

    _pack_ = 1
    _fields_ = [
        ("hor_size", c_uint16),
        ("ver_size", c_uint16),
    ]


class SegmentationModeSettings(BaseStructure):
    """Segmentation Mode Settings"""

    _pack_ = 1
    _fields_ = [
        ("hor_pos", c_int16),
        ("ver_pos", c_int16),
    ]


class TrackerCtrlCmdModeSettings(Union):
    """Tracker Control Command Mode Settings union"""

    _pack_ = 1
    _fields_ = [
        ("detector_near_mode", DetNearModeSettings),
        ("center_mode", CenterModeSettings),
        ("segmentation_mode", SegmentationModeSettings),
    ]


class TrackerCtrlCmdPayload(BaseStructure):
    """Tracker Control Command payload"""

    _pack_ = 1
    _fields_ = [
        ("commands", TrackerCtrlCommands),
        ("mode_settings", TrackerCtrlCmdModeSettings),
    ]


class TrackerCtrlCmdPacket(ServicesCommandsPacket):
    """Tracker Control Command packet"""

    _pack_ = 1
    _fields_ = [
        ("payload", TrackerCtrlCmdPayload),
    ]
    CMD_ID = CommandIDs.TRACKER_CTRL_CMD
    _payload_class = TrackerCtrlCmdPayload


# ---  ---
class OSSettingsBits(BaseStructure):
    _pack_ = 1
    _fields_ = [("fan", c_uint8, 1), ("heating", c_uint8, 1), ("reserved", c_uint8, 6)]


class OSSettingsFlags(Union):
    _pack_ = 1
    _fields_ = [("flags", OSSettingsBits), ("bits", c_uint8)]


class IRSettingsBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("shutter", c_uint16, 2),
        ("sharp_intensity_mode", c_uint16, 2),
        ("lut", c_uint16, 4),
        ("calibration", c_uint16, 1),
        ("reserved", c_uint16, 7),
    ]


class IRSettingsFlags(Union):
    _pack_ = 1
    _fields_ = [("flags", IRSettingsBits), ("bits", c_uint16)]


class IRSettingsCmd(BaseStructure):
    _pack_ = 1
    _fields_ = [("cmd", IRSettingsFlags)]


class TVSettingsBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("wb_mode", c_uint8, 1),
        ("wb", c_uint8, 1),
        ("auto_diaphragm", c_uint8, 1),
        ("reserved", c_uint8, 5),
    ]


class TVSettingsFlags(Union):
    _pack_ = 1
    _fields_ = [("flags", TVSettingsBits), ("bits", c_uint8)]


class TVDiaphragmCtrlBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("diaphragm_mode", c_uint8, 1),
        ("direction", c_uint8, 1),
        ("speed", c_uint8, 6),
    ]


class TVDiaphragmCtrlFlags(Union):
    _pack_ = 1
    _fields_ = [("flags", TVDiaphragmCtrlBits), ("bits", c_uint8)]


class TVFocusCtrlBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("focusing_mode", c_uint8, 1),
        ("direction", c_uint8, 1),
        ("speed", c_uint8, 6),
    ]


class TVFocusCtrlFlags(Union):
    _pack_ = 1
    _fields_ = [("flags", TVFocusCtrlBits), ("bits", c_uint8)]


class TVSettingsCmd(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("cmd", TVSettingsFlags),
        ("diaphragm_state", TVDiaphragmCtrlFlags),
        ("focusing_state", TVFocusCtrlFlags),
    ]


class OSSettingsPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("os", OSSettingsFlags),
        ("ir_narrow", IRSettingsCmd),
        ("ir_wide", IRSettingsCmd),
        ("tv_narrow", TVSettingsCmd),
        ("tv_wide", TVSettingsCmd),
    ]


class OSSettingsCmdPacket(ServicesCommandsPacket):
    _pack_ = 1
    _fields_ = [("payload", OSSettingsPayload)]
    CMD_ID = CommandIDs.OS_SET_CTRL_CMD
    _payload_class = OSSettingsPayload


# --- Detector Control Command ---
class DetectorCtrlCommandsBits(BaseStructure):
    """Detector Control Command bits"""

    _pack_ = 1
    _fields_ = [
        ("enable", c_uint8, 1),
        ("channel", c_uint8, 1),
        ("reserved", c_uint8, 6),
    ]


class DetectorCtrlCommands(Union):
    """Detector Control Command union"""

    _pack_ = 1
    _fields_ = [
        ("flags", DetectorCtrlCommandsBits),
        ("bits", c_uint8),
    ]


class DetectorCtrlCmdPayload(BaseStructure):
    """Detector Control Command payload"""

    _pack_ = 1
    _fields_ = [("commands", DetectorCtrlCommands)]


class DetectorCtrlCmdPacket(ServicesCommandsPacket):
    """Detector Control Command packet"""

    _pack_ = 1
    _fields_ = [
        ("payload", DetectorCtrlCmdPayload),
    ]
    CMD_ID = CommandIDs.DETECTOR_CTRL_CMD
    _payload_class = DetectorCtrlCmdPayload


_CMD_ID_TO_PACKET: Dict[int, Type[ServicesCommandsPacket]] = {
    CommandIDs.DIST_MNG_SRV_CMD: DistMngSrvCmdPacket,
    CommandIDs.BAL_COR_MNG_SRV_CMD: BalCorMngSrvCmdPacket,
    CommandIDs.BCM_MAN_SETS_CMD: BCMManualSetsCmdPacket,
    CommandIDs.WORKMODE_MNG_SRV_CMD_C: WorkmodeMngSrvCmdCPacket,
    CommandIDs.WORKMODE_MNG_SRV_CMD_G: WorkmodeMngSrvCmdGPacket,
    CommandIDs.WORKMODE_MNG_SRV_CMD_D: WorkmodeMngSrvCmdDPacket,
    CommandIDs.LRF_CTRL_CMD: LrfCtrlCmdPacket,
    CommandIDs.OS_CAP_CTRL_CMD: OsCapCtrlCmdPacket,
    CommandIDs.SECTOR_SEARCH_CTRL_CMD: SectorSearchCmdPacket,
    CommandIDs.TOPOREF_CMD: ToporefCmdPacket,
    CommandIDs.TARGET_SECTOR_CMD: TargetSectorCmdPacket,
    CommandIDs.TRACKER_CTRL_CMD: TrackerCtrlCmdPacket,
    CommandIDs.OS_SET_CTRL_CMD: OSSettingsCmdPacket,
    CommandIDs.DETECTOR_CTRL_CMD: DetectorCtrlCmdPacket,
}


# --- Packet parsing function ---
def parse_services_command_packet(data: bytes) -> Optional[ServicesCommandsPacket]:
    """Parse raw bytes to appropriate Services Commands packet type"""
    if len(data) < ctypes.sizeof(ServicesCommandsHeader):
        return None

    # Extract command ID from header
    cmd_id = int.from_bytes(
        data[
            SERVICES_COMMANDS_ID_IDX : SERVICES_COMMANDS_ID_IDX
            + SERVICES_COMMANDS_ID_SIZE
        ],
        "little",
    )

    # Find appropriate packet class
    packet_class = _CMD_ID_TO_PACKET.get(cmd_id)
    if packet_class is None:
        return None

    try:
        return packet_class.from_buffer_copy(data)
    except (ValueError, ctypes.ArgumentError):
        return None
