import ctypes
from ctypes import (
    Union,
    c_float,
    c_int8,
    c_int16,
    c_int32,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
)
from enum import IntEnum
from typing import List

from .base import (
    BaseStructure,
    TracktorHeader,
    TracktorPacket,
)


# Device IDs
class DeviceIDs(IntEnum):
    IMAGE_PROCESSING_BRD = 0x17
    T2_ANTENNA_CTRL_BLOCK = 0x40
    NAVIGATION_BLOCK = 0x69
    NAVIGATION_SERVICE = 0x01
    BALLISTICS_CORRECTIONS_MANAGER_SRV = 0xB1
    TARGET_INFO_MANAGER_SRV = 0xA0
    WORKMODE_MANAGER_SRV = 0xA2
    DISTANCE_MANAGER_SRV = 0xE8
    AWP_C = 0x2C
    AWP_G = 0xDE
    AWP_D = 0xFC
    COMMANDER_CONTROLLER = 0x44
    COMMANDER_GUIDANCE_DEVICE = 0xE1
    TURRET_CONTROL_BLOCK = 0x8B
    BLACK_BOX_LOGGER = 0xC7
    T55_CONTROL_BLOCK = 0xCB
    OPTICAL_STATION = 0x57
    DETECTOR = 0x54
    TRACKER = 0x1D
    UPS_BLOCK = 0x27
    CHASSIS_BLOCK = 0xDD


# Packet IDs
class PacketIDs(IntEnum):
    TD_T55_PACKET = 0x953C
    TD_CC_PACKET = 0xB0D5
    TD_T2_PACKET = 0x9CB1
    TD_NS_PACKET = 0x60BD
    TD_TCB_PACKET = 0xCBD7
    TD_OS_PACKET = 0x05A9
    TD_WORKMODE_PACKET = 0x6CAF
    TD_TRACKER_PACKET = 0xA334
    TD_TRACKER_EXT_PACKET = 0xA335
    TD_DETECTOR_PACKET = 0xCE56
    TD_BAL_CALC_PACKET = 0xE1BB
    TD_RF_PACKET = 0x3D64
    TD_BAL_COR_PACKET = 0xB281
    TD_DISTANCE_PACKET = 0xD1E3
    TD_TRACKS_SPD_PACKET = 0x0270
    TD_GEO_PACKET = 0xC05A
    TD_RTK_HEADING_PACKET = 0xBF0E
    TD_IMU_PACKET = 0x8E41
    TD_INCLINOMETER_PACKET = 0xA87E
    TD_STREAM_SETTINGS_PACKET = 0xF4CE
    TD_CHASSIS_PACKET = 0x370B
    TD_UPS_PACKET = 0x7828
    TD_GEN_CTRL_PACKET = 0x2A90
    TD_CALIB_CTRL_PACKET = 0x3902
    TD_SECTOR_SEARCH_CTRL_PACKET = 0x0827
    TD_GYROSTAB_CTRL_PACKET = 0x360A


# --- T55 Block ---
class T55StatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", c_uint16, 1),
        ("btn_enc_status", c_uint16, 1),
        ("led_status", c_uint16, 1),
        ("reserved", c_uint16, 13),
    ]


class T55Status(Union):
    _pack_ = 1
    _fields_ = [("flags", T55StatusBits), ("bits", c_uint16)]


class T55ButtonsBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("tracking_init_mode", c_uint16, 1),
        ("drives_mode", c_uint16, 1),
        ("enable_threadle_fire", c_uint16, 1),
        ("cooling", c_uint16, 1),
        ("tracking_ctrl", c_uint16, 1),
        ("lrf_measure", c_uint16, 1),
        ("fire", c_uint16, 1),
        ("enc_blend_btn", c_uint16, 1),
        ("enc_zoom_btn", c_uint16, 1),
        ("enc_manual_target_speed_btn", c_uint16, 1),
        ("ballistics_aiming", c_uint16, 1),
        ("reserved", c_uint16, 5),
    ]


class T55Buttons(Union):
    _pack_ = 1
    _fields_ = [("flags", T55ButtonsBits), ("bits", c_uint16)]


class T55Encoders(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("enc_blend", c_int8),
        ("enc_zoom", c_int8),
        ("enc_manual_target_speed", c_int8),
    ]


class T55Payload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", T55Status),
        ("buttons", T55Buttons),
        ("encoders", T55Encoders),
    ]


class T55Packet(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", T55Payload),
    ]
    MSG_ID = PacketIDs.TD_T55_PACKET
    _payload_class = T55Payload


# --- Chassis Block ---
class ChassisBlockStatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", c_uint8, 1),
        ("generator_status", c_uint8, 1),
        ("fuel_supply_status", c_uint8, 1),
        ("engine_status", c_uint8, 1),
        ("engine_started_by_driver", c_uint8, 1),
        ("reserved", c_uint8, 3),
    ]


class ChassisBlockStatus(Union):
    _pack_ = 1
    _fields_ = [("flags", ChassisBlockStatusBits), ("bits", c_uint8)]


class ChassisBlockPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [("status", ChassisBlockStatus), ("generator_rpm", c_uint16)]


class ChassisBlockPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [("payload", ChassisBlockPayload)]
    MSG_ID = PacketIDs.TD_CHASSIS_PACKET
    _payload_class = ChassisBlockPayload


# --- Uninterruptible Power Supply Block ---
class UPSStatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", c_uint32, 1),
        ("power_source", c_uint32, 2),
        ("workmode", c_uint32, 2),
        ("input_p24V", c_uint32, 1),  # Входные, +24В
        ("input_n24V", c_uint32, 1),  # Входные -24В
        ("p24V_top_ant_col", c_uint32, 1),  # вверх антенной колонки, +24В
        ("p24V_bot_ant_col", c_uint32, 1),  # низ антенной колонки, +24В
        ("p24V_conditioning", c_uint32, 1),  # кондиционер, +24В
        ("p24V_reserve", c_uint32, 1),  # резервный канал силового питания, +24В
        ("sys_24V_black_box", c_uint32, 1),  # питание черного ящика, 24В системное
        ("sd_to_battery", c_uint32, 1),  # коммутация SD-1000-24 и АКБ
        ("w_aimr_24V", c_uint32, 1),  # питание АРМ-Н, 24В системное
        ("w_rang_24V", c_uint32, 1),  # питание АРМ-Д, 24В системное
        ("omu_24V", c_uint32, 1),  # блок оптико-механический, 24В системное
        ("w_comm_24V", c_uint32, 1),  # питание АРМ-К, 24В системное
        ("p24V_tcu", c_uint32, 1),  # питание БУБ, 24В
        ("turn_on_sd", c_uint32, 1),  # ВКЛ/ВЫКЛ SD-1000-24
        ("reserved", c_uint32, 13),
    ]


class UPSStatus(Union):
    _pack_ = 1
    _fields_ = [("flags", UPSStatusBits), ("bits", c_uint16)]


class UPSPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", UPSStatus),
        ("p24_v", c_int16),
        ("n24_v", c_int16),
        ("sd_v", c_int16),
        ("bat_v", c_int16),
        ("current1", c_int16),
        ("current2", c_int16),
        ("current3", c_int16),
        ("current4", c_int16),
        ("current5", c_int16),
        ("current6", c_int16),
    ]


class UPSPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [("payload", UPSPayload)]
    MSG_ID = PacketIDs.TD_UPS_PACKET
    _payload_class = UPSPayload


# --- Commander Controller ---
class CCStatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("commander_operator_shoot", c_uint16, 1),
        ("gydrodrives_on", c_uint16, 1),
        ("gydrodrives_off", c_uint16, 1),
        ("hatch_opened", c_uint16, 1),
        ("cooling", c_uint16, 1),
        ("shoot_curcuit", c_uint16, 1),
        ("release", c_uint16, 1),
        ("stalling", c_uint16, 1),
        ("reserved", c_uint16, 8),
    ]


class CCStatus(Union):
    _pack_ = 1
    _fields_ = [("flags", CCStatusBits), ("bits", c_uint16)]


class CCPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", CCStatus),
        ("guns_shoot_counter", c_uint16 * 4),
        ("temperature", c_int16 * 4),
    ]


class CCPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [("payload", CCPayload)]
    MSG_ID = PacketIDs.TD_CC_PACKET
    _payload_class = CCPayload


# --- T2 Antenna Control Block ---
class T2StatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("azimuth_motor_status", c_uint16, 1),
        ("elevation_motor_status", c_uint16, 1),
        ("azimuth_encoder_status", c_uint16, 1),
        ("elevation_encoder_status", c_uint16, 1),
        ("antenna_rising_state", c_uint16, 2),
        ("antenna_locking_state", c_uint16, 1),
        ("gyrostab_status", c_uint16, 1),
        ("reserved", c_uint16, 8),
    ]


class T2Status(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", T2StatusBits),
        ("bits", c_uint16),
    ]


class T2Data(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("azimuth", c_uint16),
        ("elevation", c_int16),
        ("antenna_up_percent", c_uint8),
        ("azimuth_velocity", c_int32),
        ("elevation_velocity", c_int32),
    ]


class T2Payload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", T2Status),
        ("data", T2Data),
    ]


class T2Packet(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", T2Payload),
    ]
    MSG_ID = PacketIDs.TD_T2_PACKET
    _payload_class = T2Payload


# --- Navigation Service ---
class NSStatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("inclinometr_status", c_uint16, 1),
        ("gnss_status", c_uint16, 1),
        ("imu_status", c_uint16, 1),
        ("data_source", c_uint16, 2),
        ("rtk_state", c_uint16, 2),
        ("reserved", c_uint16, 9),
    ]


class NSStatus(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", NSStatusBits),
        ("bits", c_uint16),
    ]


class NSData(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("latitude", c_int32),
        ("longitude", c_int32),
        ("altitude", c_int16),
        ("speed", c_int16),
        ("chassis_heading", c_uint16),
        ("chassis_pitch", c_int16),
        ("chassis_roll", c_int16),
    ]


class NSPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", NSStatus),
        ("data", NSData),
    ]


class NSPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", NSPayload),
    ]
    MSG_ID = PacketIDs.TD_NS_PACKET
    _payload_class = NSPayload


# --- Turret Control Block ---
class TCBStatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("gydro_drive_status", c_uint16, 1),
        ("azimuth_drive_status", c_uint16, 1),
        ("elevation_drive_status", c_uint16, 1),
        ("azimuth_encoder_status", c_uint16, 1),
        ("elevation_encoder_status", c_uint16, 1),
        ("ballistics_aiming", c_uint16, 1),
        ("mirror_mode", c_uint16, 1),
        ("dso2_status", c_uint16, 2),
        ("reserved", c_uint16, 7),
    ]


class TCBStatus(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", TCBStatusBits),
        ("bits", c_uint16),
    ]


class TCBData(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("azimuth", c_uint16),
        ("elevation", c_int16),
        ("azimuth_velocity", c_int32),
        ("elevation_velocity", c_int32),
    ]


class TCBPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", TCBStatus),
        ("data", TCBData),
    ]


class TCBPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", TCBPayload),
    ]
    MSG_ID = PacketIDs.TD_TCB_PACKET
    _payload_class = TCBPayload


# --- Optical Station ---
class OSStatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("cap_state", c_uint16, 1),
        ("lrf_status", c_uint16, 1),
        ("tv_narrow_status", c_uint16, 1),
        ("tv_wide_status", c_uint16, 1),
        ("ir_narrow_status", c_uint16, 1),
        ("ir_wide_status", c_uint16, 1),
        ("tv_narrow_stream_status", c_uint16, 1),
        ("tv_wide_stream_status", c_uint16, 1),
        ("ir_narrow_stream_status", c_uint16, 1),
        ("ir_wide_stream_status", c_uint16, 1),
        ("search_status", c_uint16, 1),
        ("fan", c_uint16, 1),
        ("heating", c_uint16, 1),
        ("reserved", c_uint16, 3),
    ]


class OSStatus(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", OSStatusBits),
        ("bits", c_uint16),
    ]


class TVCamBlockStateBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("diaphragm_mode", c_uint8, 1),
        ("focusing_mode", c_uint8, 1),
        ("reserved", c_uint8, 6),
    ]


class TVCamBlockStateFlags(Union):
    _pack_ = 1
    _fields_ = [("flags", TVCamBlockStateBits), ("bits", c_uint8)]


class TVCamBlockState(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("cam_state", TVCamBlockStateFlags),
        ("diaphragm_state", c_uint16),
        ("focusing_state", c_uint16),
    ]


class IRCamStateBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("enhancement_mode", c_uint16, 2),
        ("shutter", c_uint16, 1),
        ("lut", c_uint16, 4),
        ("calibration", c_uint16, 1),
        ("auto_shutter", c_uint16, 1),
        ("reserved", c_uint16, 7),
    ]


class IRCamState(Union):
    _pack_ = 1
    _fields_ = [("flags", IRCamStateBits), ("bits", c_uint16)]


class OSPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", OSStatus),
        ("tv_narrow", TVCamBlockState),
        ("tv_wide", TVCamBlockState),
        ("ir_narrow", IRCamState),
        ("ir_wide", IRCamState),
        ("work_time_sec", c_uint32),
    ]


class OSPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", OSPayload),
    ]
    MSG_ID = PacketIDs.TD_OS_PACKET
    _payload_class = OSPayload


# --- Workmode ---
class WorkmodeStatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("mirror_mode", c_uint16, 1),
        ("ground_mode", c_uint16, 1),
        ("tracking_init_mode", c_uint16, 1),
        ("drives_mode", c_uint16, 1),
        ("ballistics_aiming", c_uint16, 1),
        ("treadle_fire", c_uint16, 1),
        ("commander_gunner_fire", c_uint16, 1),
        ("angles_measurement_mode", c_uint16, 1),
        ("reserved", c_uint16, 8),
    ]


class WorkmodeStatus(Union):
    _pack_ = 1
    _fields_ = [("flags", WorkmodeStatusBits), ("bits", c_uint16)]


class WorkmodePayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", WorkmodeStatus),
        ("burst_mode", c_uint8),
        ("target_angle", c_uint8),
        ("target_bisector", c_uint16),
        ("target_sector_width", c_uint16),
    ]


class WorkmodePacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", WorkmodePayload),
    ]
    MSG_ID = PacketIDs.TD_WORKMODE_PACKET
    _payload_class = WorkmodePayload


# --- Tracker ---
class TrackerStatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", c_uint8, 2),
        ("channel", c_uint8, 1),
        ("reserved", c_uint8, 5),
    ]


class TrackerStatus(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", TrackerStatusBits),
        ("bits", c_uint8),
    ]


class TargetData(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("delta_azimuth", c_int16),
        ("delta_elevation", c_int16),
        ("angular_width", c_uint16),
        ("angular_height", c_uint16),
        ("x", c_int16),
        ("y", c_int16),
        ("z", c_int16),
        ("x_velocity", c_int16),
        ("y_velocity", c_int16),
        ("z_velocity", c_int16),
    ]


class TrackerPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", TrackerStatus),
        ("target", TargetData),
    ]


class TrackerPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", TrackerPayload),
    ]
    MSG_ID = PacketIDs.TD_TRACKER_PACKET
    _payload_class = TrackerPayload


# --- Tracket Ext ---

class TargetExtData(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("delta_azimuth", c_int16),
        ("delta_elevation", c_int16),
        ("angular_width", c_uint16),
        ("angular_height", c_uint16),
        ("timestamp_us", c_uint64),
    ]

class TrackerExtPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", TrackerStatus),
        ("target", TargetExtData),
    ]

class TrackerExtPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", TrackerExtPayload),
    ]

    MSG_ID = PacketIDs.TD_TRACKER_EXT_PACKET
    _payload_class = TrackerExtPayload


# --- Detector ---
class DetectorStatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("is_enabled", c_uint8, 1),
        ("channel", c_uint8, 1),
        ("reserved", c_uint8, 6),
    ]


class DetectorStatus(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", DetectorStatusBits),
        ("bits", c_uint8),
    ]


MAX_DETECTIONS = 16


class DetectionStatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [("status", c_uint8, 2), ("reserved", c_uint8, 6)]


class DetectionStatus(Union):
    _pack_ = 1
    _fields_ = [("flags", DetectionStatusBits), ("bits", c_uint8)]


class DetectionData(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", DetectionStatus),
        ("id", c_uint16),
        ("det_class", c_uint8),
        ("det_type", c_uint8),
        ("class_confidence", c_uint8),
        ("type_confidence", c_uint8),
        ("age", c_uint16),
        ("delta_azimuth", c_int16),
        ("delta_elevation", c_int16),
        ("angular_width", c_uint16),
        ("angular_height", c_uint16),
        ("distance", c_uint16),
    ]


class DetectorPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", DetectorStatus),
        ("count", c_uint8),
        ("detections", DetectionData * MAX_DETECTIONS),
    ]

    def add_detections(self, detections: List[DetectionData]):
        for i in range(len(detections)):
            self.detections[i] = detections[i]
        self.count = len(detections)


class DetectorPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", DetectorPayload),
    ]
    MSG_ID = PacketIDs.TD_DETECTOR_PACKET
    _payload_class = DetectorPayload


# --- Ballistics Calculator ---
class BCStatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("is_solution_valid", c_uint8, 1),
        ("is_in_impact_area", c_uint8, 1),
        ("reserved", c_uint8, 6),
    ]


class BCStatus(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", BCStatusBits),
        ("bits", c_uint8),
    ]


class BCPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", BCStatus),
        ("aiming_azimuth", c_uint16),
        ("aiming_elevation", c_int16),
        ("distance", c_uint16),
        ("delta_azimuth", c_int16),
        ("delta_elevation", c_int16),
    ]


class BCPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", BCPayload),
    ]
    MSG_ID = PacketIDs.TD_BAL_CALC_PACKET
    _payload_class = BCPayload


# --- Rangefinder ---
class RFStatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", c_uint8, 1),
        ("is_pointer", c_uint8, 1),
        ("is_single", c_uint8, 1),
        ("continious_freq", c_uint8, 5),
    ]


class RFStatus(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", RFStatusBits),
        ("bits", c_uint8),
    ]


class RFPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", RFStatus),
        ("counter", c_uint8),
        ("distance", c_float),
    ]


class RFPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", RFPayload),
    ]
    MSG_ID = PacketIDs.TD_RF_PACKET
    _payload_class = RFPayload


# --- Ballistics Corrections ---
class BCCorPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("correction_azimuth", c_int16),
        ("correction_elevation", c_int16),
        ("guns_shoot_counter", c_uint16 * 4),
        ("ammo_temperature", c_int16),
        ("velocity_deviation", c_int16),
        ("air_density_deviation", c_int16),
    ]


class BCCorPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", BCCorPayload),
    ]
    MSG_ID = PacketIDs.TD_BAL_COR_PACKET
    _payload_class = BCCorPayload


# --- Distance Packet ---
class DistInfoBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("source", c_uint8, 2),
        ("is_actual", c_uint8, 1),
        ("reserved", c_uint8, 5),
    ]


class DistInfo(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", DistInfoBits),
        ("bits", c_uint8),
    ]


class DistPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("info", DistInfo),
        ("distance", c_uint16),
    ]


class DistPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", DistPayload),
    ]
    MSG_ID = PacketIDs.TD_DISTANCE_PACKET
    _payload_class = DistPayload


# --- Tracks Speeds Packet ---
class TracksSpeedsInfoBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("left_odometer_status", c_uint8, 1),
        ("right_odometer_status", c_uint8, 1),
        ("reserved", c_uint8, 6),
    ]


class TracksSpeedsInfo(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", TracksSpeedsInfoBits),
        ("bits", c_uint8),
    ]


class TracksSpeedsPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", TracksSpeedsInfo),
        ("left_track_speed", c_int16),
        ("right_track_speed", c_int16),
    ]


class TracksSpeedsPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", TracksSpeedsPayload),
    ]
    MSG_ID = PacketIDs.TD_TRACKS_SPD_PACKET
    _payload_class = TracksSpeedsPayload


# --- Geolocation Packet ---
class GeoStatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("gnss_status", c_uint8, 1),
        ("reserved", c_uint8, 7),
    ]


class GeoStatus(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", GeoStatusBits),
        ("bits", c_uint8),
    ]


class GeoPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", GeoStatus),
        ("latitude", c_int32),
        ("longitude", c_int32),
        ("altitude", c_int16),
        ("speed", c_int16),
        ("pos_cov_nn", c_float),
        ("pos_cov_ne", c_float),
        ("pos_cov_nd", c_float),
        ("pos_cov_ee", c_float),
        ("pos_cov_ed", c_float),
        ("pos_cov_dd", c_float),
    ]


class GeoPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", GeoPayload),
    ]
    MSG_ID = PacketIDs.TD_GEO_PACKET
    _payload_class = GeoPayload


# --- RTK Heading Packet ---
class RTKHeadingBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("is_heading_valid", c_uint8, 1),
        ("heading_source", c_uint8, 2),
        ("rtk_state", c_uint8, 2),
        ("reserved", c_uint8, 3),
    ]


class RTKHeadingStatus(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", RTKHeadingBits),
        ("bits", c_uint8),
    ]


class RTKHeadingPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", RTKHeadingStatus),
        ("heading", c_uint16),
        ("heading_acc", c_uint16),
    ]


class RTKHeadingPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", RTKHeadingPayload),
    ]
    MSG_ID = PacketIDs.TD_RTK_HEADING_PACKET
    _payload_class = RTKHeadingPayload


# --- IMU Packet ---
class IMUStatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("imu_status", c_uint8, 1),
        ("reserved", c_uint8, 7),
    ]


class IMUStatus(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", IMUStatusBits),
        ("bits", c_uint8),
    ]


class Orientation(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("roll", c_float),
        ("pitch", c_float),
        ("yaw", c_float),
    ]


class Acceleration(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
    ]


class AngularVelocity(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
    ]


class IMUPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", IMUStatus),
        ("orientation", Orientation),
        ("acceleration", Acceleration),
        ("angular_velocity", AngularVelocity),
    ]


class IMUPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", IMUPayload),
    ]
    MSG_ID = PacketIDs.TD_IMU_PACKET
    _payload_class = IMUPayload


# --- Inclinometer Packet ---
class InclinometerStatusBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", c_uint8, 1),
        ("reserved", c_uint8, 7),
    ]


class InclinometerStatus(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", InclinometerStatusBits),
        ("bits", c_uint8),
    ]


class InclinometerPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [("status", InclinometerStatus), ("pitch", c_float), ("roll", c_float)]


class InclinometerPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", InclinometerPayload),
    ]
    MSG_ID = PacketIDs.TD_INCLINOMETER_PACKET
    _payload_class = InclinometerPayload


# --- Generator Control Packet ---
class GenCtrlBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("generator_on", c_uint8, 1),
        ("fuel_supply_on", c_uint8, 1),
        ("engine_on", c_uint8, 1),
        ("reserved", c_uint8, 5),
    ]


class GenCtrlFlags(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", GenCtrlBits),
        ("bits", c_uint8),
    ]


class GeneratorCtrlPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [("cmd", GenCtrlFlags)]


class GeneratorCtrlPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", GeneratorCtrlPayload),
    ]
    MSG_ID = PacketIDs.TD_GEN_CTRL_PACKET
    _payload_class = GeneratorCtrlPayload


# --- Calibration Control Packet ---
class CalibCtrlBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("t55_calibration", c_uint8, 1),
        ("antenna_calibration", c_uint8, 1),
        ("turret_calibration", c_uint8, 1),
        ("alignment_calibration", c_uint8, 1),
        ("reserved", c_uint8, 4),
    ]


class CalibCtrlFlags(Union):
    _pack_ = 1
    _fields_ = [("flags", CalibCtrlBits), ("bits", c_uint8)]


class CalibCtrlPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [("cmd", CalibCtrlFlags)]


class CalibCtrlPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [("payload", CalibCtrlPayload)]
    MSG_ID = PacketIDs.TD_CALIB_CTRL_PACKET
    _payload_class = CalibCtrlPayload


# --- Stream Settings Packet ---
class StreamSettingsFlagsBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("synchonization", c_uint8, 1),
        ("reserved", c_uint8, 7),
    ]


class StreamSettingsFlags(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", StreamSettingsFlagsBits),
        ("bits", c_uint8),
    ]


class StreamSettingsPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("cmd", StreamSettingsFlags),
        ("zoom", c_uint8),
        ("blend", c_uint8),
        ("gradient", c_uint8),
    ]


class StreamSettingsPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", StreamSettingsPayload),
    ]
    MSG_ID = PacketIDs.TD_STREAM_SETTINGS_PACKET
    _payload_class = StreamSettingsPayload


# --- Sector Search Control Packet ---
class SectorSearchCtrlFlagsBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("status", c_uint8, 2),
        ("is_circle", c_uint8, 1),
        ("reserved", c_uint8, 5),
    ]


class SectorSearchCtrlFlags(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", SectorSearchCtrlFlagsBits),
        ("bits", c_uint8),
    ]


class SectorSearchCtrlPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("cmd", SectorSearchCtrlFlags),
        ("azimuth_start", c_uint16),
        ("azimuth_end", c_uint16),
        ("elevation_start", c_int16),
        ("elevation_end", c_int16),
        ("azimuth_ang_speed", c_uint16),
        ("elevation_ang_speed", c_uint16),
    ]


class SectorSearchCtrlPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [
        ("payload", SectorSearchCtrlPayload),
    ]
    MSG_ID = PacketIDs.TD_SECTOR_SEARCH_CTRL_PACKET
    _payload_class = SectorSearchCtrlPayload


# --- Gyrostabilization Control Packet ---
class GyrostabCtrlFlagsBits(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("enable", c_uint8, 1),
        ("reserved", c_uint8, 7),
    ]


class GyrostabCtrlFlags(Union):
    _pack_ = 1
    _fields_ = [
        ("flags", GyrostabCtrlFlagsBits),
        ("bits", c_uint8),
    ]


class GyrostabCtrlPayload(BaseStructure):
    _pack_ = 1
    _fields_ = [("cmd", GyrostabCtrlFlags)]


class GyrostabCtrlPacket(TracktorPacket):
    _pack_ = 1
    _fields_ = [("payload", GyrostabCtrlPayload)]
    MSG_ID = PacketIDs.TD_GYROSTAB_CTRL_PACKET
    _payload_class = GyrostabCtrlPayload


# --- Packet Type Registry & Parsing ---

_MSG_ID_TO_PACKET = {
    UPSPacket.MSG_ID: UPSPacket,
    ChassisBlockPacket.MSG_ID: ChassisBlockPacket,
    T55Packet.MSG_ID: T55Packet,
    CCPacket.MSG_ID: CCPacket,
    T2Packet.MSG_ID: T2Packet,
    NSPacket.MSG_ID: NSPacket,
    TCBPacket.MSG_ID: TCBPacket,
    OSPacket.MSG_ID: OSPacket,
    WorkmodePacket.MSG_ID: WorkmodePacket,
    TrackerPacket.MSG_ID: TrackerPacket,
    TrackerExtPacket.MSG_ID: TrackerExtPacket,
    DetectorPacket.MSG_ID: DetectorPacket,
    BCPacket.MSG_ID: BCPacket,
    RFPacket.MSG_ID: RFPacket,
    BCCorPacket.MSG_ID: BCCorPacket,
    DistPacket.MSG_ID: DistPacket,
    TracksSpeedsPacket.MSG_ID: TracksSpeedsPacket,
    GeoPacket.MSG_ID: GeoPacket,
    RTKHeadingPacket.MSG_ID: RTKHeadingPacket,
    IMUPacket.MSG_ID: IMUPacket,
    InclinometerPacket.MSG_ID: InclinometerPacket,
    GeneratorCtrlPacket.MSG_ID: GeneratorCtrlPacket,
    CalibCtrlPacket.MSG_ID: CalibCtrlPacket,
    StreamSettingsPacket.MSG_ID: StreamSettingsPacket,
    SectorSearchCtrlPacket.MSG_ID: SectorSearchCtrlPacket,
    GyrostabCtrlPacket.MSG_ID: GyrostabCtrlPacket,
}


def parse_tracktor_packet(buffer: bytes):
    if len(buffer) < ctypes.sizeof(TracktorHeader):
        raise ValueError("Buffer too short for TracktorHeader")
    # Parse header to get msg_id
    header = TracktorHeader.from_buffer_copy(buffer)
    packet_cls = _MSG_ID_TO_PACKET.get(header.msg_id)
    if packet_cls is None:
        return None
        # raise ValueError(f"Unknown msg_id: 0x{header.msg_id:X}")

    if packet_cls == DetectorPacket:
        count = int(buffer[ctypes.sizeof(TracktorHeader) + 1])
        buffer += bytes((MAX_DETECTIONS - count) * ctypes.sizeof(DetectionData))

    expected_len = ctypes.sizeof(packet_cls)
    if len(buffer) < expected_len:
        raise ValueError(
            f"Buffer too short for packet: expected {expected_len}, got {len(buffer)}"
        )
    return packet_cls.from_buffer_copy(buffer)
