from ctypes import (
    c_bool,
    c_int8,
    c_int16,
    c_int32,
    c_uint8,
    c_uint16,
    c_uint32,
    LittleEndianStructure,
)
from typing import List, Type

from src.framework.protocols.ntmp.base import NTMPStatus, NTMPMessage, NTMPHeader, NTMPButtons


class InfTarget(LittleEndianStructure):
    """
    Структура данных с информацией по цели
    """

    _fields_ = [
        # уникальный номер цели
        ("id", c_uint16),
        # идентификатор типа цели
        ("type", c_uint8),
        # коэффициент "уверенности" в решении по цели [0..100]
        ("confidence", c_uint8),
        # разница по азимуту в милиградусах
        ("delta_azimuth", c_int32),
        # разница по углу места в милиградусах
        ("delta_elevation", c_int32),
        # ширина цели в милиградусах
        ("angular_width", c_uint32),
        # высота цели в милиградусах
        ("angular_height", c_uint32),
        # мировая координата X (м)
        ("x", c_int16),
        # мировая координата Y (м)
        ("y", c_int16),
        # мировая координата Z (м)
        ("z", c_int16),
        # мировая скорость по оси X (м/с)
        ("vx", c_int16),
        # мировая скорость по оси Y (м/с)
        ("vy", c_int16),
        # мировая скорость по оси Z (м/с)
        ("vz", c_int16),
    ]

    _pack_ = 1

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.id}, "
            f"type={self.type}, "
            f"confidence={self.confidence}, "
            f"delta_azimuth={self.delta_azimuth}, "
            f"delta_elevation={self.delta_elevation}, "
            f"angular_width={self.angular_width}, "
            f"angular_height={self.angular_height}, "
            f"x={self.x}, y={self.y}, z={self.z}, "
            f"vx={self.vx}, vy={self.vy}, vz={self.vz})"
        )


class InfTracker(NTMPMessage):
    """
    Структура данных информационной посылки от системы слежения за целью
    """

    _header_ = NTMPHeader(src_id=0x0000, dst_id=0x0000, msg_id=0xE5838D3B)

    class STATUS(NTMPStatus):
        _fields_ = [
            ("is_selftest_ok", c_uint16, 1),
            ("is_tracker_enabled", c_uint16, 1),
            ("is_target_locked", c_uint16, 1),
            ("reserved", c_uint16, 13),
        ]

    _fields_ = [
        # Статус трекера
        ("status", STATUS),
        # задержка обработки данных в мкс
        ("latency", c_uint32),
        # информация по сопровождаемой цели
        ("locked_target", InfTarget),
    ]

    _pack_ = 1


class InfDetector(NTMPMessage):
    """
    Структура данных информационной посылки от системы обнаружения целей
    """

    _header_ = NTMPHeader(src_id=0x0000, dst_id=0x0000, msg_id=0x9FD491D1)

    class STATUS(NTMPStatus):
        _fields_ = [
            ("is_selftest_ok", c_uint16, 1),
            ("is_detector_enabled", c_uint16, 1),
            # индекс нейросети
            ("dnn_id", c_uint16, 4),
            ("reserved", c_uint16, 10),
        ]

    @classmethod
    def gen_const_size_cls(cls, targets_count) -> Type[NTMPMessage]:
        class InfDetectorConstantSize(NTMPMessage):
            _header_ = cls._header_

            _fields_ = [
                ("status", cls.STATUS),
                # задержка обработки данных в мкс
                ("latency", c_uint32),
                # количество элементов в массиве
                ("count", c_uint32),
                # массив целей
                ("targets", InfTarget * targets_count),
            ]

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.count = targets_count

        return InfDetectorConstantSize

    MAX_TARGET_COUNT = 20

    sub_classes: List[Type[NTMPMessage]] = []

    @classmethod
    def from_targets(
        cls,
        status: STATUS,
        targets: List[InfTarget],
        latency: int = 0,
    ) -> "NTMPMessage":
        targets_count = len(targets)
        if targets_count < cls.MAX_TARGET_COUNT:
            detections = cls.sub_classes[targets_count](status=status, latency=latency)
        else:
            detections = cls.gen_const_size_cls(targets_count)(
                status=status, latency=latency
            )
        for i in range(targets_count):
            detections.targets[i] = targets[i]
        return detections


# Create cached sub_classes after the class is defined
InfDetector.sub_classes = [
    InfDetector.gen_const_size_cls(i) for i in range(InfDetector.MAX_TARGET_COUNT)
]


class InfGpsNav(NTMPMessage):
    """
    Информационная посылка от системы навигации
    """

    _header_ = NTMPHeader(src_id=0x0, dst_id=0x0, msg_id=0x9EC39D27)

    class STATUS(NTMPStatus):
        _fields_ = [
            ("is_selftest_ok", c_uint16, 1),
            ("is_valid", c_uint16, 1),
        ]

    _fields_ = [
        ("status", STATUS),
        ("latitude", c_int32),
        ("longitude", c_int32),
        ("altitude", c_int16),
        ("speed", c_uint32),
        ("heading", c_uint32),
        ("satellite_count", c_uint8),
    ]

    _pack_ = 1


class InfCompassNav(NTMPMessage):
    """
    Информационная посылка от компаса
    """

    _header_ = NTMPHeader(src_id=0x0, dst_id=0x0, msg_id=0xC91A8B0E)

    class STATUS(NTMPStatus):
        _fields_ = [
            ("is_selftest_ok", c_uint16, 1),
            ("is_valid", c_uint16, 1),
        ]

    _fields_ = [("status", STATUS), ("machine_azimuth", c_uint32)]

    _pack_ = 1


class InfArmK(NTMPMessage):
    _header_ = NTMPHeader(src_id=0x0050, dst_id=0x0, msg_id=0xF0C59869)

    class STATUS(NTMPStatus):
        _fields_ = [("is_selftest_ok", c_uint16, 1)]

    _fields_ = [
        ("status", STATUS),
        ("is_target_designation_approved", c_bool),
        ("target_designation_mode", c_bool),
        ("target_designation_azimuth", c_uint32),
        ("target_designation_elevation", c_int32),
        ("target_designation_type", c_int8),
        ("latitude", c_int32),
        ("longitude", c_int32),
        ("height", c_int32),
        ("azimuth", c_uint32),
        ("min_area_resp", c_uint32),
        ("max_area_resp", c_uint32),
        ("mirror_azimuth", c_uint32),
        ("delta_initial_speed", c_int16),
        ("gnss_manual_mode", c_bool),
    ]

    _pack_ = 1


class InfGNSSSolution(NTMPMessage):
    """
    Результирущая посылка навигационной системы
    """

    _header_ = NTMPHeader(src_id=0x0100, dst_id=0x0, msg_id=0xC14FF9D5)

    class STATUS(NTMPStatus):
        _fields_ = [
            ("is_selftest_ok", c_uint16, 1),
            ("is_manual", c_uint16, 1),
        ]

    _fields_ = [
        ("status", STATUS),
        ("latitude", c_int32),
        ("longitude", c_int32),
        ("altitude", c_int16),
        ("speed", c_uint32),
        ("heading", c_uint32),
        ("machine_azimuth", c_uint32),
        ("satellite_count", c_uint8),
    ]

    _pack_ = 1



class InfT55Brd(NTMPMessage):
    """
    Структура данных от интерфейсной платы блока Т55
    """

    _header_ = NTMPHeader(src_id=0x0000, dst_id=0x0000, msg_id=0x21E16BBA)

    class STATUS(NTMPStatus):
        _fields_ = [
            ("is_selftest_ok", c_uint16, 1),
            ("is_butt_i2c_test_ok", c_uint16, 1),
            ("is_butt_enc_i2c_test_ok", c_uint16, 1),
            ("is_abs_enc_spi1_test_ok", c_uint16, 1),
            ("is_abs_enc_spi2_test_ok", c_uint16, 1),

            ("is_start_pos_joyst_test_ok", c_uint16, 1),
            ("is_start_pos_butt_test_ok", c_uint16, 1),

            # в процессе калибровки Т55 блока равно True
            ("is_calibration_now", c_uint16, 1),
            # при включении блока - 0
            ("nazemn_mode", c_uint16, 1),

            # Индикатор "ВНИМАНИЕ"
            ("attention", c_uint16, 1),
            # Режим работы (0,1,2,3)
            # 0: ПОЛУАВТ УМ - от себя, ПОЛУАВТ АЗ - от себя
            # 1: ПОЛУАВТ УМ - от себя, ПОЛУАВТ АЗ - на себя
            # 2: ПОЛУАВТ УМ - на себя, ПОЛУАВТ АЗ - от себя
            # 3: ПОЛУАВТ УМ - на себя, ПОЛУАВТ АЗ - на себя
            ("work_mode", c_uint16, 4),

            ("left_machine_gun_enabled", c_uint16, 1),
            ("right_machine_gun_enabled", c_uint16, 1),
        ]

    class BUTTONS(NTMPButtons):
        _fields_ = [
            # АРМ-Н ВКЛ/ВЫКЛ отображения детектирования
            ("detector", c_uint16, 1),
            # I РЕЖИМ / II РЕЖИМ
            ("mode_1_2", c_uint16, 1),
            # ПОЛУАВТ / АВТ ГП
            ("semi_auto_auto_hydr_driv", c_uint16, 1),
            # АВТ
            ("automatic", c_uint16, 1),
            # НАВЕД
            ("guidance", c_uint16, 1),
            # ОГОНЬ
            ("shoot", c_uint16, 1),
            # ОХЛ
            ("cooling", c_uint16, 1),
            # КНОПКА/ПЕДАЛЬ
            ("shoot_butt_treadle", c_uint16, 1),

            ("but_enc_manual_target_speed", c_uint16, 1),
            ("but_enc_blend_factor", c_uint16, 1),
            ("but_enc_zoom_factor", c_uint16, 1),

            # Срабатывает при крайнем правом/левом положении
            ("extr_pos_roll", c_uint16, 1),
            # Срабатывает при крайнем положении по pitch
            ("extr_pos_pitch", c_uint16, 1),
        ]

    _fields_ = [
        # Статус интерфейсной платы
        ("status", STATUS),
        # Кнопки управления
        ("buttons", BUTTONS),
        # при включении блока - 0
        ("manual_target_speed_mode", c_uint8),
        # дельта скорости цели (ракурсные кольца)
        ("enc_manual_target_speed", c_int8),
        # дельта степени смешивания ТВ-ТПВ
        ("enc_blend_factor", c_int8),
        # дельта масштаба изображения
        ("enc_zoom_factor", c_int8),
        # от себя/на себя "чебурашки" в милиградусах [-47000..47000]
        ("pitch", c_int32),
        # влево/вправо "чебурашки" в милиградусах [-47000..47000]
        ("roll", c_int32),
    ]

    _pack_ = 1



class InfTurretBrd(NTMPMessage):
    """
    Структура данных от интерфейсной платы блока управления башней
    """

    _header_ = NTMPHeader(src_id=0x0000, dst_id=0x0000, msg_id=0xE2BD631D)

    class STATUS(NTMPStatus):
        _fields_ = [
            ("is_selftest_ok", c_uint16, 1),
            ("is_gyro_motor_ok", c_uint16, 1),
            ("is_azimuth_motor_ok", c_uint16, 1),
            ("is_elevation_motor_ok", c_uint16, 1),
            ("is_azimuth_encoder_ok", c_uint16, 1),
            ("is_elevation_encoder_ok", c_uint16, 1),

            # при включении блока - 0
            ("srp_mode", c_uint16, 1),
            # при включении блока - 1
            ("mirror_mode", c_uint16, 1),

            # мотор находится в рабочем цикле - true
            ("is_dso20_work", c_uint16, 1),
            # fill remaining bits to 16
            ("reserved", c_uint16, 7),
        ]

    _fields_ = [
        # Статус интерфейсной платы
        ("status", STATUS),
        # милиградусы, при включении блока - 0
        ("correction_azimuth", c_int16),
        ("correction_elevation", c_int16),
        # азимут башни относительно самохода в милиградусах [0..360000]
        ("azimuth", c_uint32),
        # угол места пушек относительно самохода в милиградусах [-90000..90000]
        ("elevation", c_int32),
        # угловая скорость башни по азимуту в милиградусах/сек
        ("azimuth_velocity", c_int32),
        # угловая скорость пушек по углу места в милиградусах/сек
        ("elevation_velocity", c_int32),
    ]

    _pack_ = 1


class InfAntennaBrd(NTMPMessage):
    """
    Структура данных от интерфейсной платы блока антенной колонки
    """

    _header_ = NTMPHeader(src_id=0x0000, dst_id=0x0000, msg_id=0xF19DDB67)

    class STATUS(NTMPStatus):
        _fields_ = [
            ("is_selftest_ok", c_uint16, 1),
            ("is_azimuth_motor_ok", c_uint16, 1),
            ("is_elevation_motor_ok", c_uint16, 1),
            ("is_azimuth_encoder_ok", c_uint16, 1),
            ("is_elevation_encoder_ok", c_uint16, 1),
            ("is_led_up", c_uint16, 1),
            ("is_led_down", c_uint16, 1),
            ("is_led_epsilon0", c_uint16, 1),
            ("is_led_lock", c_uint16, 1),
            ("is_set_stopper1", c_uint16, 1),
            ("is_set_stopper2", c_uint16, 1),
            ("reserved_1", c_uint16, 1),
            ("reserved_2", c_uint16, 1),
            ("reserved_3", c_uint16, 1),
            ("reserved_4", c_uint16, 1),
            ("reserved_5", c_uint16, 1),
        ]

    class BUTTONS(NTMPButtons):
        _fields_ = [
            ("is_butt_move_up", c_uint16, 1),
            ("is_butt_move_down", c_uint16, 1),
            ("reserved", c_uint16, 14),
        ]

    _fields_ = [
        # Статус интерфейсной платы
        ("status", STATUS),
        # Кнопки управления
        ("buttons", BUTTONS),
        # азимут антенной колонки в милиградусах [0..360000]
        ("azimuth", c_uint32),
        # угол места в милиградусах [-90000..90000]
        ("elevation", c_int32),
        # угловая скорость по азимуту
        ("azimuth_velocity", c_int32),
        # угловая скорость по углу места
        ("elevation_velocity", c_int32),
    ]

    _pack_ = 1


class InfSwift(NTMPMessage):
    """
    Структура данных информационной посылки от оптической станции "Стриж"
    """

    _header_ = NTMPHeader(src_id=0x0000, dst_id=0x0000, msg_id=0xC84248B6)  # Replace with actual msg_id if known

    class STATUS(NTMPStatus):
        _fields_ = [
            ("is_selftest_ok", c_uint16, 1),
            # крышка ОТКР - 1, ЗАКР - 0
            ("is_cap_opened", c_uint16, 1),
            # 1 - есть картинка от камеры, 0 - нет картинки
            ("is_tv_w_ok", c_uint16, 1),
            # 1 - есть картинка от камеры, 0 - нет картинки
            ("is_ir_w_ok", c_uint16, 1),
            # 1 - есть картинка от камеры, 0 - нет картинки
            ("is_tv_n_ok", c_uint16, 1),
            # 1 - есть картинка от камеры, 0 - нет картинки
            ("is_ir_n_ok", c_uint16, 1),
            # 1 - стрим запущен и работает, 0 - стрим не работает
            ("is_tv_w_stream_on", c_uint16, 1),
            # 1 - стрим запущен и работает, 0 - стрим не работает
            ("is_ir_w_stream_on", c_uint16, 1),
            # 1 - стрим запущен и работает, 0 - стрим не работает
            ("is_tv_n_stream_on", c_uint16, 1),
            # 1 - стрим запущен и работает, 0 - стрим не работает
            ("is_ir_n_stream_on", c_uint16, 1),
            # fill to 16 bits
            ("reserved", c_uint16, 6),
        ]

    _fields_ = [
        # Статус интерфейсной платы
        ("status", STATUS),
        # текущий выбранный тип цели (могут менять все армы)
        ("target_type", c_uint8),
        # текущий режим захвата
        ("vehicle_work_mode", c_uint8),
        # источник (CameraId) изображения для трекера
        ("tracker_source", c_int8),
    ]

    _pack_ = 1


class InfArmD(NTMPMessage):
    """
    Структура данных информационной посылки от АРМ-Д
    """

    _header_ = NTMPHeader(src_id=0x0000, dst_id=0x0000, msg_id=0xAED6A44D)  # Replace with actual msg_id if known

    _fields_ = [
        ("status", c_uint16),
        # Актуально ли значение дальности АРМ Д
        ("is_send_target_distance", c_uint8),
        ("distance", c_int32),
        ("roi_delta_azimuth", c_int32),
        ("roi_delta_elevation", c_int32),
        ("roi_angular_width", c_uint16),
        ("roi_angular_height", c_uint16),
    ]

    _pack_ = 1
