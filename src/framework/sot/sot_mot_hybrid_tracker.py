from typing import Optional

import cv2 as cv
import numpy as np

from src.framework.core.box import Box, BoxTopLeft
from src.framework.sot.single_object_tracker import ISingleObjectTracker


class SotMotHybridTracker:
    """Wraps the tracker for evaluation and running purposes."""

    def __init__(self, sot_tracker: ISingleObjectTracker) -> None:
        # assert run_id is None or isinstance(run_id, int)

        # self.name = name
        # self.parameter_name = parameter_name
        # self.dataset_name = dataset_name
        # self.run_id = run_id
        # self.display_name = display_name

        self._tracker = sot_tracker

        # 1. IoU порог
        self.iou_threshold = 0.6
        # 2. Hysteresis
        self.max_consecutive_failures = 7
        self.consecutive_failures = 0
        # 3. Grace Period
        self.grace_period_frames = 10
        self.frames_since_init = 0
        # 4. Минимальная длительность трека
        self.min_track_duration = 15
        self.frames_tracked = 0
        # 5. Порог уверенности MOT
        self.mot_confidence_threshold = 0.5
        # 6. Сглаживание
        self.smoothing_alpha = 0.7
        self.smoothed_bbox: Optional[Box] = None
        # 7. MOT TRACK ID
        self.active_track_id = -1
        self.id_mismatch_count = 0
        self.max_id_mismatches = 3
        self.target_lost_alert = False
        self.mot_lost_frames = 0

        self.frame_w = 1280
        self.frame_h = 1024

        # self.tracker_instance = None  # Экземпляр трекера STARK
        self.is_tracking = False  # Флаг активного трекинга
        self.params = None  # Кэшированные параметры

    # def create_tracker(self, params):
    #     tracker = self.tracker_class(params, self.dataset_name)
    #     return tracker
    #
    # def get_parameters(self):
    #     """Get parameters."""
    #     if self.params is None:
    #         param_module = importlib.import_module('stark.test.parameter.{}'.format(self.name))
    #         self.params = param_module.parameters(self.parameter_name)
    #         self.params.tracker_name = self.name
    #         self.params.param_name = self.parameter_name
    #     return self.params

    # === МЕТОДЫ ВАЛИДАЦИИ (Без изменений) ===

    # def _calculate_iou(self, box1, box2):
    #     if not box1 or not box2:
    #         return 0.0
    #     x1, y1, w1, h1 = box1
    #     x2, y2, w2, h2 = box2
    #
    #     xi1, yi1 = max(x1, x2), max(y1, y2)
    #     xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    #
    #     inter_w = max(0, xi2 - xi1)
    #     inter_h = max(0, yi2 - yi1)
    #     inter_area = inter_w * inter_h
    #
    #     area1 = w1 * h1
    #     area2 = w2 * h2
    #     union_area = area1 + area2 - inter_area
    #
    #     return inter_area / (union_area + 1e-9)

    # def _check_bbox_sanity(self, bbox):
    #     if not bbox or len(bbox) != 4:
    #         return False
    #     x, y, w, h = bbox
    #     area = w * h
    #     frame_area = self.frame_w * self.frame_h
    #     if x < -w * 2 or y < -h * 2 or x > self.frame_w + w * 2 or y > self.frame_h + h * 2:
    #         return False
    #     if area > frame_area * 0.55:
    #         return False
    #     return True

    def _check_bbox_sanity(self, bbox: Box):
        x, y, w, h = bbox.as_top_left().as_tuple()
        area = bbox.area()
        frame_area = self.frame_w * self.frame_h
        if (
            x < -w * 2
            or y < -h * 2
            or x > self.frame_w + w * 2
            or y > self.frame_h + h * 2
        ):
            return False
        if area > frame_area * 0.55:
            return False
        return True

    def _validate_with_mot(self, stark_bbox: Box, mot_data):
        if isinstance(mot_data, dict):
            mot_bbox = mot_data.get("bbox", None)
            mot_conf = mot_data.get("conf", 1.0)
            mot_id = mot_data.get("track_id", -1)
            mot_lost = mot_data.get("lost", False)
            self.mot_lost_frames = mot_data.get("lost_frames", 0)
        else:
            mot_bbox = mot_data if mot_data else []
            mot_conf = 1.0
            mot_id = -1
            mot_lost = False
            self.mot_lost_frames = 0

        if self.frames_since_init < self.grace_period_frames:
            self.consecutive_failures = 0
            self.id_mismatch_count = 0
            return True

        if not mot_bbox or mot_lost:
            self.consecutive_failures = 0
            self.id_mismatch_count = 0
            return True

        if mot_conf < self.mot_confidence_threshold:
            self.consecutive_failures = 0
            self.id_mismatch_count = 0
            return True

        id_match = True
        if self.active_track_id != -1 and mot_id != -1:
            if self.active_track_id != mot_id:
                id_match = False
                self.id_mismatch_count += 1
                print(
                    f"STARK: ID не совпадает! Активный: {self.active_track_id}, MOT: {mot_id}"
                )
            else:
                self.id_mismatch_count = 0
        elif mot_id != -1:
            self.active_track_id = mot_id

        # iou = self._calculate_iou(stark_bbox, mot_bbox)
        iou = Box.iou(stark_bbox, mot_bbox)

        if id_match and iou >= self.iou_threshold:
            self.consecutive_failures = 0
            return True

        if id_match and iou < self.iou_threshold:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                print(
                    f"STARK: IoU низкий ({iou:.2f}) в течение {self.consecutive_failures} кадров. Сброс."
                )
                self.target_lost_alert = True
                return False
            else:
                print(
                    f"STARK: IoU низкий ({iou:.2f}), кадр {self.consecutive_failures}/{self.max_consecutive_failures}. Ждем..."
                )
                return True

        if not id_match and iou >= self.iou_threshold:
            print(
                f"✅ STARK: ID изменился, но IoU высокий. Обновляем ID: {self.active_track_id} → {mot_id}"
            )
            self.active_track_id = mot_id
            self.consecutive_failures = 0
            self.id_mismatch_count = 0
            return True

        if not id_match and iou < self.iou_threshold:
            if self.id_mismatch_count >= self.max_id_mismatches:
                print(
                    f"❌ STARK: ID не совпадает + IoU низкий ({iou:.2f}). СБРОС ТРЕКА."
                )
                self.target_lost_alert = True
                return False
            else:
                print(
                    f"STARK: ID не совпадает, но ждем подтверждения ({self.id_mismatch_count}/{self.max_id_mismatches})"
                )
                self.consecutive_failures += 1
                return True

        self.consecutive_failures = 0
        return True

    def _smooth_bbox(self, bbox):
        if self.smoothed_bbox is None:
            self.smoothed_bbox = bbox
        else:
            self.smoothed_bbox = self.smoothed_bbox.interpolate(
                bbox, self.smoothing_alpha
            )
        return self.smoothed_bbox

    # def _extract_bbox_from_data(self, data):
    #     if isinstance(data, dict):
    #         return data.get("bbox", [])
    #     elif isinstance(data, list):
    #         return data if len(data) == 4 else []
    #     else:
    #         return []

    @staticmethod
    def _build_init_info(box) -> Box:
        return BoxTopLeft(*box)

    def reset_tracking_state(self):
        """Сброс состояния трекинга при потере цели."""
        self.smoothed_bbox = None
        self.active_track_id = -1
        self.target_lost_alert = True
        self.consecutive_failures = 0
        self.id_mismatch_count = 0
        self.frames_tracked = 0
        self.frames_since_init = 0
        self.is_tracking = False

    def track(self, frame: np.ndarray, mot_data: dict):
        """
        Обрабатывает один кадр. Вызывается из внешнего цикла/потока.
        :param frame: Кадр изображения (numpy array, BGR от камеры)
        :param mot_data: Словарь от MOT трекера {'bbox': [x,y,w,h], 'track_id': int, 'conf': float, 'lost': bool, 'frame_time_ns': int}
        :return: dict {'bbox': [x,y,w,h], 'frame': displayed_frame, 'status': str, 'is_tracking': bool}
        """
        self.frame_w, self.frame_h = frame.shape[1], frame.shape[0]

        # 2. Извлечение данных MOT
        mot_bbox = mot_data.get("bbox")
        mot_id = mot_data.get("track_id", -1) if isinstance(mot_data, dict) else -1
        mot_conf = mot_data.get("conf", 1.0) if isinstance(mot_data, dict) else 1.0
        mot_lost = mot_data.get("lost", False) if isinstance(mot_data, dict) else False
        mot_lost_frames = (
            mot_data.get("lost_frames", 0) if isinstance(mot_data, dict) else 0
        )

        # 3. Подготовка кадров (BGR для отрисовки, RGB для STARK)
        # frame_disp = frame.copy()

        frame_rgb = (
            cv.cvtColor(frame, cv.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
        )

        result_bbox = None
        status_text = "Waiting..."

        # 4. Логика состояния
        if not self.is_tracking:
            # Попытка инициализации, если есть валидный бокс от MOT
            if mot_bbox and not mot_lost:
                print(f"✅ STARK: Инициализация цели Box={mot_bbox}, ID={mot_id}")
                init_info = mot_bbox
                try:
                    print("INFO_STARK_BOX_MOT", init_info)
                    self._tracker.initialize(frame_rgb, mot_bbox)
                    self.is_tracking = True
                    self.active_track_id = mot_id
                    self.frames_since_init = 0
                    self.frames_tracked = 0
                    self.consecutive_failures = 0
                    self.id_mismatch_count = 0
                    self.smoothed_bbox = None
                    self.target_lost_alert = False

                    result_bbox = mot_bbox
                    status_text = f"Init ID:{mot_id}"

                    # Отрисовка инициализации

                except Exception as e:
                    print(f"STARK: Ошибка инициализации: {e}")
                    status_text = f"Init Error: {e}"
            else:
                status_text = "Waiting for MOT..."
        else:
            # Режим трекинга
            try:
                # out = self.tracker_instance.track(frame_rgb)
                out = self._tracker.track(frame_rgb)
                stark_bbox = out.box

                if stark_bbox:
                    self.frames_since_init += 1
                    self.frames_tracked += 1

                    # Проверка адекватности бокса
                    if not self._check_bbox_sanity(stark_bbox):
                        print("STARK: Бокс невалиден. Сброс трека.")
                        self.reset_tracking_state()
                        status_text = "Reset: Sanity Fail"
                    else:
                        # Валидация с MOT
                        mot_info = {
                            "bbox": mot_bbox,
                            "conf": mot_conf,
                            "track_id": mot_id,
                            "lost": mot_lost,
                            "lost_frames": mot_lost_frames,
                        }
                        # Валидация включается после min_track_duration
                        if self.frames_tracked >= self.min_track_duration:
                            if not self._validate_with_mot(stark_bbox, mot_info):
                                print(
                                    "STARK: Валидация с MOT не пройдена. Сброс трека."
                                )
                                self.reset_tracking_state()
                                status_text = "Reset: MOT Validation"
                            else:
                                # Успешная валидация
                                # smoothed_state = self._smooth_bbox(stark_bbox)
                                # state = [int(s) for s in smoothed_state]
                                # result_bbox = state
                                result_bbox = self._smooth_bbox(stark_bbox)
                                alert_text = (
                                    " LOST!"
                                    if self.target_lost_alert or mot_lost
                                    else ""
                                )
                                status_text = f"ID:{self.active_track_id} | Fail:{self.consecutive_failures}/{self.max_consecutive_failures} | MOT Lost:{mot_lost_frames}{alert_text}"
                        else:
                            # Период прогрева (Grace Period для валидации)
                            # smoothed_state = self._smooth_bbox(stark_bbox)
                            # state = [int(s) for s in smoothed_state]
                            # result_bbox = state
                            result_bbox = self._smooth_bbox(stark_bbox)
                            status_text = (
                                f"Tracking (Warmup) | Frames: {self.frames_tracked}"
                            )
                else:
                    # Трекер потерял цель внутри себя
                    print("STARK: Tracker lost target internally.")
                    self.reset_tracking_state()
                    status_text = "Reset: Tracker Lost"
            except Exception as e:
                print(f"❌ STARK: Ошибка трекинга: {e}")
                self.reset_tracking_state()
                status_text = f"Error: {e}"

        return {
            "bbox": result_bbox,
            "is_tracking": self.is_tracking,
            "status": status_text,
        }
