import logging
import numpy as np
import torch
import torchvision.ops as ops
import cv2
from typing import List
from rknnlite.api import RKNNLite


class RKNNBoxes:
    """Имитирует ultralytics.engine.results.Boxes"""
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.cls = torch.tensor(cls, dtype=torch.float32)


class RKNNResult:
    """Имитирует ultralytics.engine.results.Results"""
    def __init__(self, boxes: RKNNBoxes):
        self.boxes = boxes


def dfl_numpy(position: np.ndarray) -> np.ndarray:
    """
    Чистый NumPy аналог DFL (Distribution Focal Loss).
    position shape: (1, 64, H, W)
    return shape: (1, 4, H, W)
    """
    n, c, h, w = position.shape
    mc = c // 4  # 16 bins
    
    # Reshape to (1, 4, 16, H, W)
    y = position.reshape(n, 4, mc, h, w)
    
    # Softmax along axis 2 (the 16 bins) с численной стабильностью
    y_max = np.max(y, axis=2, keepdims=True)
    y_exp = np.exp(y - y_max)
    y_softmax = y_exp / np.sum(y_exp, axis=2, keepdims=True)
    
    # Expectation: sum(prob * bin_index)
    bin_indices = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    out = np.sum(y_softmax * bin_indices, axis=2)  # Shape: (1, 4, H, W)
    
    return out


def process_branch_fast(pos_tensor: np.ndarray, cls_tensor: np.ndarray, 
                        obj_tensor: np.ndarray, img_size=(640, 640)):
    """
    Парсинг одной ветки YOLO11 (как в скрипте коллеги), но с учетом objectness head.
    """
    grid_h, grid_w = pos_tensor.shape[2:4]
    
    # Точное воспроизведение создания grid как у коллеги (indexing='xy' по умолчанию)
    col, row = np.meshgrid(
        np.arange(0, grid_w, dtype=np.float32), 
        np.arange(0, grid_h, dtype=np.float32)
    )
    grid = np.stack([col, row], axis=0)[None, ...]  # Shape: (1, 2, grid_h, grid_w)
    
    stride_x = img_size[0] / grid_w
    stride_y = img_size[1] / grid_h
    stride = np.array([stride_x, stride_y], dtype=np.float32).reshape(1, 2, 1, 1)

    # DFL декодирование
    pos_decoded = dfl_numpy(pos_tensor)

    # Формула координат как у коллеги
    x1y1 = (grid + 0.5 - pos_decoded[:, 0:2, :, :]) * stride
    x2y2 = (grid + 0.5 + pos_decoded[:, 2:4, :, :]) * stride
    
    boxes = np.concatenate([x1y1, x2y2], axis=1)  # Shape: (1, 4, H, W)

    # Flatten
    boxes_flat = boxes.transpose(0, 2, 3, 1).reshape(-1, 4)
    cls_flat = cls_tensor.transpose(0, 2, 3, 1).reshape(-1, cls_tensor.shape[1])  # [H*W, num_classes]
    obj_flat = obj_tensor.transpose(0, 2, 3, 1).reshape(-1, 1)                    # [H*W, 1]

    return boxes_flat, cls_flat, obj_flat


class RKNNYoloModel:
    """
    Прямая замена ultralytics.YOLO для RKNN.
    Парсинг выходных данных адаптирован под официальный пример rknn_model_zoo для YOLO11.
    """
    def __init__(
        self,
        rknn_path: str,
        task: str = "detect",
        verbose: bool = False,
        conf: float = 0.25,
        imgsz: int = 640,
        core_mask: int = 1,
    ):
        self._logger = logging.getLogger("rknn_yolo")
        self._conf = conf
        self._imgsz = imgsz
        self._cr_msk = core_mask
        self._first_inference_done = False
        self._inference_count = 0
        self._nms_iou_thresh = 0.5

        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {rknn_path}")

        ret = self._rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime")

        self._logger.info(
            "Loaded RKNN model %s on core %d (imgsz=%d, conf=%.2f)",
            rknn_path, core_mask, imgsz, conf,
        )

    def __call__(self, batch_imgs: List[np.ndarray], **kwargs) -> List[RKNNResult]:
        results = []

        for img in batch_imgs:
            orig_h, orig_w = img.shape[:2]


            img_rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)

            # 2. Ресайз до размера модели
            if orig_h != self._imgsz or orig_w != self._imgsz:
                img_rgb = cv2.resize(
                    img_rgb,
                    (self._imgsz, self._imgsz),
                    interpolation=cv2.INTER_LINEAR,
                )


            img_batch = np.expand_dims(img_rgb, axis=0)


            if img_batch.dtype != np.uint8:
                img_batch = img_batch.astype(np.uint8)

            # 4. Inference
            try:
                outputs = self._rknn.inference(inputs=[img_batch])
            except Exception as e:
                self._logger.error(f"RKNN inference failed: {e}")
                results.append(RKNNResult(RKNNBoxes(np.empty((0, 4)), np.empty(0), np.empty(0))))
                continue

            if outputs is None or len(outputs) == 0:
                results.append(RKNNResult(RKNNBoxes(np.empty((0, 4)), np.empty(0), np.empty(0))))
                continue



            # 6. Postprocessing
            result = self._postprocess(outputs, orig_h, orig_w)
            results.append(result)
            self._inference_count += 1

        return results

    def _postprocess(self, outputs: List[np.ndarray], orig_h: int, orig_w: int) -> RKNNResult:
        """
        Обработка выходов YOLO11 с 9 головами (3 масштаба × 3 головы: bbox, cls, obj).
        """
        all_boxes, all_scores, all_classes = [], [], []
        
        # Для 9 выходов: pair_per_branch = 3
        # Берем outputs[0,3,6] = bbox, outputs[1,4,7] = cls, outputs[2,5,8] = obj
        pair_per_branch = len(outputs) // 3

        for i in range(3):
            pos_tensor = outputs[i * pair_per_branch]
            cls_tensor = outputs[i * pair_per_branch + 1]
            obj_tensor = outputs[i * pair_per_branch + 2]

            b_flat, c_flat, o_flat = process_branch_fast(
                pos_tensor, cls_tensor, obj_tensor, 
                img_size=(self._imgsz, self._imgsz)
            )
            

            if c_flat.max() > 1.0:
                cls_probs = 1.0 / (1.0 + np.exp(-np.clip(c_flat, -500, 500)))
            else:
                cls_probs = c_flat

            if o_flat.max() > 1.0:
                obj_probs = 1.0 / (1.0 + np.exp(-np.clip(o_flat, -500, 500)))
            else:
                obj_probs = o_flat

            # Финальная уверенность = вероятность класса * вероятность объекта
            final_scores = cls_probs * obj_probs
            
            scores = final_scores.max(axis=1)
            classes = final_scores.argmax(axis=1)

            all_boxes.append(b_flat)
            all_scores.append(scores)
            all_classes.append(classes)

        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_classes = np.concatenate(all_classes, axis=0)

        # Отладочный вывод
        if self._inference_count % 10 == 0:
            self._logger.info(
                f"[DEBUG] Before filter: {len(all_boxes)} boxes, "
                f"score range: [{all_scores.min():.4f}, {all_scores.max():.4f}], "
                f"mean: {all_scores.mean():.4f}"
            )

        # 1. Фильтр по confidence
        mask = all_scores >= self._conf
        if not np.any(mask):
            if self._inference_count % 10 == 0:
                self._logger.warning(f"No boxes passed conf filter {self._conf}")
            return RKNNResult(RKNNBoxes(np.empty((0, 4)), np.empty(0), np.empty(0)))

        filtered_boxes = all_boxes[mask]
        filtered_scores = all_scores[mask]
        filtered_classes = all_classes[mask]

        if self._inference_count % 10 == 0:
            self._logger.info(f"[DEBUG] After conf filter: {len(filtered_boxes)} boxes")

        # 2. === NMS (Non-Maximum Suppression) ===
        boxes_tensor = torch.from_numpy(filtered_boxes).float()
        scores_tensor = torch.from_numpy(filtered_scores).float()
        
        # Multi-class NMS трюк
        max_coordinate = boxes_tensor.max()
        offsets = torch.from_numpy(filtered_classes).long() * (int(max_coordinate) + 1)
        boxes_for_nms = boxes_tensor + offsets[:, None]
        print("POST_PROCESSS")
        
        keep_indices = ops.nms(boxes_for_nms, scores_tensor, self._nms_iou_thresh).numpy()
        
        final_boxes = filtered_boxes[keep_indices]
        final_scores = filtered_scores[keep_indices]
        final_classes = filtered_classes[keep_indices]
        
        if self._inference_count % 10 == 0:
            self._logger.info(f"[DEBUG] After NMS: {len(final_boxes)} boxes")
            if len(final_scores) > 0:
                self._logger.info(
                    f"[DEBUG] Final score range: [{final_scores.min():.4f}, {final_scores.max():.4f}]"
                )

        # 3. Масштабирование от imgsz обратно к orig_size
        scale_x = orig_w / self._imgsz
        scale_y = orig_h / self._imgsz
        final_boxes[:, [0, 2]] *= scale_x
        final_boxes[:, [1, 3]] *= scale_y

        # 4. ОБРЕЗКА до границ изображения
        final_boxes[:, [0, 2]] = np.clip(final_boxes[:, [0, 2]], 0, orig_w)
        final_boxes[:, [1, 3]] = np.clip(final_boxes[:, [1, 3]], 0, orig_h)
        
        # 5. Удаляем ббоксы с нулевой площадью после обрезки
        valid_mask = (final_boxes[:, 2] > final_boxes[:, 0]) & (final_boxes[:, 3] > final_boxes[:, 1])
        final_boxes = final_boxes[valid_mask]
        final_scores = final_scores[valid_mask]
        final_classes = final_classes[valid_mask]

        return RKNNResult(RKNNBoxes(final_boxes, final_scores, final_classes))

    def __del__(self):
        if hasattr(self, '_rknn'):
            self._rknn.release()
