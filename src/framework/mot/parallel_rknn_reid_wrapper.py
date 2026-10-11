import logging
import numpy as np
import cv2
import time
from rknnlite.api import RKNNLite


class ParallelRKNNReIDWrapper:
    """
    ReID-экстрактор для модели с динамическим батчем [1..25].
    - Использует ОДНУ модель, скомпилированную с dynamic_input.
    - Автоматически выбирает минимальный поддерживаемый размер >= n_boxes.
    - Добивает нулями ТОЛЬКО до выбранного размера.
    - Обрезает результат до реального количества объектов.
    """
    
    # Поддерживаемые размеры батча (должны совпадать с dynamic_input при конвертации)
    _SUPPORTED_BATCHES = list(range(1, 25))  # [1, 2, 3, ..., 25]
    
    def __init__(
        self,
        rknn_path: str,
        input_size: tuple = (256, 128),
        core_mask: int = 4,
    ):
        self._logger = logging.getLogger("parallel_rknn_reid")
        self._input_h, self._input_w = input_size
        self._core_mask = core_mask
        
        self._logger.info(f"Загрузка динамической ReID модели на ядро {core_mask}...")
        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN (code {ret})")
        ret = self._rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"Failed to init runtime on core {core_mask} (code {ret})")
        
        self._logger.info(
            f"✅ ReID Wrapper готов: динамический batch [1..32], "
            f"ядро {self._core_mask}"
        )
    
    def _get_target_batch(self, n_boxes: int) -> int:
        """Возвращает минимальный поддерживаемый batch_size >= n_boxes."""
        if n_boxes <= 25:
            return n_boxes  # Модель поддерживает любой размер от 1 до 25
        return 25  # Максимальный размер
    
    def get_features(self, boxes, img) -> np.ndarray:
        t_total_start = time.perf_counter()
        
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 512), dtype=np.float32)
        
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        
        n_boxes = len(boxes)


        target_batch = self._get_target_batch(n_boxes)
        
        # Готовим кропы
        crops = []
        empty_crop = np.zeros((self._input_h, self._input_w, 3), dtype=np.uint8)
        
        for i in range(n_boxes):
            x1, y1, x2, y2 = boxes[i]
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(img.shape[1], int(round(x2))), min(img.shape[0], int(round(y2)))
            
            if x2 <= x1 or y2 <= y1:
                crops.append(empty_crop)
            else:
                crop = img[y1:y2, x1:x2]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = cv2.resize(crop, (self._input_w, self._input_h), interpolation=cv2.INTER_LINEAR)
                crops.append(crop)
        

        
        t_cpu_end = time.perf_counter()
        
        # Формируем батч и делаем инференс
        batch = np.stack(crops, axis=0).astype(np.uint8)  # (target_batch, H, W, C)
        #batch = np.transpose(batch, (0, 3, 1, 2)).astype(np.uint8)  # (target_batch, C, H, W)
        
        t_npu_start = time.perf_counter()
        

        try:
            outputs = self._rknn.inference(inputs=[batch])
            if outputs is None or len(outputs) == 0 or outputs[0] is None:
                self._logger.error(
                    f"RKNN inference returned None! "
                    f"target_batch={target_batch}, n_boxes={n_boxes}"
                )
                return np.empty((0, 512), dtype=np.float32)
            all_embeddings = outputs[0].astype(np.float32)
        except Exception as e:
            self._logger.error(f"RKNN inference failed: {e}")
            return np.empty((0, 512), dtype=np.float32)
        

        valid_embeddings = all_embeddings
        t_npu_end = time.perf_counter()
        
        # L2 нормализация
        norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        features = valid_embeddings / norms
        
        t_total_end = time.perf_counter()
        
        cpu_time_ms = (t_cpu_end - t_total_start) * 1000
        npu_time_ms = (t_npu_end - t_cpu_end) * 1000
        total_time_ms = (t_total_end - t_total_start) * 1000
        
        self._logger.info(
            f"⏱️ ReID Dynamic Batch={target_batch:2d}: Total={total_time_ms:5.1f}ms | "
            f"CPU={cpu_time_ms:4.1f}ms | NPU={npu_time_ms:4.1f}ms | "
            f"Objects={n_boxes}/{target_batch}"
        )
        
        return features
    
    def __call__(self, boxes, img):
        return self.get_features(boxes, img)
    
    def release(self):
        if hasattr(self, '_rknn') and self._rknn is not None:
            self._rknn.release()
    
    def __del__(self):
        self.release()
