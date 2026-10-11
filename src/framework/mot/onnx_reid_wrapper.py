import logging
import numpy as np
import cv2
import onnxruntime as ort

class ONNXReIDWrapper:
    def __init__(self, onnx_path: str, input_size: tuple = (256, 128), **kwargs):
        self._logger = logging.getLogger("onnx_reid")
        self._input_h, self._input_w = input_size
        
        # Используем CPU. Если вдруг захочешь CUDA, добавь 'CUDAExecutionProvider'
        providers = ['CPUExecutionProvider']
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        
        self._logger.info(
            f"Loaded ONNX ReID model '{onnx_path}' (input {self._input_w}x{self._input_h}) on CPU"
        )

    def get_features(self, boxes, img):
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 0), dtype=np.float32)

        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)

        n_boxes = boxes.shape[0]
        crops = []

        # 1. Подготовка кропов (идентично RKNN версии)
        for i in range(n_boxes):
            x1, y1, x2, y2 = boxes[i]
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(img.shape[1], int(round(x2)))
            y2 = min(img.shape[0], int(round(y2)))

            if x2 <= x1 or y2 <= y1:
                crop = np.zeros((self._input_h, self._input_w, 3), dtype=np.uint8)
            else:
                crop = img[y1:y2, x1:x2]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = cv2.resize(crop, (self._input_w, self._input_h), interpolation=cv2.INTER_LINEAR)

            # Нормализация для ONNX (обычно требуется float32 в диапазоне 0..1)
            crop = crop.astype(np.float32) / 255.0
            crops.append(crop)

        # 2. Стек в батч (N, H, W, C) -> (N, C, H, W) для ONNX
        batch = np.stack(crops, axis=0).transpose(0, 3, 1, 2)

        # 3. Инференс
        outputs = self._session.run(None, {self._input_name: batch})
        features = outputs[0].astype(np.float32)

        # 4. L2-нормализация (обязательно для ReID)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        features = features / norms

        return features

    def __call__(self, boxes, img):
        return self.get_features(boxes, img)
