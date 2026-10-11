# framework/mot/rknn_reid_wrapper.py
import logging
import numpy as np
import cv2
from rknnlite.api import RKNNLite


class RKNNReIDWrapper:
    def __init__(
            self,
            rknn_path: str,
            input_size: tuple = (256, 128),
            core_mask: int = 4,
            normalize_input: bool = False,
    ):
        self._logger = logging.getLogger("rknn_reid")
        self._input_h, self._input_w = input_size
        self._normalize_input = normalize_input
        self._cor_mask = core_mask
        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN ReID model from {rknn_path} (code {ret})")

        ret = self._rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN ReID runtime on core {core_mask} (code {ret})")

        self._logger.info(
            "Loaded RKNN ReID model '%s' (input %dx%d) on NPU core %d",
            rknn_path, self._input_w, self._input_h, core_mask,
        )

    def get_features(self, boxes, img):
        """
        Args:
            boxes: np.ndarray формы (N, 4) — [x1, y1, x2, y2]
            img: np.ndarray формы (H, W, 3) — BGR изображение
        
        Returns:
            np.ndarray формы (N, D) — эмбеддинги
        """
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 0), dtype=np.float32)

        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)

        n_boxes = boxes.shape[0]
        all_embeddings = []


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

            if self._normalize_input:
                crop = crop.astype(np.float32) / 255.0

            # Добавляем batch dimension: (H, W, 3) -> (1, H, W, 3)
            batch = np.expand_dims(crop, axis=0)
            if batch.dtype != np.uint8 and not self._normalize_input:
                batch = batch.astype(np.uint8)


            # Инференс
            outputs = self._rknn.inference(inputs=[batch])
            embedding = outputs[0]
            
            # Flatten если нужно
            if embedding.ndim > 2:
                embedding = embedding.reshape(1, -1)
            elif embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            all_embeddings.append(embedding)

        # Stack all embeddings: (N, D)
        features = np.vstack(all_embeddings).astype(np.float32)

        # L2-нормализация
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        features = features / norms

        return features

    def __call__(self, boxes, img):
        return self.get_features(boxes, img)

    def release(self):
        if hasattr(self, "_rknn") and self._rknn is not None:
            self._rknn.release()
            self._rknn = None

    def __del__(self):
        self.release()
