import logging
import numpy as np
import cv2
from rknnlite.api import RKNNLite


class XFeatRKNN:
    """
    Обёртка для XFeat heatmap модели в формате RKNN.
    - Вход: batch=2 кадра (prev, curr), размер 360x640
    - Выход: keypoints и/или дескрипторы для обоих кадров
    """
    
    def __init__(
            self,
            rknn_path: str,
            input_size: tuple = (360, 640),  # (H, W)
            core_mask: int = 4,
            top_k: int = 1024,
    ):
        self._logger = logging.getLogger("xfeat_rknn")
        self._input_h, self._input_w = input_size
        self._core_mask = core_mask
        self._top_k = top_k
        
        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load XFeat RKNN: {rknn_path} (code {ret})")
        
        ret = self._rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"Failed to init XFeat runtime on core {core_mask} (code {ret})")
        
        self._logger.info(
            "Loaded XFeat RKNN '%s' (input %dx%d, batch=2) on core %d",
            rknn_path, self._input_w, self._input_h, core_mask,
        )
    
    def _preprocess_frame(self, img_bgr: np.ndarray) -> np.ndarray:
        """Препроцессинг одного кадра: BGR → RGB → resize → NCHW uint8"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(
            img_rgb, 
            (self._input_w, self._input_h), 
            interpolation=cv2.INTER_LINEAR
        )
        # NHWC → NCHW
        img_nchw = np.transpose(img_rgb, (2, 0, 1))
        return img_nchw
    
    def detect_pair(self, img_prev: np.ndarray, img_curr: np.ndarray):
        """
        Детектирует keypoints на паре кадров за ОДИН вызов NPU.
        
        Returns:
            kp_prev, kp_curr: keypoints для каждого кадра 
                              (список координат или heatmap)
            desc_prev, desc_curr: дескрипторы (если модель их выдаёт)
        """
        # Формируем батч из 2 кадров
        frame_prev = self._preprocess_frame(img_prev)
        frame_curr = self._preprocess_frame(img_curr)
        batch = np.stack([frame_prev, frame_curr], axis=0).astype(np.uint8)  # (2, 3, 360, 640)
        
        # Один вызов NPU
        outputs = self._rknn.inference(inputs=[batch])
        
        # Парсинг выходов (зависит от конкретной модели XFeat)
        # Обычно XFeat heatmap выдаёт: [heatmap_prev, heatmap_curr, desc_prev, desc_curr]
        # или [heatmaps (2, H, W), descriptors (2, N, 64)]
        return self._parse_outputs(outputs)
    
    def _parse_outputs(self, outputs):
        """
        Парсит выходы модели. АДАПТИРОВАТЬ ПОД ТВОЮ МОДЕЛЬ!
        
        Типичные варианты выходов XFeat heatmap:
        - Вариант A: 2 выхода [heatmaps (2,H,W), descriptors (2,N,64)]
        - Вариант B: 4 выхода [heatmap_prev, heatmap_curr, desc_prev, desc_curr]
        """
        # 🔧 ЗАМЕНИ ЭТОТ КОД ПОД РЕАЛЬНУЮ СТРУКТУРУ ВЫХОДОВ ТВОЕЙ МОДЕЛИ
        # Для начала просто логируем, чтобы понять структуру
        
        self._logger.info(f"XFeat outputs: {len(outputs)} tensors")
        for i, o in enumerate(outputs):
            self._logger.info(f"  output[{i}].shape = {o.shape}, dtype = {o.dtype}")
        
        # Заглушка — нужно адаптировать под реальную модель
        if len(outputs) == 2:
            # heatmaps + descriptors
            heatmaps = outputs[0]  # (2, H, W)
            descriptors = outputs[1]  # (2, N, 64)
            
            kp_prev = self._heatmap_to_keypoints(heatmaps[0])
            kp_curr = self._heatmap_to_keypoints(heatmaps[1])
            
            desc_prev = descriptors[0]
            desc_curr = descriptors[1]
            
            return kp_prev, kp_curr, desc_prev, desc_curr
        else:
            raise NotImplementedError(
                f"Неизвестная структура выходов XFeat: {len(outputs)} тензоров. "
                f"Адаптируй _parse_outputs под свою модель."
            )
    
    def _heatmap_to_keypoints(self, heatmap: np.ndarray, top_k: int = None):
        """
        Превращает heatmap в список координат keypoints.
        Берём top_k точек с максимальной активацией.
        """
        if top_k is None:
            top_k = self._top_k
        
        # Flatten и поиск top_k индексов
        flat = heatmap.flatten()
        top_indices = np.argpartition(flat, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
        
        # Превращаем индексы в координаты (y, x)
        ys = top_indices // heatmap.shape[1]
        xs = top_indices % heatmap.shape[1]
        
        keypoints = np.stack([xs, ys], axis=1).astype(np.float32)
        
        # Масштабируем обратно к оригинальному размеру кадра, если нужно
        # (зависит от того, как модель обучена)
        
        return keypoints
    
    def match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray):
        """
        Сопоставление дескрипторов между двумя кадрами.
        Возвращает пары индексов (idx1, idx2).
        """
        # Нормализация дескрипторов (L2)
        desc1 = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8)
        desc2 = desc2 / (np.linalg.norm(desc2, axis=1, keepdims=True) + 1e-8)
        
        # Brute-force matching через матрицу расстояний
        # dist[i,j] = 1 - dot(desc1[i], desc2[j])
        similarity = desc1 @ desc2.T  # (N1, N2)
        
        # Для каждого дескриптора в desc1 ищем лучший match в desc2
        best_matches = np.argmax(similarity, axis=1)
        best_scores = similarity[np.arange(len(desc1)), best_matches]
        
        # Фильтр по порогу сходства
        mask = best_scores > 0.7  # Порог настраивается
        idx1 = np.where(mask)[0]
        idx2 = best_matches[mask]
        
        return idx1, idx2
    
    def release(self):
        if hasattr(self, '_rknn') and self._rknn is not None:
            self._rknn.release()
            self._rknn = None
    
    def __del__(self):
        self.release()
