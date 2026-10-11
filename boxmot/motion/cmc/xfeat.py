# XFeat CMC (Camera Motion Compensation) on RKNN NPU
# batch=2 модель: принимает пару кадров, возвращает keypoints + descriptors

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from boxmot.motion.cmc.base_cmc import BaseCMC
from boxmot.utils import logger as LOGGER


class XFeat(BaseCMC):
    """
    CMC на базе XFeat heatmap, сконвертированного в RKNN INT8 (batch=2).
    
    За один вызов NPU обрабатывает пару кадров (prev, curr),
    извлекает keypoints и дескрипторы, затем вычисляет аффинное преобразование.
    """

    def __init__(
        self,
        rknn_path: str = "/path/to/local-file",
        core_mask: int = 4,
        input_size: tuple = (360, 640),  # (H, W)
        top_k: int = 1024,
        match_threshold: float = 0.7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self._logger = logging.getLogger("xfeat_cmc")
        self._input_h, self._input_w = input_size
        self._top_k = top_k
        self._match_threshold = match_threshold
        
        # Инициализация RKNN модели
        from rknnlite.api import RKNNLite
        
        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load XFeat RKNN: {rknn_path} (code {ret})")
        
        ret = self._rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"Failed to init XFeat runtime on core {core_mask} (code {ret})")
        
        self._prev_frame = None
        self._prev_kp = None
        self._prev_desc = None
        
        LOGGER.info(
            f"XFeat CMC initialized: model={rknn_path}, "
            f"input={self._input_w}x{self._input_h}, "
            f"top_k={top_k}, core={core_mask}"
        )
    
    def apply(self, img: np.ndarray, dets: np.ndarray = None) -> np.ndarray:
        """
        Вычисляет аффинное преобразование между предыдущим и текущим кадром.
        
        Args:
            img: текущий кадр (BGR, HxWx3)
            dets: детекции (не используются в XFeat CMC)
        
        Returns:
            warp: матрица аффинного преобразования 2x3
        """
        if self._prev_frame is None:
            # Первый кадр — сохраняем и возвращаем identity
            self._prev_frame = img.copy()
            kp, desc = self._extract_features_single(img)
            self._prev_kp = kp
            self._prev_desc = desc
            return np.eye(2, 3, dtype=np.float32)
        
        # Детектируем keypoints на паре кадров (batch=2) — ОДИН вызов NPU
        kp_prev, kp_curr, desc_prev, desc_curr = self._detect_pair(
            self._prev_frame, img
        )
        
        # Сопоставляем дескрипторы
        idx1, idx2 = self._match_descriptors(desc_prev, desc_curr)
        
        if len(idx1) < 4:
            LOGGER.warning(
                f"XFeat CMC: недостаточно матчей ({len(idx1)}), "
                f"возвращаю identity warp"
            )
            self._update_state(img, kp_curr, desc_curr)
            return np.eye(2, 3, dtype=np.float32)
        
        # Берём координаты сопоставленных keypoints
        pts1 = kp_prev[idx1].copy()
        pts2 = kp_curr[idx2].copy()
        
        # Масштабируем координаты обратно к оригинальному размеру кадра
        orig_h, orig_w = img.shape[:2]
        scale_x = orig_w / self._input_w
        scale_y = orig_h / self._input_h
        
        pts1[:, 0] *= scale_x
        pts1[:, 1] *= scale_y
        pts2[:, 0] *= scale_x
        pts2[:, 1] *= scale_y
        
        # Вычисляем аффинное преобразование через RANSAC
        warp, inliers = cv2.estimateAffinePartial2D(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0
        )
        
        if warp is None:
            LOGGER.warning("XFeat CMC: estimateAffinePartial2D вернул None")
            self._update_state(img, kp_curr, desc_curr)
            return np.eye(2, 3, dtype=np.float32)
        
        n_inliers = int(np.sum(inliers)) if inliers is not None else 0
        LOGGER.debug(f"XFeat CMC: {len(idx1)} matches, {n_inliers} inliers")
        
        self._update_state(img, kp_curr, desc_curr)
        return warp.astype(np.float32)
    
    def _preprocess_frame(self, img_bgr: np.ndarray) -> np.ndarray:
        """BGR → RGB → resize → NCHW uint8"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(
            img_rgb,
            (self._input_w, self._input_h),
            interpolation=cv2.INTER_LINEAR
        )
        return np.transpose(img_rgb, (2, 0, 1))  # NHWC → NCHW
    
    def _detect_pair(self, img_prev: np.ndarray, img_curr: np.ndarray):
        """
        Один вызов NPU для пары кадров (batch=2).
        Возвращает: kp_prev, kp_curr, desc_prev, desc_curr
        """
        frame_prev = self._preprocess_frame(img_prev)
        frame_curr = self._preprocess_frame(img_curr)
        batch = np.stack([frame_prev, frame_curr], axis=0).astype(np.uint8)
        
        outputs = self._rknn.inference(inputs=[batch])
        return self._parse_outputs(outputs)
    
    def _parse_outputs(self, outputs):
        """
        Парсит выходы модели XFeat heatmap.
        
        ⚠️ АДАПТИРОВАТЬ ПОД СТРУКТУРУ ТВОЕЙ МОДЕЛИ!
        Сначала запусти тестовый скрипт (см. ниже), чтобы увидеть реальные shapes.
        """
        # 🔍 ВРЕМЕННАЯ ЗАГЛУШКА — логируем структуру выходов
        if not hasattr(self, '_logged_outputs'):
            LOGGER.info(f"XFeat outputs: {len(outputs)} tensors")
            for i, o in enumerate(outputs):
                LOGGER.info(
                    f"  output[{i}]: shape={o.shape}, dtype={o.dtype}, "
                    f"range=[{o.min():.4f}, {o.max():.4f}]"
                )
            self._logged_outputs = True
        
        # Стандартная структура XFeat heatmap:
        # Вариант A (2 выхода): [heatmaps (2,H,W), descriptors (2,N,64)]
        # Вариант B (4 выхода): [heatmap_prev, heatmap_curr, desc_prev, desc_curr]
        
        if len(outputs) == 2:
            heatmaps = outputs[0]   # (2, H, W) или (H, W)
            descriptors = outputs[1]  # (2, N, 64) или (N, 64)
            
            if heatmaps.ndim == 3:
                kp_prev = self._heatmap_to_keypoints(heatmaps[0])
                kp_curr = self._heatmap_to_keypoints(heatmaps[1])
            else:
                kp_prev = self._heatmap_to_keypoints(heatmaps)
                kp_curr = kp_prev.copy()
            
            if descriptors.ndim == 3:
                desc_prev = descriptors[0]
                desc_curr = descriptors[1]
            else:
                desc_prev = descriptors
                desc_curr = descriptors.copy()
            
            return kp_prev, kp_curr, desc_prev, desc_curr
        
        elif len(outputs) == 4:
            kp_prev = self._heatmap_to_keypoints(outputs[0].squeeze())
            kp_curr = self._heatmap_to_keypoints(outputs[1].squeeze())
            desc_prev = outputs[2].squeeze()
            desc_curr = outputs[3].squeeze()
            return kp_prev, kp_curr, desc_prev, desc_curr
        
        else:
            raise NotImplementedError(
                f"Неизвестная структура выходов XFeat: {len(outputs)} тензоров. "
                f"Адаптируй _parse_outputs под свою модель."
            )
    
    def _heatmap_to_keypoints(self, heatmap: np.ndarray) -> np.ndarray:
        """Превращает heatmap в top_k координат keypoints."""
        flat = heatmap.flatten()
        k = min(self._top_k, len(flat))
        top_indices = np.argpartition(flat, -k)[-k:]
        top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
        
        ys = top_indices // heatmap.shape[1]
        xs = top_indices % heatmap.shape[1]
        
        return np.stack([xs, ys], axis=1).astype(np.float32)
    
    def _match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray):
        """Brute-force matching с L2-нормализацией."""
        # Нормализация
        desc1 = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8)
        desc2 = desc2 / (np.linalg.norm(desc2, axis=1, keepdims=True) + 1e-8)
        
        # Матрица сходства (cosine similarity)
        similarity = desc1 @ desc2.T
        
        # Лучший match для каждого дескриптора в desc1
        best_matches = np.argmax(similarity, axis=1)
        best_scores = similarity[np.arange(len(desc1)), best_matches]
        
        # Фильтр по порогу
        mask = best_scores > self._match_threshold
        idx1 = np.where(mask)[0]
        idx2 = best_matches[mask]
        
        return idx1, idx2
    
    def _extract_features_single(self, img: np.ndarray):
        """Извлечение фич на одном кадре (дублируем кадр для batch=2)."""
        kp_prev, kp_curr, desc_prev, desc_curr = self._detect_pair(img, img)
        return kp_curr, desc_curr
    
    def _update_state(self, img, kp, desc):
        """Обновление состояния для следующего кадра."""
        self._prev_frame = img.copy()
        self._prev_kp = kp
        self._prev_desc = desc
    
    def __del__(self):
        if hasattr(self, '_rknn') and self._rknn is not None:
            self._rknn.release()
            self._rknn = None
