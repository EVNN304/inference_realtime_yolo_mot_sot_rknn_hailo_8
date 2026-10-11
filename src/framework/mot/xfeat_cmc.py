import logging
import numpy as np
import cv2
from src.framework.mot.xfeat_rknn import XFeatRKNN


class XFeatCMC:
    """
    CMC (Global Motion Compensation) на базе XFeat + RKNN.
    Совместим с интерфейсом BoxMOT CMC.
    """
    
    def __init__(
            self,
            rknn_path: str,
            core_mask: int = 4,
            input_size: tuple = (360, 640),
            top_k: int = 1024,
    ):
        self._logger = logging.getLogger("xfeat_cmc")
        self._xfeat = XFeatRKNN(
            rknn_path=rknn_path,
            input_size=input_size,
            core_mask=core_mask,
            top_k=top_k,
        )
        self._prev_frame = None
        self._prev_kp = None
        self._prev_desc = None
        self._last_homography = np.eye(3, dtype=np.float32)
    
    def apply(self, img: np.ndarray, dets: np.ndarray = None) -> np.ndarray:
        """
        Вычисляет гомографию между предыдущим и текущим кадром.
        
        Args:
            img: текущий кадр (BGR, HxWx3)
            dets: детекции (не используются в XFeat CMC, но нужны для интерфейса)
        
        Returns:
            H: матрица гомографии 3x3
        """
        if self._prev_frame is None:
            # Первый кадр — просто сохраняем
            self._prev_frame = img.copy()
            kp, desc = self._extract_features_single(img)
            self._prev_kp = kp
            self._prev_desc = desc
            return np.eye(3, dtype=np.float32)
        
        # Детектируем keypoints на паре кадров (batch=2)
        kp_prev, kp_curr, desc_prev, desc_curr = self._xfeat.detect_pair(
            self._prev_frame, img
        )
        
        # Сопоставляем дескрипторы
        idx1, idx2 = self._xfeat.match_descriptors(desc_prev, desc_curr)
        
        if len(idx1) < 4:
            # Недостаточно матчей — возвращаем identity
            self._logger.warning(
                f"XFeat CMC: недостаточно матчей ({len(idx1)}), "
                f"возвращаю identity warp"
            )
            self._update_state(img, kp_curr, desc_curr)
            return np.eye(3, dtype=np.float32)
        
        # Берём координаты сопоставленных keypoints
        pts1 = kp_prev[idx1]
        pts2 = kp_curr[idx2]
        
        # Вычисляем гомографию через RANSAC
        H, inliers = cv2.findHomography(
            pts1, pts2, 
            cv2.RANSAC, 
            ransacReprojThreshold=3.0
        )
        
        if H is None:
            self._logger.warning("XFeat CMC: findHomography вернул None")
            self._update_state(img, kp_curr, desc_curr)
            return np.eye(3, dtype=np.float32)
        
        n_inliers = np.sum(inliers) if inliers is not None else 0
        self._logger.debug(
            f"XFeat CMC: {len(idx1)} matches, {n_inliers} inliers"
        )
        
        self._update_state(img, kp_curr, desc_curr)
        return H.astype(np.float32)
    
    def _extract_features_single(self, img: np.ndarray):
        """Извлечение фич на одном кадре (использует batch=2, дублируя кадр)"""
        kp_prev, kp_curr, desc_prev, desc_curr = self._xfeat.detect_pair(img, img)
        return kp_curr, desc_curr
    
    def _update_state(self, img, kp, desc):
        """Обновление состояния для следующего кадра"""
        self._prev_frame = img.copy()
        self._prev_kp = kp
        self._prev_desc = desc
    
    def release(self):
        if self._xfeat is not None:
            self._xfeat.release()
