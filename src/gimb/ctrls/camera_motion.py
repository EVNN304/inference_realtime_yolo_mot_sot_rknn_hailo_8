"""
Camera Motion Compensation (CMC) using Fast Resized ORB Feature Matching and RANSAC.
Fixed internal resolution: 320x180 | Fixed feature budget: 50 points.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class OrbCameraMotionEstimator:
    """
    Estimates 2D camera ego-motion (translation, rotation, scale) between consecutive frames.
    Processes downscaled 320x180 grayscale images with cached descriptor extraction.
    """

    INTERNAL_WIDTH: int = 320
    INTERNAL_HEIGHT: int = 180
    NUM_FEATURES: int = 50
    RANSAC_REPROJ_THRESH: float = 3.0
    TARGET_EXCLUDE_MARGIN: float = 0.25

    def __init__(self, num_features: int = NUM_FEATURES):
        self.num_features = num_features
        self.identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        self.orb = cv2.ORB_create(
            nfeatures=self.num_features,
            scaleFactor=1.2,
            nlevels=1,
            edgeThreshold=15,
            patchSize=15,
            scoreType=cv2.ORB_FAST_SCORE,
            fastThreshold=20
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.prev_kps = None
        self.prev_des = None

    def reset(self) -> None:
        """Clears descriptor cache."""
        self.prev_kps = None
        self.prev_des = None

    def _get_background_mask(
        self,
        frame_h: int,
        frame_w: int,
        exclude_bbox: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if exclude_bbox is None or len(exclude_bbox) != 4:
            return None

        mask = np.ones((self.INTERNAL_HEIGHT, self.INTERNAL_WIDTH), dtype=np.uint8) * 255

        # Border margin in 320x180 space
        border = 6
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0

        # Scale exclude bbox down to 320x180
        scale_x = float(self.INTERNAL_WIDTH) / float(frame_w)
        scale_y = float(self.INTERNAL_HEIGHT) / float(frame_h)

        x1, y1, x2, y2 = exclude_bbox
        bw = (x2 - x1) * scale_x
        bh = (y2 - y1) * scale_y
        mx = bw * self.TARGET_EXCLUDE_MARGIN
        my = bh * self.TARGET_EXCLUDE_MARGIN

        ex1 = max(0, int(round(x1 * scale_x - mx)))
        ey1 = max(0, int(round(y1 * scale_y - my)))
        ex2 = min(self.INTERNAL_WIDTH, int(round(x2 * scale_x + mx)))
        ey2 = min(self.INTERNAL_HEIGHT, int(round(y2 * scale_y + my)))

        mask[ey1:ey2, ex1:ex2] = 0
        return mask

    def estimate(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        exclude_bbox: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Estimates camera transformation matrix M (2x3 Affine) from prev_frame to curr_frame.
        Returns: (M, num_inliers)
        """
        frame_h, frame_w = prev_frame.shape[:2]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if curr_frame.ndim == 3 else curr_frame

        scale_x = float(frame_w) / float(self.INTERNAL_WIDTH)
        scale_y = float(frame_h) / float(self.INTERNAL_HEIGHT)

        curr_small = cv2.resize(curr_gray, (self.INTERNAL_WIDTH, self.INTERNAL_HEIGHT), interpolation=cv2.INTER_LINEAR)
        mask_small = self._get_background_mask(frame_h, frame_w, exclude_bbox)

        # 1. Reuse previous keypoints/descriptors if available
        if self.prev_des is None or self.prev_kps is None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
            prev_small = cv2.resize(prev_gray, (self.INTERNAL_WIDTH, self.INTERNAL_HEIGHT), interpolation=cv2.INTER_LINEAR)
            self.prev_kps, self.prev_des = self.orb.detectAndCompute(prev_small, mask_small)

        # 2. Compute only on current frame
        curr_kps, curr_des = self.orb.detectAndCompute(curr_small, mask_small)

        prev_kps, prev_des = self.prev_kps, self.prev_des

        # Update cache for next step
        self.prev_kps = curr_kps
        self.prev_des = curr_des

        if prev_des is None or curr_des is None or len(prev_kps) < 6 or len(curr_kps) < 6:
            return self.identity_matrix.copy(), 0

        matches = self.matcher.match(prev_des, curr_des)
        if len(matches) < 6:
            return self.identity_matrix.copy(), 0

        # Rescale matched points back to original image coordinates
        p0 = np.array([[prev_kps[m.queryIdx].pt[0] * scale_x, prev_kps[m.queryIdx].pt[1] * scale_y] for m in matches], dtype=np.float32)
        p1 = np.array([[curr_kps[m.trainIdx].pt[0] * scale_x, curr_kps[m.trainIdx].pt[1] * scale_y] for m in matches], dtype=np.float32)

        # 3. Fit 2D Rigid / Euclidean transform via RANSAC
        M, inliers = cv2.estimateAffinePartial2D(
            p0,
            p1,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.RANSAC_REPROJ_THRESH,
            maxIters=500,
            confidence=0.99
        )

        if M is None or inliers is None:
            return self.identity_matrix.copy(), 0

        num_inliers = int(np.sum(inliers))
        if num_inliers < 4:
            return self.identity_matrix.copy(), 0

        return M.astype(np.float32), num_inliers

    @staticmethod
    def warp_point(pt: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Transforms 2D coordinate [x, y] by 2x3 Affine matrix M."""
        x, y = float(pt[0]), float(pt[1])
        new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        return np.array([new_x, new_y], dtype=np.float32)

    @staticmethod
    def warp_box(bbox: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Transforms bounding box [x1, y1, x2, y2] by 2x3 Affine matrix M."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)

        new_cx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
        new_cy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]

        scale = float(np.sqrt(M[0, 0]**2 + M[1, 0]**2))
        new_w = w * scale
        new_h = h * scale

        new_x1 = new_cx - new_w * 0.5
        new_y1 = new_cy - new_h * 0.5
        new_x2 = new_cx + new_w * 0.5
        new_y2 = new_cy + new_h * 0.5

        return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)
