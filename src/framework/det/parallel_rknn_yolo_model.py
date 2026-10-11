import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List

from src.framework.det.rknn_yolo_model import RKNNYoloModel, RKNNResult


class ParallelRKNNYoloModel:
    """
    Параллельный RKNN детектор, использующий несколько ядер NPU одновременно.
    Распределяет слайсы SAHI между ядрами для ускорения инференса.
    """
    def __init__(
        self,
        rknn_path: str,
        core_masks: List[int],
        task: str = "detect",
        verbose: bool = False,
        conf: float = 0.25,
        imgsz: int = 640,
    ):
        self._logger = logging.getLogger("parallel_rknn_yolo")
        self._core_masks = core_masks
        self._models = []
        
        # Создаем по одной модели на каждое ядро
        for core_mask in core_masks:
            model = RKNNYoloModel(
                rknn_path=rknn_path,
                task=task,
                verbose=verbose,
                conf=conf,
                imgsz=imgsz,
                core_mask=core_mask,
            )
            self._models.append(model)
        
        # ThreadPoolExecutor для параллельного инференса
        self._executor = ThreadPoolExecutor(max_workers=len(core_masks))
        
        self._logger.info(
            f"Parallel RKNN model initialized with {len(core_masks)} cores: {core_masks}"
        )

    def __call__(self, batch_imgs: List[np.ndarray], **kwargs) -> List[RKNNResult]:
        """
        Распределяет слайсы между ядрами и запускает параллельный инференс.
        """
        if not batch_imgs:
            return []
        
        n_models = len(self._models)
        n_imgs = len(batch_imgs)
        
        # Если слайсов меньше или равно количеству моделей, используем только одну
        if n_imgs <= 1:
            return self._models[0](batch_imgs, **kwargs)
        
        # Распределяем слайсы по моделям (подряд, не чередуя)
        chunk_size = (n_imgs + n_models - 1) // n_models
        chunks = []
        for i in range(n_models):
            start = i * chunk_size
            end = min(start + chunk_size, n_imgs)
            if start < end:
                chunks.append((i, batch_imgs[start:end]))
            else:
                chunks.append((i, []))
        
        # Параллельный инференс
        futures = []
        for model_idx, chunk in chunks:
            if chunk:
                future = self._executor.submit(self._models[model_idx], chunk, **kwargs)
                futures.append((model_idx, future))
        
        # Собираем результаты в правильном порядке
        results = [None] * n_imgs
        for model_idx, future in futures:
            chunk_results = future.result()
            start = model_idx * chunk_size
            for j, res in enumerate(chunk_results):
                idx = start + j
                if idx < n_imgs:
                    results[idx] = res
        
        return results

    def __del__(self):
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
