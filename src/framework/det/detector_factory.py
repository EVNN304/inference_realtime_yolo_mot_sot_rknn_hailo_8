import config.config as config
from src.framework.det.yolo_sahi import YoloSAHI
from src.gimb.domain.cameras import TrOsCamId


def create_yolo_sahi_model(model_path: str, confidence: float, imgsz: int, cam_id: TrOsCamId) -> YoloSAHI:
    """
    Универсальная фабрика. Возвращает YoloSAHI с параллельным RKNN детектором.
    """
    if config.Detector.detector_type == "rknn":
        from src.framework.det.parallel_rknn_yolo_model import ParallelRKNNYoloModel
        

        core_masks = [1]
        

        HARD_CONF = 0.45
        
        # Создаем параллельный детектор
        parallel_model = ParallelRKNNYoloModel(
            rknn_path=model_path,
            core_masks=core_masks,
            task="detect",
            verbose=False,
            conf=HARD_CONF,
            imgsz=imgsz,
        )

        return YoloSAHI(
            model=parallel_model,
            nms="classic",
            runtime_args={
                "conf": HARD_CONF,
                "imgsz": imgsz,
                "verbose": False,
            }
        )

