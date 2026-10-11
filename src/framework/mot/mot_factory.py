import config.config as config
import os

# Ключи, которые используются только для инициализации ReID и не должны передаваться в трекер
_MOT_IGNORE_KEYS = {
    "reid_model_path",
    "reid_input_size",
    "reid_core_mask",
    "reid_core_masks",
    "reid_batch_size",
}

def create_mot_tracker(cam_id):
    """
    Создает MOT-трекер с ReID-экстрактором.
    Использует единую динамическую RKNN модель с поддержкой batch [1..25].
    """
    tracker_kwargs = {
        k: v for k, v in config.mot_settings.items() if k not in _MOT_IGNORE_KEYS
    }

    reid_model_path = config.mot_settings.get("reid_model_path")
    reid_extractor = None

    if reid_model_path and os.path.exists(reid_model_path):
        # === ВАРИАНТ 1: RKNN (NPU) с динамическим батчем ===
        if reid_model_path.endswith(".rknn"):
            from src.framework.mot.parallel_rknn_reid_wrapper import ParallelRKNNReIDWrapper
            

            reid_extractor = ParallelRKNNReIDWrapper(
                rknn_path=reid_model_path,
                input_size=config.mot_settings.get("reid_input_size", (256, 128)),
                core_mask=config.mot_settings.get("reid_core_mask", 2),
            )
            
        # === ВАРИАНТ 2: ONNX (CPU) ===
        elif reid_model_path.endswith(".onnx"):
            from src.framework.mot.onnx_reid_wrapper import ONNXReIDWrapper
            reid_extractor = ONNXReIDWrapper(
                onnx_path=reid_model_path,
                input_size=config.mot_settings.get("reid_input_size", (256, 128)),
            )
            print(f"[MOT Factory] ✅ Using ONNX ReID backend (CPU): {reid_model_path}")
        else:
            print(f"[MOT Factory] ⚠️ Warning: Unknown ReID model format: {reid_model_path}")
    else:
        print(f"[MOT Factory] ⚠️ Warning: ReID model not found at {reid_model_path}")

    # Импортируем трекер (HybridSort)
    from src.framework.mot.hybridsort import HybridSortTracker
    
    tracker = HybridSortTracker(
        reid_weights=None,
        custom_reid_extractor=reid_extractor,
        **tracker_kwargs,
    )

    if reid_extractor is not None:
        tracker._tracker.model = reid_extractor
        
    return tracker
