"""Copy to local.py; override the defaults using existing config classes."""
# ROOT and configuration classes are provided by config/config.py.
detector_backend = "hailo"
mot_backend = "rknn"
frame_source = "video"
DatasetVideoPath.tv_n = str(ROOT / "data/input.mp4")
Detector.use_sahi = False
Detector.confidence = [0.5]
Detector.model_path = [str(ROOT / "models/best_yolo11s.rknn")]
Hailo.yolo_hef = str(ROOT / "models/yolov11s_per.hef")
Hailo.reid_hef = str(ROOT / "models/osnet_msmt17.hef")
Hailo.reid_manifest = str(ROOT / "models/osnet_msmt17.json")
Hailo.confidence = 0.5
Hailo.full_frame_batch = 4
Hailo.full_frame_max_wait_ms = 120
Hailo.inflight = 2
Hailo.input_max_age_ms = 250
mot_settings["reid_model_path"] = str(ROOT / "models/osnet_x0_25_msmt17_dynamic.rknn")
Tracking.nanotrack_backbone = str(ROOT / "models/nanotrackv3_backbone_dyn.rknn")
Tracking.nanotrack_head = str(ROOT / "models/nanotrackv3_head_fp16.onnx")
CameraCapture.device = "/dev/video0"
CameraCapture.width = 1920
CameraCapture.height = 1080
CameraCapture.fps = 60
CameraCapture.fourcc = "MJPG"
