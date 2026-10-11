from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

use_faulthandler = False
frame_source = "video"   # rtsp or video or image
enable_rendering = True  # вкл/выкл рендеринг
alignment_path = "build"

class RtspPath:
    tv_n = "rtsp://127.0.0.1:8554/stream"  # путь до видеопотока

class DatasetFolderPath:
    tv_n = ""           # путь до набора с картинками

class DatasetVideoPath:
    tv_n = str(ROOT / "data" / "input.mp4")   ## путь для тестового видео 

logging = {
    "tv_det": "DEBUG",
    "sot": "WARNING",
    "track": "INFO",
    "state": "INFO",
}

class Tracking:
    enable_two_channel_tracking = False
    detections_timeout_ns = 800_000_000
    mot_forgot_time_ns = 2_000_000_000
    extrapolation_min_time_ns = 1_000_000_000
    extrapolation_timeout_ns = 4_000_000_000
    mot_iou_thresh = 0.55
    sot_box_sanity_thresh = 0.4
    frame_step = 2
    tracker = "nanotrack"
    nanotrack_backbone = str(ROOT / "models/nanotrackv3_backbone_dyn.rknn")
    nanotrack_head = str(ROOT / "models/nanotrackv3_head_fp16.onnx")

class Detector:
    channel = "narrow"
    enable_local_detector = True
    detector_type = "rknn"
    ir_tv_simultaneous = False
    

    confidence = [0.5]
    downscale_ratio = [None]
    model_path = [
  
        str(ROOT / "models" / "best_yolo11s.rknn")  # веса модели детектора rknn 
    ]
    rknn_core_mask = [1]  # Ядро
    use_sahi = False

if Detector.detector_type == "rknn":
    mot_settings = {
        "track_thresh": 0.55,
        "half": False,
        "delta_t": 10,
        "max_age": 150,
        "device": "cpu",
        "with_reid": True,
        "cmc_method": "orb",
        "reid_model_path": str(ROOT / "models" / "osnet_x0_25_msmt17_dynamic.rknn"),
        "reid_input_size": (256, 128),
        "reid_core_mask": 2,  
    }

# Detector + ReID accelerator. Existing RKNN settings above are unchanged.
# HybridSORT association always stays on CPU; NanoTrack backend is unchanged.
detector_backend = "rknn"  # "rknn" or "hailo": YOLO
mot_backend = "rknn"       # "rknn" or "hailo": OSNet; HybridSORT association stays on CPU.

class Hailo:
    yolo_hef = str(ROOT / "models" / "yolov11s_per.hef")
    reid_hef = str(ROOT / "models" / "osnet_msmt17.hef")
    reid_manifest = str(ROOT / "models" / "osnet_msmt17.json")
    confidence = 0.5
    iou = 0.45
    # Full-frame batches contain different source frames; SAHI batches contain tiles.
    full_frame_batch = 4
    full_frame_max_wait_ms = 120  # Flush partial batch with ignored padding on a slow source.
    sahi_batch = 4
    inflight = 2
    overlap = 0.0
    timeout_ms = 10000
    reid_batch = 4
    reid_inflight = 2
    mot_queue_size = 4
    mot_max_wait_ms = 120
    max_reid_objects = 16  # Same cap as the existing project; 0 = all detections.
    torch_threads = 1
    # The existing project video/RTSP/shared-memory source remains authoritative.
    tracker_options = dict(
        cmc_method="sift", per_class=False, nr_classes=6,
        det_thresh=0.5, low_thresh=0.1, track_thresh=0.5,
        max_age=30, min_hits=3, iou_threshold=0.3,
    )

# Optional direct USB camera source: set frame_source = "camera" above.
class CameraCapture:
    device = "/dev/video0"
    width = 1920
    height = 1200
    fps = 60
    fourcc = "MJPG"

# Optional machine-specific overrides; this file is ignored by Git.
_local = ROOT / "config/local.py"
if _local.is_file():
    exec(compile(_local.read_text(), str(_local), "exec"), globals())
