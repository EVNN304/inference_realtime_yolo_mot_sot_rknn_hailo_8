import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.framework.core.slice import slice_image, ImageSlice
from src.framework.det.cv_utils import draw_detections
from src.framework.det.yolo_sahi import YoloSAHI
from src.framework.mot.bytetrack import ByteTrackTracker
from src.framework.mot.cv_utils import draw_tracks
from src.framework.mot.hybridsort import HybridSortTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO + SAHI detection on a video and save output frames."
    )

    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to YOLO model (.pt)",
    )

    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to input video",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where output images will be saved",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=288,
        help="Inference image size (default: 288)",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.model_path.exists():
        print(f"Model not found: {args.model_path}")
        return 1

    if not args.video_path.exists():
        print(f"Video not found: {args.video_path}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    img_sz = args.imgsz

    detector = YoloSAHI(
        model=YOLO(
            str(args.model_path),
            verbose=False,
        ),
        nms="classic",
        runtime_args={"conf": 0.5, "imgsz": img_sz, "half": True, "verbose": False},
    )

    mot = HybridSortTracker(reid_weights="models/osnet_x0_25_msmt17.pt", longterm_bank_length=60, max_age=120, max_obs=150)

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {args.video_path}")
        return 1

    counter = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        slices = slice_image(img, window_wh=(img_sz, img_sz), overlay=(0.1, 0.1))
        # slices.append(ImageSlice.from_image(img))

        detections = detector.infer(
            slices,
            timestamp_ns=time.perf_counter_ns(),
        )

        # try:
        tracks = mot.track(img, detections)
        # except Exception as e:
        #     print(f"Exception: {e}")

        img_tracks = draw_tracks(img, tracks, class_names=["0", "1", "2"])

        cv2.imshow("track", img_tracks)
        cv2.waitKey(1)

        # out_path = args.output_dir / f"{counter}.jpg"
        # cv2.imwrite(str(out_path), img_det)
        # print(f"Saved {out_path}")

        counter += 1

    cap.release()
    return 0


if __name__ == "__main__":
    main()
