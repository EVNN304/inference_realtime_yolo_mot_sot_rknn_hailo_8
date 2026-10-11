import argparse
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.framework.core.box import Box, BoxTopLeft
from src.framework.sot.stark_tracker import StarkTracker


MODELS_DIR = Path("models")


def parse_box(text: str) -> tuple[float, float, float, float]:
    parts = text.split(",")

    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Box must be in format: x,y,w,h"
        )

    try:
        x, y, w, h = map(float, parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "Box values must be numbers"
        ) from e

    return x, y, w, h


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run STARK single-object tracker on a video file.",
    )

    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--init-box", type=parse_box, required=True)

    parser.add_argument(
        "--model-name",
        type=str,
        default="stark_s",
        choices=("stark_s", "stark_lightning_X_trt"),
    )

    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=MODELS_DIR / "stark" / "starks_baseline.yaml",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "stark" / "STARKS_ep0500.pth.tar",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
    )

    return parser.parse_args(argv)


def draw_box(frame: np.ndarray, box: Box, label: str | None = None) -> None:
    x, y, w, h = box.as_tuple()

    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if label is not None:
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    stark = StarkTracker(
        model_name=args.model_name,
        config=StarkTracker.Config(
            yaml_path=args.yaml_path,
            model_path=args.model_path,
        ),
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        raise RuntimeError("Failed to read first frame")

    init_box = BoxTopLeft(*args.init_box)

    stark.initialize(first_frame, init_box)

    frame_index = 0

    fps = 0.0
    alpha = 0.9  # smoothing

    prev_time = time.perf_counter()

    if args.show:
        vis_first = first_frame.copy()
        draw_box(vis_first, init_box, label="init")
        draw_fps(vis_first, fps)
        cv2.imshow("STARK Tracker", vis_first)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            return 0

    while True:

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_index += 1

        t0 = time.perf_counter()

        track = stark.track(frame=frame)

        t1 = time.perf_counter()

        dt = t1 - t0
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        fps = alpha * fps + (1 - alpha) * inst_fps

        print(
            f"frame={frame_index} "
            f"box={track.box.as_tuple()} "
            f"conf={track.conf:.3f} "
            f"fps={fps:.1f}"
        )

        if args.show:
            vis = frame.copy()

            draw_box(vis, track.box, label=f"{track.conf:.2f}")
            draw_fps(vis, fps)

            cv2.imshow("STARK Tracker", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    main()
