from enum import Enum
from typing import Optional, Union

import cv2
import numpy as np


def _codec_format(a: str, b: str, c: str, d: Union[str, int]) -> int:
    if isinstance(d, int):
        return ord(a) | (ord(b) << 8) | (ord(c) << 16) | (d << 24)
    return ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24)


class ImageFormat(Enum):
    GRAY = _codec_format("G", "R", "E", "Y")
    RGB = _codec_format("R", "G", "B", 24)
    BGR = _codec_format("B", "G", "R", 24)
    RGBA = _codec_format("R", "G", "B", "A")
    BGRA = _codec_format("B", "G", "R", "A")
    I420 = _codec_format("I", "4", "2", "0")


IMAGE_FORMAT_TO_NUM_CHANNELS_MAP = {
    ImageFormat.GRAY: 1,
    ImageFormat.RGB: 3,
    ImageFormat.BGR: 3,
    ImageFormat.RGBA: 4,
    ImageFormat.BGRA: 4,
    ImageFormat.I420: 1
}


class FormatManager:
    def __init__(self, image_format: ImageFormat):
        self._image_format = image_format

    def preprocess_frame(
            self, frame: np.ndarray,
            target_frame_format: Optional[ImageFormat]) -> np.ndarray:
        preprocessed_frame = self._cvt_color(frame, self._image_format,
                                             target_frame_format)
        return preprocessed_frame

    def _cvt_color(self, frame: np.ndarray, source_format: ImageFormat,
                   target_format: ImageFormat) -> np.ndarray:
        if target_format is None or target_format is source_format:
            return frame
        source_format_name = 'YUV' if source_format is ImageFormat.I420 else source_format.name
        target_format_name = 'YUV' if target_format is ImageFormat.I420 else target_format.name
        add_format_name = 'I420' if (source_format is ImageFormat.I420 or
                                     target_format is ImageFormat.I420) else ''
        color_code_str = f'color_{source_format_name}2{target_format_name}_{add_format_name}'.upper(
        ).rstrip('_')
        color_code = getattr(cv2, color_code_str, None)
        if color_code is None:
            raise ValueError(
                f"There's no such color code {color_code_str} in cv2.")
        frame = cv2.cvtColor(frame, color_code)
        return frame

    def get_channels_count(self, image_format: ImageFormat) -> int:
        return IMAGE_FORMAT_TO_NUM_CHANNELS_MAP[image_format]
