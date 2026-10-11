import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True, slots=True)
class SliceCoords:
    x_offset: int
    y_offset: int


@dataclass(frozen=True, slots=True)
class ImageSlice:
    image: np.ndarray
    coords: SliceCoords

    @classmethod
    def from_image(cls, image: np.ndarray) -> "ImageSlice":
        return cls(image, SliceCoords(0, 0))


def slice_image(
    img: np.ndarray,
    window_wh: Tuple[int, int],
    overlay: Tuple[float, float] = (0.1, 0.1),
    copy_slices: bool = True,
) -> List[ImageSlice]:
    result = []
    coords = calc_scan_areas((0, 0, img.shape[1], img.shape[0]), window_wh, overlay)
    for k in coords:
        slice_img = img[k[1] : k[3], k[0] : k[2]]
        if copy_slices:
            slice_img = slice_img.copy()
        result.append(ImageSlice(slice_img, coords=SliceCoords(k[0], k[1])))

    return result


def calc_scan_areas(
    scan_area_bbox, window_wh=(800, 800), overlay=(0.1, 0.1)
) -> List[Tuple[int, int, int, int]]:
    """
    Вычисление подобластей заданного размера (окон) для сканирования в заданной области
    с перекрытием не менее указанного в параметре overlay
    :param bbox: область сканирования
    :param window_wh: (ширина, высота) окна сканирования
    :param overlay: коэффициенты перекрытия (overlay_x, overlay_y) по соответствующим осям
    :return: массив bbox_w окон
    """
    x_areas = calc_linear_scan_areas(
        (scan_area_bbox[0], scan_area_bbox[2]), window_wh[0], overlay[0]
    )
    y_areas = calc_linear_scan_areas(
        (scan_area_bbox[1], scan_area_bbox[3]), window_wh[1], overlay[1]
    )

    areas = []
    for y_a in y_areas:
        for x_a in x_areas:
            a_bbox = (x_a[0], y_a[0], x_a[1], y_a[1])
            areas.append(a_bbox)

    return areas


def calc_linear_scan_areas(
    scan_area: tuple, window_w=800, overlay=0.1
) -> List[Tuple[int, int]]:
    """
    Вычисление подобластей заданного размера (окон) для сканирования в заданной области
    с перекрытием не менее указанного в параметре overlay вдоль оси
    :param scan_area: область сканирования x1<->x2
    :param window: размер окна сканирования
    :param overlay: коэффициенты перекрытия
    :return:
    """

    scan_area_w = scan_area[1] - scan_area[0]
    over = overlay * window_w

    n_x = (scan_area_w - window_w) / (window_w - over)
    n_x = math.ceil(n_x)
    over = window_w - (scan_area_w - window_w) / n_x
    window_arr = []
    for i in range(n_x + 1):
        start_x = scan_area[0] + i * (window_w - over)
        stop_x = start_x + window_w
        window_arr.append((int(start_x), int(stop_x)))
    return window_arr
