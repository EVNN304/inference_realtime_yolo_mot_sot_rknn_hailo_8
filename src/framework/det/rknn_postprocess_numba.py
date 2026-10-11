import numpy as np
from numba import njit
import torch
import torchvision.ops as ops

# Глобальный кэш для сеток (чтобы не пересоздавать meshgrid каждый кадр)
_GRID_CACHE = {}

def get_grid_and_stride(grid_h, grid_w, img_size):
    key = (grid_h, grid_w, img_size[0], img_size[1])
    if key not in _GRID_CACHE:
        col, row = np.meshgrid(np.arange(0, grid_w, dtype=np.float32), 
                               np.arange(0, grid_h, dtype=np.float32))
        grid = np.stack([col, row], axis=0)[None, ...]  # (1, 2, grid_h, grid_w)
        
        stride_x = img_size[0] / grid_w
        stride_y = img_size[1] / grid_h
        stride = np.array([stride_x, stride_y], dtype=np.float32).reshape(1, 2, 1, 1)
        
        _GRID_CACHE[key] = (grid, stride)
    return _GRID_CACHE[key]

@njit(fastmath=True)
def dfl_numba(position):
    """
    Оптимизированный DFL через Numba. 
    ВАЖНО: без parallel=True, чтобы избежать зависания компилятора Numba на вложенных циклах.
    """
    n, c, h, w = position.shape
    mc = c // 4  # 16 bins
    
    y = position.reshape((n, 4, mc, h, w))
    out = np.zeros((n, 4, h, w), dtype=np.float32)
    
    for b in range(n):
        for ch in range(4):
            for row in range(h):
                for col in range(w):
                    # Находим max для стабильности exp
                    max_val = y[b, ch, 0, row, col]
                    for bin_idx in range(1, mc):
                        if y[b, ch, bin_idx, row, col] > max_val:
                            max_val = y[b, ch, bin_idx, row, col]
                    
                    # Один проход: считаем exp, sum и weighted_sum одновременно
                    sum_exp = 0.0
                    weighted_sum = 0.0
                    for bin_idx in range(mc):
                        exp_val = np.exp(y[b, ch, bin_idx, row, col] - max_val)
                        sum_exp += exp_val
                        weighted_sum += exp_val * bin_idx
                    
                    out[b, ch, row, col] = weighted_sum / sum_exp
    return out

def process_branch_numba(pos_tensor, cls_tensor, obj_tensor, img_size=(640, 640)):
    """Быстрый парсинг ветки с использованием кэша и Numba."""
    grid_h, grid_w = pos_tensor.shape[2:4]
    grid, stride = get_grid_and_stride(grid_h, grid_w, img_size)

    pos_decoded = dfl_numba(pos_tensor)

    x1y1 = (grid + 0.5 - pos_decoded[:, 0:2, :, :]) * stride
    x2y2 = (grid + 0.5 + pos_decoded[:, 2:4, :, :]) * stride
    
    boxes = np.concatenate([x1y1, x2y2], axis=1)

    boxes_flat = boxes.transpose(0, 2, 3, 1).reshape(-1, 4)
    cls_flat = cls_tensor.transpose(0, 2, 3, 1).reshape(-1, cls_tensor.shape[1])
    obj_flat = obj_tensor.transpose(0, 2, 3, 1).reshape(-1, 1)
    
    return boxes_flat, cls_flat, obj_flat

def postprocess_single_image(outputs, imgsz, conf_thresh, nms_iou_thresh):
    """
    Постпроцессинг для ОДНОГО изображения.
    outputs: List[np.ndarray] — список из 9 numpy массивов shape (1, C, H, W)
    """
    if outputs is None or len(outputs) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)
    
    pair_per_branch = len(outputs) // 3
    all_boxes, all_scores, all_classes = [], [], []
    
    for i in range(3):
        pos_tensor = outputs[i * pair_per_branch]
        cls_tensor = outputs[i * pair_per_branch + 1]
        obj_tensor = outputs[i * pair_per_branch + 2]
        
        b_flat, c_flat, o_flat = process_branch_numba(pos_tensor, cls_tensor, obj_tensor, (imgsz, imgsz))
        
        # Умножение cls на obj
        cls_probs = c_flat if c_flat.max() <= 1.0 else 1.0 / (1.0 + np.exp(-np.clip(c_flat, -500, 500)))
        obj_probs = o_flat if o_flat.max() <= 1.0 else 1.0 / (1.0 + np.exp(-np.clip(o_flat, -500, 500)))
        
        final_scores = cls_probs * obj_probs
        scores = final_scores.max(axis=1)
        classes = final_scores.argmax(axis=1)
        
        all_boxes.append(b_flat)
        all_scores.append(scores)
        all_classes.append(classes)
        
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    
    # Фильтр по confidence
    mask = scores >= conf_thresh
    if not np.any(mask):
        return np.empty((0, 4)), np.empty(0), np.empty(0)
        
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_classes = classes[mask]
    
    # NMS
    if len(filtered_boxes) > 0:
        boxes_tensor = torch.from_numpy(filtered_boxes).float()
        scores_tensor = torch.from_numpy(filtered_scores).float()
        max_coordinate = boxes_tensor.max()
        offsets = torch.from_numpy(filtered_classes).long() * (int(max_coordinate) + 1)
        boxes_for_nms = boxes_tensor + offsets[:, None]
        keep_indices = ops.nms(boxes_for_nms, scores_tensor, nms_iou_thresh).numpy()
        
        return filtered_boxes[keep_indices], filtered_scores[keep_indices], filtered_classes[keep_indices]
    else:
        return filtered_boxes, filtered_scores, filtered_classes

def postprocess_batch_numba(images_outputs, imgsz, conf_thresh, nms_iou_thresh):
    """
    Постпроцессинг батча изображений.
    images_outputs: List[List[np.ndarray]] — список изображений, каждое — список из 9 outputs
    """
    all_results = []
    for outputs in images_outputs:
        boxes, scores, classes = postprocess_single_image(outputs, imgsz, conf_thresh, nms_iou_thresh)
        all_results.append((boxes, scores, classes))
    return all_results
