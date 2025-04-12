import torch
import numpy as np
from .bbox_iou import box_iou


def grid_xy(dtype=torch.int32, device='cpu'):
    """
    Функция для получения сетки с добавочными значениями. С её помощью координаты
    x_cell и y_cell (кооординаты центра ограничивающей рамки внутри ячейки) можно
    перевести в координаты X и Y относительно всего изображения.
    Returns:
        grid: torch.Tensor, size (1, 7, 7, 2, 2).
    """
    # Создаю векторы с добавочными значениями при смещении на одну ячейку
    # вдоль X и Y.
    shifts_x = torch.arange(0, 7, dtype=dtype, device=device) / 7.0    # size (7,)
    shifts_y = torch.arange(0, 7, dtype=dtype, device=device) / 7.0    # size (7,)

    # Создаю сетку с добавочными значениями для всех ячеек при смещении на одну
    # ячейку вдоль X и Y. Size (7, 7)
    shifts_x, shifts_y = torch.meshgrid(shifts_x, shifts_y, indexing="xy")

    shifts_x = shifts_x.reshape((1, 7, 7, 1, 1))   # size (1, 7, 7, 1, 1)
    shifts_y = shifts_y.reshape((1, 7, 7, 1, 1))   # size (1, 7, 7, 1, 1)

    shifts_x = shifts_x.repeat(1, 1, 1, 2, 1)   # size (1, 7, 7, 2, 1)
    shifts_y = shifts_y.repeat(1, 1, 1, 2, 1)   # size (1, 7, 7, 2, 1)

    grid = torch.cat([shifts_x, shifts_y], dim=-1)   # size (1, 7, 7, 2, 2)
    return grid

def clip_boxes(boxes, shape=[1, 1]):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def process_batch(detections, labels, iouv):
    """
    Return a correct prediction matrix given detections and labels at various IoU thresholds.

    Args:
        detections (np.ndarray): Array of shape (N, 6) where each row corresponds to a detection with format
            [x1, y1, x2, y2, conf, class].
        labels (np.ndarray): Array of shape (M, 5) where each row corresponds to a ground truth label with format
            [class, x1, y1, x2, y2].
        iouv (np.ndarray): Array of IoU thresholds to evaluate at.

    Returns:
        correct (np.ndarray): A binary array of shape (N, len(iouv)) indicating whether each detection is a true positive
            for each IoU threshold. There are 10 IoU levels used in the evaluation.

    Example:
        ```python
        detections = np.array([[50, 50, 200, 200, 0.9, 1], [30, 30, 150, 150, 0.7, 0]])
        labels = np.array([[1, 50, 50, 200, 200]])
        iouv = np.linspace(0.5, 0.95, 10)
        correct = process_batch(detections, labels, iouv)
        ```

    Notes:
        - This function is used as part of the evaluation pipeline for object detection models.
        - IoU (Intersection over Union) is a common evaluation metric for object detection performance.
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)
