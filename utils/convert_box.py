import torch
from .general import grid_xy


def xywh2xyxy(inp):
    """
    Преобразование формата ограничивающих рамок. (x, y, w, h) -> (x, y, x, y)
    Parameters:
        inp: torch.Tensor, size (batch_size, num_boxes, 4) или (num_boxes, 4).
             Формат ограничивающих рамок (x, y, w, h).
    Returns:
        out: torch.Tensor, size (batch_size, num_boxes, 4) или (num_boxes, 4).
             Формат ограничивающих рамок (x, y, x, y).
    """
    
    out = torch.empty_like(inp)

    xy = inp[..., :2]  # координаты x и y центра ограничивающей рамки.
    wh = inp[..., 2:] / 2  # половина высоты и ширины ограничивающей рамки.

    out[..., :2] = xy - wh  # координаты x и y левого верхнего угла.
    out[..., 2:] = xy + wh  # координаты x и y правого нижнего угла.
    return out

def xyxy2xywh(inp):
    """
    Преобразование формата ограничивающих рамок. (x, y, x, y) -> (x, y, w, h)
    Parameters:
        inp: torch.Tensor, size (batch_size, num_boxes, 4) или (num_boxes, 4).
             Формат ограничивающих рамок (x, y, x, y).
    Returns:
        out: torch.Tensor, size (batch_size, num_boxes, 4) или (num_boxes, 4).
             Формат ограничивающих рамок (x, y, w, h).
    """

    out = torch.empty_like(inp)

    # координата x центра ограничивающей рамки.
    out[..., 0] = (inp[..., 0] + inp[..., 2]) / 2
    # координата y центра ограничивающей рамки.
    out[..., 1] = (inp[..., 1] + inp[..., 3]) / 2

    # ширина ограничивающей рамки.
    out[..., 2] = inp[..., 2] - inp[..., 0]
    # высота ограничивающей рамки.
    out[..., 3] = inp[..., 3] - inp[..., 1]
    return out

def xcell_ycell_wh2xyxy(inp):
    """
    Преобразование формата ограничивающих рамок. (x_cell, y_cell, w, h) -> (x, y, x, y)
    Parameters:
        inp: torch.Tensor, size (batch_size, 7, 7, 2, 4).
             Формат ограничивающих рамок (x_cell, y_cell, w, h).
    Returns:
        out: torch.Tensor, size (batch_size, 7, 7, 2, 4).
             Формат ограничивающих рамок (x, y, x, y).
    """
    out = torch.empty_like(inp, dtype=inp.dtype, device=inp.device)

    grid = grid_xy(dtype=inp.dtype, device=inp.device)

    xy = inp[..., :2]/7 + grid  # координаты x и y центра ограничивающей рамки.
    wh = inp[..., 2:4]/2  # половина высоты и ширины ограничивающей рамки.

    out[..., :2] = xy - wh
    out[..., 2:4] = xy + wh

    return out

def xcell_ycell_wh2xywh(inp):
    """
    Преобразование формата ограничивающих рамок. (x_cell, y_cell, w, h) -> (x, y, w, h)
    Parameters:
        inp: torch.Tensor, size (batch_size, 7, 7, 2, 4).
             Формат ограничивающих рамок (x_cell, y_cell, w, h).
    Returns:
        out: torch.Tensor, size (batch_size, 7, 7, 2, 4).
             Формат ограничивающих рамок (x, y, w, h).
    """
    out = torch.empty_like(inp, dtype=torch.float32, device=inp.device)

    grid_xy = grid_xy(dtype=inp.dtype, device=inp.device)

    out[..., :2] = inp[..., :2]/7 + grid_xy
    out[..., 2:4] = inp[..., 2:4]

    return out














