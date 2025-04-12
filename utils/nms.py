import torch
from .bbox_iou import box_iou
from .convert_box import xywh2xyxy


def nms(boxes, scores, threshold=0.5):
    _, sorted_idx = scores.sort(descending=True)

    keep = []
    while sorted_idx.numel() > 0:
        if sorted_idx.numel() == 1:
            keep.append(sorted_idx)
            break

        idx = sorted_idx[0]
        keep.append(idx)

        boxs1 = boxes[sorted_idx[0]].unsqueeze(dim=0)    # size (1, 4)
        boxs2 = boxes[sorted_idx[1:]]                    # size (M, 4)
        iou = box_iou(boxs1, boxs2)                      # size (1, M)

        i = (iou < threshold).nonzero()[:, 1]
        if i.numel() == 0:
            break

        sorted_idx = sorted_idx[i+1]

    return torch.tensor(keep, dtype=torch.int)

def non_max_suppression(
        pred,
        score_threshold=0.25,
        iou_threshold=0.45,
        agnostic=False,
        max_wh=7600,
        classes=None

):

    """
    Parameters:
        pred: torch.Tensor, size (batch-size, 4 + num_classes, num_boxes).
              Параметры ограничивающих рамок в формате (x, y, widht, height).
        score_threshold: float (по умол. 0.25), значения в интервале от 0 до 1.
        iou_threshold: float (по умол. 0.45), значения в интервале от 0 до 1.
        agnostic: bool (по умол. False), если False, то NMS считается c учетом
                  классов, если True, то NMS считается без учета классов.
        max_wh: int (по умол. 7600), максимально возможная ширина и высота
                входного изображения.
        classes: List[int] (по умол. None), список c индексами классов, которые
                 нужно учитывать. Если None, то учитываются все классы.
    Returns:
        output: List[torch.Tensor], список, длина которого batch_size, содержащий
                тензоры c результатоми преобразований для каждого элемента батча.
                Размер тензора (num_boxes, 4 + score + class)
    """
    # bs - batch size, размер батча.
    # nc - num_class, количество классов.
    # nb - num_boxes, количесто предсказанных ограничивающих рамок.

    # Если передан аргумент classes, то создаем тензор с индексами классов.
    if classes is not None:
        classes = torch.tensor(classes, device=pred.device)

    bs = pred.shape[0]
    nc = pred.shape[1] - 4
    candidates = pred[:, 4:].amax(dim=1) > score_threshold  # (bs, nb)

    pred = pred.transpose(-1, -2)  # (bs, 4+nc, nb) -> (bs, nb, 4+nc)
    pred = torch.cat((xywh2xyxy(pred[..., :4]), pred[..., 4:]), dim=-1)  # xywh -> xyxy

    output = [torch.zeros((0, 6), device=pred.device)] * bs
    # Перебираем предсказания для каждого изображения в батче.
    for idx, pr in enumerate(pred):
        # idx - индекс изображения в батче.
        # pr - предсказания для изображения. Size (nb, 4+nc)

        # Отбираем только те предсказания, у которых уверенность в
        # предсказанном классе (score) больше порогового (score_threchold).
        pr = pr[candidates[idx]]

        # Если нет подходящих предсказаний, переходим к следующему изображению.
        if not pr.shape[0]:
            continue

        # Разделяем предсказания для одного изображения на предсказанные
        # ограничивающие рамки и предсказания классов.
        # box - предсказанные ограничивающие рамки, size (nb, 4).
        # cls - предсказания классов, size (nb, nc)
        box, cls = pr.split((4, nc), dim=1)

        # Получаем индекс предсказанного класса и предсказанное значение (score).
        # При этом сохраняем размерность тензоров (с помощью аргумента keepdim).
        score, idx_cls = cls.max(dim=1, keepdim=True)

        # Расширенный тензор с предсказаниями, size (nb, 4 + score + idx_cls)
        pr = torch.cat((box, score, idx_cls.float()), dim=1)

        # Если указан аргумент classes, то в тензоре pr оставляем только
        # указанные классы.
        if classes is not None:
            i = (pr[:, 5:6] == classes).any(dim=1)
            pr = pr[i]

        # Если нет подходящих предсказаний, переходим к следующему изображению.
        if not pr.shape[0]:
            continue

        # Non maximum suppression.
        scores = pr[:, 4]
        scaling_for_classes = pr[:, 5:6] * (0 if agnostic else max_wh)
        boxes = pr[:, :4] + scaling_for_classes

        idx_nms = nms(boxes, scores, iou_threshold)

        output[idx] = pr[idx_nms]

    return output

def nms_yolov1(
        pred,
        score_threshold=0.25,
        iou_threshold=0.45,
        agnostic=False,
        max_wh=7600,
        classes=None

):

    """
    Parameters:
        pred: torch.Tensor, size (batch-size, num_boxes, 5 + num_classes).
              Параметры ограничивающих рамок в формате (x1, y1, x2, y2).
        score_threshold: float (по умол. 0.25), значения в интервале от 0 до 1.
        iou_threshold: float (по умол. 0.45), значения в интервале от 0 до 1.
        agnostic: bool (по умол. False), если False, то NMS считается c учетом
                  классов, если True, то NMS считается без учета классов.
        max_wh: int (по умол. 7600), максимально возможная ширина и высота
                входного изображения.
        classes: List[int] (по умол. None), список c индексами классов, которые
                 нужно учитывать. Если None, то учитываются все классы.
    Returns:
        output: List[torch.Tensor], список, длина которого batch_size, содержащий
                тензоры c результатоми преобразований для каждого элемента батча.
                Размер тензора (num_boxes, 4 + score + class)
    """
    # bs - batch size, размер батча.
    # nc - num_class, количество классов.
    # nb - num_boxes, количесто предсказанных ограничивающих рамок.

    # Если передан аргумент classes, то создаем тензор с индексами классов.
    if classes is not None:
        classes = torch.tensor(classes, device=pred.device)

    bs = pred.shape[0]
    # nc = pred.shape[1] - 5
    candidates = pred[..., 4] > score_threshold  # (bs, nb)

    output = [torch.zeros((0, 6), device=pred.device)] * bs
    # Перебираем предсказания для каждого изображения в батче.
    for idx, pr in enumerate(pred):
        # idx - индекс изображения в батче.
        # pr - предсказания для изображения. Size (nb, 4+nc)

        # Отбираем только те предсказания, у которых уверенность в
        # предсказанном классе (score) больше порогового (score_threchold).
        pr = pr[candidates[idx]]

        # Если нет подходящих предсказаний, переходим к следующему изображению.
        if not pr.shape[0]:
            continue
        
        pr[:, 5:] *= pr[:, 4:5]
        
        # Разделяем предсказания для одного изображения на предсказанные
        # ограничивающие рамки и предсказания классов.
        # boxes - предсказанные ограничивающие рамки, size (nb, 4).
        # cls - предсказания классов, size (nb, nc)
        boxes = pr[..., :4]
        cls = pr[:, 5:]

        # Получаем индекс предсказанного класса и предсказанное значение (score).
        # При этом сохраняем размерность тензоров (с помощью аргумента keepdim).
        score, idx_cls = cls.max(dim=1, keepdim=True)

        # Расширенный тензор с предсказаниями, size (nb, 4 + score + idx_cls)
        pr = torch.cat((boxes, score, idx_cls.float()), dim=1)[score.view(-1) > score_threshold]

        # Если указан аргумент classes, то в тензоре pr оставляем только
        # указанные классы.
        if classes is not None:
            i = (pr[:, 5:6] == classes).any(dim=1)
            pr = pr[i]

        # Если нет подходящих предсказаний, переходим к следующему изображению.
        if not pr.shape[0]:
            continue

        # Non maximum suppression.
        scores = pr[:, 4]
        scaling_for_classes = pr[:, 5:6] * (0 if agnostic else max_wh)
        boxes = pr[:, :4] + scaling_for_classes

        idx_nms = nms(boxes, scores, iou_threshold)

        output[idx] = pr[idx_nms]

    return output




















