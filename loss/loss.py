import torch
import torch.nn as nn
from utils.convert_box import xcell_ycell_wh2xyxy
from utils.bbox_iou import bbox_iou

class YOLOv1Loss(nn.Module):
    """
    Класс реализующий функцию потерь для нейронной сети YOLOv1.
    """
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, pred, target):
        device = pred.device
        batch_size = pred.shape[0]

        # ========== Pred =========== #
        pred_boxes_conf = pred[..., :5*self.B]  # size: (bs, 7, 7, 10)
        pred_boxes_conf = pred_boxes_conf.reshape(batch_size, self.S, self.S, self.B, -1)   # size: (bs, 7, 7, 2, 5)
        pred_boxes = pred_boxes_conf[..., :4]   # size: (bs, 7, 7, 2, 4)
        xy = pred_boxes[..., :2]
        wh = pred_boxes[..., 2:].square()
        boxes_xywh = torch.cat([xy, wh], dim=-1)
        pred_class = pred[..., 5*self.B:]   # size: (bs, 7, 7, 20)

        # ========== Target =========== #
        mask_obj = torch.any((target > 0), dim=-1)  # size: (bs, 7, 7)

        target_boxes_conf = target[..., :5*self.B]  # size: (bs, 7, 7, 10)
        target_boxes_conf = target_boxes_conf.reshape(batch_size, self.S, self.S, self.B, -1)   # size: (bs, 7, 7, 2, 5)
        target_boxes = target_boxes_conf[..., :4]   # size: (bs, 7, 7, 2, 4)
        target_class = target[..., 5*self.B:]   # size: (bs, 7, 7, 20)

        # Меняю формат ограничивающих рамок.
        pred_boxes_xyxy = xcell_ycell_wh2xyxy(boxes_xywh)   # size: (bs, 7, 7, 2, 4)
        target_boxes_xyxy = xcell_ycell_wh2xyxy(target_boxes)   # size: (bs, 7, 7, 2, 4)

        iou = bbox_iou(pred_boxes_xyxy, target_boxes_xyxy)  # size: (bs, 7, 7, 2)
        max_idx = iou.argmax(dim=-1, keepdim=True)

        box_idx = torch.zeros(iou.shape, device=device).scatter(-1, max_idx, 1.0)

        obj_box_idx = (mask_obj[..., None] * box_idx).bool()    # size: (bs, 7, 7, 2)
        no_obj_box_idx = ~obj_box_idx   # size: (bs, 7, 7, 2)

        # XYWH Loss
        obj_box_pred_boxes = pred_boxes[obj_box_idx]
        obj_box_target_boxes = target_boxes[obj_box_idx]

        pred_x = obj_box_pred_boxes[..., 0]
        target_x = obj_box_target_boxes[..., 0]

        pred_y = obj_box_pred_boxes[..., 1]
        target_y = obj_box_target_boxes[..., 1]

        pred_w_sqrt = obj_box_pred_boxes[..., 2]
        target_w_sqrt = obj_box_target_boxes[..., 2].sqrt()

        pred_h_sqrt = obj_box_pred_boxes[..., 3]
        target_h_sqrt = obj_box_target_boxes[..., 3].sqrt()

        x_loss = (pred_x - target_x).square().sum()
        y_loss = (pred_y - target_y).square().sum()
        w_sqrt_loss = (pred_w_sqrt - target_w_sqrt).square().sum()
        h_sqrt_loss = (pred_h_sqrt - target_h_sqrt).square().sum()

        xywh_loss = x_loss + y_loss + w_sqrt_loss + h_sqrt_loss

        # Loss для параметра conf
        pred_conf = pred_boxes_conf[..., 4][obj_box_idx]
        target_conf = iou[obj_box_idx]

        conf_loss = (pred_conf - target_conf).square().sum()

        # Classification Loss
        pred_class = pred_class[mask_obj]
        target_class = target_class[mask_obj]

        class_loss = (pred_class - target_class).square().sum()

        # No Objectness Loss
        pred_no_obj_conf = pred_boxes_conf[..., 4][no_obj_box_idx]

        no_obj_conf_loss = pred_no_obj_conf.square().sum()

        loss = (
            self.lambda_coord * xywh_loss
            + conf_loss
            + class_loss
            + self.lambda_noobj * no_obj_conf_loss
        )

        loss = loss / batch_size

        # Значения для мониторинга функции потерь.
        loss_item = loss.item()
        x_loss = (x_loss/batch_size).item()
        y_loss = (y_loss/batch_size).item()
        w_loss = (w_sqrt_loss/batch_size).item()
        h_loss = (h_sqrt_loss/batch_size).item()
        conf_loss = (conf_loss/batch_size).item()
        class_loss = (class_loss/batch_size).item()
        no_obj_conf_loss = (no_obj_conf_loss/batch_size).item()

        return loss, loss_item, x_loss, y_loss, w_loss, h_loss, conf_loss, class_loss, no_obj_conf_loss
