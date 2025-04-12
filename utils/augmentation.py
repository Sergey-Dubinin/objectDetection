import torch
import torch.nn as nn
from torchvision.transforms import v2


class DataAugment(nn.Module):
    """
    Класс для расширения набора данных. Включает в себя преобразования
    изображений и ограничивающих рамок, а так же нормализацию изображений.
    """

    def __init__(self, mode=None, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.mode = 'train' if mode is None else mode
        self.transform = {
            'train': v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                # ...
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomAffine(
                    degrees=0,
                    translate=(0.2, 0.2),
                    scale=(0.8, 1.2)
                ),
                v2.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue=(-0.2, 0.2),
                ),
                v2.Resize(size=(448, 448)),
                v2.SanitizeBoundingBoxes(),
                # ...
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std)
            ]),
            'test': v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                # ...
                v2.Resize(size=(448, 448)),
                v2.SanitizeBoundingBoxes(),
                # ...
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std)
            ])
        }

    def forward(self, *args, **kwargs):
        new = self.transform[self.mode](*args, **kwargs)  # transform
        return new
    