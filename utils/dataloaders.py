import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from PIL import Image
from torchvision import tv_tensors
from .convert_box import xyxy2xywh
from .augmentation import DataAugment


class PascalVOC:
    def __init__(self, dirs):
        if isinstance(dirs, str):
            dirs = [dirs]
        if not isinstance(dirs, list):
            raise TypeError('Параметр dirs должен быть или строкой, или список содержащий путь\пути до папки с изображениями.')

        self.dirs = dirs
        self.img_formats = ".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"

        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]

        self.classes = sorted(classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

    def create_data(self):
        """
        Данный метод каждое изображения, у которого есть разметка, кеширует на
        диск и сохраняет разметку в формате:
        img_data = {
            'path_img': str, путь до изображения,
            'img_name': str, имя файла с изображением,
            'size_img': [int, int], размер изображения,
            'depth': int, кол-во каналов,
            'objs_ann': [
                {
                    'label': int, класс объекта,
                    'bbox': [int, int, int, int], параметры ограничивающей рамки (x1y1x2y2)
                },
                { label и bbox второго объекта },
                ...
                { label и bbox N-го объекта }
            ]
        }

        Метод возвращает список с разметкой изображений:
        target_data = [
            { img_data_1 },
            { img_data_2 },
            ...
            { img_data_M }
        ]
        """
        # Список с разметкой всех изображений.
        target_data = []
        for path in self.dirs:
            # Список с разметкой изображений в одной папке.
            target_data_on_path = []
            self.path = Path(path)
            self.path_anns = self.path.joinpath('Annotations')
            self.path_imgs = self.path.joinpath('JPEGImages')
            for im_f in self.path_imgs.iterdir():
                # Проверяю формат файла.
                true_format = im_f.suffix in self.img_formats

                # Получаю путь до файла с разметкой.
                ann_f = Path(self.path_anns, f'{im_f.name.split(".")[0]}.xml')

                # Проверяю наличие файла с разметкой.
                exists = ann_f.exists()

                if not true_format or not exists:
                    continue

                img_data = {}
                # Парсинг файла с аннотацией.
                ann_ps = ET.parse(ann_f)

                # Сохраняю путь до файла с изображения.
                img_data['path_img'] = im_f

                # Кэширование изображений на диск
                np_f = Path(im_f).with_suffix(".npy")
                if not np_f.exists():
                    np.save(np_f, np.array(Image.open(im_f).convert('RGB')))

                # Сохраняю имя файла с изображением.
                img_data['img_name'] = ann_ps.find('filename').text

                # Сохраняю размер изображения и кол-во каналов.
                w = int(ann_ps.find('size').find('width').text)
                h = int(ann_ps.find('size').find('height').text)
                c = int(ann_ps.find('size').find('depth').text)
                img_data['size_img'] = [w, h]
                img_data['depth'] = c


                # Список для хранения аннотации каждого объекта на изображении.
                objs_ann = []

                for obj in ann_ps.findall('object'):
                    # Словарь для хранения аннотации одного объекта.
                    obj_ann = {}
                    # Сохраняю класс объекта.
                    obj_ann['label'] = self.class_to_idx[obj.find('name').text]

                    bbox_ann = obj.find('bndbox')
                    bbox = [
                        int(float(bbox_ann.find('xmin').text))-1,   # x1
                        int(float(bbox_ann.find('ymin').text))-1,   # y1
                        int(float(bbox_ann.find('xmax').text))-1,   # x2
                        int(float(bbox_ann.find('ymax').text))-1    # y2
                    ]
                    # Сохраняю параметры ограничивающей рамки (x1y1x2y2).
                    obj_ann['bbox'] = bbox

                    objs_ann.append(obj_ann)
                # Сохраняю аннотации всех объектов на изображении.
                img_data['objs_ann'] = objs_ann

                target_data_on_path.append(img_data)

            if len(target_data_on_path) == 0:
                print(f"\033[33m\033[1mWarning\033[0m: В папке \033[34m'{self.path}'\033[0m изображений имеющих разметку не найдено.")
                continue
            target_data += target_data_on_path

        assert target_data, f"\033[31mИзображений имеющих разметку не найдено.\033[0m"

        print(f"В наборе данных \033[34m\033[1m{len(target_data)}\033[0m размеченных изображений.")

        return target_data


class DatasetPascalVOCYOLOv1(Dataset):
    def __init__(self, data, mode=None, transform=None):
        self.data = data
        self.transform = transform if transform else DataAugment(mode)
        self.S = 7
        self.B = 2
        self.C = 20

        classes = [
                'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
                'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
            ]

        self.classes = sorted(classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Получаю данные об изображении.
        data = self.data[index]
        # Загружаю изображение.
        path_img = data['path_img']
        np_f = Path(path_img).with_suffix(".npy")
        if np_f.exists():
            img = np.load(np_f).transpose((2, 0, 1))
            img = torch.as_tensor(img)
        else:
            img = Image.open(path_img).convert('RGB')


        # Получаю ширину и высоту изображения.
        W, H = data['size_img']
        # Получаю список с рамками.
        bboxes = [obj_ann['bbox'] for obj_ann in data['objs_ann']]
        bboxes = tv_tensors.BoundingBoxes(
            bboxes,
            format="XYXY",
            canvas_size=[H, W]
        )

        # Получаю список с классами объктов.
        labels = [obj_ann['label'] for obj_ann in data['objs_ann']]

        target_bl = {
            'boxes': bboxes,
            'labels': torch.tensor(labels)
        }

        img, target_bl = self.transform(img, target_bl)

        bboxes = target_bl['boxes']
        labels = target_bl['labels']

        # Получаю рамки в относительных значениях.
        whwh = torch.tensor([448, 448, 448, 448])
        bboxes_xyxy = bboxes / whwh
        # Тензор (x1, y1, x2, y2, labels)
        boxes_classes = torch.cat([bboxes_xyxy, labels.unsqueeze(dim=1)], dim=1)

        # Получаю рамки в формате xywh.
        bboxes = xyxy2xywh(bboxes_xyxy)

        target = torch.zeros(self.S, self.S, 5 * self.B + self.C)

        n_l = len(labels)
        cls_labels = torch.zeros(n_l, self.C)
        cls_labels[[*range(n_l)], labels] = 1

        conf = torch.ones(n_l, 1)

        # Ширина и высота одной ячейки в относительных значениях.
        w_cell = 1 / self.S
        h_cell = 1 / self.S

        wh_cell = torch.tensor([w_cell, h_cell])
        # Получение индексов "столбцов" и "строк" тех ячеек, где расположены
        # центры объектов.
        idx = torch.floor(bboxes[:, :2] / wh_cell)
        ij = torch.tensor_split(idx.long(), 2, dim=1)

        # Преобразование координат X и Y в X_CELL и Y_CELL.
        bboxes[:, :2] = (bboxes[:, :2] % wh_cell) / wh_cell
        bboxes_conf_labels = torch.cat([bboxes, conf, bboxes, conf, cls_labels], dim=1)
        target[ij[1], ij[0]] = bboxes_conf_labels[:, None, :]

        return img, target, boxes_classes

    @staticmethod
    def collate_fn(batch):
        """
        Статический метод для формирования батча данных.
        """
        data = list(zip(*batch))
        imgs = torch.stack(data[0])
        targets = torch.stack(data[1])
        box_class = data[2]
        return imgs, targets, box_class


def create_loader(
    path,
    split=[0.8, 0.2],
    batch_size=16,
    workers=1,
    pin_memory=True,
    shuffle_train=True
):
    data = PascalVOC(path).create_data()
    train_data, val_data = random_split(data, split)

    train_set = DatasetPascalVOCYOLOv1(train_data, mode='train')
    val_set = DatasetPascalVOCYOLOv1(val_data, mode='test')

    train_loader = DataLoader(
        train_set,
        batch_size,
        shuffle=shuffle_train,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=DatasetPascalVOCYOLOv1.collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=DatasetPascalVOCYOLOv1.collate_fn
    )
    return train_loader, val_loader
