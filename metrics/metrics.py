import numpy as np


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Функция для расчёта Average Precision для каждого класса при разных
    значениях метрики IoU между предсказанными рамками и ground_True рамками.

    Parameters:
        tp: numpy.array, size (N, 10), является ли каждая из N предсказанных
            рамок TP при разных пороговых значениях IoU. При оценке используется
            IoU_threshold от 0.50 до 0.95 с шагом 0.05 (@[.50:.05:.95]).
        conf: numpy.array, size (N, ), вероятность предсказанного класса.
        pred_cls: numpy.array, size (N, ), предсказанный класс.
        target_cls: numpy.array, size (M, ), класс ground_True рамки.
    Returns:
        ap: numpy.array, size (nc, 10), значение Average Precision для каждого
        класса при разных значениях метрики IoU между предсказанными рамками и
        ground_True рамками.
    """
    # Индексы отсортированные по убыванию значения вероятности класса.
    idx = np.argsort(-conf)

    # Сортировка тензоров согласно индексам idx.
    tp = tp[idx]
    conf = conf[idx]
    pred_cls = pred_cls[idx]

    # Получение классов ground_True рамок и их количество.
    un_cls, num_cls = np.unique(target_cls, return_counts=True)
    # Количество классов.
    nc = un_cls.shape[0]

    ap = np.zeros((nc, tp.shape[1]))    # size (nc, 10)

    # Считаем AP для каждого класса с разным значением IoU_threshold.
    for ci, cls in enumerate(un_cls):
        i = pred_cls == cls
        # Кол-во ground_True рамок с классом cls.
        n_l = num_cls[ci]
        # Кол-во предсказанных рамок с классом cls.
        n_p = i.sum()

        if n_p == 0 or n_l == 0:
            continue

        # Накопленная сумма TP для класса cls
        tpc = tp[i].cumsum(axis=0)          # size (n_p, 10)
        # Накопленная сумма FP для класса cls
        fpc = (1 - tp[i]).cumsum(axis=0)    # size (n_p, 10)

        # Precision
        precision = tpc / (tpc + fpc)  # size (n_p, 10)

        # Recall
        recall = tpc / (n_l + eps)  # size (n_p, 10)

        # Считаем AP для разного значения IoU_threshold.
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    return ap   # size (nc, 10)

def compute_ap(recall, precision):
    # Добавляем крайние значения.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Измените порядок расположения элементов в массиве на противоположный.
    # Были значения от 1 до 0, стали от 0 до 1.
    mpre = np.flip(mpre)

    # Продвигаясь по массиву слева направо заменяю элемент на максимальный 
    # до него.
    # Пр: [0, 2, 1, 1, 4, 3, 3] -> [0, 2, 2, 2, 4, 4, 4]
    mpre = np.maximum.accumulate(mpre)

    # Возвращаю массив к изначальному порядку элементов, от 1 до 0.
    mpre = np.flip(mpre)

    # Интерполяция по 101 точке.
    x = np.linspace(0, 1, 101)

    # Подсчёт площади.
    # В Numpy вурсии 2.0 и выше np.trapz нужно заменить на np.trapezoid
    ap = np.trapz(np.interp(x, mrec, mpre), x)

    return ap
