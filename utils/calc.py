from collections import OrderedDict
import numpy as np
import torch
from typing import Union


def model_proximal_term(state_dict1: OrderedDict, state_dict2: OrderedDict, miu: float):
    norm = 0.0
    for k, w1 in state_dict1.items():
        if w1.dtype not in [torch.float16, torch.float32, torch.float64]:
            continue
        w2 = state_dict2[k]
        norm += torch.linalg.norm(w1 - w2) ** 2
    norm *= miu / 2
    return norm


def divide_int(total: int, num: int, portion: Union[np.ndarray, list] = None):
    assert num > 0
    if portion is None:
        portion = np.ones(num)
    assert len(portion) == num

    p = np.asarray(portion)
    p = np.cumsum(p)
    p = np.divide(p, p[-1])
    res = np.round(np.multiply(p, total)).astype(np.int64)
    res[1:] = res[1:] - res[:-1]
    res = res.tolist()

    return res


def distance_matrix(mat: np.ndarray):
    return group_distance(mat, mat.copy())


def group_distance(mat1: np.ndarray, mat2: np.ndarray):
    assert mat1.shape[1] == mat2.shape[1]

    n1, n2 = mat1.shape[0], mat2.shape[0]
    mat1 = np.repeat(mat1, n2, axis=0)
    mat1 = np.reshape(mat1, (n1, n2, -1))
    dist = np.sqrt(np.sum((mat1 - mat2) ** 2, axis=2))
    return dist
