import numpy as np
import datetime
import torch
from typing import Union


def sec2eta(sec: float):
    sec = max(sec, 0)
    s = int(sec)

    m = s // 60
    s -= m * 60
    h = m // 60
    m -= h * 60
    d = h // 24
    h -= d * 24

    eta = '{:d}:{:02d}:{:02d}'.format(h, m, s)
    if d > 0:
        eta = '{:d}d '.format(d) + eta

    return eta


def current_ymd_hms():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def show_time():
    time_stamp = '\033[1;31;40m[' + str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ']\033[0m'
    return time_stamp


def data_size_with_unit(size: int):
    size = max(size, 0)
    units = ['B', 'KiB', 'MiB', 'GiB']
    category = 1024

    for i, u in enumerate(units):
        if size < category:
            return f'{round(size, 2)}{u}'
        if i < len(units) - 1:
            size /= category

    return f'{round(size, 2)}{units[-1]}'


def print_long_vector(vector: Union[list, np.ndarray, torch.Tensor],
                      keep: int = 3,
                      decimals: int = None):
    if isinstance(vector, list):
        vector = np.asarray(vector)
    elif isinstance(vector, torch.Tensor):
        vector = vector.numpy()
    assert len(vector.shape) == 1

    if vector.dtype in [np.float16, np.float32, np.float64] and decimals is not None:
        vector = np.round(vector, decimals=decimals)

    ret = '['
    if vector.shape[0] > keep * 2:
        vector_end = [str(x) for x in vector[:keep]]
        vector_end = ' '.join(vector_end)
        ret += vector_end
        ret += '...'
        vector_end = [str(x) for x in vector[-keep:]]
        vector_end = ' '.join(vector_end)
        ret += vector_end
    else:
        vector_str = [str(x) for x in vector]
        vector_str = ' '.join(vector_str)
        ret += vector_str
    ret += ']'

    return ret
