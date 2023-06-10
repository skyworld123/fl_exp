import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ['TrainSetDivision', 'TestSet']


class TrainSetDivision(Dataset):
    """
    An abstract Dataset class for divided subsets of the original dataset,
    which are held by clients.

    Args:
        dataset: The original train set.
        targets: The targets of original train set.
        idxs: Indices to index the subset.
    """

    def __init__(self, dataset, targets, idxs):
        self.dataset = dataset
        div_targets = np.asarray(targets)[idxs]
        self.targets = div_targets.tolist()
        self.idxs = torch.LongTensor(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class TestSet(Dataset):
    """
    Args:
        dataset: The original test set.
        targets: The targets of original test set.
    """

    def __init__(self, dataset, targets):
        self.dataset = dataset
        self.targets = targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]
