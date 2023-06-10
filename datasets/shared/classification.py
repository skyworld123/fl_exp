import numpy as np

from .custom import TrainSetDivision, TestSet

__all__ = ['ClsTrainSetDivision', 'ClsTestSet']


class ClsTrainSetDivision(TrainSetDivision):
    """
    An abstract Dataset class for divided subsets of the original dataset (for classification tasks),
    which are held by clients.

    Args:
        dataset: The original train set.
        targets: The targets of original train set.
        idxs: Indices to index the subset.
    """

    def __init__(self, dataset, targets, idxs, num_classes):
        super(ClsTrainSetDivision, self).__init__(dataset, targets, idxs)
        self.num_classes = num_classes
        self.np_idxs = idxs

        div_targets = np.asarray(self.targets)
        unique_div_targets = np.unique(div_targets)
        samples_by_class = {}
        for t in unique_div_targets:
            class_idx = np.nonzero(np.equal(div_targets, t))
            samples_by_class[t] = idxs[class_idx]
        self.samples_by_class = samples_by_class


class ClsTestSet(TestSet):
    """
    Args:
        dataset: The original test set.
        targets: The targets of original test set.
    """

    def __init__(self, dataset, targets):
        super(ClsTestSet, self).__init__(dataset, targets)

        unique_targets = np.unique(targets)
        samples_by_class = {}
        for t in unique_targets:
            samples_by_class[t] = np.nonzero(np.equal(targets, t))[0]
        self.samples_by_class = samples_by_class
