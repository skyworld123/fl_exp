import abc
from abc import ABC
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

__all__ = ['ClassificationMetrics', 'ClassificationMetricsTopK', 'ClassificationReport',
           'ClassificationReportTopK']


class PredGtMetric(ABC):
    def __init__(self):
        self.pred = np.empty(0, dtype=np.int64)
        self.gt = np.empty(0, dtype=np.int64)

    def update(self, pred, gt):
        pred, gt = np.asarray(pred, dtype=np.int64), np.asarray(gt, dtype=np.int64)
        assert pred.shape == gt.shape, 'Prediction and ground truth should have the same shape'
        self.pred = np.hstack([self.pred, pred])
        self.gt = np.hstack([self.gt, gt])

    @abc.abstractmethod
    def evaluate(self):
        pass


class TopKMetric(ABC):
    def __init__(self,
                 top_k=None):
        self.pred = np.empty(0, dtype=np.int64)
        self.gt = np.empty(0, dtype=np.int64)

        if top_k is None:
            top_k = []
        elif isinstance(top_k, int):
            top_k = [top_k]
        elif isinstance(top_k, list):
            assert all(isinstance(x, int) for x in top_k)
        top_k.sort()
        for k in top_k:
            assert isinstance(k, int) and k >= 1

        self.top_k = top_k
        self.top_k_hit_count = [0 for _ in top_k]

    def update(self, logit, gt):
        logit = np.asarray(logit)
        pred = np.argmax(logit, axis=1).astype(np.int64)
        gt = np.asarray(gt, dtype=np.int64)

        assert pred.shape == gt.shape, 'Prediction and ground truth should have the same shape'
        self.pred = np.hstack([self.pred, pred])
        self.gt = np.hstack([self.gt, gt])

        # top-k
        idx_rank = np.fliplr(np.argsort(logit, axis=1))
        for ki, k in enumerate(self.top_k):
            idx = np.nonzero(idx_rank == gt.repeat(idx_rank.shape[1]).reshape(logit.shape))[1]
            self.top_k_hit_count[ki] += np.count_nonzero(idx < k)

    @abc.abstractmethod
    def evaluate(self):
        pass


class ClassificationMetrics(PredGtMetric):
    """
    A metric class that reports overall metrics of classification tasks.
    """

    def __init__(self,
                 multi_class=True,
                 precision_average='weighted',
                 recall_average='weighted',
                 f1_average='weighted'):
        super(ClassificationMetrics, self).__init__()

        self.p_average = precision_average if multi_class else 'binary'
        self.r_average = recall_average if multi_class else 'binary'
        self.f1_average = f1_average if multi_class else 'binary'

    def evaluate(self):
        acc = accuracy_score(self.gt, self.pred)
        p = precision_score(self.gt, self.pred, average=self.p_average)
        r = recall_score(self.gt, self.pred, average=self.r_average)
        f1 = f1_score(self.gt, self.pred, average=self.f1_average)
        fmt = '%.6f'
        result = {
            'accuracy': float(fmt % acc),
            'precision': float(fmt % p),
            'recall': float(fmt % r),
            'f1': float(fmt % f1),
        }
        return result


class ClassificationMetricsTopK(TopKMetric):
    """
    A metric class that reports overall metrics (including top-k) of classification tasks.
    """

    def __init__(self,
                 multi_class=True,
                 top_k=None,
                 precision_average='weighted',
                 recall_average='weighted',
                 f1_average='weighted'):
        super(ClassificationMetricsTopK, self).__init__(top_k)

        self.p_average = precision_average if multi_class else 'binary'
        self.r_average = recall_average if multi_class else 'binary'
        self.f1_average = f1_average if multi_class else 'binary'

    def evaluate(self):
        acc = accuracy_score(self.gt, self.pred)
        p = precision_score(self.gt, self.pred, average=self.p_average)
        r = recall_score(self.gt, self.pred, average=self.r_average)
        f1 = f1_score(self.gt, self.pred, average=self.f1_average)
        fmt = '%.6f'
        result = {
            'accuracy': float(fmt % acc),
            'precision': float(fmt % p),
            'recall': float(fmt % r),
            'f1': float(fmt % f1),
        }

        for ki, k in enumerate(self.top_k):
            top_k = self.top_k_hit_count[ki] / len(self.gt)
            result[f'top-{k}'] = float(fmt % top_k)

        return result


class ClassificationReport(PredGtMetric):
    """
    A metric class that reports class-wise metrics of classification tasks.
    """

    def __init__(self):
        super(ClassificationReport, self).__init__()

    def evaluate(self):
        result = classification_report(self.gt, self.pred, output_dict=True)
        return result


class ClassificationReportTopK(TopKMetric):
    """
    A metric class that reports class-wise metrics and top-k metric of classification tasks.
    """

    def __init__(self,
                 top_k=None):
        super(ClassificationReportTopK, self).__init__(top_k)

    def evaluate(self):
        result = classification_report(self.gt, self.pred, output_dict=True)

        fmt = '%.6f'
        for ki, k in enumerate(self.top_k):
            top_k = self.top_k_hit_count[ki] / len(self.gt)
            result[f'top-{k}'] = float(fmt % top_k)

        return result
