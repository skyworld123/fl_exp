import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseTester
from datasets import ClsTestSet
from utils.metrics import *


class ClassificationTester(BaseTester):
    """
    A tester for classification tasks.
    """

    best_metric_options = {
        'name': 'accuracy',
        'compare': '+',
    }

    def __init__(self,
                 dataloader: DataLoader,
                 model: torch.nn.Module,
                 device=None,
                 verbose=False,
                 metric_params=None,
                 video=False):
        super(ClassificationTester, self).__init__(
            dataloader, model, device, verbose,
            metric_params
        )
        self.video = video
        if self.video:
            self.metric_typ = ClassificationMetricsTopK
        else:
            self.metric_typ = ClassificationMetrics

    def test_model_video(self, metric, verbose):
        with torch.no_grad():
            bar = tqdm(total=len(self.dataloader)) if verbose else None
            for inp, label in self.dataloader:
                gt = label.tolist()

                if isinstance(inp, torch.Tensor):
                    if len(inp.shape) == 6:  # (N,num_seg,C,T,H,W)
                        logits = []
                        for i in range(inp.shape[1]):
                            inp_seg = inp[:, i].to(self.device)
                            out = self.model(inp_seg)
                            logit = out.detach().cpu().numpy()
                            logits.append(logit)
                        logit = np.stack(logits).mean(axis=0)
                    else:  # (N,C,T,H,W)
                        inp = inp.to(self.device)
                        out = self.model(inp)
                        logit = out.detach().cpu().numpy()

                elif isinstance(inp, (list, tuple)):  # for PoseC3D RGB+pose
                    if len(inp[0].shape) == 6:  # (N,num_seg,C,T,H,W)
                        logits = []
                        for i in range(inp[0].shape[1]):
                            inp_seg = [item[:, i].to(self.device) for item in inp if isinstance(item, torch.Tensor)]
                            out = self.model(inp_seg)
                            logit = out.detach().cpu().numpy()
                            logits.append(logit)
                        logit = np.stack(logits).mean(axis=0)
                    else:  # (N,C,T,H,W)
                        inp = [item.to(self.device) for item in inp if isinstance(item, torch.Tensor)]
                        out = self.model(inp)
                        logit = out.detach().cpu().numpy()

                metric.update(logit, gt)
                if verbose:
                    bar.update()
            if verbose:
                bar.close()

    def test_model(self, metric, verbose):
        with torch.no_grad():
            bar = tqdm(total=len(self.dataloader)) if verbose else None
            for inp, label in self.dataloader:
                gt = label.tolist()
                if isinstance(inp, torch.Tensor):
                    inp = inp.to(self.device)
                elif isinstance(inp, list) or isinstance(inp, tuple):
                    inp = [item.to(self.device) for item in inp if isinstance(item, torch.Tensor)]
                out = self.model(inp)
                pred = out.argmax(1).detach().cpu().tolist()
                metric.update(pred, gt)

                if verbose:
                    bar.update()
            if verbose:
                bar.close()

    def test(self, verbose=None):
        print('Testing...')
        if verbose is None:
            verbose = self.verbose

        metric = self.metric_typ(**self.metric_params)
        self.model.eval()

        if self.video:
            self.test_model_video(metric, verbose)
        else:
            self.test_model(metric, verbose)

        result = metric.evaluate()
        scalars = result.copy()
        best_metric = result['accuracy']

        console = 'Test: '
        items = []
        for k, v in scalars.items():
            items.append(f'{k}={"%.6f" % v}')
        console += ', '.join(items)
        print(console)

        return result, scalars, best_metric


class ClassWiseClassificationTester(ClassificationTester):
    """
    Reports class-wise metrics.
    """

    best_metric_options = {
        'name': 'accuracy',
        'compare': '+',
    }

    def __init__(self,
                 dataloader: DataLoader,
                 model: torch.nn.Module,
                 device=None,
                 verbose=False,
                 metric_params=None,
                 video=False):
        super(ClassWiseClassificationTester, self).__init__(
            dataloader, model, device, verbose,
            metric_params, video
        )

        if self.video:
            assert 'top_k' in self.metric_params
            self.top_k = self.metric_params['top_k']
        else:
            self.top_k = None

    def test(self, verbose=None, print_detail=True):
        if verbose is None:
            verbose = self.verbose
        if print_detail:
            print('Testing...')

        if self.video:
            metric = ClassificationReportTopK(self.top_k)
        else:
            metric = ClassificationReport()
        self.model.eval()

        if self.video:
            self.test_model_video(metric, verbose)
        else:
            self.test_model(metric, verbose)

        result = metric.evaluate()
        scalars = {
            'accuracy': result['accuracy'],
            'weighted-precision': result['weighted avg']['precision'],
            'weighted-recall': result['weighted avg']['recall'],
            'weighted-f1-score': result['weighted avg']['f1-score'],
        }
        best_metric = result['accuracy']

        if print_detail:
            console = 'Test: '
            items = []
            for k, v in scalars.items():
                items.append(f'{k}={"%.6f" % v}')
            console += ', '.join(items)
            print(console)

        return result, scalars, best_metric
