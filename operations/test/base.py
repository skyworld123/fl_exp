import abc
from abc import ABC
import copy
import torch
from torch.utils.data import DataLoader


class BaseTester(ABC):
    best_metric_options = {
        'name': str(),
        'compare': str(),
    }

    def __init__(self,
                 dataloader: DataLoader,
                 model: torch.nn.Module,
                 device=None,
                 verbose=False,
                 metric_params=None):
        self.dataloader = dataloader
        self.model = model
        if device is None:
            print('[WARN] Device not specified in tester. CPU will be used by default.')
            device = 'cpu'
        self.device = device
        self.metric_params = {} if metric_params is None else metric_params
        self.verbose = verbose  # default verbose setting

        self.model.to(self.device)

    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)

    def export_model(self):
        state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.to('cpu')
        state_dict = copy.deepcopy(state_dict)
        return state_dict

    @abc.abstractmethod
    def test(self):
        pass
