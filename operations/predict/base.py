import abc
from abc import ABC
import torch


class BasePredictor(ABC):
    def __init__(self,
                 inputs: list,
                 transform,
                 model: torch.nn.Module,
                 device=None,
                 show_output=True,
                 output_dir=None):
        self.inputs = inputs
        self.transform = transform
        self.model = model
        if device is None:
            print('[WARN] Device not specified in predictor. CPU will be used by default.')
            device = 'cpu'
        self.device = device
        self.show_output = show_output
        self.output_dir = output_dir

        self.model.to(self.device)

    @abc.abstractmethod
    def predict(self):
        pass
