import abc
from abc import ABC
import copy
import torch
from torch.utils.data import DataLoader

from utils.lr_schedulers import step_in_batch


class BaseTrainer(ABC):
    def __init__(self,
                 dataloader: DataLoader,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler=None,
                 device=None,
                 verbose=False,
                 public_model=True,
                 grad_acc_num_iters=None):
        self.dataloader = dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        if device is None:
            print('[WARN] Device not specified in trainer. CPU will be used by default.')
            device = 'cpu'
        self.device = device
        self.verbose = verbose  # default verbose setting
        self.public_model = public_model
        self.grad_acc_num_iters = grad_acc_num_iters

        self.epoch = 0
        self.lr_step_in_batch = self.lr_scheduler is not None \
            and type(self.lr_scheduler) in step_in_batch

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
    def train_epoch(self):
        pass
