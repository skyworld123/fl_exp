import copy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .fedavg import FedAvgTrainer
from utils import model_proximal_term


class FedADMMTrainer(FedAvgTrainer):
    def __init__(self,
                 dataloader: DataLoader,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 rho: float,
                 lr_scheduler=None,
                 device=None,
                 verbose=False,
                 public_model=True,
                 grad_acc_num_iters=None):
        super(FedADMMTrainer, self).__init__(
            dataloader, model, criterion, optimizer, lr_scheduler,
            device, verbose, public_model, grad_acc_num_iters
        )
        self.rho = rho
        self.server_state_dict = None
        self.multiplier = None
        self.updated_multiplier = None

    def receive_server_state_dict(self, state_dict):
        self.server_state_dict = state_dict

    def initialize_multiplier(self):
        assert self.server_state_dict is not None
        self.multiplier = copy.deepcopy(self.server_state_dict)
        for k, v in self.multiplier.items():
            torch.fill_(self.multiplier[k], 0)

    def train_epoch(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        batch_loss = []
        if self.updated_multiplier is not None:
            self.multiplier = copy.deepcopy(self.updated_multiplier)

        self.model.train()
        bar = tqdm(total=len(self.dataloader)) if verbose else None
        for bi, (inp, label) in enumerate(self.dataloader):
            if isinstance(inp, torch.Tensor):
                inp = inp.to(self.device)
            elif isinstance(inp, list) or isinstance(inp, tuple):
                inp = [item.to(self.device) for item in inp if isinstance(item, torch.Tensor)]
            label = label.to(self.device)
            if bi == 0:
                self.optimizer.zero_grad()

            output = self.model(inp)
            loss = self.criterion(output, label)

            # Lagrangian multiplier term
            mt = 0.0
            state_dict = self.export_model()
            for k, w in state_dict.items():
                if w.dtype not in [torch.float32, torch.float16, torch.float64]:
                    continue
                m = self.multiplier[k]
                sw = self.server_state_dict[k]
                mt += torch.sum(torch.mul(m, w - sw))
            loss += mt

            # proximal term
            state_dict = self.export_model()
            norm = model_proximal_term(state_dict, self.server_state_dict, self.rho)
            loss += norm

            if self.grad_acc_num_iters is not None:
                loss /= self.grad_acc_num_iters
                loss.backward()
                if (bi + 1) % self.grad_acc_num_iters == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.lr_step_in_batch:
                self.lr_scheduler.step(self.epoch + bi / len(self.dataloader))

            batch_loss.append(loss.item())

            if verbose:
                bar.update()
        if verbose:
            bar.close()

        # update multiplier
        self.updated_multiplier = copy.deepcopy(self.multiplier)
        state_dict = self.export_model()
        for k, w in state_dict.items():
            sw = self.server_state_dict[k]
            delta = self.rho * (w - sw)
            delta = delta.type_as(self.updated_multiplier[k])
            self.updated_multiplier[k] += delta

        if self.lr_scheduler and not self.lr_step_in_batch:
            self.lr_scheduler.step()
        self.epoch += 1

        avg_loss = sum(batch_loss) / len(batch_loss)
        return avg_loss
