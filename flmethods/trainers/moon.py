import copy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .fedavg import FedAvgTrainer


class MOONTrainer(FedAvgTrainer):
    def __init__(self,
                 dataloader: DataLoader,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 miu: float,
                 temperature: float,
                 lr_scheduler=None,
                 device=None,
                 verbose=False,
                 public_model=True,
                 grad_acc_num_iters=None):
        super(MOONTrainer, self).__init__(
            dataloader, model, criterion, optimizer, lr_scheduler,
            device, verbose, public_model, grad_acc_num_iters
        )
        self.miu = miu
        self.temperature = temperature

        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        if self.public_model:
            # save memory
            self.last_local_model = self.export_model()
            self.global_model = self.export_model()
        else:
            self.last_local_model = copy.deepcopy(self.model)
            self.global_model = copy.deepcopy(self.model)
            self.last_local_model.to(device)
            self.global_model.to(device)
        self.last_model_valid = False

    def receive_server_state_dict(self, state_dict):
        if self.public_model:
            self.global_model = state_dict
        else:
            self.global_model.train()
            self.global_model.load_state_dict(state_dict)

    def train_epoch(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        batch_loss = []

        # get proj of last_local_model and global_model in advance
        proj_last_list = []
        proj_global_list = []
        if self.public_model:
            self.model.train()
            self.load_model(self.last_local_model)
            self.model.eval()
            for inp, label in self.dataloader:
                if isinstance(inp, torch.Tensor):
                    inp = inp.to(self.device)
                elif isinstance(inp, list) or isinstance(inp, tuple):
                    inp = [item.to(self.device) for item in inp if isinstance(item, torch.Tensor)]
                self.model(inp)
                proj_last = self.model.get_proj().detach().cpu()
                proj_last_list.append(proj_last)

            self.model.train()
            self.load_model(self.global_model)
            self.model.eval()
            for inp, label in self.dataloader:
                if isinstance(inp, torch.Tensor):
                    inp = inp.to(self.device)
                elif isinstance(inp, list) or isinstance(inp, tuple):
                    inp = [item.to(self.device) for item in inp if isinstance(item, torch.Tensor)]
                self.model(inp)
                proj_global = self.model.get_proj().detach().cpu()
                proj_global_list.append(proj_global)
        else:
            self.last_local_model.eval()
            self.global_model.eval()
            for inp, label in self.dataloader:
                if isinstance(inp, torch.Tensor):
                    inp = inp.to(self.device)
                elif isinstance(inp, list) or isinstance(inp, tuple):
                    inp = [item.to(self.device) for item in inp if isinstance(item, torch.Tensor)]
                self.last_local_model(inp)
                proj_last = self.last_local_model.get_proj()
                proj_last_list.append(proj_last)

            for inp, label in self.dataloader:
                if isinstance(inp, torch.Tensor):
                    inp = inp.to(self.device)
                elif isinstance(inp, list) or isinstance(inp, tuple):
                    inp = [item.to(self.device) for item in inp if isinstance(item, torch.Tensor)]
                self.global_model(inp)
                proj_global = self.global_model.get_proj()
                proj_global_list.append(proj_global)

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

            if self.last_model_valid:
                proj = self.model.get_proj()
                proj_last = proj_last_list[bi].to(self.device)
                proj_global = proj_global_list[bi].to(self.device)

                pos = self.cos(proj, proj_global)
                neg = self.cos(proj, proj_last)
                proj_logit = torch.vstack((pos, neg)).T
                proj_logit /= self.temperature
                proj_label = torch.zeros(proj_logit.shape[0]).long().to(self.device)
                loss2 = self.cross_entropy_loss(proj_logit, proj_label)
                loss += self.miu * loss2

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

        if self.lr_scheduler and not self.lr_step_in_batch:
            self.lr_scheduler.step()
        self.epoch += 1

        # record last local model
        if self.public_model:
            self.last_local_model = self.export_model()
        else:
            self.last_local_model.train()
            self.last_local_model.load_state_dict(self.model.state_dict())
        self.last_model_valid = True

        avg_loss = sum(batch_loss) / len(batch_loss)
        return avg_loss
