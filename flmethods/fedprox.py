from .base import BaseManager
from .fedavg import FedAvgClient
from .fedavg import FedAvgServer as FedProxServer
from .tools import *
from .trainers import FedProxTrainer


class FedProxManager(BaseManager):
    def __init__(self,
                 global_epoch: int,
                 local_epoch: int,
                 num_clients: int,
                 frac: float,
                 miu: float):
        super(FedProxManager, self).__init__(global_epoch)
        self.local_epoch = local_epoch
        self.num_clients = num_clients
        self.frac = frac
        self.miu = miu

    def generate_clients(self, cfg):
        dataloaders = generate_train_dataloaders_default(cfg, self.num_clients)

        # generate clients
        print(f'Generating {self.num_clients} clients...')
        params = {'miu': self.miu}
        trainers = generate_trainers_default(cfg, dataloaders, FedProxTrainer, params)
        clients = []
        for ci in range(self.num_clients):
            client = FedProxClient(trainers[ci], self.local_epoch)
            clients.append(client)

        return clients

    def generate_server(self, cfg):
        dataloader = generate_test_dataloader_default(cfg)

        # generate server
        print('Generating server...')
        tester = generate_tester_default(cfg, dataloader)
        test_interval = cfg.args['save_interval'] if cfg.args['eval'] else 0
        server = FedProxServer(tester, self.global_epoch, self.frac, test_interval)

        return server


class FedProxClient(FedAvgClient):
    def __init__(self,
                 trainer: FedProxTrainer,
                 local_epoch: int):
        super(FedProxClient, self).__init__(trainer, local_epoch)
        self.trainer = trainer

    def process(self):
        chosen = self.buffer('chosen', False)
        if not chosen:
            return

        # synchronize epoch (for lr_scheduler)
        self.trainer.epoch = self.buffer('global_epoch') * self.epochs

        # receive parameters from the server and train locally
        state_dict = self.buffer()
        self.trainer.load_model(state_dict)
        self.trainer.receive_server_state_dict(state_dict)
        if self.trainer.lr_scheduler is not None:
            global_lr_scheduler = self.buffer('global_lr_scheduler')
            if global_lr_scheduler is not None:
                self.trainer.lr_scheduler.load_state_dict(global_lr_scheduler)
                for param_group in self.trainer.optimizer.param_groups:
                    param_group['lr'] = global_lr_scheduler['_last_lr'][0]

        loss = 0
        for epoch in range(self.epochs):
            epoch_loss = self.trainer.train_epoch()
            loss += epoch_loss
        avg_loss = loss / self.epochs
        self.avg_loss = avg_loss

        # record loss
        avg_loss = '%.6f' % avg_loss
        print(f'Client {self.cid}: avg_loss={avg_loss}')
