from .base import BaseManager
from .fedprox import FedProxClient as MOONClient
from .fedavg import FedAvgServer as MOONServer
from .tools import *
from .trainers import MOONTrainer


class MOONManager(BaseManager):
    def __init__(self,
                 global_epoch: int,
                 local_epoch: int,
                 num_clients: int,
                 frac: float,
                 miu: float,
                 temperature: float):
        super(MOONManager, self).__init__(global_epoch)
        self.local_epoch = local_epoch
        self.num_clients = num_clients
        self.frac = frac
        self.miu = miu
        self.temperature = temperature

    def generate_clients(self, cfg):
        dataloaders = generate_train_dataloaders_default(cfg, self.num_clients)

        # generate clients
        print(f'Generating {self.num_clients} clients...')
        params = {'miu': self.miu, 'temperature': self.temperature}
        trainers = generate_trainers_default(cfg, dataloaders, MOONTrainer, params)
        clients = []
        for ci in range(self.num_clients):
            client = MOONClient(trainers[ci], self.local_epoch)
            clients.append(client)

        return clients

    def generate_server(self, cfg):
        dataloader = generate_test_dataloader_default(cfg)

        # generate server
        print('Generating server...')
        tester = generate_tester_default(cfg, dataloader)
        test_interval = cfg.args['save_interval'] if cfg.args['eval'] else 0
        server = MOONServer(tester, self.global_epoch, self.frac, test_interval)

        return server
