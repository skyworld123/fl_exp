import copy

from .base import BaseManager
from .fedavg import FedAvgClient, FedAvgServer
from .tools import *
from .trainers import FedADMMTrainer
from operations.test import BaseTester


class FedADMMManager(BaseManager):
    def __init__(self,
                 global_epoch: int,
                 local_epoch: int,
                 num_clients: int,
                 frac: float,
                 rho: float,
                 average_aggregate: bool = False):
        """
        average_aggregate: When set to false, η=1; when true, η=|S^t|/m
            (|S^t| denotes the data amount of a client and m the number of clients)
        """
        super(FedADMMManager, self).__init__(global_epoch)
        self.local_epoch = local_epoch
        self.num_clients = num_clients
        self.frac = frac
        self.rho = rho
        self.average_aggregate = average_aggregate

    def generate_clients(self, cfg):
        dataloaders = generate_train_dataloaders_default(cfg, self.num_clients)

        # generate clients
        print(f'Generating {self.num_clients} clients...')
        params = {'rho': self.rho}
        trainers = generate_trainers_default(cfg, dataloaders, FedADMMTrainer, params)
        clients = []
        for ci in range(self.num_clients):
            client = FedADMMClient(trainers[ci], self.local_epoch, self.rho)
            clients.append(client)

        return clients

    def generate_server(self, cfg):
        dataloader = generate_test_dataloader_default(cfg)

        # generate server
        print('Generating server...')
        tester = generate_tester_default(cfg, dataloader)
        test_interval = cfg.args['save_interval'] if cfg.args['eval'] else 0
        server = FedADMMServer(tester, self.global_epoch, self.frac, self.average_aggregate, test_interval)

        return server


class FedADMMClient(FedAvgClient):
    def __init__(self,
                 trainer: FedADMMTrainer,
                 local_epoch: int,
                 rho: float):
        super(FedADMMClient, self).__init__(trainer, local_epoch)
        self.trainer = trainer
        self.rho = rho

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
        if self.trainer.multiplier is None:
            self.trainer.initialize_multiplier()
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

    def send(self):
        chosen = self.buffer('chosen', False)
        if not chosen:
            return

        state_dict = self.trainer.export_model()
        server_state_dict = self.trainer.server_state_dict
        multiplier = self.trainer.multiplier
        updated_multiplier = self.trainer.updated_multiplier
        for k in state_dict:
            delta = -server_state_dict[k] + (updated_multiplier[k] - multiplier[k]) / self.rho
            delta = delta.type_as(state_dict[k])
            state_dict[k] += delta

        self.server.receive(self, state_dict)
        self.server.receive(self, self.trainer.optimizer.param_groups[0]['lr'], 'learning_rate')
        self.server.receive(self, self.avg_loss, 'loss')
        self.server.receive(self, self._data_amount(), 'data_amount')
        if self.trainer.lr_scheduler is not None:
            self.server.receive(self, self.trainer.lr_scheduler.state_dict(), 'lr_scheduler')

        self.set_buffer(False, 'chosen')


class FedADMMServer(FedAvgServer):
    def __init__(self,
                 tester: BaseTester,
                 global_epoch: int,
                 frac: float,
                 average_aggregate=False,
                 test_interval: int = 1):
        super(FedADMMServer, self).__init__(tester, global_epoch, frac, test_interval)
        self.average_aggregate = average_aggregate

    def post_process(self):
        ret = {}
        scalars = {}
        ret['scalars'] = scalars

        # record learning rate
        learning_rates = [self.buffer(c, 'learning_rate') for c in self.chosen_clients]
        learning_rate = sum(learning_rates) / len(learning_rates)
        scalars['learning_rate'] = learning_rate

        # record a client's lr state to global lr state
        lr_scheduler = self.buffer(self.chosen_clients[0], 'lr_scheduler')
        if lr_scheduler is not None:
            self.global_lr_scheduler = lr_scheduler

        # record loss
        losses = [self.buffer(c, 'loss') for c in self.chosen_clients]
        avg_loss = sum(losses) / len(losses)
        avg_loss = float('%.6f' % avg_loss)
        print(f'Global epoch {self.epoch + 1}/{self.epochs}: '
              f'avg_loss={avg_loss}, lr={learning_rate}')
        scalars['train_loss'] = avg_loss

        # aggregate parameters from the chosen clients' models
        w_list = [self.buffer(c) for c in self.chosen_clients]
        if self.average_aggregate:
            dp = [1] * len(self.chosen_clients)
        else:
            dp = [self.buffer(c, 'data_amount') for c in self.chosen_clients]
        dp = [x / sum(dp) for x in dp]

        w_avg = copy.deepcopy(w_list[0])
        for key in w_avg.keys():
            w_avg[key] = w_list[0][key] * dp[0]
        for key in w_avg.keys():
            for i in range(1, len(w_list)):
                w_avg[key] += w_list[i][key] * dp[i]

        self.tester.model.train()
        state_dict = self.tester.export_model()
        for k in state_dict:
            delta = w_avg[k]
            delta = delta.type_as(state_dict[k])
            state_dict[k] += delta
        self.tester.load_model(state_dict)

        # test the aggregated model on the test set
        if self.test_interval > 0 \
                and ((self.epoch + 1) % self.test_interval == 0 or self.epoch + 1 == self.epochs):
            _, res_scalars, best_metric = self.tester.test()
            scalars.update(res_scalars)
            ret['best_metric'] = best_metric

        return ret
