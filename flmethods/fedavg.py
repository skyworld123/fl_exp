import copy
import random
from torch.utils.data import DataLoader

from .base import BaseManager, BaseClient, BaseServer
from .tools import *
from .trainers import FedAvgTrainer
from operations.test import BaseTester


class FedAvgManager(BaseManager):
    def __init__(self,
                 global_epoch: int,
                 local_epoch: int,
                 num_clients: int,
                 frac: float,
                 local_bs: int = None):
        super(FedAvgManager, self).__init__(global_epoch)
        self.local_epoch = local_epoch
        self.num_clients = num_clients
        self.frac = frac
        self.local_bs = local_bs

    def generate_clients(self, cfg):
        # train set
        train_set = cfg.train_set

        # divisions of train set
        params = train_set['params']
        if train_set['root'] is not None:
            params.update({'root': train_set['root']})
        params.update({'train_transform': train_set['transform']})
        wrapper = train_set['type'](**params)
        params = train_set['division']
        if params is None:
            print('Dataset division options not specified in configuration file. Use iid by default.')
            params = {'method': 'iid'}
        params['num_clients'] = self.num_clients
        divisions = wrapper.get_train_set_division(**params)

        batch_size = train_set.get('batch_size', 1)
        if self.local_bs is None:
            self.local_bs = batch_size
        num_workers = cfg.args['num_workers']
        cfg_sampler = train_set.get('sampler')

        dataloaders = []
        for division in divisions.values():
            sampler = get_sampler(cfg_sampler, division, default='random')
            dataloader = DataLoader(division, batch_size=self.local_bs, sampler=sampler, num_workers=num_workers)
            dataloaders.append(dataloader)

        # generate clients
        print(f'Generating {self.num_clients} clients...')
        trainers = generate_trainers_default(cfg, dataloaders, FedAvgTrainer)
        clients = []
        for ci in range(self.num_clients):
            client = FedAvgClient(trainers[ci], self.local_epoch)
            clients.append(client)

        return clients

    def generate_server(self, cfg):
        dataloader = generate_test_dataloader_default(cfg)

        # generate server
        print('Generating server...')
        tester = generate_tester_default(cfg, dataloader)
        test_interval = cfg.args['save_interval'] if cfg.args['eval'] else 0
        server = FedAvgServer(tester, self.global_epoch, self.frac, test_interval)

        return server


class FedAvgClient(BaseClient):
    def __init__(self,
                 trainer: FedAvgTrainer,
                 local_epoch: int):
        super(FedAvgClient, self).__init__()
        self.trainer = trainer
        self.epochs = local_epoch

        self.avg_loss = 0.0

    def _data_amount(self):
        dataset = self.trainer.dataloader.dataset
        return len(dataset)

    def state(self) -> dict:
        state = {
            'lr_scheduler': self.trainer.lr_scheduler.state_dict() if self.trainer.lr_scheduler is not None else None
        }
        return state

    def load_state(self, state: dict):
        if self.trainer.lr_scheduler is not None:
            lr_scheduler_state_dict = state.get('lr_scheduler')
            if lr_scheduler_state_dict is None:
                # In this case, maybe you added a scheduler setting in the configuration file before resuming
                pass
            self.trainer.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    def process(self):
        chosen = self.buffer('chosen', False)
        if not chosen:
            return

        # synchronize epoch (for lr_scheduler)
        self.trainer.epoch = self.buffer('global_epoch') * self.epochs

        # receive parameters from the server and train locally
        state_dict = self.buffer()
        self.trainer.load_model(state_dict)
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

        # send model parameters and loss to the server
        self.server.receive(self, self.trainer.export_model())
        self.server.receive(self, self.trainer.optimizer.param_groups[0]['lr'], 'learning_rate')
        self.server.receive(self, self.avg_loss, 'loss')
        self.server.receive(self, self._data_amount(), 'data_amount')
        if self.trainer.lr_scheduler is not None:
            self.server.receive(self, self.trainer.lr_scheduler.state_dict(), 'lr_scheduler')

        self.set_buffer(False, 'chosen')


class FedAvgServer(BaseServer):
    def __init__(self,
                 tester: BaseTester,
                 global_epoch: int,
                 frac: float,
                 test_interval: int = 1):
        super(FedAvgServer, self).__init__()
        self.tester = tester
        self.epochs = global_epoch
        self.frac = frac
        self.test_interval = test_interval

        self.chosen_clients = []
        self.global_lr_scheduler = None

    def state(self) -> dict:
        state = {
            'model': self.tester.export_model(),
            'global_lr_scheduler': self.global_lr_scheduler,
        }
        return state

    def load_state(self, state: dict):
        model_state_dict = state.get('model')
        if model_state_dict is None:
            raise RuntimeError('No model found in the resumed server state.')
        self.tester.load_model(model_state_dict)
        self.global_lr_scheduler = state.get('global_lr_scheduler')

    def pre_process(self):
        # choose clients
        num_chosen_clients = max(int(len(self.clients) * self.frac), 1)
        self.chosen_clients = random.sample(list(self.clients.keys()), num_chosen_clients)

    def send(self):
        # send the parameters of the global model to chosen clients
        # one state_dict copy shared by all clients, so not allowed to be modified by any of the clients
        state_dict = self.tester.export_model()
        for client in self.chosen_clients:
            client.receive(True, 'chosen')
            client.receive(state_dict)
            client.receive(self.epoch, 'global_epoch')
            client.receive(self.global_lr_scheduler, 'global_lr_scheduler')

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
        dp = [self.buffer(c, 'data_amount') for c in self.chosen_clients]
        dp = [x / sum(dp) for x in dp]

        w_avg = copy.deepcopy(w_list[0])
        for key in w_avg.keys():
            w_avg[key] = w_list[0][key] * dp[0]
        for key in w_avg.keys():
            for i in range(1, len(w_list)):
                w_avg[key] += w_list[i][key] * dp[i]

        self.tester.model.train()
        self.tester.load_model(w_avg)

        # test the aggregated model on the test set
        if self.test_interval > 0 \
                and ((self.epoch + 1) % self.test_interval == 0 or self.epoch + 1 == self.epochs):
            _, res_scalars, best_metric = self.tester.test()
            scalars.update(res_scalars)
            ret['best_metric'] = best_metric

        return ret
