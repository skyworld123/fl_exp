import copy
import numpy as np

from .base import BaseManager
from .fedprox import FedProxClient
from .fedavg import FedAvgServer
from .tools import *
from .trainers import FedBSTrainer
from operations.test import BaseTester


class FedBSManager(BaseManager):
    def __init__(self,
                 global_epoch: int,
                 local_epoch: int,
                 num_clients: int,
                 frac: float,
                 miu: float,
                 epsilon: float,
                 rounds: int):
        super(FedBSManager, self).__init__(global_epoch)
        self.local_epoch = local_epoch
        self.num_clients = num_clients
        self.frac = frac
        self.miu = miu
        self.epsilon = epsilon
        self.rounds = rounds

    def generate_clients(self, cfg):
        dataloaders = generate_train_dataloaders_default(cfg, self.num_clients)

        # generate clients
        print(f'Generating {self.num_clients} clients...')
        params = {'miu': self.miu}
        trainers = generate_trainers_default(cfg, dataloaders, FedBSTrainer, params)
        clients = []
        for ci in range(self.num_clients):
            client = FedBSClient(trainers[ci], self.local_epoch)
            clients.append(client)

        return clients

    def generate_server(self, cfg):
        dataloader = generate_test_dataloader_default(cfg)

        # generate server
        print('Generating server...')
        tester = generate_tester_default(cfg, dataloader)
        test_interval = cfg.args['save_interval'] if cfg.args['eval'] else 0
        server = FedBSServer(
            tester, self.global_epoch, self.frac, self.epsilon, self.rounds, test_interval)

        return server


class FedBSClient(FedProxClient):
    def __init__(self,
                 trainer: FedBSTrainer,
                 local_epoch: int):
        super(FedBSClient, self).__init__(trainer, local_epoch)
        self.trainer = trainer

    def switch_to_fedprox(self):
        self.trainer.use_fedprox = True


class FedBSServer(FedAvgServer):
    def __init__(self,
                 tester: BaseTester,
                 global_epoch: int,
                 frac: float,
                 epsilon: float,
                 rounds: int,
                 test_interval: int = 1):
        super(FedBSServer, self).__init__(tester, global_epoch, frac, test_interval)
        self.epsilon = epsilon
        self.rounds = rounds
        self.past_rounds = 0
        self.use_fedprox = False

    def state(self) -> dict:
        state = {
            'model': self.tester.export_model(),
            'global_lr_scheduler': self.global_lr_scheduler,
            'past_rounds': self.past_rounds,
            'use_fedprox': self.use_fedprox,
        }
        return state

    def load_state(self, state: dict):
        model_state_dict = state.get('model')
        if model_state_dict is None:
            raise RuntimeError('No model found in the resumed server state.')
        self.tester.load_model(model_state_dict)
        self.global_lr_scheduler = state.get('global_lr_scheduler')

        self.past_rounds = state['past_rounds']
        self.use_fedprox = state['use_fedprox']

    def switch_to_fedprox(self):
        self.use_fedprox = True
        for client in self.clients:
            client.switch_to_fedprox()

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
        if self.use_fedprox:
            dp = [1] * len(self.chosen_clients)
        else:
            dp = losses.copy()
        dp = [x / sum(dp) for x in dp]

        w_avg = copy.deepcopy(w_list[0])
        for key in w_avg.keys():
            w_avg[key] = w_list[0][key] * dp[0]
        for key in w_avg.keys():
            for i in range(1, len(w_list)):
                w_avg[key] += w_list[i][key] * dp[i]

        # if std(F(w)) < epsilon, switch to FedProx
        if not self.use_fedprox:
            loss_std = np.std(losses)
            print(f'loss_std: {loss_std}')
            if loss_std < self.epsilon:
                self.past_rounds += 1
                if self.past_rounds >= self.rounds:
                    self.switch_to_fedprox()
                    print('---FedBS: Switched to FedProx---')
            else:
                self.past_rounds = 0
            print(f'past_rounds: {self.past_rounds}')

        self.tester.model.train()
        self.tester.load_model(w_avg)

        # test the aggregated model on the test set
        if self.test_interval > 0 \
                and ((self.epoch + 1) % self.test_interval == 0 or self.epoch + 1 == self.epochs):
            _, res_scalars, best_metric = self.tester.test()
            scalars.update(res_scalars)
            ret['best_metric'] = best_metric

        return ret
