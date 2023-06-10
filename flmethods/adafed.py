import copy
from sklearn.exceptions import UndefinedMetricWarning
import torch
from tqdm import tqdm
import warnings

from .base import BaseManager
from .fedavg import FedAvgClient, FedAvgServer
from .tools import *
from .trainers import AdaFedTrainer
from operations.test import ClassWiseClassificationTester
from utils import print_long_vector


class AdaFedManager(BaseManager):
    """
    Currently the AdaFed method is designed for classification tasks only.
    It uses (weighted) cross entropy loss as criterion.
    """
    def __init__(self,
                 global_epoch: int,
                 local_epoch: int,
                 num_clients: int,
                 frac: float,
                 epsilon: float = 1e-8):
        super(AdaFedManager, self).__init__(global_epoch)
        self.local_epoch = local_epoch
        self.num_clients = num_clients
        self.frac = frac
        self.epsilon = epsilon

    def generate_clients(self, cfg):
        dataloaders = generate_train_dataloaders_default(cfg, self.num_clients)

        # generate clients
        print(f'Generating {self.num_clients} clients...')
        trainers = generate_trainers_default(cfg, dataloaders, AdaFedTrainer)
        clients = []
        for ci in range(self.num_clients):
            client = AdaFedClient(trainers[ci], self.local_epoch)
            clients.append(client)

        return clients

    def generate_server(self, cfg):
        dataloader = generate_test_dataloader_default(cfg)

        # generate server
        print('Generating server...')
        cfg_model = cfg.model
        cfg_tester = cfg.tester
        device = cfg.device
        verbose = cfg.args['verbose_val']

        pretrained_path = cfg_model['pretrained']
        model = cfg_model['type'](**cfg_model['params'])
        if pretrained_path is not None:
            print(f'Loading pretrained model {pretrained_path}...')
            state_dict = torch.load(pretrained_path)
            model.load_state_dict(state_dict)

        tester = ClassWiseClassificationTester(dataloader, model, device,
                                               verbose,
                                               **cfg_tester['params'])

        test_interval = cfg.args['save_interval'] if cfg.args['eval'] else 0
        server = AdaFedServer(tester, self.global_epoch, self.frac, test_interval, self.epsilon)

        return server


class AdaFedClient(FedAvgClient):
    def __init__(self,
                 trainer: AdaFedTrainer,
                 local_epoch: int):
        super(AdaFedClient, self).__init__(trainer, local_epoch)
        self.trainer = trainer

    def process(self):
        chosen = self.buffer('chosen', False)
        if not chosen:
            return

        # synchronize epoch (for lr_scheduler)
        self.trainer.epoch = self.buffer('global_epoch') * self.epochs

        # receive parameters from the server and train locally
        state_dict = self.buffer()
        ce_weight = self.buffer('ce_weight')
        self.trainer.load_model(state_dict)
        self.trainer.receive_ce_weight(ce_weight)
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
        if self.trainer.lr_scheduler is not None:
            self.server.receive(self, self.trainer.lr_scheduler.state_dict(), 'lr_scheduler')

        self.set_buffer(False, 'chosen')


class AdaFedServer(FedAvgServer):
    def __init__(self,
                 tester: ClassWiseClassificationTester,
                 global_epoch: int,
                 frac: float,
                 test_interval: int = 1,
                 epsilon: float = 1e-8):
        super(AdaFedServer, self).__init__(tester, global_epoch, frac, test_interval)
        self.tester = tester
        self.epsilon = epsilon

        self.client_ce_weight = None

    def state(self) -> dict:
        state = {
            'model': self.tester.export_model(),
            'global_lr_scheduler': self.global_lr_scheduler,
            'client_ce_weight': self.client_ce_weight,
        }
        return state

    def load_state(self, state: dict):
        model_state_dict = state.get('model')
        if model_state_dict is None:
            raise RuntimeError('No model found in the resumed server state.')
        self.tester.load_model(model_state_dict)
        self.global_lr_scheduler = state.get('global_lr_scheduler')

        self.client_ce_weight = state['client_ce_weight']

    def send(self):
        state_dict = self.tester.export_model()
        for client in self.chosen_clients:
            client.receive(True, 'chosen')
            client.receive(state_dict)
            client.receive(self.epoch, 'global_epoch')
            client.receive(self.global_lr_scheduler, 'global_lr_scheduler')
            client.receive(self.client_ce_weight, 'ce_weight')

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
        dp = []
        eps = 1e-8
        print('Testing models from chosen clients...')
        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
        bar = tqdm(total=len(self.chosen_clients))
        for c, w in zip(self.chosen_clients, w_list):
            self.tester.model.train()
            self.tester.load_model(w)
            test_result, _, _ = self.tester.test(print_detail=False)
            dp.append(test_result['accuracy'] + eps)
            bar.update()
        bar.close()
        warnings.resetwarnings()
        dp = [x / sum(dp) for x in dp]

        w_avg = copy.deepcopy(w_list[0])
        for key in w_avg.keys():
            w_avg[key] = w_list[0][key] * dp[0]
        for key in w_avg.keys():
            for i in range(1, len(w_list)):
                w_avg[key] += w_list[i][key] * dp[i]

        # test the aggregated model on the test set
        self.tester.model.train()
        self.tester.load_model(w_avg)
        test_result, res_scalars, best_metric = self.tester.test()
        f1_score = []
        for k, v in test_result.items():
            if k.isdigit():
                ci = int(k)
                f1 = v['f1-score']
                f1_score.append((ci, f1))
        f1_score.sort(key=lambda x: x[0])
        f1_score = [x[1] for x in f1_score]
        ce_weight = torch.FloatTensor(f1_score)
        ce_weight = 1 / (ce_weight + self.epsilon)
        self.client_ce_weight = ce_weight
        print(f'Cross entropy weights: {print_long_vector(self.client_ce_weight, keep=5, decimals=3)}')

        if self.test_interval > 0 \
                and ((self.epoch + 1) % self.test_interval == 0 or self.epoch + 1 == self.epochs):
            scalars.update(res_scalars)
            ret['best_metric'] = best_metric

        return ret
