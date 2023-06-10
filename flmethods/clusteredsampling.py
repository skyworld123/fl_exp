import numpy as np

from .base import BaseManager
from .fedavg import FedAvgClient
from .fedavg import FedAvgServer
from .tools import *
from .trainers import ClusteredSamplingTrainer
from operations.test import BaseTester
from numpy.random import choice
from utils.clusteredsampling import get_gradients
from utils.clusteredsampling import get_matrix_similarity_from_grads
from scipy.cluster.hierarchy import linkage
from utils.clusteredsampling import get_clusters_with_alg2
import copy


class ClusteredSamplingManager(BaseManager):
    def __init__(self,
                 global_epoch: int,
                 local_epoch: int,
                 num_clients: int,
                 frac: float,
                 sim_type: str = 'L1'):
        super(ClusteredSamplingManager, self).__init__(global_epoch)
        self.local_epoch = local_epoch
        self.num_clients = num_clients
        self.frac = frac
        self.sim_type = sim_type

    def generate_clients(self, cfg):
        dataloaders = generate_train_dataloaders_default(cfg, self.num_clients)

        # generate clients
        print(f'Generating {self.num_clients} clients...')
        trainers = generate_trainers_default(cfg, dataloaders, ClusteredSamplingTrainer)
        clients = []
        for ci in range(self.num_clients):
            client = ClusteredSamplingClient(trainers[ci], self.local_epoch)
            clients.append(client)

        return clients

    def generate_server(self, cfg):
        dataloader = generate_test_dataloader_default(cfg)

        # generate server
        print('Generating server...')
        tester = generate_tester_default(cfg, dataloader)
        test_interval = cfg.args['save_interval'] if cfg.args['eval'] else 0
        seed = cfg.args['seed']
        server = ClusteredSamplingServer(
            tester, self.global_epoch, self.frac,
            self.sim_type, seed, test_interval)

        return server

    def prepare(self, cfg):
        super().prepare(cfg)

        for c in self.clients:
            self.server.receive(c, c.data_amount(), 'data_amount')
        self.server.n_samples = np.array([self.server.buffer(c, 'data_amount') for c in self.clients])
        self.server.local_models = [None] * len(self.clients)


class ClusteredSamplingClient(FedAvgClient):
    def __init__(self,
                 trainer: ClusteredSamplingTrainer,
                 local_epoch: int):
        super(ClusteredSamplingClient, self).__init__(trainer, local_epoch)
        self.trainer = trainer

    def data_amount(self):
        return self._data_amount()

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


class ClusteredSamplingServer(FedAvgServer):
    def __init__(self,
                 tester: BaseTester,
                 global_epoch: int,
                 frac: float,
                 sim_type: str,
                 seed: int,
                 test_interval: int = 1):
        super(ClusteredSamplingServer, self).__init__(tester, global_epoch, frac)
        self.tester = tester
        self.epochs = global_epoch
        self.frac = frac
        self.test_interval = test_interval
        self.clusters = []
        self.chosen_cluster_clients = []
        self.global_lr_scheduler = None
        # self.gradients = []  # grad of each client using for calculate clients' similarity matrix
        self.local_models = []
        sim_type_list = ['L1', 'L2', 'cosine']
        assert sim_type in sim_type_list, f'Unknown sim_type "{sim_type}". ' \
                                          f'Please choose one in {", ".join(sim_type_list)}.'
        self.sim_type = sim_type
        self.seed = seed
        self.n_samples = np.array(0)

    def state(self) -> dict:
        state = {
            'model': self.tester.export_model(),
            'global_lr_scheduler': self.global_lr_scheduler,
            'clusters': [[c.cid for c in cluster] for cluster in self.clusters],
            'chosen_cluster_clients': [[c.cid for c in cluster] for cluster in self.chosen_cluster_clients],
            # 'gradients': self.gradients,
            'local_models': self.local_models,
        }
        return state

    def load_state(self, state: dict):
        model_state_dict = state.get('model')
        if model_state_dict is None:
            raise RuntimeError('No model found in the resumed server state.')
        self.tester.load_model(model_state_dict)
        self.global_lr_scheduler = state.get('global_lr_scheduler')

        cid2c = {list(self.clients.keys()).index(c): c for c in self.clients}

        self.clusters = [[cid2c[cid] for cid in cluster] for cluster in state['clusters']]
        self.chosen_cluster_clients = [[cid2c[cid] for cid in cluster] for cluster in state['chosen_cluster_clients']]
        self.local_models = state['local_models']

    def cluster(self):
        global_model = self.tester.export_model()
        n_clients = len(self.clients)  # 100
        gradients = get_gradients(global_model, self.local_models)
        # Get the clients' similarity matrix
        sim_matrix = get_matrix_similarity_from_grads(gradients, distance_type=self.sim_type)
        # Get the dendrogram tree associated
        if len(sim_matrix) == 0:
            raise RuntimeError("distance matrix is empty.")
        linkage_matrix = linkage(sim_matrix, "ward")
        weights = self.n_samples / np.sum(self.n_samples)
        n_sampled = round(self.frac * n_clients)
        distri_clusters = get_clusters_with_alg2(linkage_matrix, n_sampled, weights)
        return distri_clusters

    def choose_cluster_clients(self):
        distri_clusters = self.cluster()

        n_clients = len(distri_clusters[0])
        n_sampled = len(distri_clusters)

        sampled_clients = []
        for k in range(n_sampled):
            sampled_clients.append(int(choice(n_clients, 1, p=distri_clusters[k])))
        return sampled_clients

    def pre_process(self):
        orig_clients_list = list(self.clients.keys())
        sampled_clients = self.choose_cluster_clients()
        chosen_clients = []
        for idx in sampled_clients:
            chosen_clients.append(orig_clients_list[idx])
        self.chosen_clients = chosen_clients

    def send(self):
        # send the parameters of the global model to chosen clients
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

        # updating local_models
        for ci, c in enumerate(self.chosen_clients):
            self.local_models[c.cid] = w_list[ci]

        self.tester.model.train()
        self.tester.load_model(w_avg)

        # test the aggregated model on the test set
        if self.test_interval > 0 \
                and ((self.epoch + 1) % self.test_interval == 0 or self.epoch + 1 == self.epochs):
            _, res_scalars, best_metric = self.tester.test()
            scalars.update(res_scalars)
            ret['best_metric'] = best_metric

        return ret
