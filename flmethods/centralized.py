from tensorboardX import SummaryWriter
import time
from torch.utils.data import DataLoader

from .tools import *
from .trainers import FedAvgTrainer
from operations.test import BaseTester
from utils import load_resume, save_ckpt_multiple, remove_dir_recursive, ensure_dir, get_random_state, \
    set_random_state, output_epoch_dir, output_best_dir, sec2eta, show_time


class CentralizedManager:
    def __init__(self,
                 epochs: int):
        self.client = None

        self.epochs = epochs
        self.last_epoch = -1
        self.args = None
        self.checkpoint_dirs = []
        self.writer = None
        self.best_metric_name = str()
        self.best_metric_compare = str()
        self.last_best_metric = (-1, None)  # epoch, metric value
        self.last_best_scalars = None

    def _prepared(self):
        if self.client is None:
            return False, 'Client not found'
        if self.args is None:
            return False, 'Arguments not specified'
        return True, 'Prepared'

    def synchronize_epoch(self, epoch: int):
        self.client.epoch = epoch

    def generate_client(self, cfg):
        # train set
        cfg_train_set = cfg.train_set

        params = cfg_train_set['params']
        if cfg_train_set['root'] is not None:
            params.update({'root': cfg_train_set['root']})
        params.update({'train_transform': cfg_train_set['transform']})
        wrapper = cfg_train_set['type'](**params)
        train_set, _ = wrapper.load_train_and_targets()

        batch_size = cfg_train_set.get('batch_size', 1)
        num_workers = cfg.args['num_workers']
        cfg_sampler = cfg_train_set.get('sampler')
        sampler = get_sampler(cfg_sampler, train_set)
        train_dataloader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

        # test set
        test_dataloader = generate_test_dataloader_default(cfg)

        # generate client
        print(f'Generating client...')
        trainer = generate_trainer_default(cfg, train_dataloader, FedAvgTrainer)
        
        cfg_model = cfg.model
        cfg_tester = cfg.tester
        tester_type = cfg_tester['type']
        device = cfg.device
        verbose = cfg.args['verbose_val']
        model = cfg_model['type'](**cfg_model['params'])
        tester = tester_type(test_dataloader, model, device, verbose,
                             **cfg_tester['params'])

        test_interval = cfg.args['save_interval'] if cfg.args['eval'] else 0
        client = CentralizedClient(trainer, tester, self.epochs, test_interval)

        return client

    def prepare(self, cfg):
        """
        prepare the only client, record cfg
        :param cfg: Config object
        """
        self.client = self.generate_client(cfg)

        self.args = cfg.args
        resume = self.args['resume']
        if resume is not None:
            client_state = load_resume(resume, 'client_state')
            model = load_resume(resume, 'model')
            if model is not None:
                client_state['model'] = model
            self.client.load_state(client_state)
            self.last_epoch = load_resume(resume, 'info')['last_epoch']
            set_random_state(load_resume(resume, 'random'))
            print(f'Resuming from dir {resume} (last epoch: {self.last_epoch})...')
        if self.args['board']:
            self.writer = SummaryWriter(logdir=self.args['output_dir'])
        opt = cfg.tester['type'].best_metric_options
        self.best_metric_name, self.best_metric_compare = opt['name'], opt['compare']

    def train(self):
        prepared, err = self._prepared()
        if not prepared:
            raise RuntimeError(f'Manager not prepared yet ({err}). Please run "prepare" method before training.')

        time_start = time.time()
        for epoch in range(self.last_epoch + 1, self.epochs):
            time_now = time.time()
            if epoch > self.last_epoch + 1:
                eta_sec = (self.epochs - epoch) / (epoch - self.last_epoch - 1) * (time_now - time_start)
                print(f'\n{show_time()}Epoch: {epoch + 1}/{self.epochs}, ETA: {sec2eta(eta_sec)}')
            else:
                print(f'\n{show_time()}Epoch: {epoch + 1}/{self.epochs}')

            self.synchronize_epoch(epoch)
            ret = self.client.process()
            if ret:
                self.record(ret, epoch)

    def record(self, ret: dict, epoch: int):
        """
        Record scalars and/or checkpoint (states).
        ret (dict):
            scalars: Scalars to be written.
            best_metric: Value of the best metric.
        """
        # scalars
        scalars = ret.get('scalars')
        scalars_bak = scalars.copy()
        if scalars is not None and self.args['board']:
            for key, value in scalars.items():
                self.writer.add_scalar(key, value, epoch + 1)

        # checkpoint
        save_interval = self.args['save_interval']
        if (save_interval > 0 and (epoch + 1) % save_interval == 0) \
                or epoch + 1 == self.epochs:
            oe_dir = output_epoch_dir(self.args['output_dir'], epoch)
            max_num_ckpt = self.args['keep_checkpoint_max']
            if len(self.checkpoint_dirs) >= max_num_ckpt:
                si = len(self.checkpoint_dirs) - max_num_ckpt + 1
                for removed_dir in self.checkpoint_dirs[:si]:
                    remove_dir_recursive(removed_dir)
                self.checkpoint_dirs = self.checkpoint_dirs[si:]
            ensure_dir(oe_dir)

            client_state = self.client.state()
            model = client_state.pop('model') if 'model' in client_state else None
            ckpt = {
                'client_state': client_state,
                'server_state': None,
                'model': model,
                'random': get_random_state(),
                'info': {'last_epoch': epoch},
            }

            save_ckpt_multiple(oe_dir, ckpt)
            self.checkpoint_dirs.append(oe_dir)
            print(f'Checkpoint at epoch {epoch + 1} saved to {oe_dir}.')

            # best model
            best_metric = ret.get('best_metric')
            if best_metric is not None:
                last_best_metric = self.last_best_metric[1]
                if last_best_metric is None:
                    update_cond = True
                    diff = 'first evaluation'
                else:
                    if self.best_metric_compare == '+':
                        update_cond = best_metric > last_best_metric
                        diff = '{:.4f}'.format(best_metric - last_best_metric)
                        diff = f'+{diff}'
                    elif self.best_metric_compare == '-':
                        update_cond = best_metric < last_best_metric
                        diff = '{:.4f}'.format(best_metric - last_best_metric)
                        diff = f'+{diff}'
                    else:
                        update_cond = best_metric > last_best_metric
                        diff = '{:.4f}'.format(best_metric - last_best_metric)
                        diff = f'+{diff}'
                    diff = f'{self.best_metric_name}: ' + diff

                if update_cond:
                    ob_dir = output_best_dir(self.args['output_dir'])
                    ensure_dir(ob_dir)
                    save_ckpt_multiple(ob_dir, ckpt)
                    self.last_best_metric = (epoch, best_metric)
                    self.last_best_scalars = scalars_bak
                    print(f'New best model saved! ({diff})')

                if self.last_best_scalars is None:
                    last_best_console = f'{self.best_metric_name}: {self.last_best_metric[1]}'
                else:
                    last_best_console_items = []
                    for k, v in self.last_best_scalars.items():
                        if k in ['train_loss', 'learning_rate']:
                            continue
                        last_best_console_items.append(f'{k}={"%.6f" % v}')
                    last_best_console = ', '.join(last_best_console_items)
                print(f'Best model is at epoch {self.last_best_metric[0] + 1} ({last_best_console}).')

    def post_process(self):
        if self.writer is not None:
            self.writer.close()


class CentralizedClient:
    def __init__(self,
                 trainer: FedAvgTrainer,
                 tester: BaseTester,
                 epochs: int,
                 test_interval: int = 1):
        self.trainer = trainer
        self.tester = tester
        self.epochs = epochs
        self.test_interval = test_interval

        self.epoch = 0

    def state(self) -> dict:
        state = {
            'model': self.trainer.export_model(),
            'lr_scheduler': self.trainer.lr_scheduler.state_dict() if self.trainer.lr_scheduler else None,
        }
        return state

    def load_state(self, state: dict):
        state_dict = state.get('model')
        if state_dict is None:
            raise RuntimeError('No model found in the resumed state.')
        self.trainer.load_model(state_dict)
        if self.trainer.lr_scheduler is not None:
            state_dict = state.get('lr_scheduler')
            if state_dict is None:
                # In this case, maybe you added a scheduler setting in the configuration file before resuming
                pass
            self.trainer.lr_scheduler.load_state_dict(state_dict)

    def process(self):
        ret = {}
        scalars = {}
        ret['scalars'] = scalars

        avg_loss = self.trainer.train_epoch()

        avg_loss = float('%.6f' % avg_loss)
        learning_rate = self.trainer.optimizer.param_groups[0]['lr']
        scalars['learning_rate'] = learning_rate
        print(f'Epoch {self.epoch + 1}/{self.epochs}: '
              f'avg_loss={avg_loss}, lr={learning_rate}')
        scalars['train_loss'] = avg_loss

        if self.test_interval > 0 \
                and ((self.epoch + 1) % self.test_interval == 0 or self.epoch + 1 == self.epochs):
            self.tester.model.train()
            self.tester.model.load_state_dict(self.trainer.model.state_dict())
            _, res_scalars, best_metric = self.tester.test()
            scalars.update(res_scalars)
            ret['best_metric'] = best_metric

        return ret
