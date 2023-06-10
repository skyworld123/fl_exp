import abc
from abc import ABC
from tensorboardX import SummaryWriter
import time
from typing import List

from utils import MessageSize
from utils import load_resume, save_ckpt_multiple, remove_dir_recursive, ensure_dir, get_random_state, \
    set_random_state, output_epoch_dir, output_best_dir, sec2eta, data_size_with_unit, current_ymd_hms, show_time


class BaseClient(ABC):
    def __init__(self):
        super(BaseClient, self).__init__()
        self.cid = -1
        self.server = None
        self._buffer = {'_default': None}
        self._received_bytes = 0
        self._message_size = MessageSize()

    def register_server(self, server):
        assert isinstance(server, BaseServer)
        self.server = server

    def state(self) -> dict:
        pass

    def load_state(self, state: dict):
        pass

    def received_bytes(self) -> int:
        return self._received_bytes

    @abc.abstractmethod
    def process(self):
        pass

    @abc.abstractmethod
    def send(self):
        pass

    def receive(self, msg, key='_default'):
        self._buffer[key] = msg
        self._received_bytes += self._message_size.message_size(msg)

    def buffer(self, key='_default', default=None):
        return self._buffer.get(key, default)

    def set_buffer(self, value, key='_default'):
        if key not in self._buffer:
            return
        self._buffer[key] = value

    def clean_buffer(self, key='_default'):
        if key in self._buffer:
            del self._buffer[key]

    def set_cid(self, cid: int):
        self.cid = cid


class BaseServer(ABC):
    def __init__(self):
        super(BaseServer, self).__init__()
        self.clients = {}
        self._buffers = {}
        self._received_bytes = {}
        self._message_size = MessageSize()
        self._next_cid = 0
        self.epoch = 0

    def register_clients(self, clients):
        if not isinstance(clients, list):
            clients = [clients]
        for client in clients:
            assert isinstance(client, BaseClient)
            if client in self.clients:
                print(f'[WARN] Cannot re-register client (id(client)={id(client)}). Skip this client.')
                continue
            cid = self._next_cid
            self._next_cid += 1
            self.clients[client] = cid
            client.set_cid(cid)
            _buffer = {'_default': None}
            self._buffers[cid] = _buffer

    def unregister_clients(self, clients):
        if not isinstance(clients, list):
            clients = [clients]
        for client in clients:
            assert isinstance(client, BaseClient)
            if client not in self.clients:
                print(f'[WARN] Cannot unregister unknown client (id(client)={id(client)}).')
                continue
            cid = self.clients[client]
            self.clients.pop(client)
            self._buffers.pop(cid)

    def state(self) -> dict:
        pass

    def load_state(self, state: dict):
        pass

    def received_bytes(self, client: BaseClient = None) -> int:
        if client is None:
            return sum(self._received_bytes.values())
        return self._received_bytes.get(client, 0)

    @abc.abstractmethod
    def pre_process(self):
        pass

    @abc.abstractmethod
    def send(self):
        pass

    @abc.abstractmethod
    def post_process(self):
        pass

    def receive(self, client, msg, key='_default'):
        cid = self.clients[client]
        self._buffers[cid][key] = msg
        if cid not in self._received_bytes:
            self._received_bytes[cid] = 0
        self._received_bytes[cid] += self._message_size.message_size(msg)

    def buffer(self, client, key='_default', default=None):
        cid = self.clients[client]
        return self._buffers[cid].get(key, default)

    def set_buffer(self, client, value, key='_default'):
        cid = self.clients[client]
        _buffer = self._buffers[cid]
        if key not in _buffer:
            return
        _buffer[key] = value

    def clean_buffer(self, key='_default'):
        for client in self.clients:
            cid = self.clients[client]
            if key in self._buffers[cid]:
                del self._buffers[cid][key]


class BaseManager(ABC):
    def __init__(self,
                 global_epoch: int):
        super(BaseManager, self).__init__()
        self.clients = []
        self.server = None

        self.global_epoch = global_epoch
        self.last_epoch = -1
        self.args = None
        self.checkpoint_dirs = []
        self.writer = None
        self.best_metric_name = str()
        self.best_metric_compare = str()
        self.last_best_metric = (-1, None)  # epoch, metric value
        self.last_best_scalars = None  # scalar values
        self.target_metric = None
        self.target_metric_reached = None

    def _prepared(self):
        if len(self.clients) == 0:
            return False, 'No clients'
        if self.server is None:
            return False, 'No server'
        if self.args is None:
            return False, 'Arguments not specified'
        return True, 'Prepared'

    def synchronize_epoch(self, epoch: int):
        self.server.epoch = epoch
        for client in self.clients:
            if hasattr(client, 'trainer'):
                client.trainer.epoch = epoch

    def total_message_size(self):
        ms = 0
        for c in self.clients:
            ms += c.received_bytes()
        ms += self.server.received_bytes()

        return ms

    @abc.abstractmethod
    def generate_clients(self, cfg) -> List[BaseClient]:
        pass

    @abc.abstractmethod
    def generate_server(self, cfg) -> BaseServer:
        pass

    def prepare(self, cfg):
        """
        prepare clients and server, record cfg
        :param cfg: Config object
        """
        self.clients = self.generate_clients(cfg)
        self.server = self.generate_server(cfg)
        for client in self.clients:
            client.register_server(self.server)
        self.server.register_clients(self.clients)

        self.args = cfg.args
        resume = self.args['resume']
        if resume is not None:
            client_state_list = load_resume(resume, 'client_state')
            server_state = load_resume(resume, 'server_state')
            if server_state is None:
                server_state = {}
            model = load_resume(resume, 'model')
            if model is not None:
                server_state['model'] = model
            for ci, client in enumerate(self.clients):
                client.load_state(client_state_list[ci])
            self.server.load_state(server_state)
            self.last_epoch = load_resume(resume, 'info')['last_epoch']
            set_random_state(load_resume(resume, 'random'))
            print(f'Resuming from dir {resume} (last epoch: {self.last_epoch + 1})...')
        if self.args['board']:
            comment = current_ymd_hms()
            self.writer = SummaryWriter(logdir=self.args['output_dir'], comment=comment)

        opt = cfg.tester['type'].best_metric_options
        self.best_metric_name, self.best_metric_compare = opt['name'], opt['compare']
        target_metric = cfg.target_metric
        if target_metric is None:
            target_metric = []
        elif isinstance(target_metric, (int, float)):
            target_metric = [target_metric, ]
        else:
            target_metric = list(target_metric)
        reverse = self.best_metric_compare == '-'
        target_metric.sort(reverse=reverse)
        self.target_metric = target_metric
        self.target_metric_reached = [None] * len(target_metric)

    def train(self):
        prepared, err = self._prepared()
        if not prepared:
            raise RuntimeError(f'Manager not prepared yet ({err}). Please run "prepare" method before training.')

        max_recent_count = 5
        recent_queue = []
        for epoch in range(self.last_epoch + 1, self.global_epoch):
            time_now = time.time()
            if len(recent_queue) > 0:
                eta_sec = (self.global_epoch - epoch) / len(recent_queue) * (time_now - recent_queue[0])
                print(f'\n{show_time()}Global epoch: {epoch + 1}/{self.global_epoch}, ETA: {sec2eta(eta_sec)}')
            else:
                print(f'\n{show_time()}Global epoch: {epoch + 1}/{self.global_epoch}')
            recent_queue.append(time_now)
            if len(recent_queue) > max_recent_count:
                recent_queue = recent_queue[-max_recent_count:]

            self.synchronize_epoch(epoch)
            self.server.pre_process()
            self.server.send()
            for client in self.clients:
                client.process()
                client.send()
            ret = self.server.post_process()
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
            scalars['total_message_size'] = self.total_message_size()
            for key, value in scalars.items():
                self.writer.add_scalar(key, value, epoch + 1)

        # checkpoint
        save_interval = self.args['save_interval']
        if (save_interval > 0 and (epoch + 1) % save_interval == 0) \
                or epoch + 1 == self.global_epoch:
            oe_dir = output_epoch_dir(self.args['output_dir'], epoch)
            max_num_ckpt = self.args['keep_checkpoint_max']
            if len(self.checkpoint_dirs) >= max_num_ckpt:
                si = len(self.checkpoint_dirs) - max_num_ckpt + 1
                for removed_dir in self.checkpoint_dirs[:si]:
                    remove_dir_recursive(removed_dir)
                self.checkpoint_dirs = self.checkpoint_dirs[si:]
            ensure_dir(oe_dir)

            server_state = self.server.state()
            model = server_state.pop('model') if 'model' in server_state else None
            ckpt = {
                'client_state': [],
                'random': get_random_state(),
                'info': {'last_epoch': epoch},
            }
            if model is not None:
                ckpt['model'] = model
            if len(server_state) > 0:
                ckpt['server_state'] = server_state
            for client in self.clients:
                state = client.state()
                ckpt['client_state'].append(state)

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
                        diff = '{:.4f}'.format(last_best_metric - best_metric)
                        diff = f'-{diff}'
                    else:
                        update_cond = best_metric > last_best_metric
                        diff = '{:.4f}'.format(best_metric - last_best_metric)
                        diff = f'+{diff}'
                    diff = f'{self.best_metric_name} ' + diff

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

                for ti, t in enumerate(self.target_metric):
                    if self.target_metric_reached[ti] is not None:
                        continue
                    if (self.best_metric_compare == '+' and self.last_best_metric[1] >= t) \
                            or (self.best_metric_compare == '-' and self.last_best_metric[1] <= t):
                        self.target_metric_reached[ti] = epoch + 1
                        print(f'Target metric ({self.best_metric_name}) reached {t}.')
                    else:
                        break

        if epoch == 0:
            print(f'Message size in the first epoch: {data_size_with_unit(self.total_message_size())}')

    def post_process(self):
        if self.writer is not None:
            self.writer.close()

        print(f'Total message size: {data_size_with_unit(self.total_message_size())}')
        if self.target_metric:
            print(f'Epoch(s) when target metric ({self.best_metric_name}) is reached:',)
            reached_list = []
            for ti, t in enumerate(self.target_metric):
                reached = self.target_metric_reached[ti]
                reached = str(reached) if reached is not None else '-'
                reached_list.append(f'{t}: {reached}')
            print('; '.join(reached_list))
