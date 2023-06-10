import codecs
from copy import deepcopy
import os
import torch
from typing import Union, List
import yaml

import datasets
from operations.test import tester_dict
from operations.predict import predictor_dict
import flmethods
import models
import models.losses as losses
import utils.optimizers as optimizers
import utils.lr_schedulers as lr_schedulers
import utils.transforms as transforms

__all__ = ['Config']


class Config:
    """
    A class for loading configurations in training/testing. Only support yml/yaml files.
    The following hyper-parameters are available in the config file:
        fl_method: Config for federated learning method, require: type.
        train_set: Config for train set, require: type/root(/division/batch_size/transform). "division" is a sub dict,
            require: method/num_clients.
        test_set: Config for test set, require: type/root(/batch_size/transform).
        model: Config for model, require: type/backbone(/num_classes).
        loss: Config for loss, require: type.
        optimizer: Config for optimizer, require: type.
        lr_scheduler: Config for learning rate scheduler, require: type.
    You can specify additional arguments after the required sub-keys above.
    A config file can inherit other config file(s). You can use the "_base_" key to specify the relative path of the
    parent config file(s). When there exists identical keys in both parent and child config file, the config in the
    child config file has higher priority. If you do not allow (part of) a config file to inherit its parent, use
    "_inherited_: False" at the corresponding place.

    Args:
        path (str): The path of the config file (.yml/.yaml).
    """

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Config file {path} not exist.')
        ext = os.path.splitext(path)[1]
        if ext not in ['.yml', '.yaml']:
            raise RuntimeError('Config file should be in .yml/.yaml format.')

        self.path = path
        loader = ConfigLoader(path)
        self.dic = loader.load()

    def _check_path(self, path):
        """
        If path does not exist as a relevant path (to the config file), treat it as an absolute path
        (or a relevant path to the working directory)
        """
        cfg_dir = os.path.dirname(self.path)
        rel_path_res = os.path.join(cfg_dir, path)
        return rel_path_res if os.path.exists(rel_path_res) else path

    def _check_attr(self, key, raise_err=True):
        value = self.dic.get(key)
        if value is None:
            if raise_err:
                raise RuntimeError(f'No {key} specified in the config file.')
            else:
                return value
        if isinstance(value, dict):
            value = deepcopy(value)
        return value

    @property
    def fl_method(self):
        params = self._check_attr('fl_method')
        typ = params.pop('type')
        typ = typ + 'Manager'
        typ = getattr(flmethods, typ)
        return {'type': typ, 'params': params}

    @property
    def device(self):
        args = self.args
        device = args.get('device') if args is not None else None
        if device is None:
            # default device
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'
        return torch.device(device)

    @staticmethod
    def get_transform(trans=None):
        default_transform = transforms.Compose(transforms.ToTensor())
        if trans is None:
            return default_transform

        assert isinstance(trans, list), 'Configured "transforms" should be a list to be composed'

        components = []
        for t in trans:
            if isinstance(t, dict):
                for typ, params in t.items():
                    component = getattr(transforms, typ)(**params)
                    components.append(component)
            elif isinstance(t, str):
                typ = t
                params = {}
                component = getattr(transforms, typ)(**params)
                components.append(component)
            else:
                raise RuntimeError('Items of transforms should be either a single type or a dict with parameters of '
                                   'the types')
        transform = transforms.Compose(components)
        return transform

    @property
    def train_set(self):
        params = self._check_attr('train_set')
        typ = params.pop('type')
        typ = typ + 'Wrapper'
        typ = getattr(datasets, typ)
        root = params.pop('root') if 'root' in params else None
        batch_size = params.pop('batch_size') if 'batch_size' in params else 1
        global_batch_size = params.pop('global_batch_size') if 'global_batch_size' in params else None
        trans = params.pop('transform') if 'transform' in params else None
        trans = self.get_transform(trans)
        division = params.pop('division') if 'division' in params else None
        sampler = params.pop('sampler') if 'sampler' in params else None
        ret = {
            'type': typ,
            'root': root,
            'batch_size': batch_size,
            'global_batch_size': global_batch_size,
            'transform': trans,
            'division': division,
            'sampler': sampler,
            'params': params,
        }
        return ret

    @property
    def test_set(self):
        params = self._check_attr('test_set')
        typ = params.pop('type')
        typ = typ + 'Wrapper'
        typ = getattr(datasets, typ)
        root = params.pop('root') if 'root' in params else None
        batch_size = params.pop('batch_size') if 'batch_size' in params else 1
        trans = params.pop('transform') if 'transform' in params else None
        trans = self.get_transform(trans)
        sampler = params.pop('sampler') if 'sampler' in params else None
        ret = {
            'type': typ,
            'root': root,
            'batch_size': batch_size,
            'transform': trans,
            'sampler': sampler,
            'params': params,
        }
        return ret

    @property
    def model(self):
        params = self._check_attr('model')
        typ = params.pop('type')
        typ = getattr(models, typ)
        pretrained = params.pop('pretrained') if 'pretrained' in params else None
        if pretrained is not None:
            assert isinstance(pretrained, str), '"pretrained" should be the path of the pretrained model'
            pretrained_path = self._check_path(pretrained)
            if not os.path.exists(pretrained_path):
                raise RuntimeError(f'Pretrained model path {pretrained_path} not exist')
        else:
            pretrained_path = None

        # get alternative parameter from configured datasets
        checked_param_list = ['in_channels', 'num_classes']
        co_varnames = typ.__init__.__code__.co_varnames
        for checked_param in checked_param_list:
            if checked_param in co_varnames and checked_param not in params.keys():
                class_attr = checked_param.upper()
                train_set_typ = self.train_set['type']
                test_set_typ = self.test_set['type']
                alt1 = getattr(train_set_typ, class_attr) if class_attr in train_set_typ.__dict__ else None
                alt2 = getattr(test_set_typ, class_attr) if class_attr in test_set_typ.__dict__ else None
                alt_list = [alt1, alt2]
                if all(alt_list) is None:
                    raise RuntimeError(f'Config does not know the parameter {checked_param} of the model type {typ}. '
                                       f'Please specify it under the "model" key.')
                for alt in alt_list:
                    if alt is not None:
                        params[checked_param] = alt
                        break

        return {'type': typ, 'pretrained': pretrained_path, 'params': params}

    @property
    def loss(self):
        params = self._check_attr('loss')
        typ = params.pop('type')
        typ = getattr(losses, typ)
        return {'type': typ, 'params': params}

    @property
    def optimizer(self):
        params = self._check_attr('optimizer')
        typ = params.pop('type')
        typ = getattr(optimizers, typ)
        return {'type': typ, 'params': params}

    @property
    def lr_scheduler(self):
        # returned params do not contain optimizer
        params = self._check_attr('lr_scheduler', raise_err=False)
        if params is None:
            return None
        typ = params.pop('type')
        typ = getattr(lr_schedulers, typ)
        return {'type': typ, 'params': params}

    @property
    def tester(self):
        test_set = self._check_attr('test_set')
        cfg_tester = test_set.get('tester')
        cfg_params = {}
        if isinstance(cfg_tester, dict):
            cfg_params = cfg_tester
        test_set_typ = test_set.pop('type')
        def_tester = tester_dict[test_set_typ]
        tester_typ, params = def_tester
        params.update(cfg_params)
        return {'type': tester_typ, 'params': params}

    @property
    def predictor(self):
        test_set = self._check_attr('test_set')
        cfg_tester = test_set.get('predictor')
        cfg_params = {}
        if isinstance(cfg_tester, dict):
            cfg_params = cfg_tester
        test_set_typ = test_set.pop('type')
        def_predictor = predictor_dict[test_set_typ]
        predictor_typ, params = def_predictor
        params.update(cfg_params)
        return {'type': predictor_typ, 'params': params}

    @property
    def target_metric(self):
        target = self._check_attr('target_metric', raise_err=False)
        err_flag = False
        cond = target is None or isinstance(target, (int, float, list, tuple))
        err_flag = err_flag or not cond
        if isinstance(target, (list, tuple)):
            err_flag = err_flag or not all([isinstance(x, (int, float)) for x in target])
        if err_flag:
            raise RuntimeError(f'Input target_metric "{target}" not in correct format. '
                               f'Ensure it is a single number or a list of numbers.')

        return target

    def merge_args(self, args):
        """
        Merge config with input arguments.
        """
        self.dic['args'] = args.__dict__

    @property
    def args(self):
        return self.dic.get('args')


class ConfigLoader:
    def __init__(self, path: str):
        self.abs_stack = []  # List[str] (abstract path of yaml paths)
        self.rel_stack = []  # List[str] (raw yaml paths)
        self.path_idx = {}  # map: str -> int (normalized yaml path -> path index)
        self.parents_p = {}  # map: int -> Set[int]
        self.children_p = {}  # map: int -> Set[int]

        self.path = path

    @staticmethod
    def _load_yaml(path: str):
        f = codecs.open(path, 'r', 'utf-8')
        dic = yaml.load(f, Loader=yaml.FullLoader)
        return dic

    @staticmethod
    def _check_path(path: str, cfg_path: str = None):
        """
        If path does not exist as a relevant path, treat it as an absolute path.
        If absolute path does not exist, return None.
        """
        if cfg_path:
            cfg_dir = os.path.dirname(cfg_path)
            rel_path = os.path.join(cfg_dir, path)
        else:
            rel_path = path
        if os.path.exists(rel_path):
            return rel_path
        if os.path.exists(path):
            return path
        return None

    @staticmethod
    def _normalize_path(path: str):
        path = os.path.abspath(path)
        path = path.replace('\\', '/')
        return path

    def _relpath(self, path: str):
        cur_dir = self._normalize_path('.')
        if not cur_dir.endswith('/'):
            cur_dir += '/'
        path = self._normalize_path(path)
        if path.startswith(cur_dir):
            return path[len(cur_dir):]
        else:
            return path

    def _add_path(self, paths: Union[List[str], str], parent: str = None):
        if isinstance(paths, str):
            paths = [paths]

        parent = self._normalize_path(parent) if parent is not None else None
        parent_idx = self.path_idx[parent] if parent is not None else -1

        new_num = 0
        for path in paths:
            raw_path = path
            path = self._check_path(path, parent)
            if path is None:
                raise FileNotFoundError(f'_base_ path "{raw_path}" in config file {parent} not found.')
            path = self._normalize_path(path)

            idx = self.path_idx.get(path, len(self.abs_stack))
            new_flag = idx == len(self.abs_stack)
            new_num += new_flag
            if not new_flag:
                start_idx = idx
                end_idx = parent_idx
                last_idx = {start_idx: -1}
                stack = [start_idx]

                while len(stack) > 0:
                    cur_idx = stack[-1]
                    stack = stack[:-1]
                    stack_ext = []

                    child_found = False
                    for child in self.children_p[cur_idx]:
                        if child not in last_idx:
                            last_idx[child] = cur_idx
                            stack_ext.append(child)
                        if child == end_idx:
                            child_found = True
                            break
                    if child_found:
                        break
                    stack_ext.reverse()
                    stack.extend(stack_ext)

                if end_idx in last_idx:
                    # circular base import
                    circular_path = [end_idx]
                    cur_idx = end_idx
                    while last_idx[cur_idx] != -1:
                        circular_path.append(last_idx[cur_idx])
                        cur_idx = last_idx[cur_idx]
                    circular_path.reverse()
                    circular_path.append(start_idx)
                    circular_path = [self.rel_stack[x] for x in circular_path]
                    raise RuntimeError(f'Circular _base_ import detected '
                                       f'(import circle: {"->".join(circular_path)}). '
                                       f'Please check your configured _base_ path carefully.')

            if new_flag:
                self.rel_stack.append(self._relpath(path))
                self.abs_stack.append(path)
                self.path_idx[path] = idx
                self.parents_p[idx] = set()
                self.children_p[idx] = set()
            if parent_idx != -1:
                self.parents_p[idx].add(parent_idx)
                self.children_p[parent_idx].add(idx)

        return new_num

    def _update_dic(self, base_dic, new_dic, prev_keys=None):
        """
        Update base_dic from new_dic.
        """
        base_dic = base_dic.copy()
        new_dic = new_dic.copy()
        if prev_keys is None:
            prev_keys = []

        if '_inherited_' in base_dic:
            base_dic.pop('_inherited_')
        if not new_dic.get('_inherited_', True):
            new_dic.pop('_inherited_')
            return new_dic

        for key, val in new_dic.items():
            if key not in base_dic:
                base_dic[key] = val
                continue

            d1 = isinstance(val, dict)
            d2 = isinstance(base_dic[key], dict)
            keys = prev_keys + [key]
            if d1 ^ d2:
                raise RuntimeError(f'The value type of {"->".join(keys)} in the current/base config file must align '
                                   f'(found {type(val)}, {type(base_dic[key])})')
            if d1:
                prev_keys.append(key)
                base_dic[key] = self._update_dic(base_dic[key], val, prev_keys=prev_keys)
            else:
                base_dic[key] = val

        return base_dic

    def load(self):
        self._add_path(self.path)
        qs, qe = 0, 1

        # check base import
        dic_stack = []
        while qs < qe:
            parent_path = self.abs_stack[qs]
            dic = self._load_yaml(parent_path)
            qs += 1
            if '_base_' in dic:
                base_path = dic.pop('_base_')
                new_num = self._add_path(base_path, parent_path)
                qe += new_num
            dic_stack.append(dic)

        # load dic
        for si in range(len(dic_stack) - 1, -1, -1):
            children = self.children_p[si].copy()
            for child in children:
                try:
                    dic_stack[si] = self._update_dic(dic_stack[child], dic_stack[si])
                except RuntimeError as e:
                    dic_path = self.abs_stack[si]
                    raise RuntimeError(f'Error when processing {dic_path}: {e}')

        return dic_stack[0]
