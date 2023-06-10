import abc
# import numpy as np

from .base import BaseWrapper
from .shared import ClsTrainSetDivision, ClsTestSet
from utils import arrange_dataset_by_target, divide


class ClsWrapper(BaseWrapper):
    def __init__(self,
                 root: str,
                 train_transform=None,
                 test_transform=None):
        super(ClsWrapper, self).__init__(
            root, train_transform, test_transform)

    @abc.abstractmethod
    def load_train_and_targets(self):
        pass

    @abc.abstractmethod
    def load_test_and_targets(self):
        pass

    def get_train_set_division(self,
                               method: str,
                               num_clients: int,
                               **kwargs) -> dict:
        """
        :param method: division method
        :param num_clients: number of clients
        :param kwargs: extra parameters for generating non-iid divisions
            {long_tail, gamma_c, shuffle, gamma_d, tau}
        :return:
            (dict) divisions of the original dataset, which is to be distributed to all clients
        """
        method = method.replace('_', '-')
        assert method in self.division_methods
        assert num_clients > 0

        arrange_params_dict = {
            'long_tail': 'long_tail',
            'gamma': 'gamma_d',
            'shuffle': 'shuffle',
        }
        divide_params_dict = {
            'classes_per_client': 'classes_per_client',
            'seed': 'seed',
            'gamma': 'gamma_c',
            'tau': 'tau',
        }
        arrange_params = {}
        divide_params = {}
        for key, cfg_key in arrange_params_dict.items():
            if cfg_key in kwargs:
                arrange_params[key] = kwargs[cfg_key]
        for key, cfg_key in divide_params_dict.items():
            if cfg_key in kwargs:
                divide_params[key] = kwargs[cfg_key]

        train_set, targets = self.load_train_and_targets()
        samples_by_class = arrange_dataset_by_target(targets, **arrange_params)
        samples_by_client = divide(method, samples_by_class, num_clients, **divide_params)
        num_classes = len(samples_by_class)

        divisions = {}
        for ci, samples in samples_by_client.items():
            divisions[ci] = ClsTrainSetDivision(train_set, targets, samples, num_classes)

        return divisions

    def get_test_set(self):
        """
        :return: test set
        """
        test_set, targets = self.load_test_and_targets()
        dataset = ClsTestSet(test_set, targets)
        return dataset
