import abc
from abc import ABC


class BaseWrapper(ABC):
    def __init__(self,
                 root: str,
                 train_transform=None,
                 test_transform=None):
        super(BaseWrapper, self).__init__()
        self.root = root
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.division_methods = ['iid', 'non-iid', 'non-iid-hard']

    @abc.abstractmethod
    def get_train_set_division(self,
                               method: str,
                               num_clients: int,
                               **kwargs):
        pass

    @abc.abstractmethod
    def get_test_set(self):
        pass
