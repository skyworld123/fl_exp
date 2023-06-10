import numpy as np
from torchvision.datasets import CIFAR10

from .classification import ClsWrapper


class CIFAR10Wrapper(ClsWrapper):
    IN_CHANNELS = 3
    NUM_CLASSES = 10

    def __init__(self,
                 root: str,
                 train_transform=None,
                 test_transform=None):
        super(CIFAR10Wrapper, self).__init__(root, train_transform, test_transform)
        self.split_list = ['train', 'test']

    def load_split_and_targets(self, split: str):
        assert split in self.split_list
        train = True if split == 'train' else False
        transform = self.train_transform if split == 'train' else self.test_transform
        split_set = CIFAR10(self.root, train=train, transform=transform, download=True)
        targets = np.asarray(split_set.targets, dtype=np.int64)
        return split_set, targets

    def load_train_and_targets(self):
        return self.load_split_and_targets('train')

    def load_test_and_targets(self):
        return self.load_split_and_targets('test')
