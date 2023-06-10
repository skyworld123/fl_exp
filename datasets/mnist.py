from torchvision.datasets import MNIST, FashionMNIST

from .classification import ClsWrapper


class MNISTWrapper(ClsWrapper):
    IN_CHANNELS = 1
    NUM_CLASSES = len(MNIST.classes)

    def __init__(self,
                 root: str,
                 train_transform=None,
                 test_transform=None):
        super(MNISTWrapper, self).__init__(root, train_transform, test_transform)
        self.split_list = ['train', 'test']

    def load_split_and_targets(self, split: str):
        assert split in self.split_list
        train = True if split == 'train' else False
        transform = self.train_transform if split == 'train' else self.test_transform
        split_set = MNIST(self.root, train=train, transform=transform, download=True)
        targets = split_set.targets.numpy()
        return split_set, targets

    def load_train_and_targets(self):
        return self.load_split_and_targets('train')

    def load_test_and_targets(self):
        return self.load_split_and_targets('test')


class FashionMNISTWrapper(MNISTWrapper):
    IN_CHANNELS = 1
    NUM_CLASSES = len(FashionMNIST.classes)

    def __init__(self,
                 root: str,
                 train_transform=None,
                 test_transform=None):
        super(FashionMNISTWrapper, self).__init__(root, train_transform, test_transform)
        self.split_list = ['train', 'test']

    def load_split_and_targets(self, split: str):
        assert split in self.split_list
        train = True if split == 'train' else False
        transform = self.train_transform if split == 'train' else self.test_transform
        split_set = FashionMNIST(self.root, train=train, transform=transform, download=True)
        targets = split_set.targets.numpy()
        return split_set, targets
