import numpy as np
import os

from torchvision.datasets.folder import find_classes, make_dataset
from torch.utils.data import Dataset

from .classification import ClsWrapper
from .tools import read_video


class CustomKinetics400(Dataset):
    """
    Kinetics400 video dataset.
    """

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform=None,
                 sample_ratio: float = 1.0):
        super(CustomKinetics400, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.sample_ratio = min(max(sample_ratio, 0), 1)
        split = 'train' if self.train else 'val'
        self.split_root = os.path.join(self.root, split)

        extensions = ('mp4', 'mkv')
        self.classes, class_to_idx = find_classes(self.split_root)
        self.samples = make_dataset(self.split_root, class_to_idx, extensions, is_valid_file=None)

        self.available_indices = np.arange(len(self.samples))
        self.sample_ratio = min(max(sample_ratio, 0), 1)
        if self.sample_ratio < 1:
            num_samples = max(round(len(self.available_indices) * self.sample_ratio), 1)
            self.available_indices = np.random.choice(self.available_indices, num_samples, replace=False)

    def __len__(self) -> int:
        return len(self.available_indices)

    def __getitem__(self, idx: int):
        idx = self.available_indices[idx]
        video_path, label = self.samples[idx]
        video, info = read_video(video_path)
        if self.transform is not None:
            video = self.transform(video)

        return video, label


class Kinetics400Wrapper(ClsWrapper):
    IN_CHANNELS = 3
    NUM_CLASSES = 400

    def __init__(self,
                 root: str,
                 train_sample_ratio: float = 1.0,
                 test_sample_ratio: float = 1.0,
                 train_transform=None,
                 test_transform=None):
        super(Kinetics400Wrapper, self).__init__(root, train_transform, test_transform)
        self.data_root = os.path.join(self.root, 'videos')
        self.train_sample_ratio = train_sample_ratio
        self.test_sample_ratio = test_sample_ratio

        self.split_list = ['train', 'test']

    def load_split_and_targets(self, split: str):
        assert split in self.split_list
        train = True if split == 'train' else False
        transform = self.train_transform if split == 'train' else self.test_transform
        sample_ratio = self.train_sample_ratio if split == 'train' else self.test_sample_ratio
        print(f'Loading Kinetics400 (split: {split}, sample_ratio: {sample_ratio})...')
        split_set = CustomKinetics400(
            self.data_root,
            sample_ratio=sample_ratio,
            train=train, transform=transform)

        video_targets = [split_set.samples[x][1] for x in split_set.indices]
        video_targets = np.asarray(video_targets, dtype=np.int64)
        video_targets = video_targets[split_set.available_indices]
        targets = video_targets.tolist()

        return split_set, targets

    def load_train_and_targets(self):
        return self.load_split_and_targets('train')

    def load_test_and_targets(self):
        return self.load_split_and_targets('test')
