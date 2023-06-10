import os

from .classification import ClsWrapper
from .skeleton import SkeletonDataset, RGBSkeletonDataset
from .tools import read_video


class Kinetics400SkeletonDataset(SkeletonDataset):
    """
    Skeletons of Kinetics400 dataset.
    """
    def __init__(self,
                 file_path: str,
                 train: bool = True,
                 transform=None,
                 sample_ratio: float = 1.0):
        super(Kinetics400SkeletonDataset, self).__init__(file_path, train, transform, sample_ratio)

    def _get_split(self):
        return 'train' if self.train else 'val'


class Kinetics400SkeletonWrapper(ClsWrapper):
    IN_CHANNELS = 17
    NUM_CLASSES = 400

    def __init__(self,
                 root: str,
                 train_transform=None,
                 test_transform=None,
                 train_sample_ratio: float = 1.0,
                 test_sample_ratio: float = 1.0):
        if os.path.isdir(root):
            root = os.path.join(root, 'kinetics400.pkl')
        super(Kinetics400SkeletonWrapper, self).__init__(root, train_transform, test_transform)
        self.file_path = root
        self.train_sample_ratio = train_sample_ratio
        self.test_sample_ratio = test_sample_ratio

        self.split_list = ['train', 'test']

    def load_split_and_targets(self, split: str):
        assert split in self.split_list
        train = True if split == 'train' else False
        transform = self.train_transform if split == 'train' else self.test_transform
        sample_ratio = self.train_sample_ratio if split == 'train' else self.test_sample_ratio
        print(f'Loading Kinetics400SkeletonDataset (split: {split}, sample_ratio: {sample_ratio})...')
        split_set = Kinetics400SkeletonDataset(
            self.file_path,
            train=train,
            transform=transform,  sample_ratio=sample_ratio
        )

        targets = [x['label'] for x in split_set.data]

        return split_set, targets

    def load_train_and_targets(self):
        return self.load_split_and_targets('train')

    def load_test_and_targets(self):
        return self.load_split_and_targets('test')


class Kinetics400RGBSkeletonDataset(RGBSkeletonDataset):
    """
    RGB+Skeletons of Kinetics400 dataset.
    The fold of RGB follows that of Skeletons.
    """
    def __init__(self,
                 rgb_root: str,
                 skeleton_path: str,
                 train: bool = True,
                 transform=None,
                 sample_ratio: float = 1.0):
        super(Kinetics400RGBSkeletonDataset, self).__init__(rgb_root, skeleton_path,
                                                            train, transform, sample_ratio)

    def _get_split(self):
        return 'train' if self.train else 'val'

    def load_rgb_video(self, frame_dir: str):
        video_path = os.path.join(self.rgb_root, self._get_split(), frame_dir)
        video, info = read_video(video_path)  # (t,h,w,c)
        return video


class Kinetics400RGBSkeletonWrapper(ClsWrapper):
    NUM_CLASSES = 400

    def __init__(self,
                 rgb_root: str,
                 skeleton_path: str,
                 train_transform=None,
                 test_transform=None,
                 train_sample_ratio: float = 1.0,
                 test_sample_ratio: float = 1.0):
        super(Kinetics400RGBSkeletonWrapper, self).__init__('Not applicable', train_transform, test_transform)
        self.rgb_root = rgb_root
        self.skeleton_path = skeleton_path
        self.train_sample_ratio = train_sample_ratio
        self.test_sample_ratio = test_sample_ratio

        self.split_list = ['train', 'test']

    def load_split_and_targets(self, split: str):
        assert split in self.split_list
        train = True if split == 'train' else False
        transform = self.train_transform if split == 'train' else self.test_transform
        sample_ratio = self.train_sample_ratio if split == 'train' else self.test_sample_ratio
        print(f'Loading Kinetics400RGBSkeletonDataset (split: {split}, sample_ratio: {sample_ratio})...')
        split_set = Kinetics400RGBSkeletonDataset(
            self.rgb_root, self.skeleton_path,
            train=train,
            transform=transform, sample_ratio=sample_ratio
        )

        targets = [x['label'] for x in split_set.skeleton_data]

        return split_set, targets

    def load_train_and_targets(self):
        return self.load_split_and_targets('train')

    def load_test_and_targets(self):
        return self.load_split_and_targets('test')
