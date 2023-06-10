import os

from .classification import ClsWrapper
from .skeleton import SkeletonDataset, RGBSkeletonDataset
from .tools import read_video


class UCF101SkeletonDataset(SkeletonDataset):
    """
    Skeletons of UCF101 dataset.
    """
    def __init__(self,
                 file_path: str,
                 fold: int = 1,
                 train: bool = True,
                 transform=None,
                 sample_ratio: float = 1.0):
        self.fold = fold
        super(UCF101SkeletonDataset, self).__init__(file_path, train, transform, sample_ratio)

    def _get_split(self):
        assert self.fold in [1, 2, 3], 'UCF101SkeletonDataset only supports fold within [1,2,3].'
        split = 'train' if self.train else 'test'
        return f'{split}{self.fold}'


class UCF101SkeletonWrapper(ClsWrapper):
    IN_CHANNELS = 17
    NUM_CLASSES = 101

    def __init__(self,
                 root: str,
                 fold: int = 1,
                 train_transform=None,
                 test_transform=None,
                 train_sample_ratio: float = 1.0,
                 test_sample_ratio: float = 1.0):
        if os.path.isdir(root):
            root = os.path.join(root, 'ucf101.pkl')
        super(UCF101SkeletonWrapper, self).__init__(root, train_transform, test_transform)
        assert 1 <= fold <= 3
        self.file_path = root
        self.fold = fold
        self.train_sample_ratio = train_sample_ratio
        self.test_sample_ratio = test_sample_ratio

        self.split_list = ['train', 'test']

    def load_split_and_targets(self, split: str):
        assert split in self.split_list
        train = True if split == 'train' else False
        transform = self.train_transform if split == 'train' else self.test_transform
        sample_ratio = self.train_sample_ratio if split == 'train' else self.test_sample_ratio
        print(f'Loading UCF101SkeletonDataset (split: {split}, sample_ratio: {sample_ratio})...')
        split_set = UCF101SkeletonDataset(
            self.file_path,
            train=train, fold=self.fold,
            transform=transform,  sample_ratio=sample_ratio
        )

        targets = [x['label'] for x in split_set.data]

        return split_set, targets

    def load_train_and_targets(self):
        return self.load_split_and_targets('train')

    def load_test_and_targets(self):
        return self.load_split_and_targets('test')


class UCF101RGBSkeletonDataset(RGBSkeletonDataset):
    """
    RGB+Skeletons of UCF101 dataset.
    The fold of RGB follows that of Skeletons.
    """
    def __init__(self,
                 rgb_root: str,
                 skeleton_path: str,
                 fold: int = 1,
                 train: bool = True,
                 transform=None,
                 sample_ratio: float = 1.0):
        self.fold = fold
        super(UCF101RGBSkeletonDataset, self).__init__(rgb_root, skeleton_path,
                                                       train, transform, sample_ratio)

    def _get_split(self):
        assert self.fold in [1, 2, 3], 'UCF101RGBSkeletonDataset only supports fold within [1,2,3].'
        split = 'train' if self.train else 'test'
        return f'{split}{self.fold}'

    def load_rgb_video(self, frame_dir: str):
        cls = frame_dir.split('_')[1]
        video_path = os.path.join(self.rgb_root, cls, f'{frame_dir}.avi')
        video, info = read_video(video_path)  # (t,h,w,c)
        return video


class UCF101RGBSkeletonWrapper(ClsWrapper):
    NUM_CLASSES = 101

    def __init__(self,
                 rgb_root: str,
                 skeleton_path: str,
                 fold: int = 1,
                 train_transform=None,
                 test_transform=None,
                 train_sample_ratio: float = 1.0,
                 test_sample_ratio: float = 1.0):
        super(UCF101RGBSkeletonWrapper, self).__init__('Not applicable', train_transform, test_transform)
        self.rgb_root = rgb_root
        self.skeleton_path = skeleton_path
        self.fold = fold
        self.train_sample_ratio = train_sample_ratio
        self.test_sample_ratio = test_sample_ratio

        self.split_list = ['train', 'test']

    def load_split_and_targets(self, split: str):
        assert split in self.split_list
        train = True if split == 'train' else False
        transform = self.train_transform if split == 'train' else self.test_transform
        sample_ratio = self.train_sample_ratio if split == 'train' else self.test_sample_ratio
        print(f'Loading UCF101RGBSkeletonDataset (split: {split}, sample_ratio: {sample_ratio})...')
        split_set = UCF101RGBSkeletonDataset(
            self.rgb_root, self.skeleton_path,
            train=train, fold=self.fold,
            transform=transform, sample_ratio=sample_ratio
        )

        targets = [x['label'] for x in split_set.skeleton_data]

        return split_set, targets

    def load_train_and_targets(self):
        return self.load_split_and_targets('train')

    def load_test_and_targets(self):
        return self.load_split_and_targets('test')
