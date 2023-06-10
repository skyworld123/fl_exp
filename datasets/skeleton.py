from abc import abstractmethod
import copy
import numpy as np
import pickle
from torch.utils.data import Dataset


class SkeletonDataset(Dataset):
    """
    Skeleton dataset.
    """
    def __init__(self,
                 file_path: str,
                 train: bool = True,
                 transform=None,
                 sample_ratio: float = 1.0):
        super(SkeletonDataset, self).__init__()
        self.file_path = file_path
        self.train = train
        self.transform = transform
        self.sample_ratio = sample_ratio
        self.start_index = 0
        self.modality = 'Pose'

        self.split = self._get_split()
        self.data = self.load_file()

        self.available_indices = np.arange(len(self.data))
        self.sample_ratio = min(max(sample_ratio, 0), 1)
        if self.sample_ratio < 1:
            num_samples = max(round(len(self.available_indices) * self.sample_ratio), 1)
            self.available_indices = np.random.choice(self.available_indices, num_samples, replace=False)

    @abstractmethod
    def _get_split(self):
        pass

    def load_file(self):
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)

        split, data = data['split'], data['annotations']
        identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
        data = [x for x in data if x[identifier] in split[self.split]]

        return data

    def __len__(self):
        return len(self.available_indices)

    def __getitem__(self, idx):
        idx = self.available_indices[idx]
        results: dict = copy.deepcopy(self.data[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        return self.transform(results)


class RGBSkeletonDataset(Dataset):
    """
    RGB+Skeleton dataset.
    """
    def __init__(self,
                 rgb_root: str,
                 skeleton_path: str,
                 train: bool = True,
                 transform=None,
                 sample_ratio: float = 1.0):
        super(RGBSkeletonDataset, self).__init__()
        self.rgb_root = rgb_root
        self.skeleton_path = skeleton_path
        self.train = train
        self.transform = transform
        self.sample_ratio = sample_ratio
        self.start_index = 0

        self.split = self._get_split()
        self.skeleton_data = self.load_file()

        self.available_indices = np.arange(len(self.skeleton_data))
        self.sample_ratio = min(max(sample_ratio, 0), 1)
        if self.sample_ratio < 1:
            num_samples = max(round(len(self.available_indices) * self.sample_ratio), 1)
            self.available_indices = np.random.choice(self.available_indices, num_samples, replace=False)

    @abstractmethod
    def _get_split(self):
        pass

    def load_file(self):
        with open(self.skeleton_path, 'rb') as f:
            data = pickle.load(f)

        split, data = data['split'], data['annotations']
        identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
        data = [x for x in data if x[identifier] in split[self.split]]

        return data

    @abstractmethod
    def load_rgb_video(self, frame_dir: str):
        pass

    def __len__(self):
        return len(self.available_indices)

    def __getitem__(self, idx):
        idx = self.available_indices[idx]
        results: dict = copy.deepcopy(self.skeleton_data[idx])
        results['start_index'] = self.start_index

        rgb_data = self.load_rgb_video(results['frame_dir'])  # (t,h,w,c)
        results['rgb'] = rgb_data

        ret = self.transform(results)
        assert isinstance(ret, list) and len(ret) == 3, \
            'Transformed data should be a list containing rgb, skeleton image, label'
        rgb, skeleton_image, label = ret
        return (rgb, skeleton_image), label
