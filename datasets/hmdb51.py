import glob
import numpy as np
import os
from typing import List

from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes, make_dataset

from .classification import ClsWrapper
from .tools import read_video


# class CustomHMDB51(VisionDataset):
#     """
#     `HMDB51 <http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>`_
#     dataset.
#
#     HMDB51 is an action recognition video dataset.
#     This dataset consider every video as a collection of video clips of fixed size, specified
#     by ``frames_per_clip``, where the step in frames between each clip is given by
#     ``step_between_clips``.
#
#     To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
#     and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
#     elements will come from video 1, and the next three elements from video 2.
#     Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
#     frames in a video might be present.
#
#     Internally, it uses a VideoClips object to handle clip creation.
#
#     Args:
#         root (string): Root directory of the HMDB51 Dataset.
#         annotation_path (str): Path to the folder containing the split files.
#         frames_per_clip (int): Number of frames in a clip.
#         step_between_clips (int): Number of frames between each clip.
#         fold (int, optional): Which fold to use. Should be between 1 and 3.
#         train (bool, optional): If ``True``, creates a dataset from the train split,
#             otherwise from the ``test`` split.
#         transform (callable, optional): A function/transform that takes in a TxHxWxC video
#             and returns a transformed version.
#
#     Returns:
#         tuple: A 3-tuple with the following entries:
#
#             - video (Tensor[T, H, W, C]): The `T` video frames
#             - audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
#               and `L` is the number of points
#             - label (int): class of the video clip
#     """
#
#     data_url = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
#     splits = {
#         "url": "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar",
#         "md5": "15e67781e70dcfbdce2d7dbb9b3344b5"
#     }
#     TRAIN_TAG = 1
#     TEST_TAG = 2
#
#     def __init__(
#         self,
#         root: str,
#         annotation_path: str,
#         frames_per_clip: int,
#         sample_ratio: float = 1.0,
#         step_between_clips: int = 1,
#         frame_rate: Optional[int] = None,
#         fold: int = 1,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         _precomputed_metadata: Optional[Dict[str, Any]] = None,
#         num_workers: int = 0,
#         _video_width: int = 0,
#         _video_height: int = 0,
#         _video_min_dimension: int = 0,
#         _audio_samples: int = 0,
#     ) -> None:
#         super(CustomHMDB51, self).__init__(root)
#         if fold not in (1, 2, 3):
#             raise ValueError("fold should be between 1 and 3, got {}".format(fold))
#
#         extensions = ('avi',)
#         self.classes, class_to_idx = find_classes(self.root)
#         self.samples = make_dataset(
#             self.root,
#             class_to_idx,
#             extensions,
#         )
#
#         video_paths = [path for (path, _) in self.samples]
#         video_clips = VideoClips(
#             video_paths,
#             frames_per_clip,
#             step_between_clips,
#             frame_rate,
#             _precomputed_metadata,
#             num_workers=num_workers,
#             _video_width=_video_width,
#             _video_height=_video_height,
#             _video_min_dimension=_video_min_dimension,
#             _audio_samples=_audio_samples,
#         )
#         # we bookkeep the full version of video clips because we want to be able
#         # to return the meta data of full version rather than the subset version of
#         # video clips
#         self.full_video_clips = video_clips
#         self.fold = fold
#         self.train = train
#         self.indices = self._select_fold(video_paths, annotation_path, fold, train)
#         self.video_clips = video_clips.subset(self.indices)
#         self.transform = transform
#
#         self.available_indices = np.arange(self.video_clips.num_clips())
#         self.sample_ratio = min(max(sample_ratio, 0), 1)
#         if self.sample_ratio < 1:
#             num_samples = max(round(len(self.available_indices) * self.sample_ratio), 1)
#             self.available_indices = np.random.choice(self.available_indices, num_samples, replace=False)
#
#     @property
#     def metadata(self) -> Dict[str, Any]:
#         return self.full_video_clips.metadata
#
#     def _select_fold(self, video_list: List[str], annotations_dir: str, fold: int, train: bool) -> List[int]:
#         target_tag = self.TRAIN_TAG if train else self.TEST_TAG
#         split_pattern_name = "*test_split{}.txt".format(fold)
#         split_pattern_path = os.path.join(annotations_dir, split_pattern_name)
#         annotation_paths = glob.glob(split_pattern_path)
#         selected_files = set()
#         for filepath in annotation_paths:
#             with open(filepath) as fid:
#                 lines = fid.readlines()
#             for line in lines:
#                 video_filename, tag_string = line.split()
#                 tag = int(tag_string)
#                 if tag == target_tag:
#                     selected_files.add(video_filename)
#
#         indices = []
#         for video_index, video_path in enumerate(video_list):
#             if os.path.basename(video_path) in selected_files:
#                 indices.append(video_index)
#
#         return indices
#
#     def __len__(self) -> int:
#         # return self.video_clips.num_clips()
#         return len(self.available_indices)
#
#     def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
#         video, _, _, video_idx = self.video_clips.get_clip(self.available_indices[idx])
#         sample_index = self.indices[video_idx]
#         _, class_index = self.samples[sample_index]
#
#         if self.transform is not None:
#             video = self.transform(video)
#
#         return video, class_index


class CustomHMDB51(Dataset):
    """
    HMDB51 video dataset.
    """
    TRAIN_TAG = 1
    TEST_TAG = 2

    def __init__(self,
                 root: str,
                 annotation_path: str,
                 fold: int = 1,
                 train: bool = True,
                 transform=None,
                 sample_ratio: float = 1.0):
        super(CustomHMDB51, self).__init__()
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))
        self.root = root
        self.fold = fold
        self.train = train
        self.transform = transform
        self.sample_ratio = min(max(sample_ratio, 0), 1)

        extensions = ('avi',)
        self.classes, class_to_idx = find_classes(self.root)
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        video_list = [x[0] for x in self.samples]
        self.indices = self._select_fold(video_list, annotation_path, self.fold, self.train)

        self.available_indices = np.arange(len(self.indices))
        self.sample_ratio = min(max(sample_ratio, 0), 1)
        if self.sample_ratio < 1:
            num_samples = max(round(len(self.available_indices) * self.sample_ratio), 1)
            self.available_indices = np.random.choice(self.available_indices, num_samples, replace=False)

    def _select_fold(self, video_list: List[str], annotations_dir: str, fold: int, train: bool) -> List[int]:
        target_tag = self.TRAIN_TAG if train else self.TEST_TAG
        split_pattern_name = "*test_split{}.txt".format(fold)
        split_pattern_path = os.path.join(annotations_dir, split_pattern_name)
        annotation_paths = glob.glob(split_pattern_path)
        selected_files = set()
        for filepath in annotation_paths:
            with open(filepath) as fid:
                lines = fid.readlines()
            for line in lines:
                video_filename, tag_string = line.split()
                tag = int(tag_string)
                if tag == target_tag:
                    selected_files.add(video_filename)

        indices = []
        for video_index, video_path in enumerate(video_list):
            if os.path.basename(video_path) in selected_files:
                indices.append(video_index)

        return indices

    def __len__(self) -> int:
        return len(self.available_indices)

    def __getitem__(self, idx: int):
        idx = self.available_indices[idx]
        video_path, label = self.samples[self.indices[idx]]
        video, info = read_video(video_path)
        if self.transform is not None:
            video = self.transform(video)

        return video, label


class HMDB51Wrapper(ClsWrapper):
    IN_CHANNELS = 3
    NUM_CLASSES = 51

    def __init__(self,
                 root: str,
                 fold: int = 1,
                 train_sample_ratio: float = 1.0,
                 test_sample_ratio: float = 1.0,
                 train_transform=None,
                 test_transform=None):
        super(HMDB51Wrapper, self).__init__(root, train_transform, test_transform)
        self.data_root = os.path.join(self.root, 'hmdb51_org')
        self.annotation_path = os.path.join(self.root, 'testTrainMulti_7030_splits')
        self.fold = fold
        self.train_sample_ratio = train_sample_ratio
        self.test_sample_ratio = test_sample_ratio

        self.split_list = ['train', 'test']

    def load_split_and_targets(self, split: str):
        assert split in self.split_list
        train = True if split == 'train' else False
        transform = self.train_transform if split == 'train' else self.test_transform
        sample_ratio = self.train_sample_ratio if split == 'train' else self.test_sample_ratio
        print(f'Loading HMDB51 (split: {split}, sample_ratio: {sample_ratio})...')
        split_set = CustomHMDB51(
            self.data_root, self.annotation_path,
            fold=self.fold, sample_ratio=sample_ratio,
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
