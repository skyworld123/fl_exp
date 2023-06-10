from torchvision.transforms import *

import collections
import copy
import cv2
from itertools import repeat
import numpy as np
# import random
import torch
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _scale_size(size, scale):
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    w, h = size
    return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)


def rescale_size(old_size, scale, return_scale=False):
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = collections.abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def seq2tuple(seq):
    # "(x, y)"
    if isinstance(seq, str):
        seq = eval(seq)
    # [x, y]
    if isinstance(seq, list):
        seq = tuple(seq)
    return seq


def imflip_(img, direction='horizontal'):
    """Inplace flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image (inplace).
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return cv2.flip(img, 1, img)
    elif direction == 'vertical':
        return cv2.flip(img, 0, img)
    else:
        return cv2.flip(img, -1, img)


def iminvert(img):
    """Invert (negate) an image.

    Args:
        img (ndarray): Image to be inverted.

    Returns:
        ndarray: The inverted image.
    """
    return np.full_like(img, 255) - img


class VideoSampler:
    """
    Uniformly sample frames from a video clip.
    """

    def __init__(self,
                 num_seg: int,
                 seg_len: int,
                 frame_interval: int = None,
                 test_mode: bool = False,
                 test_collapse: bool = False,
                 linspace_sample: bool = False):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.frame_interval = frame_interval
        self.test_mode = test_mode
        self.test_collapse = test_collapse
        self.linspace_sample = linspace_sample

        if not test_mode:
            assert num_seg == 1
        if test_collapse:
            assert num_seg == 1, f'A validation input has to be 6d if num_seg set to >1 ({num_seg}). ' \
                                 f'If need to collapse the input to 5d, set num_seg to 1.'

    @staticmethod
    def _get(frames_idx: np.ndarray, video: torch.Tensor):
        """
        frames_idx: (num_seg, seg_len) or (seg_len)
        video: (C, T, H, W)
        """
        frames_idx = torch.from_numpy(frames_idx).to(torch.int64)
        assert len(frames_idx.shape) in [1, 2]
        if len(frames_idx.shape) == 2:
            frames_total = []
            for i in range(frames_idx.shape[0]):
                frames_idx_seg = frames_idx[i]
                frames = video[:, frames_idx_seg]
                frames_total.append(frames)
            frames_total = torch.stack(frames_total, dim=0)
            return frames_total
        else:
            frames = video[:, frames_idx]
            return frames

    def _get_train_clips(self, num_frames):
        ori_seg_len = self.seg_len * self.frame_interval
        avg_interval = (num_frames - ori_seg_len + 1) // self.num_seg

        if avg_interval > 0:
            base_offsets = np.arange(self.num_seg) * avg_interval
            clip_offsets = base_offsets + np.random.randint(avg_interval,
                                                            size=self.num_seg)
        elif num_frames > max(self.num_seg, ori_seg_len):
            clip_offsets = np.sort(
                np.random.randint(num_frames - ori_seg_len + 1,
                                  size=self.num_seg))
        elif avg_interval == 0:
            ratio = (num_frames - ori_seg_len + 1.0) / self.num_seg
            clip_offsets = np.around(np.arange(self.num_seg) * ratio)
        else:
            clip_offsets = np.zeros((self.num_seg,), dtype=np.int64)
        return clip_offsets

    def _get_test_clips(self, num_frames):
        ori_seg_len = self.seg_len * self.frame_interval
        avg_interval = (num_frames - ori_seg_len + 1) / float(self.num_seg)
        if num_frames > ori_seg_len - 1:
            base_offsets = np.arange(self.num_seg) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int64)
        else:
            clip_offsets = np.zeros((self.num_seg,), dtype=np.int64)
        return clip_offsets

    def __call__(self, video: torch.Tensor):
        frames_len = video.shape[1]

        if self.frame_interval is not None:
            if not self.test_mode:
                offsets = self._get_train_clips(frames_len)
            else:
                offsets = self._get_test_clips(frames_len)
        elif self.linspace_sample:
            offsets = np.linspace(0, frames_len - 1, self.num_seg).astype(np.int64)
        else:
            average_dur = int(frames_len / self.num_seg)
            offsets = np.arange(self.num_seg) * max(average_dur, 1)
            if not self.test_mode and average_dur >= self.seg_len:
                offsets += np.random.randint(0, average_dur - self.seg_len + 1, self.seg_len)
            elif self.test_mode and average_dur >= self.seg_len:
                offsets += (average_dur - 1) // 2

        frames_idx = offsets[:, None] + np.arange(
            self.seg_len)[None, :] * self.frame_interval  # (frames_len,seg_num)
        frames_idx = np.mod(frames_idx, frames_len)
        if not self.test_mode \
                or (self.test_collapse and frames_idx.shape[0] == 1):
            frames_idx = frames_idx[0]  # (frames_len,)
        return self._get(frames_idx, video)


class UniformCrop:
    """
    Perform uniform spatial sampling on the images,
    select the two ends of the long side and the middle position (left middle right or top middle bottom) 3 regions.
    """

    def __init__(self, size):
        """
        size: target size (h, w)
        """
        if isinstance(size, int):
            size = (size, size)
        else:
            size = seq2tuple(size)
            if not isinstance(size, tuple):
                raise TypeError(f'size must be int or tuple[int] or list[int], but got {type(size)}')
        self.size = size

    def __call__(self, video: torch.Tensor):
        """
        video: (num_seg, C, T, H, W)
        """
        if len(video.shape) == 4:
            video = video.unsqueeze(0)

        h, w = video.shape[-2:]
        crop_h, crop_w = self.size
        short_h = h < w
        assert (short_h and h == crop_h) or (not short_h and w == crop_w), \
            f'UniformCrop requires the short edge of input to be equal to the target size. ' \
            f'However, the short edge of input is {"h" if short_h else "w"} and its size ({w}) ' \
            f'is unequal to the target edge size ({crop_w}).'
        if short_h:
            w_step = (w - crop_w) // 2
            offsets = [
                (0, 0),
                (w_step, 0),
                (w_step * 2, 0),
            ]
        else:
            h_step = (h - crop_h) // 2
            offsets = [
                (0, 0),
                (0, h_step),
                (0, h_step * 2),
            ]

        video_crops = []
        for i in range(video.shape[0]):
            video_seg = video[i]
            for x_offset, y_offset in offsets:
                crop = video_seg[:, :, y_offset:y_offset + crop_h,
                                 x_offset:x_offset + crop_w]
                video_crops.append(crop)
        video = torch.stack(video_crops)

        return video


class ResizeShort(torch.nn.Module):
    """
    Resize the shorter edge of the input image to a given value.
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        if not isinstance(size, int):
            raise TypeError("Size should be int. Got {}".format(type(size)))
        if not (size > 0):
            raise ValueError("Size should be positive. Got {}".format(size))
        self.size = size

        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def forward(self, img):
        """
        Args:
            img (torch.Tensor): Image to be scaled.

        Returns:
            torch.Tensor: Rescaled image.
        """
        img_size = img.shape[-2:]
        if img_size[0] < img_size[1]:
            s0 = self.size
            s1 = round(s0 * img_size[1] / img_size[0])
        else:
            s1 = self.size
            s0 = round(s1 * img_size[0] / img_size[1])

        dst_size = [s0, s1]
        extra_dims = len(img.shape) - 4
        if extra_dims > 0:
            extra_shape = list(img.shape[:extra_dims + 1])
            shape0 = 1
            for s in extra_shape:
                shape0 *= s
            shape = (shape0, ) + tuple(img.shape[extra_dims + 1:])
            img = torch.reshape(img, shape)
            result = F.resize(img, dst_size, interpolation=self.interpolation,
                              max_size=self.max_size, antialias=self.antialias)
            new_shape = extra_shape + list(result.shape[1:])
            result = torch.reshape(result, new_shape)
            return result
        else:
            return F.resize(img, dst_size, interpolation=self.interpolation,
                            max_size=self.max_size, antialias=self.antialias)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2}, antialias={3})'.format(
            self.size, interpolate_str, self.max_size, self.antialias)


class VideoTensorTransform(torch.nn.Module):
    """
    Transformations for tensors from video dataset
    (1) permute, default dims: (3, 0, 1, 2) (if transpose set to True)
    (2) convert to float type (if normalize set to True, divide by 255)
    """

    def __init__(self,
                 transpose=True,
                 dims=(3, 0, 1, 2),
                 normalize=False):
        super(VideoTensorTransform, self).__init__()
        self.transpose = transpose
        self.dims = dims
        self.normalize = normalize

    def forward(self, clip):
        """
        Args:
            clip (Tensor): Video clip to be transformed.

        Returns:
            Tensor: Transformed video clip.
        """
        if self.transpose:
            clip = torch.permute(clip, dims=self.dims)
        default_float_dtype = torch.get_default_dtype()
        if isinstance(clip, torch.ByteTensor):
            clip = clip.to(dtype=default_float_dtype)
        if self.normalize:
            clip = clip.div(255)

        return clip


class VideoNormalize(Normalize):
    def __init__(self,
                 mean,
                 std,
                 inplace=False,
                 data_format='cthw'):
        super().__init__(mean, std, inplace)
        self.mean = mean
        self.std = std
        self.inplace = inplace
        assert data_format in ['cthw', 'thwc']
        self.data_format = data_format

    def forward(self, tensor: Tensor) -> Tensor:
        dims = (1, 0, 2, 3) if self.data_format == 'cthw' else (0, 3, 1, 2)
        extra_dims = len(tensor.shape) - 4
        if extra_dims > 0:
            dims = list(range(extra_dims)) + [x + extra_dims for x in dims]
            dims = tuple(dims)

        tensor = torch.permute(tensor, dims=dims)  # data_format -> tchw
        tensor = F.normalize(tensor, self.mean, self.std, self.inplace)
        tensor = torch.permute(tensor, dims=dims)  # tchw -> data_format
        return tensor


# class Permute(torch.nn.Module):
#     """
#     Perform torch.permute for input tensor.
#     """
#
#     def __init__(self,
#                  dims=(3, 0, 1, 2)):
#         super(Permute, self).__init__()
#         self.dims = dims
#
#     def forward(self, tensor):
#         """
#         Args:
#             tensor (Tensor): Tensor to be permuted.
#
#         Returns:
#             Tensor: Permuted tensor.
#         """
#         return torch.permute(tensor, dims=self.dims)


# Skeleton transformations

class VideoTensorTransformV2:
    """
    VideoTensorTransform for RGB+pose data.
    """

    def __init__(self,
                 transpose=True,
                 dims=(3, 0, 1, 2),
                 normalize=False):
        self.transpose = transpose
        self.dims = dims
        self.normalize = normalize

    def __call__(self, results: dict):
        rgb = results['rgb']
        if self.transpose:
            rgb = torch.permute(rgb, dims=self.dims)
        default_float_dtype = torch.get_default_dtype()
        if isinstance(rgb, torch.ByteTensor):
            rgb = rgb.to(dtype=default_float_dtype)
        if self.normalize:
            rgb = rgb.div(255)
        results['rgb'] = rgb
        return results


class VideoNormalizeV2(Normalize):
    """
    VideoNormalize for RGB+pose data.
    """

    def forward(self, results: dict):
        tensor = results['rgb']
        tensor = torch.permute(tensor, dims=(1, 0, 2, 3))  # cthw -> tchw
        tensor = F.normalize(tensor, self.mean, self.std, self.inplace)
        tensor = torch.permute(tensor, dims=(1, 0, 2, 3))  # tchw -> cthw
        results['rgb'] = tensor
        return results


class RGBPoseAlignment:
    """
    When input data contain both RGB and pose modalities, this transform
    aligns the two modalities to enable subsequent transforms.
    Specifically, correct the pose data to adapt it to RGB data:
    (1) Pad/Crop the pose frames to make the number of frames of RGB data
        equal to the number of frames of pose data;
    (2) Scale the skeleton coordinates to make the size of RGB data equal to
        the size of pose data.
    """

    def __call__(self, results: dict):
        for key in ['rgb', 'keypoint', 'keypoint_score']:
            if key not in results:
                return results

        # correction to pose data
        video_total_frames = results['rgb'].shape[1]  # (c,t,h,w)
        pose_total_frames = results['total_frames'] \
            if 'total_frames' in results else results['keypoint'].shape[1]
        if pose_total_frames != video_total_frames:
            kpt_shape = list(results['keypoint'].shape)  # (np, nf, 17, 2)
            kpt_shape[1] = video_total_frames
            kpt_score_shape = list(results['keypoint_score'].shape)  # (np, nf, 17)
            kpt_score_shape[1] = video_total_frames
            copy_bound = min(pose_total_frames, video_total_frames)
            new_kpt = np.zeros(kpt_shape, dtype=results['keypoint'].dtype)
            new_kpt[:, :copy_bound] = results['keypoint']
            # We do not want to directly set all padded frames to be zeros, since
            # zero coordinates may be misunderstood by subsequent transforms on
            # pose data. Instead, we assume that the persons in the video remain
            # still since the last frame before the padded part, and simply copy this
            # frame to all the padded frames. However, the keypoint scores of the
            # padded frames are still set to zeros.
            if copy_bound < video_total_frames:
                new_kpt[:, copy_bound:] = \
                    results['keypoint'][:, copy_bound - 1:copy_bound].repeat(video_total_frames - copy_bound, axis=1)
            results['keypoint'] = new_kpt
            new_kpt_score = np.zeros(kpt_score_shape, dtype=results['keypoint_score'].dtype)
            new_kpt_score[:, :copy_bound, :] = results['keypoint_score']
            results['keypoint_score'] = new_kpt_score
            results['total_frames'] = video_total_frames
        video_shape = results['rgb'].shape[2], results['rgb'].shape[3]
        pose_shape = results['img_shape'] if 'img_shape' in results else (0, 0)
        if video_shape != pose_shape:
            hr, wr = video_shape[0] / pose_shape[0], video_shape[1] / pose_shape[1]
            kpt = results['keypoint']
            kpt[..., 0] = kpt[..., 0] * wr
            kpt[..., 1] = kpt[..., 1] * hr
            results['keypoint'] = kpt
            results['img_shape'] = video_shape

        return results


class UniformSampleFrames:
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len_rgb (int): Frames of each sampled output clip (rgb).
        clip_len_pose (int): Frames of each sampled output clip (pose).
        num_clips (int): Number of clips to be sampled. Default: 1.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self,
                 clip_len_rgb=8,
                 clip_len_pose=32,
                 num_clips=1,
                 test_mode=False):
        self.clip_len_rgb = clip_len_rgb
        self.clip_len_pose = clip_len_pose
        self.num_clips = num_clips
        self.test_mode = test_mode

        if not self.test_mode:
            assert self.num_clips == 1

    @staticmethod
    def _get_train_clips(num_frames, clip_len):
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        if num_frames < clip_len:
            start = np.random.randint(0, num_frames)
            inds = np.arange(start, start + clip_len)
        elif clip_len <= num_frames < 2 * clip_len:
            basic = np.arange(clip_len)
            inds = np.random.choice(
                clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset
        return inds

    def _get_test_clips(self, num_frames, clip_len):
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        if num_frames < clip_len:
            # Then we use a simple strategy
            if num_frames < self.num_clips:
                start_inds = list(range(self.num_clips))
            else:
                start_inds = [
                    i * num_frames // self.num_clips
                    for i in range(self.num_clips)
                ]
            all_inds = [np.arange(i, i + clip_len) for i in start_inds]
            inds = np.concatenate(all_inds)
        elif clip_len <= num_frames < clip_len * 2:
            all_inds = []
            for i in range(self.num_clips):
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
                all_inds.append(inds)
            inds = np.concatenate(all_inds)
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            all_inds = []
            for i in range(self.num_clips):
                offset = np.random.randint(bsize)
                all_inds.append(bst + offset)
            inds = np.concatenate(all_inds)
        return inds

    def __call__(self, results: dict):
        num_frames = results['total_frames']

        if self.test_mode:
            inds_rgb = self._get_test_clips(num_frames, self.clip_len_rgb)
            inds_pose = self._get_test_clips(num_frames, self.clip_len_pose)
        else:
            inds_rgb = self._get_train_clips(num_frames, self.clip_len_rgb)
            inds_pose = self._get_train_clips(num_frames, self.clip_len_pose)

        start_index = results['start_index']
        inds_rgb = np.mod(inds_rgb, num_frames)
        inds_rgb = inds_rgb + start_index
        inds_rgb = inds_rgb.astype(np.int64)
        inds_pose = np.mod(inds_pose, num_frames)
        inds_pose = inds_pose + start_index
        inds_pose = inds_pose.astype(np.int64)

        # uniform sample frames
        if 'rgb' in results:
            results['rgb'] = results['rgb'][:, inds_rgb]
        if 'keypoint' in results:
            results['keypoint'] = results['keypoint'][:, inds_pose].astype(np.float32)
        if 'keypoint_score' in results:
            results['keypoint_score'] = results['keypoint_score'][:, inds_pose].astype(np.float32)

        results['frame_inds_rgb'] = inds_rgb
        results['frame_inds_pose'] = inds_pose
        results['clip_len_rgb'] = self.clip_len_rgb
        results['clip_len_pose'] = self.clip_len_pose
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len_rgb={self.clip_len_rgb}, '
                    f'pose_clip_len={self.clip_len_pose}, '
                    f'num_clips={self.num_clips}, '
                    f'test_mode={self.test_mode})')
        return repr_str


# class PoseDecode:
#     """
#     This transformation uniformly samples RGB and pose frames with the indices
#     given in UniformSampleFrames.
#
#     Required keys are "keypoint", "frame_inds_rgb" (optional),
#     "frame_inds_pose" (optional), "keypoint_score"
#     (optional), added or modified keys are "keypoint", "keypoint_score" (if
#     applicable).
#     """
#
#     def __call__(self, results: dict):
#         for ind_key, keys in [('frame_inds_rgb', ('rgb', )),
#                               ('frame_inds_pose', ('keypoint', 'keypoint_score'))]:
#             if ind_key not in results:
#                 results[ind_key] = np.arange(results[ind_key])
#
#             if results[ind_key].ndim != 1:
#                 results[ind_key] = np.squeeze(results[ind_key])
#
#             offset = results.get('offset', 0)
#             frame_inds = results[ind_key] + offset
#
#             for key in keys:
#                 if key in results:
#                     results[key] = results[key][:, frame_inds]
#                     if key in ['keypoint', 'keypoint_score']:
#                         results[key] = results[key].astype(np.float32)
#
#         return results
#
#     def __repr__(self):
#         repr_str = f'{self.__class__.__name__}()'
#         return repr_str


class ExtractLargestPose:
    """
    Extract the largest pose among possibly multiple poses in a video.
    """

    def __call__(self, results: dict):
        kp = np.copy(results['keypoint'])  # (num_poses, num_frames, num_kps, 2)
        num_poses = kp.shape[0]
        if num_poses == 1:
            return

        kp_x = kp[..., 0]  # (num_poses, num_frames, num_kps)
        kp_y = kp[..., 1]  # (num_poses, num_frames, num_kps)
        x_min = np.nanmin(kp_x, axis=2)  # (num_poses, num_frames)
        x_max = np.nanmax(kp_x, axis=2)
        y_min = np.nanmin(kp_y, axis=2)
        y_max = np.nanmax(kp_y, axis=2)
        x_span = x_max - x_min
        y_span = y_max - y_min
        areas = np.multiply(x_span, y_span)  # (num_poses, num_frames)
        mean_areas = np.nanmean(areas, axis=1)

        if np.all(np.isnan(mean_areas)):
            idx = np.random.randint(0, num_poses)
        else:
            idx = np.nanargmax(mean_areas)

        results['keypoint'] = results['keypoint'][np.newaxis, ...]
        results['keypoint_score'] = results['keypoint_score'][idx][np.newaxis, ...]


class PoseCompact:
    """Convert the coordinates of keypoints to make it more compact.
    Specifically, it first find a tight bounding box that surrounds all joints
    in each frame, then we expand the tight box by a given padding ratio. For
    example, if 'padding == 0.25', then the expanded box has unchanged center,
    and 1.25x width and height.

    Required keys in results are "img_shape", "keypoint", add or modified keys
    are "img_shape", "keypoint", "crop_quadruple".

    Args:
        padding (float): The padding size. Default: 0.25.
        threshold (int): The threshold for the tight bounding box. If the width
            or height of the tight bounding box is smaller than the threshold,
            we do not perform the compact operation. Default: 10.
        hw_ratio (float | tuple[float] | None): The hw_ratio of the expanded
            box. Float indicates the specific ratio and tuple indicates a
            ratio range. If set as None, it means there is no requirement on
            hw_ratio. Default: None.
        allow_imgpad (bool): Whether to allow expanding the box outside the
            image to meet the hw_ratio requirement. Default: True.

    Returns:
        type: Description of returned object.
    """

    def __init__(self,
                 padding=0.25,
                 threshold=10,
                 hw_ratio=None,
                 allow_imgpad=True):

        self.padding = padding
        self.threshold = threshold
        if hw_ratio is not None:
            hw_ratio = (hw_ratio, hw_ratio)

        self.hw_ratio = hw_ratio

        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0

    @staticmethod
    def _combine_quadruple(a, b):
        return (a[0] + a[2] * b[0], a[1] + a[3] * b[1], a[2] * b[2],
                a[3] * b[3])

    def __call__(self, results: dict):
        img_shape = results['img_shape']
        h, w = img_shape
        kp = results['keypoint']

        # Make NaN zero
        kp_nan_mask = np.isnan(kp)
        kp[kp_nan_mask] = 0.
        kp_x = kp[..., 0]
        kp_y = kp[..., 1]
        kp_x_nn_mask = np.logical_not(kp_nan_mask[..., 0])
        kp_y_nn_mask = np.logical_not(kp_nan_mask[..., 1])

        min_x = np.min(kp_x[kp_x_nn_mask], initial=np.Inf)
        min_y = np.min(kp_y[kp_y_nn_mask], initial=np.Inf)
        max_x = np.max(kp_x[kp_x_nn_mask], initial=-np.Inf)
        max_y = np.max(kp_y[kp_y_nn_mask], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return results

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            half_height = max(self.hw_ratio[0] * half_width, half_height)
            half_width = max(1 / self.hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        # hot update
        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)

        # crop RGB data
        if 'rgb' in results:
            rgb_size = list(results['rgb'].shape)
            rgb_size[-2] = max_y - min_y
            rgb_size[-1] = max_x - min_x
            new_rgb = torch.zeros(rgb_size, dtype=results['rgb'].dtype)
            bl, br = max(min_x, 0), min(max_x, w)
            bt, bb = max(min_y, 0), min(max_y, h)
            l, r = bl - min_x, br - min_x
            t, b = bt - min_y, bb - min_y
            new_rgb[..., t:b, l:r] = results['rgb'][..., bt:bb, bl:br]  # (c,t,h,w)
            results['rgb'] = new_rgb

        # crop pose data
        kp_x[kp_x_nn_mask] -= min_x
        kp_y[kp_y_nn_mask] -= min_y
        # results['keypoint'] = np.stack((kp_x, kp_y), axis=-1)  # Useless!

        new_shape = (max_y - min_y, max_x - min_x)
        results['img_shape'] = new_shape

        # the order is x, y, w, h (in [0, 1]), a tuple
        crop_quadruple = results.get('crop_quadruple', (0., 0., 1., 1.))
        new_crop_quadruple = (min_x / w, min_y / h, (max_x - min_x) / w,
                              (max_y - min_y) / h)
        crop_quadruple = self._combine_quadruple(crop_quadruple,
                                                 new_crop_quadruple)
        results['crop_quadruple'] = crop_quadruple
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(padding={self.padding}, '
                    f'threshold={self.threshold}, '
                    f'hw_ratio={self.hw_ratio}, '
                    f'allow_imgpad={self.allow_imgpad})')
        return repr_str


class EntityBoxRescale:
    """Rescale the entity box and proposals according to the image shape.

    Required keys are "proposals", "gt_bboxes", added or modified keys are
    "gt_bboxes". If original "proposals" is not None, "proposals" and
    will be added or modified.

    Args:
        scale_factor (np.ndarray): The scale factor used entity_box rescaling.
    """

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, results: dict):
        scale_factor = np.concatenate([self.scale_factor, self.scale_factor])

        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            results['gt_bboxes'] = gt_bboxes * scale_factor

        if 'proposals' in results:
            proposals = results['proposals']
            if proposals is not None:
                assert proposals.shape[1] == 4, (
                    'proposals shape should be in '
                    f'(n, 4), but got {proposals.shape}')
                results['proposals'] = proposals * scale_factor

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(scale_factor={self.scale_factor})'


class ResizeV2:
    """Resize images to a specific size.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "lazy",
    "resize_size". Required keys in "lazy" is None, added or modified key is
    "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear'):
        scale = seq2tuple(scale)
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation

        # self.cv2_interp_codes = {
        #     'nearest': cv2.INTER_NEAREST,
        #     'bilinear': cv2.INTER_LINEAR,
        #     'bicubic': cv2.INTER_CUBIC,
        #     'area': cv2.INTER_AREA,
        #     'lanczos': cv2.INTER_LANCZOS4,
        # }
        self.torchvision_interp_codes = {
            'nearest': InterpolationMode.NEAREST,
            'bilinear': InterpolationMode.BILINEAR,
            'bicubic': InterpolationMode.BICUBIC,
            'hamming': InterpolationMode.HAMMING,
            'lanczos': InterpolationMode.LANCZOS,
        }

    def __call__(self, results: dict):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)
        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        # if 'imgs' in results:
        #     results['imgs'] = [
        #         cv2.resize(img, (new_w, new_h), interpolation=self.cv2_interp_codes[self.interpolation])
        #         for img in results['imgs']
        #     ]
        if 'rgb' in results:  # (c,t,h,w)
            results['rgb'] = F.resize(
                results['rgb'], [new_h, new_w],
                interpolation=self.torchvision_interp_codes[self.interpolation])
        if 'keypoint' in results:
            results['keypoint'] = results['keypoint'] * self.scale_factor

        if 'gt_bboxes' in results:
            entity_box_rescale = EntityBoxRescale(self.scale_factor)
            results = entity_box_rescale(results)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation})')
        return repr_str


class CropBase:
    @staticmethod
    def _crop_rgb(rgb, crop_bbox):
        x1, y1, x2, y2 = crop_bbox
        # (c,t,h,w)
        return rgb[..., y1:y2, x1:x2]

    @staticmethod
    def _crop_kps(kps, crop_bbox):
        return kps - crop_bbox[:2]

    @staticmethod
    def _box_crop(box, crop_bbox):
        """Crop the bounding boxes according to the crop_bbox.

        Args:
            box (np.ndarray): The bounding boxes.
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """

        x1, y1, x2, y2 = crop_bbox
        img_w, img_h = x2 - x1, y2 - y1

        box_ = box.copy()
        box_[..., 0::2] = np.clip(box[..., 0::2] - x1, 0, img_w - 1)
        box_[..., 1::2] = np.clip(box[..., 1::2] - y1, 0, img_h - 1)
        return box_

    def _all_box_crop(self, results, crop_bbox):
        """Crop the gt_bboxes and proposals in results according to crop_bbox.

        Args:
            results (dict): All information about the sample, which contain
                'gt_bboxes' and 'proposals' (optional).
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """
        results['gt_bboxes'] = self._box_crop(results['gt_bboxes'], crop_bbox)
        if 'proposals' in results and results['proposals'] is not None:
            assert results['proposals'].shape[1] == 4
            results['proposals'] = self._box_crop(results['proposals'],
                                                  crop_bbox)
        return results

    def __call__(self, results: dict):
        raise NotImplementedError


class RandomResizedCropV2(CropBase):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox" and "lazy"; Required keys in "lazy" are "flip", "crop_bbox",
    added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3)):
        area_range = seq2tuple(area_range)
        aspect_ratio_range = seq2tuple(aspect_ratio_range)
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        if not is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(
            target_areas * aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(
            target_areas / aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = np.random.randint(0, img_w - crop_w + 1)
                y_offset = np.random.randint(0, img_h - crop_h + 1)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def __call__(self, results: dict):
        """Performs the RandomResizedCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if 'rgb' in results:
            results['rgb'] = self._crop_rgb(results['rgb'], crop_bbox)

        if 'keypoint' in results:
            results['keypoint'] = self._crop_kps(results['keypoint'], crop_bbox)

        # if 'imgs' in results:
        #     results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)

        if 'gt_bboxes' in results:
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range})')
        return repr_str


class CenterCropV2(CropBase):
    """Crop the center area from images.

    Required keys are "img_shape", "imgs" (optional), "keypoint" (optional),
    added or modified keys are "imgs", "keypoint", "crop_bbox", "lazy" and
    "img_shape". Required keys in "lazy" is "crop_bbox", added or modified key
    is "crop_bbox".

    Args:
        crop_size (int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        if not isinstance(crop_size, int):
            crop_size = seq2tuple(crop_size)
        self.crop_size = _pair(crop_size)
        if not is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results: dict):
        """Performs the CenterCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        img_h, img_w = results['img_shape']
        crop_w, crop_h = self.crop_size

        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        new_h, new_w = bottom - top, right - left

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        if 'rgb' in results:
            results['rgb'] = self._crop_rgb(results['rgb'], crop_bbox)

        if 'keypoint' in results:
            results['keypoint'] = self._crop_kps(results['keypoint'], crop_bbox)

        # if 'imgs' in results:
        #     results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)

        if 'gt_bboxes' in results:
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


class FlipV2:
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "keypoint", "lazy" and
    "flip_direction". Required keys in "lazy" is None, added or modified key
    are "flip" and "flip_direction". The Flip augmentation should be placed
    after any cropping / reshaping augmentations, to make sure crop_quadruple
    is calculated properly.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        flip_label_map (Dict[int, int] | None): Transform the label of the
            flipped image with the specific label. Default: None.
        left_kp (list[int] | tuple[int]): Indexes of left keypoints, used to flip keypoints.
            Default: None.
        right_kp (list[int] | tuple[int]): Indexes of right keypoints, used to flip
            keypoints. Default: None.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 flip_label_map=None,
                 left_kp=None,
                 right_kp=None):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flip_label_map = flip_label_map
        self.left_kp = left_kp
        self.right_kp = right_kp

    # def _flip_imgs(self, imgs, modality):
    #     _ = [imflip_(img, self.direction) for img in imgs]
    #     lt = len(imgs)
    #     if modality == 'Flow':
    #         # The 1st frame of each 2 frames is flow-x
    #         for i in range(0, lt, 2):
    #             imgs[i] = iminvert(imgs[i])
    #     return imgs

    def _flip_kps(self, kps, kpscores, img_width):
        kp_x = kps[..., 0]
        kp_x[kp_x != 0] = img_width - kp_x[kp_x != 0]
        new_order = list(range(kps.shape[2]))
        if self.left_kp is not None and self.right_kp is not None:
            for left, right in zip(self.left_kp, self.right_kp):
                new_order[left] = right
                new_order[right] = left
        kps = kps[:, :, new_order]
        if kpscores is not None:
            kpscores = kpscores[:, :, new_order]
        return kps, kpscores

    @staticmethod
    def _box_flip(box, img_width):
        """Flip the bounding boxes given the width of the image.

        Args:
            box (np.ndarray): The bounding boxes.
            img_width (int): The img width.
        """
        box_ = box.copy()
        box_[..., 0::4] = img_width - box[..., 2::4]
        box_[..., 2::4] = img_width - box[..., 0::4]
        return box_

    def __call__(self, results: dict):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'keypoint' in results:
            assert self.direction == 'horizontal', (
                'Only horizontal flips are'
                'supported for human keypoints')

        # modality = results['modality']
        # if modality == 'Flow':
        #     assert self.direction == 'horizontal'

        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction
        img_width = results['img_shape'][1]

        if self.flip_label_map is not None and flip:
            results['label'] = self.flip_label_map.get(results['label'],
                                                       results['label'])

        if flip:
            # if 'imgs' in results:
            #     results['imgs'] = self._flip_imgs(results['imgs'], modality)
            if 'rgb' in results:
                assert self.direction in ['horizontal', 'vertical']
                if self.direction == 'horizontal':
                    results['rgb'] = results['rgb'].flip(dims=(3,))
                else:
                    results['rgb'] = results['rgb'].flip(dims=(2,))
            if 'keypoint' in results:
                kp = results['keypoint']
                kpscore = results.get('keypoint_score', None)
                kp, kpscore = self._flip_kps(kp, kpscore, img_width)
                results['keypoint'] = kp
                if 'keypoint_score' in results:
                    results['keypoint_score'] = kpscore

        if 'gt_bboxes' in results and flip:
            assert self.direction == 'horizontal'
            width = results['img_shape'][1]
            results['gt_bboxes'] = self._box_flip(results['gt_bboxes'], width)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_flip(results['proposals'],
                                                      width)

        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'flip_label_map={self.flip_label_map})')
        return repr_str


class GeneratePoseTarget:
    """Generate pseudo heatmaps based on joint coordinates and confidence.

    Required keys are "keypoint", "img_shape", "keypoint_score" (optional),
    added or modified keys are "imgs".

    Args:
        sigma (float): The sigma of the generated gaussian map. Default: 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Default: True.
        with_kp (bool): Generate pseudo heatmaps for keypoints. Default: True.
        with_limb (bool): Generate pseudo heatmaps for limbs. At least one of
            'with_kp' and 'with_limb' should be True. Default: False.
        skeletons (tuple[tuple]): The definition of human skeletons.
            Default: ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
                      (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
                      (6, 12), (12, 14), (14, 16), (11, 12)),
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Default: False.
        target_shape (int | tuple[int]): Specified shape of the generated
            heatmap. Default: None. (use image shape in the annotation)
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.
    """

    def __init__(self,
                 sigma=0.6,
                 use_score=True,
                 with_kp=True,
                 with_limb=False,
                 double=False,
                 target_shape=None,
                 skeletons=((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
                            (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
                            (13, 15), (6, 12), (12, 14), (14, 16), (11, 12)),
                 left_kp=(1, 3, 5, 7, 9, 11, 13, 15),
                 right_kp=(2, 4, 6, 8, 10, 12, 14, 16)):

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double
        if target_shape is not None:
            target_shape = seq2tuple(target_shape)
            if isinstance(target_shape, int):
                target_shape = (target_shape, target_shape)
            assert isinstance(target_shape, tuple), 'target_shape should be an int or a tuple of ints'
        self.target_shape = target_shape

        # an auxiliary const
        self.eps = 1e-4

        assert self.with_kp or self.with_limb, (
            'At least one of "with_limb" '
            'and "with_kp" should be set as True.')
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons

    def generate_a_heatmap(self, img_h, img_w, centers, sigma, max_values):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            centers (np.ndarray | list): The coordinates of corresponding keypoints
                (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray | list): The max values of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for center, max_value in zip(centers, max_values):
            mu_x, mu_y = center[0], center[1]
            if max_value < self.eps:
                continue

            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
            patch = patch * max_value
            heatmap[st_y:ed_y, st_x:ed_x] = np.maximum(
                heatmap[st_y:ed_y, st_x:ed_x], patch)

        return heatmap

    def generate_a_limb_heatmap(self, img_h, img_w, starts, ends, sigma,
                                start_values, end_values):
        """Generate pseudo heatmap for one limb in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            starts (np.ndarray): The coordinates of one keypoint in the
                corresponding limbs (of multiple persons).
            ends (np.ndarray): The coordinates of the other keypoint in the
                corresponding limbs (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            start_values (np.ndarray): The max values of one keypoint in the
                corresponding limbs.
            end_values (np.ndarray): The max values of the other keypoint in
                the corresponding limbs.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for start, end, start_value, end_value in zip(starts, ends,
                                                      start_values, end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < self.eps:
                continue

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = ((x - start[0]) ** 2 + (y - start[1]) ** 2)

            # distance to end keypoints
            d2_end = ((x - end[0]) ** 2 + (y - end[1]) ** 2)

            # the distance between start and end keypoints.
            d2_ab = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

            if d2_ab < 1:
                full_map = self.generate_a_heatmap(img_h, img_w, [start], sigma,
                                                   [start_value])
                heatmap = np.maximum(heatmap, full_map)
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff],
                                          axis=-1) * (end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0] ** 2 + d2_line[:, :, 1] ** 2
            d2_seg = (a_dominate * d2_start + b_dominate * d2_end +
                      seg_dominate * d2_line)

            patch = np.exp(-d2_seg / 2. / sigma ** 2)
            patch = patch * value_coeff

            heatmap[min_y:max_y, min_x:max_x] = np.maximum(
                heatmap[min_y:max_y, min_x:max_x], patch)

        return heatmap

    def generate_heatmap(self, img_h, img_w, kps, sigma, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            kps (np.ndarray): The coordinates of keypoints in this frame.
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The confidence score of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmaps = []
        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                heatmap = self.generate_a_heatmap(img_h, img_w, kps[:, i],
                                                  sigma, max_values[:, i])
                heatmaps.append(heatmap)

        if self.with_limb:
            for limb in self.skeletons:
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                heatmap = self.generate_a_limb_heatmap(
                    img_h, img_w, starts, ends, sigma, start_values, end_values)
                heatmaps.append(heatmap)

        return np.stack(heatmaps, axis=-1)

    def gen_an_aug(self, results):
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = results['keypoint']
        kp_shape = all_kps.shape

        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score']
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        if self.target_shape is None:
            img_h, img_w = results['img_shape']
        else:
            img_h, img_w = self.target_shape
        num_frame = kp_shape[1]

        imgs = []
        for i in range(num_frame):
            sigma = self.sigma
            kps = all_kps[:, i]
            kpscores = all_kpscores[:, i]

            if self.use_score:
                max_values = kpscores
            else:
                max_values = np.ones(kpscores.shape, dtype=np.float32)

            hmap = self.generate_heatmap(img_h, img_w, kps, sigma, max_values)
            imgs.append(hmap)

        return imgs

    def __call__(self, results: dict):
        if not self.double:
            results['pose_imgs'] = np.stack(self.gen_an_aug(results))
        else:
            results_ = copy.deepcopy(results)
            flip = FlipV2(
                flip_ratio=1, left_kp=self.left_kp, right_kp=self.right_kp)
            results_ = flip(results_)
            results['pose_imgs'] = np.concatenate(
                [self.gen_an_aug(results),
                 self.gen_an_aug(results_)])
        results['label'] = np.array(results['label'], dtype=np.int64)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sigma={self.sigma}, '
                    f'use_score={self.use_score}, '
                    f'with_kp={self.with_kp}, '
                    f'with_limb={self.with_limb}, '
                    f'skeletons={self.skeletons}, '
                    f'double={self.double}, '
                    f'left_kp={self.left_kp}, '
                    f'right_kp={self.right_kp})')
        return repr_str


class FormatShape:
    """Format final imgs shape to the given input_format.

    Required keys are "imgs", "num_clips" and "clip_len", added or modified
    keys are "imgs" and "input_shape".

    Args:
        input_format (str): Define the final imgs format.
        collapse (bool): To collpase input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Default: False.
    """

    def __init__(self, input_format, collapse=False):
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in ['NCTHW', 'NCHW', 'NCHW_Flow', 'NPTCHW']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def __call__(self, results: dict):
        """Performs the FormatShape formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # rgb: [C x M x H x W], M = N_crops * N_clips * L
        if 'rgb' in results:
            results['rgb'] = results['rgb'].permute(1, 2, 3, 0)

        # pose_imgs: [M x H x W x C], M = N_crops * N_clips * L
        if 'pose_imgs' in results and isinstance(results['pose_imgs'], np.ndarray):
            results['pose_imgs'] = torch.from_numpy(results['pose_imgs'])

        if self.collapse:
            assert results['num_clips'] == 1

        for key in ['rgb', 'pose_imgs']:
            if key not in results:
                continue

            imgs = results[key]
            num_clips = results['num_clips']
            if key == 'rgb':
                clip_len = results['clip_len_rgb']
            else:
                clip_len = results['clip_len_pose']

            if self.input_format == 'NCTHW':
                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x L x H x W x C
                imgs = imgs.permute(0, 1, 5, 2, 3, 4)
                # N_crops x N_clips x C x L x H x W
                imgs = imgs.reshape((-1,) + imgs.shape[2:])
                # M' x C x L x H x W
                # M' = N_crops x N_clips
            elif self.input_format == 'NCHW':
                imgs = imgs.permute(0, 3, 1, 2)
                # M x C x H x W
            elif self.input_format == 'NCHW_Flow':
                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x L x H x W x C
                imgs = imgs.permute(0, 1, 2, 5, 3, 4)
                # N_crops x N_clips x L x C x H x W
                imgs = imgs.reshape((-1, imgs.shape[2] * imgs.shape[3]) +
                                    imgs.shape[4:])
                # M' x C' x H x W
                # M' = N_crops x N_clips
                # C' = L x C
            elif self.input_format == 'NPTCHW':
                num_proposals = results['num_proposals']
                imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                    imgs.shape[1:])
                # P x M x H x W x C
                # M = N_clips x L
                imgs = imgs.permute(0, 1, 4, 2, 3)
                # P x M x C x H x W

            if self.collapse:
                assert imgs.shape[0] == 1
                imgs = imgs.squeeze(0)
                # if len(results['label'].shape) > 0:
                #     label = results['label']
                #     assert label.shape[0] == 1
                #     results['label'] = label.squeeze(0)

            results[key] = imgs

        # results['input_shape'] = imgs.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str


class Collect:
    """Collect data from the loader relevant to the specific task.

    This keeps the items in ``keys`` as it is, and collect items in
    ``meta_keys`` into a meta item called ``meta_name``.This is usually
    the last stage of the data loader pipeline.
    For example, when keys='pose_imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'pose_imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_name (str): The name of the key that contains meta infomation.
            This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str]): Keys that are collected under meta_name.
            The contents of the ``meta_name`` dictionary depends on
            ``meta_keys``.
            By default this includes:

            - "filename": path to the image file
            - "label": label of the image file
            - "original_shape": original shape of the image as a tuple
                (h, w, c)
            - "img_shape": shape of the image input to the network as a tuple
                (h, w, c).  Note that images may be zero padded on the
                bottom/right, if the batch tensor is larger than this shape.
            - "pad_shape": image shape after padding
            - "flip_direction": a str in ("horiziontal", "vertival") to
                indicate if the image is fliped horizontally or vertically.
            - "img_norm_cfg": a dict of normalization information:
                - mean - per channel mean subtraction
                - std - per channel std divisor
                - to_rgb - bool indicating if bgr was converted to rgb
    """

    def __init__(self,
                 keys,
                 to_tensor=False,
                 meta_keys=('filename', 'label', 'original_shape', 'img_shape',
                            'pad_shape', 'flip_direction', 'img_norm_cfg'),
                 meta_name='img_metas'):
        self.keys = keys
        self.to_tensor = to_tensor
        self.meta_keys = meta_keys
        self.meta_name = meta_name

    def __call__(self, results: dict):
        """Performs the Collect formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        data = []
        for key in self.keys:
            value = results[key]
            if self.to_tensor and isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            data.append(value)

        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                value = results[key]
                if self.to_tensor and isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                meta[key] = value
            data.append(meta)

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys})')
