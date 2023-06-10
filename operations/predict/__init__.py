from .base import BasePredictor
from .cifar import CIFAR10Predictor
from .mnist import MNISTPredictor, FashionMnistPredictor
from .action_recognition import UCF101Predictor, UCF101RGBSkeletonPredictor, UCF101TinyRGBSkeletonPredictor

predictor_dict = {
    'MNIST': [MNISTPredictor, {}],
    'FashionMNIST': [FashionMnistPredictor, {}],
    'CIFAR10': [CIFAR10Predictor, {}],
    'UCF101': [UCF101Predictor, {}],
    'UCF101RGBSkeleton': [UCF101RGBSkeletonPredictor, {'draw_bbox': True, 'draw_skeleton': True}],
    'UCF101Tiny': [UCF101Predictor, {}],
    'UCF101TinyRGBSkeleton': [UCF101TinyRGBSkeletonPredictor,
                              {'draw_bbox': True, 'draw_skeleton': True, 'batch_size': 1, 'det_batch_size': 1}],
}
