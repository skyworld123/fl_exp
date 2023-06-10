from .base import BaseTester
from .classification import ClassificationTester, ClassWiseClassificationTester

_default_cls_test_params = {
    'metric_params': dict(
        precision_average='micro',
        recall_average='micro',
    )
}
_default_cls_video_test_params = {
    'metric_params': dict(
        precision_average='micro',
        recall_average='micro',
        top_k=5,
    ),
    'video': True,
}
_default_cls_value = [ClassificationTester, _default_cls_test_params]
_default_cls_value_video = [ClassificationTester, _default_cls_video_test_params]

tester_dict = {
    'CIFAR10': _default_cls_value,
    'FashionMNIST': _default_cls_value,
    'MNIST': _default_cls_value,
    'HMDB51': _default_cls_value_video,
    'HMDB51Skeleton': _default_cls_value_video,
    'HMDB51RGBSkeleton': _default_cls_value_video,
    'Kinetics400': _default_cls_value_video,
    'Kinetics400Skeleton': _default_cls_value_video,
    'Kinetics400RGBSkeleton': _default_cls_value_video,
    'UCF101': _default_cls_value_video,
    'UCF101Skeleton': _default_cls_value_video,
    'UCF101RGBSkeleton': _default_cls_value_video,
    'UCF101Tiny': _default_cls_value_video,
    'UCF101TinyRGBSkeleton': _default_cls_value_video,
}
