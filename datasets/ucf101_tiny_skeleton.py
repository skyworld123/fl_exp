from .ucf101_skeleton import UCF101SkeletonWrapper, UCF101RGBSkeletonWrapper


class UCF101TinySkeletonWrapper(UCF101SkeletonWrapper):
    NUM_CLASSES = 20


class UCF101TinyRGBSkeletonWrapper(UCF101RGBSkeletonWrapper):
    NUM_CLASSES = 20
