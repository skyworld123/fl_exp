from .base import BaseWrapper
from .classification import ClsWrapper

from .cifar import CIFAR10Wrapper
from .hmdb51 import HMDB51Wrapper
from .hmdb51_skeleton import HMDB51SkeletonWrapper, HMDB51RGBSkeletonWrapper
from .kinetics400 import Kinetics400Wrapper
from .kinetics400_skeleton import Kinetics400SkeletonWrapper, Kinetics400RGBSkeletonWrapper
from .mnist import MNISTWrapper, FashionMNISTWrapper
from .ucf101 import UCF101Wrapper
from .ucf101_skeleton import UCF101SkeletonWrapper, UCF101RGBSkeletonWrapper

from .ucf101_tiny import UCF101TinyWrapper
from .ucf101_tiny_skeleton import UCF101TinySkeletonWrapper, UCF101TinyRGBSkeletonWrapper

from .shared import TrainSetDivision, TestSet
from .shared import ClsTrainSetDivision, ClsTestSet
