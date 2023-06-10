import cv2

from .mnist import MNISTPredictor


class CIFAR10Predictor(MNISTPredictor):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    save_file = 'CIFAR10-predictions'

    @staticmethod
    def _read_image(path):
        image = cv2.imread(path)
        image = cv2.resize(image, (32, 32))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
