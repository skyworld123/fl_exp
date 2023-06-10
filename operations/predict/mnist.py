import cv2
import os
import torch
from tqdm import tqdm

from .base import BasePredictor
from utils import save_data_in_csv


class MNISTPredictor(BasePredictor):
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    save_file = 'MNIST-predictions'

    def __init__(self,
                 inputs: list,
                 transform,
                 model: torch.nn.Module,
                 device=None,
                 show_output=True,
                 output_dir=None):
        super(MNISTPredictor, self).__init__(inputs, transform, model,
                                             device, show_output, output_dir)

    @staticmethod
    def _read_image(path):
        image = cv2.imread(path)
        image = cv2.resize(image, (28, 28))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def predict(self):
        print('Predicting...')
        self.model.eval()

        save = self.output_dir is not None
        saved_result = [['path', 'predicted_label', 'predicted_class']] if save else None
        bar = tqdm(total=len(self.inputs)) if not self.show_output else None
        for i, file_path in enumerate(self.inputs):
            if self.show_output:
                print(f'Predicting {i + 1}/{len(self.inputs)}: {file_path}')
            image = self._read_image(file_path)
            image = self.transform(image)
            image = image.unsqueeze(0)
            out = self.model(image)
            pred = out.argmax(1).detach().cpu().tolist()[0]
            class_name = self.classes[pred]
            if self.show_output:
                print(f'result: {class_name}')
            if save:
                saved_result.append([file_path, pred, class_name])
            if not self.show_output:
                bar.update()
        if not self.show_output:
            bar.close()

        if save:
            output_path = os.path.join(self.output_dir, f'{self.save_file}.csv')
            save_data_in_csv(saved_result, output_path)
            print(f'Results saved in {output_path}.')


class FashionMnistPredictor(MNISTPredictor):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    save_file = 'FashionMNIST-predictions'
