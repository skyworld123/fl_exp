# Federated learning experiment project

A PyTorch-based federated learning experiment platform, which currently supports training, testing and predicting of the following tasks:

* classic image classification tasks, including MNIST, FashionMNIST, and CIFAR10
* action recognition tasks, including HMDB51, UCF101 and Kinetics400



# Function introduction

## Federated learning algorithms

* FedAvg
* FedProx
* MOON (part of models)
* AdaFed
* FedBS
* FedADMM (under testing)
* ClusteredSampling

Centralized training is also supported, for performance comparisons to the federated learning algorithms.

## Datasets

* MNIST
* FashionMNIST
* CIFAR10
* HMDB51
* UCF101
* Kinetics400

## Models

* original models in [FedAvg](https://github.com/AshwinRJ/Federated-Learning-PyTorch)
* ResNet18 (for cifar)
* Inception
* ResNetI3D
* Video Swin Transformer
* PoseC3D

## Tools

* extracting skeletons in RGB video action recognition datasets
* drawing Tensorboard results

## Scalability

To add a new federated learning algorithm:

* Add a new .py code file under flmethods/, in which you should write the Manager, Client and Server classes. Then, add the Manager class (i.e. import this class) in flmethods/\__init__.py.
* Add a new .py code file under flmethods/trainers, in which you should write the Trainer class. Then, add the Trainer class (i.e. import this class) in flmethods/trainers/\__init__.py.

To add a new dataset:

* Add a new .py code file under datasets/, in which you should write the class of the new dataset and the corresponding Wrapper class. Then, add the Wrapper class (i.e. import this class) in datasets/\__init__.py.
* Specify the Tester class (and the corresponding parameters) for the new dataset in operations/test/\__init__.py. If existing Tester classes can't satisfy your demands, you can write new Tester classes under operations/test/.
* (optional) Specify the Predictor class (and the corresponding parameters) for the new dataset in operations/predict/\__init__.py. If existing Predictor classes can't satisfy your demands, you can write new Predictor classes under operations/predict/.

To add a new model:

* Add a new .py code file for the new model under models/, and add this model class (i.e. import this class) in models/\__init__.py.



# Environment configuration

```shell
torch>=1.10.1
torchvision>=0.11.2
tensorboard>=2.5  # if you want to open tensorboard
tensorboardx>=2.5
cudatoolkit  # if use GPU
scikit-learn
pyyaml
tqdm
av  # for video datasets
...
```

To install mmcv: (used for Video Swin Transformer, PoseC3D)

```shell
pip install openmim
mim install mmcv-full
```



# How to conduct experiments

## Training

Before training, please set the parameters in the corresponding configuration files (in .yml/.yaml format). These parameters involve the followings:

* federated learning algorithm types and parameters
* dataset (train set and test set) types, root directories, batch sizes, division methods and transformations
* model types and parameters
* loss function types and parameters
* optimizer types and parameters
* learning rate scheduler types and parameters (optional)

Meanwhile, you need to use the input arguments in train.py to control the whole training process. These arguments include:

* **-c/--config**: Path to the configuration file.
* --output_dir: Directory to save the output results of training (including models, internal states, and Tensorboard results, exactly a checkpoint for training). If not specified, the program will automatically generate an output directory under output/.
* --resume: Resume from the specified checkpoint directory to continue training.
* **--eval**: Whether to test every save_interval epochs during training (action='store_true').
* **--save_interval**: How many iterations to save a checkpoint each time.
* --keep_checkpoint_max: The maximum number of saved checkpoints. (when the model is too big, use this argument to prevent the disk space from exhausting)
* --num_workers: Number of extra processes to load the dataset. (when training locally, we generally set it to 0)
* **--board**: Whether to use Tensorboard to record the training process (action='store_true').
* --seed: Random seed (default is 32).
* **--device**: Device used for training.
* --verbose_train: Whether to show the progress bar during training (action='store_true').
* --verbose_val: Whether to show the progress bar during validation (action='store_true').

Examples:

Centralized training:

```shell
python train.py \
    -c=configs/centralized.yml
```

Federated learning: use FedAvg, train on the dataset specified in the yaml configuration file:

```shell
python train.py \
    -c=configs/fedavg.yml
```

## Testing

If --do_eval is specified when training, the testing will be conducted automatically on the test set during training, with metric values output at the same time (no matter how much --save_interval is, the program must test and save the model at the last epoch). If --do_eval is not specified when training, you can use test.py to test the saved models after training.

The yml/yaml files used in testing should align with that used in training.

The input arguments in test.py include:

* **-c/--config**: Path to the configuration file.
* **--model**: Path to the pretrained model. (If --do_eval is specified when training, the best model will be saved automatically under [output_dir]/best_model/. In this case, if --model is not specified, the program will automatically load the best model and test it.)
* --num_workers: Number of extra processes to load the dataset. (when training locally, we generally set it to 0)
* --seed: Random seed (default is 32).
* **--device**: Device used for testing.
* --verbose: Whether to show the progress bar (action='store_true').

Examples:

To test a model trained by centralized training:

```shell
python test.py \
    -c=configs/centralized.yml \
    --model=output/Centralized_MNIST_CNNMnist/epoch10/model.pth
```

To test a model trained by FedAvg:

```shell
python test.py \
    -c=configs/fedavg.yml \
    --model=output/FedAvg_MNIST_CNNMnist_iid/epoch10/model.pth
```

If --do_eval is specified in training, the best_model/ directory will be automatically generated under output directory. In this case you can run directly:

```shell
python test.py \
    -c=configs/fedavg.yml
```

## Predicting

Using the trained models, you can use predict.py to predict the output results for your own input data on the corresponding tasks. Different output formats are designed for different tasks. For example, for image classification tasks like MNIST and CIFAR, predict.py will output the predicted classes of the input images in console, or save these results into an Excel file; for video action recognition tasks like UCF101, predict.py will output a video with action class labels as the result.

The yml/yaml files used in predicting should also align with that used in training.

The input arguments in predict.py include:

* **-c/--config**: Path to the configuration file.
* **-i/--input**: Path to the input file(s), can be a single file or a directory (batch input).
* -r/--recursive: If input is a directory, specify -r to recursively find all suitable files under this directory as input files. Otherwise, only find suitable files which are directly under this directory.
* -e/--ext: Specify the allowed extension(s) for input files.
* --model: Path to the pretrained model. If not specified, the program finds the best model according the configuration file.
* --seed: Random seed (default is 32).
* **--device**: Device used for predicting.
* --show_output: For video action recognition tasks, whether to output real-time prediction results in videos during predicting (action='store_true')
* **-o/--output_dir**: Path to the output directory (default is predict_results/).

Examples:

To use the MNIST model trained by FedAvg to predict the hand-written digit image test_digit.jpg:

```shell
python predict.py \
    -c=configs/fedavg.yml \
    -i=test_digit.jpg
    --model=output/FedAvg_MNIST_CNNMnist_iid/epoch10/model.pth
    -o=predict_results
```

## Tools

extract_pose_kinetics.py: Human detection models and Pose estimation models come from [mmdet](https://github.com/open-mmlab/mmdetection) and [mmpose](https://github.com/open-mmlab/mmpose). You can download the models in the links in advance.

