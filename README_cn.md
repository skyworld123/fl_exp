# Federated learning experiment project

基于PyTorch的联邦学习实验平台，支持MNIST、FashionMNIST、CIFAR10等经典图像分类任务以及HMDB51、UCF101、Kinetics400等大型行为识别任务模型的训练、测试和预测。



# 项目功能介绍

## 联邦学习算法

* FedAvg
* FedProx
* MOON（部分模型）
* AdaFed
* FedBS
* FedADMM（测试中）
* ClusteredSampling

支持直接使用中心化方法训练模型，方便与联邦学习算法的性能对比

## 数据集

* MNIST
* FashionMNIST
* CIFAR10
* HMDB51
* UCF101
* Kinetics400

## 模型

* [FedAvg](https://github.com/AshwinRJ/Federated-Learning-PyTorch)原版模型
* ResNet18 (for cifar)
* Inception
* ResNetI3D
* Video Swin Transformer
* PoseC3D

## 配套工具

* 提取RGB视频行为数据集中的骨架
* 绘制Tensorboard结果

## 可扩展性

添加新的联邦学习算法：

* 在flmethods/下添加新算法的.py代码，编写Manager类、Client类与Server类的代码逻辑，并在flmethods/\_\_init__.py中添加新算法的Manager类
* 在flmethods/trainers下添加新算法的.py代码，编写训练模块（Trainer）类的代码逻辑，并在flmethods/trainers/\_\_init__.py中添加新算法的Trainer类

添加新数据集：

* 在datasets/下添加新数据集的.py代码，编写数据集类和Wrapper类的代码逻辑，并在datasets/\_\_init__.py中添加新数据集的Wrapper类
* 在operations/test/\_\_init__.py中指定新数据集所使用的测试模块（Tester）类及相应参数，若现有Tester类不符合需求，可在operations/test/下编写新的Tester类
* （可选）在operations/predict/\_\_init__.py中指定新数据集所使用的预测模块（Predictor）类及相应参数，若现有Predictor类不符合需求，可在operations/predict/下编写新的Predictor类

添加新模型：

* 在models/下添加新模型的.py代码，并在models/\_\_init__.py中添加该模型的类



# 环境配置

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

安装mmcv：（Video Swin Transformer、PoseC3D需要用到）

```shell
pip install openmim
mim install mmcv-full
```



# 实验操作

## 训练

训练前需在相应配置文件（yml或yaml格式）中设置好相应参数。yml/yaml文件中的参数涉及以下内容：

* 联邦学习算法类型和参数
* 数据集（训练集、测试集）类型、存放根目录、批量大小、划分方法、预处理方法等
* 模型类型和参数
* 损失函数类型和参数
* 优化器类型和参数
* 学习率调度器类型和参数（可选）

同时需要配合函数运行入口train.py中的输入参数对整个训练过程进行控制。这些参数包括：

* **-c/--config**：配置文件的路径
* --output_dir：保存训练输出结果（模型、中间状态、Tensorboard记录等）的目录。可以不手动指定，代码会在output文件夹下自动生成一个输出目录。
* --resume：从某个保存中间状态的目录恢复并继续开始训练（适用于训练到一半程序意外退出，又不想从头重新训练的情况）
* **--eval**：是否在训练过程中每隔save_interval个轮次进行测试（使用方法：直接--eval，不是--eval=True）
* **--save_interval**：每多少个轮次保存模型和中间状态
* --keep_checkpoint_max：最多保存几组模型和中间状态（当模型过大时，可防止磁盘爆满）
* --num_workers：加载数据集的额外进程数量（本地训练时一般设为0）
* **--board**：是否使用Tensorboard记录训练过程（使用方法：直接--board，不是--board=True）
* --seed：随机数种子（默认为32）
* **--device**：训练模型使用的设备
* --verbose_train：是否显示训练过程进度条（使用方法：直接--verbose_train）
* --verbose_val：是否显示验证过程进度条（使用方法：直接--verbose_val）

示例：

传统方法（集中训练）：

```shell
python train.py \
    -c=configs/centralized.yml
```

联邦学习方法：使用FedAvg算法，在yaml配置文件中的指定数据集上训练：

```shell
python train.py \
    -c=configs/fedavg.yml
```

## 测试

如果在训练时指定了--do_eval，则训练过程中会自动在测试集上完成测试并输出指标（无论--save_interval设为多少，总会在最后一轮进行测试并保存模型）。如果训练时没有指定--do_eval可事后再用test.py进行测试。

测试使用的yml/yaml配置文件要与训练时一致。

函数执行入口test.py的输出参数包括：

* **-c/--config**：配置文件的路径
* **--model**：预训练模型的路径（若训练时指定了--do_eval再来测试，则会在输出目录下自动生成最佳模型目录best_model，此时无需再指定--model参数，程序自动加载该最佳模型进行测试）
* --num_workers：加载数据集的额外进程数量（本地测试时一般设为0）
* --seed：随机数种子（默认为32）
* **--device**：测试模型使用的设备
* --verbose：是否显示进度条

示例：

测试使用传统方法（集中训练）训练所得模型：

```shell
python test.py \
    -c=configs/centralized.yml \
    --model=output/Centralized_MNIST_CNNMnist/epoch100/model.pth
```

测试使用联邦学习方法FedAvg训练所得模型：

```shell
python test.py \
    -c=configs/fedavg.yml \
    --model=output/FedAvg_MNIST_CNNMnist_iid/epoch100/model.pth
```

如果训练时指定了--do_eval，相应的输出目录下会生成best_model目录，此时可直接运行：

```shell
python test.py \
    -c=configs/fedavg.yml
```

## 预测

使用上面训练所得的模型，可以使用predict.py，在该模型对应任务上对输入数据预测其输出结果。不同的任务有不同的输出形式，例如，对于MNIST、CIFAR等图像分类任务，预测程序会在终端输出图片分类的类别，或者将结果保存到Excel表格中；对于UCF101等视频动作识别任务，会输出带有动作类型标注的视频作为结果。

预测使用的yml/yaml配置文件也要与训练时一致。

predict.py的参数包括：

* **-c/--config**：配置文件的路径
* **-i/--input**：输入文件的路径，可以是单个文件或文件夹（批量输入）
* -r/--recursive：若输入是文件夹，指定-r可递归寻找该文件夹下所有符合要求的文件作为输入，否则只在该文件夹下一层内寻找符合要求的文件
* -e/--ext：指定输入文件的扩展名
* --model：模型路径，若不指定，则默认根据配置文件寻找训练过程中产生的最佳模型（best_model）
* --seed：随机数种子（默认为32）
* **--device**：预测使用的设备
* --show_output：对于视频行为识别任务，是否在预测过程中以视频形式实时输出预测结果（使用方法：直接--show_output）
* **-o/--output_dir**：输出文件夹路径，默认为predict_results

示例：

使用FedAvg训练所得MNIST模型，预测手写数字图像test_digit.jpg：

```shell
python predict.py \
    -c=configs/fedavg.yml \
    -i=test_digit.jpg
    --model=output/FedAvg_MNIST_CNNMnist_iid/epoch100/model.pth
    -o=predict_results
```

## 配套工具

extract_pose_kinetics.py：人体检测模型、姿态估计模型分别来源于[mmdet项目](https://github.com/open-mmlab/mmdetection)和[mmpose项目](https://github.com/open-mmlab/mmpose)，可提前将链接中的模型下载到本地。

