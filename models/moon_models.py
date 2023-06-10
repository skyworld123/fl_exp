import abc
from abc import ABC
import torch.nn as nn
import torch.nn.functional as F


class MOONModel(ABC):

    @abc.abstractmethod
    def get_proj(self):
        pass


class MLPImageMOON(nn.Module, MOONModel):
    """
    x: (n, c, h, w) (in_dim=c*h*w)
    :return: (n, out_dim)
    """

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLPImageMOON, self).__init__()
        self.layer_input = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(hidden_dim, out_dim)

        self._proj = None

    def get_proj(self):
        return self._proj

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        self._proj = x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnistMOON(nn.Module, MOONModel):
    """
    x: (n, in_channels, 28, 28)
    :return: (n, num_classes)
    """

    def __init__(self, in_channels, num_classes):
        super(CNNMnistMOON, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

        self._proj = None

    def get_proj(self):
        return self._proj

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        self._proj = x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNFashionMnistMOON(nn.Module, MOONModel):
    """
    x: (n, in_channels, 28, 28)
    :return: (n, num_classes)
    """

    def __init__(self, in_channels, num_classes):
        super(CNNFashionMnistMOON, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(7*7*32, 50)
        self.fc2 = nn.Linear(50, num_classes)

        self._proj = None

    def get_proj(self):
        return self._proj

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        self._proj = x
        x = F.relu(x)
        x = self.fc2(x)
        return x


class CNNCifarMOON(nn.Module, MOONModel):
    """
    x: (n, in_channels, 32, 32)
    :return: (n, num_classes)
    """

    def __init__(self, in_channels, num_classes):
        super(CNNCifarMOON, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self._proj = None

    def get_proj(self):
        return self._proj

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        self._proj = x
        x = F.relu(x)
        x = self.fc3(x)
        return x
