import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    x: (n, num_channels, 32, 32)
    :return: (n, num_classes)
    """

    def __init__(self, in_channels, num_classes):
        super(LeNet5, self).__init__()
        self.c1 = nn.Conv2d(in_channels, 6, (5, 5))
        self.s2 = nn.MaxPool2d(2, stride=2)
        self.c3 = nn.Conv2d(6, 16, (5, 5))
        self.s4 = nn.MaxPool2d(2, stride=2)
        self.c5 = nn.Conv2d(16, 120, (5, 5))
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.s2(x)
        x = F.relu(self.c3(x))
        x = self.s4(x)
        x = F.relu(self.c5(x))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.f6(x))
        x = self.f7(x)
        return x
