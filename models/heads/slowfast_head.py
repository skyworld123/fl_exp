import torch
import torch.nn as nn


class SlowFastHead(nn.Module):
    """
    x: ((n, in_channels1, T1, H1, W1), (n, in_channels2, T2, H2, W2)
    :return: (n, num_classes)
    """

    def __init__(self,
                 in_channels1,
                 in_channels2,
                 num_classes=400,
                 dropout=0.5):
        super(SlowFastHead, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        self.avg_pool1 = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.avg_pool2 = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(in_channels1 + in_channels2, self.num_classes)

    def forward(self, x):
        assert isinstance(x, list) or isinstance(x, tuple)
        assert len(x) == 2

        x1, x2 = x
        x1 = self.dropout1(self.avg_pool1(x1))
        x1 = x1.reshape(x1.shape[:2])
        x2 = self.dropout2(self.avg_pool2(x2))
        x2 = x2.reshape(x2.shape[:2])
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)

        return x


class SlowFastDoubleHead(nn.Module):
    """
    x: ((n, in_channels1, T1, H1, W1), (n, in_channels2, T2, H2, W2)
    :return: (n, num_classes), (n, num_classes)
    """

    def __init__(self,
                 in_channels1,
                 in_channels2,
                 num_classes=400,
                 dropout=0.5):
        super(SlowFastDoubleHead, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        self.avg_pool1 = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.avg_pool2 = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.fc1 = nn.Linear(in_channels1, self.num_classes)
        self.fc2 = nn.Linear(in_channels2, self.num_classes)

        self.train_mode = True

    def train(self, mode: bool = True):
        super().train(mode)
        self.train_mode = mode

    def forward(self, x):
        assert isinstance(x, list) or isinstance(x, tuple)
        assert len(x) == 2

        x1, x2 = x
        x1 = self.dropout1(self.avg_pool1(x1))
        x1 = x1.reshape(x1.shape[:2])
        x1 = self.fc1(x1)
        x2 = self.dropout2(self.avg_pool2(x2))
        x2 = x2.reshape(x2.shape[:2])
        x2 = self.fc2(x2)

        if self.train_mode:
            return x1, x2
        else:
            return (x1 + x2) / 2
