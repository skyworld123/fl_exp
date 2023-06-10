import torch.nn as nn


class I3DHead(nn.Module):
    """
    x: (n, in_channels, T, H, W)
    :return: (n, num_classes)
    """

    def __init__(self,
                 in_channels,
                 num_classes=400,
                 dropout=0.5):
        super(I3DHead, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.dropout = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(in_channels, self.num_classes)

    def forward(self, x):
        x = self.dropout(self.avg_pool(x))
        b, c = x.shape[:2]
        x = x.reshape((b, c))
        x = self.fc(x)

        return x
