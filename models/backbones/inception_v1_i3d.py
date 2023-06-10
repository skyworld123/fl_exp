import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['InceptionV1I3D']


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        # return super(MaxPool3dSamePadding, self).forward(x)
        return F.max_pool3d(x, self.kernel_size, self.stride,
                            0, self.dilation, self.ceil_mode,
                            self.return_indices)


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=None,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding if padding is not None else 'same'

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=self.padding,
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def forward(self, x):
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=(1, 1, 1))
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=(1, 1, 1))
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=(3, 3, 3))
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=(1, 1, 1))
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=(3, 3, 3))
        self.b3a = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=(1, 1, 1))

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionV1I3D(nn.Module):
    """
    Inception-v1 I3D backbone architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    x: (n, in_channels, T, H, W)
    :return: (n, 1024, T/8, H/32, W/32)
    """

    def __init__(self, in_channels=3):
        super(InceptionV1I3D, self).__init__()

        self.Conv3d_1a = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=(7, 7, 7),
                                stride=(2, 2, 2), padding=(3, 3, 3))
        self.maxpool3d_2a = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.Conv3d_2b = Unit3D(in_channels=64, output_channels=64, kernel_shape=(1, 1, 1))
        self.Conv3d_2c = Unit3D(in_channels=64, output_channels=192, kernel_shape=(3, 3, 3), padding=(1, 1, 1))

        self.maxpool3d_3a = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.Mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32])
        self.Mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64])

        self.maxpool3d_4a = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2))
        self.Mixed_4b = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64])
        self.Mixed_4c = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64])
        self.Mixed_4d = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64])
        self.Mixed_4e = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64])
        self.Mixed_4f = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128])

        self.maxpool3d_5a = MaxPool3dSamePadding(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.Mixed_5b = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128])
        self.Mixed_5c = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128])

    def forward(self, x):
        x = self.Conv3d_1a(x)
        x = self.maxpool3d_2a(x)
        x = self.Conv3d_2b(x)
        x = self.Conv3d_2c(x)
        x = self.maxpool3d_3a(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.maxpool3d_4a(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.maxpool3d_5a(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)

        return x
