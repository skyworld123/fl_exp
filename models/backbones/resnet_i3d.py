import os
import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Callable, Union, List, Optional, Tuple

from utils import ensure_dir

__all__ = ['ResNetI3D']

model_2d_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3d_3x3(in_planes: int,
               out_planes: int,
               spatial_stride: int = 1,
               temporal_kernel: int = 1,
               temporal_stride: int = 1,
               dilation: int = 1,
               groups: int = 1) -> nn.Conv3d:
    spatial_kernel = 3
    padding = (0, dilation, dilation)
    dilation = (1, dilation, dilation)
    return nn.Conv3d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=(temporal_kernel, spatial_kernel, spatial_kernel),
                     stride=(temporal_stride, spatial_stride, spatial_stride),
                     padding=padding,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv3d_1x1(in_planes: int,
               out_planes: int,
               spatial_stride: int = 1,
               temporal_kernel: int = 1,
               temporal_stride: int = 1) -> nn.Conv3d:
    spatial_kernel = 1
    return nn.Conv3d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=(temporal_kernel, spatial_kernel, spatial_kernel),
                     stride=(temporal_stride, spatial_stride, spatial_stride),
                     bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            spatial_stride: int = 1,
            temporal_stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = conv3d_3x3(inplanes, planes,
                                spatial_stride=spatial_stride,
                                temporal_stride=temporal_stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        self.conv2 = conv3d_3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            spatial_stride: int = 1,
            temporal_stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv1x1(inplanes, width)
        self.conv1 = conv3d_1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # self.conv2 = conv3x3(width, width,stride, groups, dilation)
        self.conv2 = conv3d_3x3(width, width,
                                spatial_stride=spatial_stride,
                                temporal_stride=temporal_stride,
                                groups=groups,
                                dilation=dilation)
        self.bn2 = norm_layer(width)
        # self.conv3 = conv1x1(width, planes * self.expansion)
        self.conv3 = conv3d_1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetI3D(nn.Module):
    supported_arch = {'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'}
    arch2block = {
        'resnet18': BasicBlock,
        'resnet34': BasicBlock,
        'resnet50': Bottleneck,
        'resnet101': Bottleneck,
        'resnet152': Bottleneck,
    }
    arch2layers = {
        'resnet18': (2, 2, 2, 2),
        'resnet34': (3, 4, 6, 3),
        'resnet50': (3, 4, 6, 3),
        'resnet101': (3, 4, 23, 3),
        'resnet152': (3, 8, 36, 3),
    }

    _cached_2d_pretrained = {}

    def __init__(
            self,
            arch: str,
            load_2d_pretrained: bool = False,
            spatial_strides: Tuple = (1, 2, 2, 2),
            temporal_strides: Tuple = (1, 2, 2, 2),
            in_channels: int = 3,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNetI3D, self).__init__()
        assert arch in self.supported_arch
        self.arch = arch
        block = self.arch2block[arch]
        layers = self.arch2layers[arch]

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(in_channels, self.inplanes,
                               kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3),
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       spatial_stride=spatial_strides[0], temporal_stride=temporal_strides[0])
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       spatial_stride=spatial_strides[1], temporal_stride=temporal_strides[1],
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       spatial_stride=spatial_strides[2], temporal_stride=temporal_strides[2],
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       spatial_stride=spatial_strides[3], temporal_stride=temporal_strides[3],
                                       dilate=replace_stride_with_dilation[2])

        self._zero_init_residual = zero_init_residual
        self._init_weights()

        if load_2d_pretrained:
            self.load_2d_pretrained()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self._zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def load_2d_pretrained(self, cache_inflated: str = None):
        if self._cached_2d_pretrained.get(self.arch) is None:
            print(f'Loading pretrained 2D {self.arch} model...')
            self._cached_2d_pretrained[self.arch] = load_state_dict_from_url(model_2d_urls[self.arch])

        state_dict = self.state_dict()
        for k, v in self._cached_2d_pretrained[self.arch].items():
            if k not in state_dict:
                continue
            module = self
            comp = k.split('.')
            for c in comp[:-1]:
                if c.isdigit():
                    idx = int(c)
                    module = module[idx]
                else:
                    module = getattr(module, c)
            if isinstance(module, nn.Conv3d):
                dim = 2
                shape_2d = list(v.shape)
                shape_dim = state_dict[k].shape[dim]
                shape_3d = shape_2d[:dim] + [shape_dim] + shape_2d[dim:]
                v = torch.repeat_interleave(v, shape_dim, dim).reshape(*shape_3d)
            state_dict[k] = v

        self.load_state_dict(state_dict)
        if cache_inflated is not None and not os.path.exists(cache_inflated):
            path_dir = os.path.dirname(cache_inflated)
            ensure_dir(path_dir)
            torch.save(state_dict, cache_inflated)
            print(f'Cached inflated {self.__class__.__name__} saved at {cache_inflated}.')

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    spatial_stride: int = 1, temporal_stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation = self.dilation * spatial_stride
            spatial_stride = 1
        if spatial_stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # conv1x1(self.inplanes, planes * block.expansion, stride),
                conv3d_1x1(self.inplanes, planes * block.expansion,
                           spatial_stride=spatial_stride,
                           temporal_stride=temporal_stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            spatial_stride=spatial_stride,
                            temporal_stride=temporal_stride,
                            downsample=downsample,
                            groups=self.groups,
                            base_width=self.base_width,
                            dilation=previous_dilation,
                            norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


if __name__ == '__main__':
    model = ResNetI3D(arch='resnet18')

    x = torch.empty((2, 3, 16, 224, 224))
    out = model(x)

    aaa = 1

