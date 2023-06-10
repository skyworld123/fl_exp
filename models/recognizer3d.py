import torch
import torch.nn as nn
from typing import Union

from .backbones import *
from .heads import I3DHead, SlowFastHead, SlowFastDoubleHead
from .moon_models import MOONModel


class Recognizer3D(nn.Module):
    """
    x: (n, in_channels, T, H, W)
    :return: (n, num_classes)
    """
    supported_backbones = [InceptionV1I3D, ResNetI3D, VideoResNet, SwinTransformer3D,
                           ResNet3dSlowOnly, ResNet3dSlowFast]
    supported_heads = [I3DHead, SlowFastHead, SlowFastDoubleHead]

    _cached_backbone_pretrained = {}

    def __init__(self,
                 backbone: Union[str, dict],
                 head: Union[str, dict],
                 num_classes=400,
                 in_channels=3,
                 dropout=0.5):
        super(Recognizer3D, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout

        if isinstance(backbone, str):
            self.backbone = self.get_backbone(backbone)
        else:
            backbone = backbone.copy()
            backbone_type = backbone.pop('type')
            pretrained = backbone.pop('pretrained') if 'pretrained' in backbone else None
            self.backbone = self.get_backbone(backbone_type, **backbone)
            if pretrained is not None:
                if self._cached_backbone_pretrained.get(backbone_type) is None:
                    print(f'Loading pretrained {backbone_type} backbone from {pretrained}...')
                    self._cached_backbone_pretrained[backbone_type] = torch.load(pretrained)
                self.backbone.load_state_dict(self._cached_backbone_pretrained[backbone_type])

        if isinstance(head, str):
            self.head = self.get_head(head)
        else:
            head = head.copy()
            head_type = head.pop('type')
            self.head = self.get_head(head_type, **head)

    def get_backbone(self, backbone: str, **kwargs):
        arch_type = None
        for arch in self.supported_backbones:
            if arch.__name__ == backbone:
                arch_type = arch
                break
        if arch_type is None:
            raise RuntimeError(f'Backbone type "{backbone}" is not supported by {self.__class__.__name__}. '
                               f'Please choose a backbone type in {self.supported_backbones}.')

        if arch_type == InceptionV1I3D:
            return InceptionV1I3D(in_channels=self.in_channels)
        elif arch_type == ResNetI3D:
            kwargs['in_channels'] = self.in_channels
            return ResNetI3D(**kwargs)
        elif arch_type == VideoResNet:
            kwargs['in_channels'] = self.in_channels
            return VideoResNet(**kwargs)
        elif arch_type == SwinTransformer3D:
            kwargs['in_chans'] = self.in_channels
            return SwinTransformer3D(**kwargs)
        elif arch_type == ResNet3dSlowOnly:
            kwargs['in_channels'] = self.in_channels
            return ResNet3dSlowOnly(**kwargs)
        elif arch_type == ResNet3dSlowFast:
            return ResNet3dSlowFast(**kwargs)
        else:
            raise NotImplementedError(f'Input backbone arch_type "{arch_type}" not supported.')

    def get_head(self, head: str, **kwargs):
        arch_type = None
        for arch in self.supported_heads:
            if arch.__name__ == head:
                arch_type = arch
                break
        if arch_type is None:
            raise RuntimeError(f'Head type "{head}" is not supported by {type(self.__class__.__name__)}. '
                               f'Please choose a head type in {self.supported_heads}.')

        kwargs['num_classes'] = self.num_classes
        kwargs['dropout'] = self.dropout
        if isinstance(self.backbone, ResNetI3D):
            if self.backbone.arch in ['resnet18', 'resnet34']:
                kwargs['in_channels'] = 512
            else:
                kwargs['in_channels'] = 2048
            return I3DHead(**kwargs)
        elif isinstance(self.backbone, VideoResNet):
            kwargs['in_channels'] = 512
            return I3DHead(**kwargs)
        elif isinstance(self.backbone, SwinTransformer3D):
            kwargs['in_channels'] = self.backbone.num_features
            return I3DHead(**kwargs)
        elif isinstance(self.backbone, ResNet3dSlowOnly):
            kwargs['in_channels'] = self.backbone.inplanes
            return I3DHead(**kwargs)
        elif isinstance(self.backbone, ResNet3dSlowFast):
            kwargs['in_channels1'] = self.backbone.slow_path.inplanes
            kwargs['in_channels2'] = self.backbone.fast_path.inplanes
            assert arch_type in [SlowFastHead, SlowFastDoubleHead]
            return arch_type(**kwargs)
        else:
            raise NotImplementedError(f'Backbone type "{self.backbone}" not supported.')

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return x


class Recognizer3DMOON(Recognizer3D, MOONModel):
    def __init__(self,
                 backbone: Union[str, dict],
                 head: Union[str, dict],
                 num_classes=400,
                 in_channels=3,
                 dropout=0.5):
        super().__init__(backbone, head, num_classes, in_channels, dropout)

        self._proj = None

    def get_proj(self):
        return self._proj

    def forward(self, x):
        x = self.backbone(x)

        # self.head: SlowFastHead, SlowFastDoubleHead
        head = self.head
        if isinstance(head, SlowFastHead):
            assert isinstance(x, list) or isinstance(x, tuple)
            assert len(x) == 2

            x1, x2 = x
            x1 = head.dropout1(head.avg_pool1(x1))
            x1 = x1.reshape(x1.shape[:2])
            x2 = head.dropout2(head.avg_pool2(x2))
            x2 = x2.reshape(x2.shape[:2])
            x = torch.cat((x1, x2), dim=1)
            self._proj = x
            x = head.fc(x)

        elif isinstance(head, SlowFastDoubleHead):
            assert isinstance(x, list) or isinstance(x, tuple)
            assert len(x) == 2

            x1, x2 = x
            x1 = head.dropout1(head.avg_pool1(x1))
            x1 = x1.reshape(x1.shape[:2])
            x2 = head.dropout2(head.avg_pool2(x2))
            x2 = x2.reshape(x2.shape[:2])
            proj = torch.cat((x1, x2), dim=1)
            self._proj = proj

            x1 = head.fc1(x1)
            x2 = head.fc2(x2)
            if head.train_mode:
                x = x1, x2
            else:
                x = (x1 + x2) / 2

        else:
            raise NotImplementedError(f'head type "{head.__class__}" not supported')

        return x
