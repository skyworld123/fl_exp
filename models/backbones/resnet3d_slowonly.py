from .resnet3d_slowfast import ResNet3dPathway


class ResNet3dSlowOnly(ResNet3dPathway):
    """SlowOnly backbone based on ResNet3dPathway.

    Args:
        *args (arguments): Arguments same as :class:`ResNet3dPathway`.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Default: (1, 7, 7).
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Default: 1.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Default: 1.
        inflate (Sequence[int]): Inflate Dims of each block.
            Default: (0, 0, 1, 1).
        **kwargs (keyword arguments): Keywords arguments for
            :class:`ResNet3dPathway`.
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(
            *args,
            lateral_stem=False,
            lateral_layers=(0, 0, 0, 0),
            **kwargs)
