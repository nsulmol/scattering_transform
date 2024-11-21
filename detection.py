import torchvision.models.detection.ssd as ssd
from kymatio.torch import Scattering2D
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np


ANCHOR_ASPECT_RATIOS = [0.5, 1, 2]


class Scattering2DPoolingBackbone(nn.Module):
    """Contains the remaining backbone for a multi-scale object detector.

    """
    def __init__(self, in_channels: int, shape: tuple(int, int), J: int,
                 L: int = 8, max_order: int = 2):
        self.in_channels = in_channels  # TODO: remove if not doing batch norm
        self.shape = np.array(shape)
        self.J = J
        self.out_shapes = []

        self.features = Scattering2D(J, shape, L, max_order, backend='torch')
        self._build()

    def _compute_out_shapes(self)
        self.out_shapes.append(self.shape // 2**self.J)

        shape = (None, None)
        while 1 not in shape:
            shape = self.out_shapes // 2
            # Force shape to be odd (aside from first, which is scattering
            # output). The modulo operation will return 1 if shape is even.
            shape = shape + abs(shape % 2 - 1)
            self.out_shapes.append(shape)

    def _build(self):
        self._compute_out_shapes()

        poolings = []
        for shape in self.out_shapes[1:]:
            poolings.append(nn.AdaptiveAvgPool2d(tuple(shape)))

        self.extra = nn.ModuleList(poolings)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.features(x)

        # Rescale? Normalize?
        rescaled = F.normalize(x)  # TODO: may not be necessary
        output = [rescaled]

        # Calculate feature maps for the remaining blocks
        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


def create_anchor_generator(backbone: Scattering2DPoolingBackbone):
    """Create an appropriate anchor generator for the provided backbone."""
    aspect_ratios = [ANCHOR_ASPECT_RATIOS for ratio in backbone.out_shapes]
    return ssd.anchor_utils.DefaultBoxGenerator(aspect_ratios)


class Scattering2DSSD(ssd.SSD):
    """Implements a scattering-based SSD model.

    Note: look up the torchvision.models.detection.ssd SSD for remaining
    arguments.

    Args:

    """

    def __init__(self, in_channels: int, shape: tuple(int, int), J: int,
                 L: int = 8, max_order: int = 2, **kwargs):

    backbone = Scattering2DPoolingBackbone(in_channels, shape, J, L, max_order)
    anchor_generator = create_anchor_generator(backbone)
    super().__init(**kwargs)
