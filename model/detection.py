import torchvision.models.detection as tv_detection
from kymatio.torch import Scattering2D
#from kymatio.numpy import Scattering2D
import torch.nn.functional as F
from torch import nn, Tensor
import torch
import numpy as np
from collections import OrderedDict

from model import ssd as ssd  # Our local SSD


ANCHOR_ASPECT_RATIOS = [0.5, 1, 2]


class Scattering2DPoolingBackbone(nn.Module):
    """Contains the remaining backbone for a multi-scale object detector.
    """

    def __init__(self, shape: tuple[int, int], J: int,
                 L: int = 8, max_order: int = 2):
        super(Scattering2DPoolingBackbone, self).__init__()
        self.shape = np.array(shape)
        self.J = J
        self.out_shapes = []
        self.out_channels = []
        self.K = 1  # Input channels
        # scattering coefficients
        self.coeffs = int(1 + L*J + (L**2 * J*(J-1))/2)

        self._build()

    def _compute_out_shapes(self):
        self.out_shapes.append(self.shape // 2**self.J)
        while 1 not in self.out_shapes[-1]:
            shape = self.out_shapes[-1] // 2
            # Force shape to be odd (aside from first, which is scattering
            # output). The modulo operation will return 1 if shape is even.
            shape = shape + abs(shape % 2 - 1)
            self.out_shapes.append(shape)

    def _compute_out_channels(self):
        self.out_channels = [self.coeffs for shape in self.out_shapes]

    def _build(self):
        self._compute_out_shapes()
        self._compute_out_channels()

        poolings = []
        for shape in self.out_shapes[1:]:
            poolings.append(nn.AdaptiveAvgPool2d(tuple(shape)))

        self.extra = nn.ModuleList(poolings)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        #print(f'input dims: {x.size()}, type: {x.type()}')

        # Hack-ey computation of scattering
        # new_x = []
        # for b in x.size(0):
        #     channels = []
        #     for c in x.size(1):
        #         tmp = x.view(x.size(-2), x.size(-1))  # Go to (W, H) from (C, W, H)
        #         channels.append(self.features.scattering(tmp))

        #     new_x.append(channels)

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
    return tv_detection.anchor_utils.DefaultBoxGenerator(aspect_ratios)


class Scattering2DSSD(ssd.SSD):
    """Implements a scattering-based SSD model.

    Note: look up the torchvision.models.detection.ssd SSD for remaining
    arguments.

    Args:

    """

    def __init__(self, shape: tuple[int, int], J: int,
                 L: int = 8, max_order: int = 2, **kwargs):
        backbone = Scattering2DPoolingBackbone(shape, J, L,
                                               max_order)
        kwargs['backbone'] = backbone
        kwargs['size'] = shape
        kwargs['anchor_generator'] = create_anchor_generator(backbone)
        super().__init__(**kwargs)

        self.features = Scattering2D(J, shape, L, max_order)
        # self.scattering = self.scattering.to(device)

    def forward(
        self, images: list[Tensor],
            targets: list[dict[str, Tensor]] | None = None
    ) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:
        """"""
        # Perform scattering here and then pass along.
        res = []
        for image in images:
            scatterings = self.features.scattering(image)

            #print(f'scatterings shape: {scatterings.size()}')
            res.append(scatterings.view(-1, scatterings.size(-2),
                                        scatterings.size(-1)))
            #print(f'scatterings shape: {scatterings.size()}')
        return super().forward(res, targets)



# TODO: Copy and fiddle around with ssd.py!
# It is calling a normalize method (and probably other things) that we do ont
# need.
