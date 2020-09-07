import numpy as np
import pydensecrf.densecrf as dcrf
import torch
from pydensecrf.utils import unary_from_softmax


class ConditionalRandomFields:
    """
    Conditional Random Fileds (CRF)

    Examples:
        >>> x = torch.zeros(size=(3,128,128))
        >>> y = torch.zeros(size=(3,128,128))

        >>> c = ConditionalRandomFields.inference(image=x, mask_prob=y)
        >>> print(c.shape)
        [3, 128, 128]

    """

    @staticmethod
    def inference(image, mask_prob, n_iteration=5):
        """

        Args:
            image (torch.Tensor):       shape -> [C, H, W]
            mask_prob (torch.Tensor):   shape -> [C, H, W]
            n_iteration (int):

        Returns:
            torch.Tensor:   shape -> [C, H, W]

        """
        image = image.permute(dims=(1, 2, 0)).type(torch.uint8).numpy()
        mask_prob = mask_prob.numpy()

        n_channels, height, width = mask_prob.shape

        unary = unary_from_softmax(mask_prob)
        unary = np.ascontiguousarray(unary)

        d = dcrf.DenseCRF2D(width, height, n_channels)

        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.copy(image, order="C"), compat=10)
        q = d.inference(n_iteration)

        return torch.from_numpy(np.array(q)).contiguous().view(size=(-1, height, width)).type(torch.float32)
