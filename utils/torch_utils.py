from typing import Tuple

import torch as T
import torch.nn as nn
import torch.nn.functional as f


def dilate(
        image: T.Tensor,
        strel: T.Tensor,
        origin: Tuple[int, int] = (0, 0),
        border_value: float = 0,
    ) -> T.Tensor:

    # first pad the image to have correct unfolding; here is where the origins is used
    image_pad = f.pad(
        image,
        [
            origin[0],
            strel.shape[0] - origin[0] - 1,
            origin[1],
            strel.shape[1] - origin[1] - 1,
        ],
        mode='constant',
        value=border_value,
    )

    # Unfold the image to be able to perform operation on neighborhoods
    image_unfold = f.unfold(
        image_pad.unsqueeze(0).unsqueeze(0),
        kernel_size=strel.shape
    )

    # Flatten the structural element since its two dimensions have been flatten when
    # unfolding
    strel_flatten = T.flatten(strel).unsqueeze(0).unsqueeze(-1)

    # Perform the greyscale operation; sum would be replaced by rest if you want erosion
    sums = image_unfold + strel_flatten

    # Take maximum over the neighborhood
    result, _ = sums.max(dim=1)

    # Reshape the image to recover initial shape
    return T.reshape(result, image.shape) - 1


def erode(
        image: T.Tensor,
        strel: T.Tensor,
        origin: Tuple[int, int] = (0, 0),
        border_value: float = 0,
    ) -> T.Tensor:

    # performs dilation on inverted image, and inverts result.
    m = image.max()
    return m - dilate(m - image, strel, origin, border_value)


def strel(size: Tuple[int, int] = (3, 3), shape: str = "rect"):
    if shape == "rect":
        out = T.ones(size)
    else:
        raise Exception(f"Shape not implemented: {shape}")
