import numpy as np


def mask_to_rle(mask):
    """
    Convert a binary mask to RLE format.
    :param mask: numpy array, 1 - mask, 0 - background
    :return: RLE string
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return [int(x) for x in runs]
