from typing import List
import numpy as np
from matplotlib import pyplot as plt


def mask_to_rle(mask: np.ndarray):
    """
    Convert a binary mask to RLE format.
    :param mask: numpy array, 1 - mask, 0 - background
    :return: RLE array
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return [int(x) for x in runs]


def show_in_grid(
    images: List[np.ndarray],
    num_rows,
    num_cols,
    show_plot=False,
    savefig_path=None,
    size_factor=10,
    x_labels=[],
):
    plt.ioff()
    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * size_factor, num_rows * size_factor)
    )
    fig.tight_layout()

    for i, img in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        if num_rows > 1:
            ax = axs[row, col]
        else:
            ax = axs[i]
        ax.axis("off")
        ax.imshow(img)

        if row == num_rows - 1:
            if len(x_labels) > 0:
                ax.set_title(x_labels[col])

    if show_plot:
        plt.show()
    else:
        fig.savefig(savefig_path)
        plt.close()
