import numpy as np

CHARS = 4  # Number of chars in the plate
H_START = 8
H_END = 44 + H_START
W_START = 85
CHAR_WIDTH = 23


def squarify(img, normalize=True):
    """Squarify an 23x44 img to 44x44"""

    filling = np.ones((44, 21, 3), dtype=img.dtype) * 255
    if normalize:
        filling /= (255/2)
        filling -= 1
    out = np.concatenate((img, filling), axis=1)
    return out


def segment_chars(batch, normalize=True):
    """
    Segment chars of a 60x180 plate

    :param batch: Array like with shape (B, H, W, 3)
        B: batch size
        W: image width
        H: image height
    :param normalize: If the images are normalized in [-1,1]
    :return: Array like with shape (B*N, 44,44,3)
        N: number of chars
    """
    chars = []
    for img in batch:
        for j in range(CHARS):
            w_start = W_START + (j * CHAR_WIDTH)
            w_end = W_START + ((j + 1) * CHAR_WIDTH)
            block = img[H_START:H_END, w_start:w_end]
            block = squarify(block, normalize)
            chars.append(block)

    return np.array(chars)