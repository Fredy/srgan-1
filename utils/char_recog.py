import numpy as np

VALID_CHARS = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
CHARS_IDX = { c: i for i, c in enumerate(VALID_CHARS)}

def chars_to_one_hot(chars, is_real, dtype=np.float32):
    """
    Return one hot vector of chars + real/fake label:
    10 digits + 24 letters (excluding I, O) + label
    
    :param chars: String of valid chars
    :param is_real: Label that indicates if this comes from ground truth
        images or generator network
    :param dtype: Dtype of the output array, defaults to np.float32

    :return: Array(chars len, 35)
    """
    nchars = len(chars)
    out = np.zeros((nchars, len(VALID_CHARS) + 1), dtype)
    for i,c in enumerate(chars):
        out[i, CHARS_IDX[c]] = 1
        out[i, -1] = is_real

    return out

