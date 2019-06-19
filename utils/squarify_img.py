import tensorlayer as tl
import numpy as np
import os
import itertools
from multiprocessing import Pool

ORIGINAL_PATH = '/home/fredy/Desktop/license_plates/'
SQUARIFY_PATH = '/home/fredy/Desktop/license_plates_squares/'
INNER_PATHS = [
    'train_Dataset/label/',
    'train_Dataset/train/',
    'test_Dataset/test_result/source',
    'test_Dataset/test_result/query',
    'test_Dataset/test/source',
    'test_Dataset/test/query',
]

IN_PATHS = [os.path.join(ORIGINAL_PATH, i) for i in INNER_PATHS]
OUT_PATHS = [os.path.join(SQUARIFY_PATH, i) for i in INNER_PATHS]


def create_dirs():
    for i in OUT_PATHS:
        os.makedirs(i, exist_ok=True)


def parallel_this(tuple_):
    img, name, out_path = tuple_
    filling = np.ones((20, 100, 3), dtype=np.int8) * 255
    out = np.concatenate((filling, img, filling))
    tl.vis.save_image(out, os.path.join(out_path, name))


def squarify_imgs(in_path, out_path):
    # NOTE: this only works for images with this size: 3X * X
    img_list = sorted(tl.files.load_file_list(
        path=in_path, regx='.*.jpg', printable=False))
    imgs = tl.vis.read_images(img_list, path=in_path, n_threads=32)

    imgs = [i[:, 80:] for i in imgs]

    with Pool() as pool:
        pool.map(parallel_this, zip(imgs, img_list, itertools.repeat(out_path)))


if __name__ == "__main__":
    create_dirs()
    for inp, outp in zip(IN_PATHS, OUT_PATHS):
        squarify_imgs(inp, outp)
