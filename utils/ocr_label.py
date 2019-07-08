import tensorlayer as tl
import numpy as np
import os
import itertools
from pytesseract import image_to_string
from multiprocessing import Pool

LABEL_PATH = '/home/fredy/Desktop/license_plates/train_Dataset/label/'

def parallel_this(tuple_):
    img, name = tuple_

    h, w, _ = img.shape
    img = 255 - tl.prepro.imresize(img, (h*5, w*5))
    text = image_to_string(img)

    return (name, text)


def ocr_label(path):
    img_list = sorted(tl.files.load_file_list(
        path=path, regx='.*.jpg', printable=False))
    imgs = tl.vis.read_images(img_list, path=path, n_threads=32)

    with Pool() as pool:
        labels = pool.map(parallel_this, zip(imgs, img_list))
    
    return labels
