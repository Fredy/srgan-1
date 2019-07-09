import tensorflow as tf
import tensorlayer as tl
import numpy as np
from skimage.measure import compare_ssim

def psnr():
    generated_path = '/home/fredy/code/projects/tesis/pr/test_res/'
    real_path = '/home/fredy/Desktop/license_plates/test_Dataset_old_or/test_result/source/'

    img_list = sorted(tl.files.load_file_list(path=generated_path, regx='.*.jpg', printable=False))

    generated_imgs = tl.vis.read_images(img_list, path=generated_path, n_threads=32)
    real_imgs = tl.vis.read_images(img_list, path=real_path, n_threads=32)
    generated_imgs = np.array(generated_imgs, dtype=np.uint8)
    real_imgs = np.array(real_imgs, dtype=np.uint8)

    return tf.image.psnr(generated_imgs, real_imgs, 255)

def ssim():
    generated_path = '/home/fredy/code/projects/tesis/pr/test_res/'
    real_path = '/home/fredy/Desktop/license_plates/test_Dataset_old_or/test_result/source/'

    img_list = sorted(tl.files.load_file_list(path=generated_path, regx='.*.jpg', printable=False))

    generated_imgs = tl.vis.read_images(img_list, path=generated_path, n_threads=32)
    real_imgs = tl.vis.read_images(img_list, path=real_path, n_threads=32)

    tmp = []
    for i,j in zip(generated_imgs, real_imgs):
        tmp.append(compare_ssim(i,j, multichannel=True))

    return np.array(tmp)


