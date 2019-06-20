#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy, multiprocessing
import tensorflow as tf
import tensorlayer as tl
from model import get_G
from model_og import get_D
from config import config
from utils.segment_chars import segment_chars

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
shuffle_buffer_size = 128

ni = int(np.sqrt(batch_size))

def train():
    # create folders to save result images and trained model
    save_dir_ginit = "/content/drive/My Drive/tesis_data/samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "/content/drive/My Drive/tesis_data/samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "/content/drive/My Drive/tesis_data/checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    # load dataset
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.jpg', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.jpg', printable=False))
    # valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_lr_imgs = tl.vis.read_images(train_lr_img_list, path=config.TRAIN.lr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    # valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)

    # dataset API and augmentation
    def generator_train():
        for lr, hr in zip(train_lr_imgs, train_hr_imgs):
            yield lr, hr
    def _map_fn_train(lr, hr):
        # hr_patch = tf.image.random_crop(img, [60, 60, 3])
        lr = lr / (255. / 2.)
        lr = lr - 1.

        hr = hr / (255. / 2.)
        hr = hr - 1.
        # hr_patch = tf.image.random_flip_left_right(hr_patch)
        # lr_patch = tf.image.resize(hr_patch, size=[30, 30])
        return lr, hr
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
    train_ds = train_ds.repeat(n_epoch_init + n_epoch)
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=4096)
    train_ds = train_ds.batch(batch_size)
    # value = train_ds.make_one_shot_iterator().get_next()

    # obtain models
    G = get_G((batch_size, None, None, 3)) # (None, 96, 96, 3)
    D = get_D((batch_size * 4, 44, 44, 3)) # (None, 384, 384, 3)
    # VGG = tl.models.vgg19(pretrained=True, end_with='pool4', mode='static')

    print(G)
    print(D)
    # print(VGG)

    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)#.minimize(mse_loss, var_list=g_vars)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)#.minimize(g_loss, var_list=g_vars)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)#.minimize(d_loss, var_list=d_vars)

    G.train()
    D.train()
    # VGG.train()

    # initialize learning (G)
    try:
        G.load_weights(checkpoint_dir + '/g_init_{}.h5'.format(tl.global_flag['mode']))
        print('G initial weights loaded!')
    except FileNotFoundError:
        n_step_epoch = round(n_epoch_init // batch_size)
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            step_time = time.time()
            step += 1
            epoch = step//n_step_epoch
            if epoch > n_epoch_init:
                break
            with tf.GradientTape() as tape:
                fake_hr_patchs = G(lr_patchs)
                mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
            grad = tape.gradient(mse_loss, G.trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {}s, mse: {} ".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
            if (epoch != 0) and (epoch % 10 == 0):
                tl.vis.save_images(fake_hr_patchs.numpy(), [ni, ni], save_dir_gan + '/train_g_init_{}.png'.format(epoch))
                G.save_weights(checkpoint_dir + '/g_init_{}.h5'.format(tl.global_flag['mode']))

    try:
        G.load_weights(checkpoint_dir + '/g_{}.h5'.format(tl.global_flag['mode'])) # in case you want to restore a training
        print('G weights loaded!')
        D.load_weights(checkpoint_dir + '/d_{}.h5'.format(tl.global_flag['mode']))
        print('D weights loaded!')
    except FileNotFoundError:
        print('G or D  weights not found')

    # adversarial learning (G, D)
    n_step_epoch = round(n_epoch // batch_size)
    for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
        step_time = time.time()
        hr_chars = segment_chars(hr_patchs)
        with tf.GradientTape(persistent=True) as tape:
            fake_patchs = G(lr_patchs)
            fake_chars = segment_chars(fake_hr_patchs)
            logits_fake = D(fake_chars)
            logits_real = D(hr_chars)
            # feature_fake = VGG((fake_patchs+1)/2.)
            # feature_real = VGG((hr_patchs+1)/2.)
            d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
            d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
            d_loss = d_loss1 + d_loss2
            g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
            mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)
            # vgg_loss = 2e-6 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
            g_loss = mse_loss + g_gan_loss
        grad = tape.gradient(g_loss, G.trainable_weights)
        g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
        grad = tape.gradient(d_loss, D.trainable_weights)
        del tape
        d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
        step += 1
        epoch = step//n_step_epoch
        print("Epoch: [{}/{}] step: [{}/{}] time: {}s, g_loss(mse:{}, vgg:{}, adv:{}) d_loss: {}".format(
            epoch, n_epoch, step, n_step_epoch, time.time() - step_time, mse_loss, 0, g_gan_loss, d_loss))

        # update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)

        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_patchs.numpy(), [ni, ni], save_dir_gan + '/train_g_1_{}.png'.format(epoch))
            G.save_weights(checkpoint_dir + '/g_{}.h5'.format(tl.global_flag['mode']))
            D.save_weights(checkpoint_dir + '/d_{}.h5'.format(tl.global_flag['mode']))



def evaluate():
    ## create folders to save result images
    save_dir = "/content/drive/My Drive/tesis_data/samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "/content/drive/My Drive/tesis_data/checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)

    ###========================== DEFINE MODEL ============================###
    imid = 64  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    size = valid_lr_img.shape
    G = get_G([1, None, None, 3])
    G.load_weights(checkpoint_dir + '/g_{}.h5'.format(tl.global_flag['mode']))
    G.eval()

    out = G(valid_lr_img).numpy()

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir + '/valid_gen.png')
    tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.png')
    tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr.png')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
