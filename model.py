import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (Input, Conv2d, BatchNorm, Elementwise,
                                SubpixelConv2d, Flatten, Dense, DeConv2d)
from tensorlayer.models import Model


# NOTE: read images to float32, and normalize it values,
# check if it is necesary to map the values in the range [-1, 1]

def get_G(input_shape):
    # TODO: try stdev and mean
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(stddev=0.02)

    input_layer = Input(input_shape)
    net = Conv2d(n_filter=64, filter_size=(9, 9), strides=(1, 1),
                 act=tf.nn.relu, W_init=w_init, name='first_conv')(input_layer)

    # ======= N residual blocks (N = 16)========
    for i in range(16):
        tmp = Conv2d(64, (3, 3), (1, 1), W_init=w_init, b_init=None)(net)
        tmp = BatchNorm(act=tf.nn.relu, gamma_init=g_init)(tmp)
        tmp = Conv2d(64, (3, 3), (1, 1), W_init=w_init, b_init=None)(tmp)
        tmp = BatchNorm(gamma_init=g_init)(tmp)
        tmp = Elementwise(tf.add)([net, tmp])  # shortcut: sum
        net = tmp
    # ======= N residual blocks ========

    # n_filter: in the original SRGAN n=256, but it uses subpixel convolutions
    # TODO: check if this works with 64 filters
    net = DeConv2d(256, (3, 3), (1, 1), act=tf.nn.relu, W_init=w_init)(net)
    # TODO: check if an activation function is neede here (tanh ?)
    net = DeConv2d(3, (1, 1), (1, 1), act=tf.nn.tanh, W_init=w_init)(net)

    generator = Model(inputs=input_layer, outputs=net, name='generator')
    return generator


def get_D(input_shape):
    # Segmentar caracteres directamente
    # License plate: W x H, N caracteres
    # 6 bloques de caracteres
    # M = 8 capas convolucionales
    # una capa completamente conectada
    # revisar [13](SRGAN): LeakyReLu y batch normalization
    # Sigmoid como clasificador

    # 35 categorías: 10 dígitos + 24 letras (no I ni O) + fake label (1: fake, 0: real)
    # "Los caracteres segmentados son entradas para DN"
    # ------------------------------
    # 1. Segmentar en N bloques
    # 2. M = 8  capas convolucionales
    # 3. capa totalmente conectada
    # 4. Sigmoid
    # Cada uno de los bloques se usa como input!!!

    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1, 0.02)

    input_layer = Input(input_shape)

    # 0
    net = Conv2d(64, (3, 3), (1, 1), act=tl.act.lrelu, W_init=w_init, b_init=None)(input_layer)

    # 1
    net = Conv2d(64, (3, 3), (2, 2), W_init=w_init, b_init=None)(net)
    net = BatchNorm(act=tl.act.lrelu, gamma_init=g_init)(net)

    # 2
    net = Conv2d(128, (3, 3), (1, 1), W_init=w_init, b_init=None)(net)
    net = BatchNorm(act=tl.act.lrelu, gamma_init=g_init)(net)

    # 3
    net = Conv2d(128, (3, 3), (2, 2), W_init=w_init, b_init=None)(net)
    net = BatchNorm(act=tl.act.lrelu, gamma_init=g_init)(net)

    # 4
    net = Conv2d(256, (3, 3), (1, 1), W_init=w_init, b_init=None)(net)
    net = BatchNorm(act=tl.act.lrelu, gamma_init=g_init)(net)

    # 5
    net = Conv2d(256, (3, 3), (2, 2), W_init=w_init, b_init=None)(net)
    net = BatchNorm(act=tl.act.lrelu, gamma_init=g_init)(net)

    # 6
    net = Conv2d(512, (3, 3), (1, 1), W_init=w_init, b_init=None)(net)
    net = BatchNorm(act=tl.act.lrelu, gamma_init=g_init)(net)

    # 7
    net = Conv2d(512, (3, 3), (2, 2), W_init=w_init, b_init=None)(net)
    net = BatchNorm(act=tl.act.lrelu, gamma_init=g_init)(net)

    net = Flatten()(net)
    net = Dense(n_units=1024, act=tl.act.lrelu)(net)
    net = Dense(n_units=35, act=tl.act.lrelu)(net)
    # TODO: Use 2 separate dense layers?
    # one for fake or real prob
    # other for one hot vector
    # Check https://arxiv.org/pdf/1708.05509.pdf

    discriminator = Model(inputs=input_layer, outputs=net, name='discriminator')

    return discriminator
