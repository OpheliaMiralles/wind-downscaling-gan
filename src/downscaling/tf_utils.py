import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow_addons.layers import SpectralNormalization


def img_size(z):
    return z.shape[2]


def channels(z):
    return z.shape[-1]


def shortcut_convolution(high_res_img, low_res_target, nb_channels_out):
    target = img_size(low_res_target) if not isinstance(low_res_target, int) else low_res_target
    if target == 1:
        kernel_size = img_size(high_res_img)
        downsampled_input = kl.TimeDistributed(
            SpectralNormalization(kl.Conv2D(nb_channels_out, kernel_size,
                                            activation=LeakyReLU(0.2))), name='shortcut_conv_1')(high_res_img)
    else:
        strides = int(tf.math.ceil((2 + img_size(high_res_img)) / (target - 1)))
        margin = 2
        padding = int(tf.math.ceil((strides * (target - 1) - img_size(high_res_img)) / 2) + 1 + margin)
        kernel_size = int(strides * (1 - target) + img_size(high_res_img) + 2 * padding)
        downsampled_input = kl.TimeDistributed(kl.ZeroPadding2D(padding=padding))(high_res_img)
        downsampled_input = kl.TimeDistributed(
            SpectralNormalization(kl.Conv2D(nb_channels_out, kernel_size, strides=strides,
                                            activation=LeakyReLU(0.2))), name='shortcut_conv')(downsampled_input)
    downsampled_input = kl.LayerNormalization()(downsampled_input)
    return downsampled_input
