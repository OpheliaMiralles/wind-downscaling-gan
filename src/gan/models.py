from math import log2

import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow_addons.layers import SpectralNormalization


def make_generator_no_noise(
        image_size: int,
        in_channels: int,
        out_channels: int,
        n_timesteps: int,
        batch_size: int = None,
        feature_channels=128
):
    # Make sure we have nice powers of 2 everywhere
    assert log2(image_size).is_integer()
    assert log2(feature_channels).is_integer()
    total_in_channels = in_channels
    img_shape = (image_size, image_size)
    tshape = (n_timesteps,) + img_shape
    input_image = kl.Input(shape=tshape + (in_channels,), batch_size=batch_size, name='input_image')
    # Concatenate inputs
    x = input_image

    # Add features and decrease image size - in 2 steps
    intermediate_features = total_in_channels * 8 if total_in_channels * 8 <= feature_channels else feature_channels
    x = kl.TimeDistributed(kl.ZeroPadding2D(padding=4))(x)
    x = kl.TimeDistributed(kl.Conv2D(intermediate_features, (9, 9), strides=2, activation=LeakyReLU(0.2)))(x)
    x = kl.BatchNormalization()(x)
    assert tuple(x.shape) == (batch_size, n_timesteps, image_size // 2, image_size // 2, intermediate_features)
    res_2 = x  # Keep residuals for later

    x = kl.TimeDistributed(kl.ZeroPadding2D(padding=2))(x)
    x = kl.TimeDistributed(kl.Conv2D(feature_channels, (5, 5), strides=2, activation=LeakyReLU(0.2)))(x)
    x = kl.BatchNormalization()(x)
    assert tuple(x.shape) == (batch_size, n_timesteps, image_size // 4, image_size // 4, feature_channels)
    res_4 = x  # Keep residuals for later

    # Recurrent unit
    x = kl.ConvLSTM2D(feature_channels, (3, 3), padding='same', return_sequences=True)(x)
    assert tuple(x.shape) == (batch_size, n_timesteps, image_size // 4, image_size // 4, feature_channels)

    # Re-increase image size and decrease features
    x = kl.TimeDistributed(kl.Conv2D(feature_channels // 2, (3, 3), padding='same', activation=LeakyReLU(0.2)))(x)
    x = kl.BatchNormalization()(x)
    assert tuple(x.shape) == (batch_size, n_timesteps, image_size // 4, image_size // 4, feature_channels // 2)

    # Re-introduce residuals from before (skip connection)
    x = kl.Concatenate()([x, res_4])
    x = kl.TimeDistributed(kl.Conv2DTranspose(feature_channels / 4, (2, 2), strides=2, activation=LeakyReLU(0.2)))(
        x)
    x = kl.BatchNormalization()(x)
    assert tuple(x.shape) == (batch_size, n_timesteps, image_size // 2, image_size // 2, feature_channels // 4)

    # Skip connection 2
    x = kl.Concatenate()([x, res_2])
    if feature_channels / 8 >= out_channels:
        x = kl.TimeDistributed(
            kl.Conv2DTranspose(feature_channels // 8, (2, 2), strides=2, activation=LeakyReLU(0.2)))(x)
        assert tuple(x.shape) == (batch_size, n_timesteps, image_size, image_size, feature_channels // 8)
    else:
        x = kl.TimeDistributed(
            kl.Conv2D(out_channels, (3, 3), padding='same', activation=LeakyReLU(0.2)))(x)
        assert tuple(x.shape) == (batch_size, n_timesteps, image_size, image_size, out_channels)
    x = kl.BatchNormalization()(x)
    x = kl.TimeDistributed(kl.Conv2D(out_channels, (3, 3), padding='same', activation='linear'),
                           name='predicted_image')(x)
    assert tuple(x.shape) == (batch_size, n_timesteps, image_size, image_size, out_channels)
    return Model(inputs=input_image, outputs=x, name='generator')


def make_generator(
        image_size: int,
        in_channels: int,
        noise_channels: int,
        out_channels: int,
        n_timesteps: int,
        batch_size: int = None,
        feature_channels=128
):
    # Make sure we have nice powers of 2 everywhere
    assert log2(image_size).is_integer()
    assert log2(feature_channels).is_integer()
    total_in_channels = in_channels + noise_channels
    img_shape = (image_size, image_size)
    tshape = (n_timesteps,) + img_shape
    input_image = kl.Input(shape=tshape + (in_channels,), batch_size=batch_size, name='input_image')
    input_noise = kl.Input(shape=tshape + (noise_channels,), batch_size=batch_size, name='input_noise')

    # Concatenate inputs
    x = kl.Concatenate()([input_image, input_noise])

    # Add features and decrease image size - in 2 steps
    intermediate_features = total_in_channels * 8 if total_in_channels * 8 <= feature_channels else feature_channels
    x = kl.TimeDistributed(kl.ZeroPadding2D(padding=4))(x)
    x = kl.TimeDistributed(kl.Conv2D(intermediate_features, (9, 9), strides=2, activation=LeakyReLU(0.2)))(x)
    x = kl.BatchNormalization()(x)
    assert tuple(x.shape) == (batch_size, n_timesteps, image_size // 2, image_size // 2, intermediate_features)
    res_2 = x  # Keep residuals for later

    x = kl.TimeDistributed(kl.ZeroPadding2D(padding=2))(x)
    x = kl.TimeDistributed(kl.Conv2D(feature_channels, (5, 5), strides=2, activation=LeakyReLU(0.2)))(x)
    x = kl.BatchNormalization()(x)
    assert tuple(x.shape) == (batch_size, n_timesteps, image_size // 4, image_size // 4, feature_channels)
    res_4 = x  # Keep residuals for later

    # Recurrent unit
    x = kl.ConvLSTM2D(feature_channels, (3, 3), padding='same', return_sequences=True)(x)
    assert tuple(x.shape) == (batch_size, n_timesteps, image_size // 4, image_size // 4, feature_channels)

    # Re-increase image size and decrease features
    x = kl.TimeDistributed(kl.Conv2D(feature_channels // 2, (3, 3), padding='same', activation=LeakyReLU(0.2)))(x)
    x = kl.BatchNormalization()(x)
    assert tuple(x.shape) == (batch_size, n_timesteps, image_size // 4, image_size // 4, feature_channels // 2)

    # Re-introduce residuals from before (skip connection)
    x = kl.Concatenate()([x, res_4])
    x = kl.TimeDistributed(kl.Conv2DTranspose(feature_channels / 4, (2, 2), strides=2, activation=LeakyReLU(0.2)))(
        x)
    x = kl.BatchNormalization()(x)
    assert tuple(x.shape) == (batch_size, n_timesteps, image_size // 2, image_size // 2, feature_channels // 4)

    # Skip connection 2
    x = kl.Concatenate()([x, res_2])
    if feature_channels / 8 >= out_channels:
        x = kl.TimeDistributed(
            kl.Conv2DTranspose(feature_channels // 8, (2, 2), strides=2, activation=LeakyReLU(0.2)))(x)
        assert tuple(x.shape) == (batch_size, n_timesteps, image_size, image_size, feature_channels // 8)
    else:
        x = kl.TimeDistributed(
            kl.Conv2D(out_channels, (3, 3), padding='same', activation=LeakyReLU(0.2)))(x)
        assert tuple(x.shape) == (batch_size, n_timesteps, image_size, image_size, out_channels)
    x = kl.BatchNormalization()(x)
    x = kl.TimeDistributed(kl.Conv2D(out_channels, (3, 3), padding='same', activation='linear'),
                           name='predicted_image')(x)
    assert tuple(x.shape) == (batch_size, n_timesteps, image_size, image_size, out_channels)
    return Model(inputs=[input_image, input_noise], outputs=x, name='generator')


def make_discriminator(
        low_res_size: int,
        high_res_size: int,
        low_res_channels: int,
        high_res_channels: int,
        n_timesteps: int,
        batch_size: int = None,
        feature_channels: int = 16
):
    assert log2(low_res_size).is_integer()
    assert log2(high_res_size).is_integer()

    low_res = kl.Input(shape=(n_timesteps, low_res_size, low_res_size, low_res_channels), batch_size=batch_size,
                       name='low_resolution_image')
    high_res = kl.Input(shape=(n_timesteps, high_res_size, high_res_size, high_res_channels), batch_size=batch_size,
                        name='high_resolution_image')
    if tuple(low_res.shape)[:-1] != tuple(high_res.shape)[:-1]:
        raise NotImplementedError("The discriminator assumes that the low res and high res images have the same size."
                                  "Perhaps you should upsample your low res image first?")
    # First branch: high res only
    hr = kl.ConvLSTM2D(high_res_channels, (3, 3), padding='same', return_sequences=True)(high_res)
    hr = kl.TimeDistributed(
        SpectralNormalization(kl.Conv2D(feature_channels, (3, 3), padding='same', activation=LeakyReLU(0.2))))(hr)

    # Second branch: Mix both inputs
    mix = kl.Concatenate()([low_res, high_res])
    mix = kl.ConvLSTM2D(feature_channels, (3, 3), padding='same', return_sequences=True)(mix)
    mix = kl.TimeDistributed(
        SpectralNormalization(kl.Conv2D(feature_channels, (3, 3), padding='same', activation=LeakyReLU(0.2))))(mix)

    # Merge everything together
    x = kl.Concatenate()([hr, mix])
    assert tuple(x.shape) == (batch_size, n_timesteps, low_res_size, low_res_size, 2 * feature_channels)

    def img_size(z):
        return z.shape[2]

    def channels(z):
        return z.shape[-1]

    while img_size(x) >= 4:
        x = kl.TimeDistributed(kl.ZeroPadding2D())(x)
        x = kl.TimeDistributed(
            SpectralNormalization(kl.Conv2D(channels(x) * 2, (7, 7), strides=3, activation=LeakyReLU(0.2))))(x)

    while img_size(x) > 2:
        x = kl.TimeDistributed(
            SpectralNormalization(kl.Conv2D(channels(x) * 2, (3, 3), strides=2, activation=LeakyReLU(0.2))))(x)
    x = kl.TimeDistributed(kl.Flatten())(x)
    print(x.shape)
    assert tuple(x.shape)[:-1] == (batch_size, n_timesteps)  # Unknown number of channels
    x = kl.TimeDistributed(kl.Dense(1, activation='linear'))(x)
    x = kl.GlobalAveragePooling1D(name='score')(x)

    return Model(inputs=[low_res, high_res], outputs=x, name='discriminator')
