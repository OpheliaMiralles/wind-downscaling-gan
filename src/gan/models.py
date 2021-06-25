import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import TimeDistributed, Lambda
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import ConvLSTM2D

from gan.blocks import res_block
from gan.layers import ReflectionPadding2D
from gan.rnn import ConvGRU, ConvGate, PaddingGate


def make_generator_model(in_channels=1, out_channels=1, num_timesteps=8, num_res_blocks=3, image_size=(None, None),
                         noise_channels=None):
    initial_state = Input(shape=(*image_size, 256))
    noise = Input(shape=(num_timesteps, *image_size, noise_channels), name="noise")
    low_res_image = Input(shape=(num_timesteps, *image_size, in_channels), name="low_res_image")
    inputs = [low_res_image, initial_state, noise]

    xt = TimeDistributed(ReflectionPadding2D(padding=(1, 1)))(low_res_image)
    xt = TimeDistributed(Conv2D(256 - noise.shape[-1], kernel_size=(3, 3)))(xt)
    xt = Concatenate()([xt, noise])
    for i in range(num_res_blocks):
        xt = res_block(256, time_dist=True, activation='relu')(xt)

    x = ConvGRU(
        initial_state.shape[1:],
        update_gate=ConvGate(),
        reset_gate=ConvGate(),
        output_gate=ConvGate(activation='linear'),
        return_sequences=True,
    )(xt, initial_state)

    block_channels = [256, 256, 128, 64, 32]
    for (i, channels) in enumerate(block_channels):
        if i > 0:
            x = TimeDistributed(UpSampling2D(interpolation='bilinear'))(x)
        x = res_block(channels, time_dist=True, activation='leakyrelu')(x)

    x = TimeDistributed(ReflectionPadding2D(padding=(1, 1)))(x)
    img_out = TimeDistributed(Conv2D(out_channels, kernel_size=(3, 3), activation='sigmoid'))(x)

    # img_out = TimeDistributed(MaxPool2D(48))(img_out)

    return Model(inputs=inputs, outputs=img_out, name='generator')


def make_discriminator_model(in_channels=1, out_channels=1, num_timesteps=8, high_res_size=(None, None),
                             low_res_size=(None, None)):
    high_res = Input(shape=(num_timesteps, *high_res_size, out_channels), name="high resolution image")
    low_res = Input(shape=(num_timesteps, *low_res_size, in_channels), name="inputs")

    x_hr = high_res
    x_lr = low_res

    # for _ in range(7):
    #     x_hr = TimeDistributed(UpSampling2D(size=(2, 2)))(x_hr)

    # x_lr = TimeDistributed(UpSampling2D())(x_lr)
    # x_lr = TimeDistributed(ReflectionPadding2D())(x_lr)

    block_channels = [32, 64, 128, 256]
    for (i, channels) in enumerate(block_channels):
        # x_hr = res_block(channels, time_dist=True, norm="spectral", stride=2)(x_hr)
        x_hr = res_block(channels, time_dist=True, norm="spectral")(x_hr)
        x_lr = res_block(channels, time_dist=True, norm="spectral")(x_lr)

    # x_lr = TimeDistributed(MaxPool2D(3))(x_lr)

    x_joint = Concatenate()([x_lr, x_hr])
    x_joint = res_block(256, time_dist=True, norm="spectral")(x_joint)
    x_joint = res_block(256, time_dist=True, norm="spectral")(x_joint)

    x_hr = res_block(256, time_dist=True, norm="spectral")(x_hr)
    x_hr = res_block(256, time_dist=True, norm="spectral")(x_hr)

    h = Lambda(lambda x: tf.zeros_like(x[:, 0, ...]))
    x_joint = ConvGRU(
        x_joint.shape[2:],
        update_gate=PaddingGate(),
        reset_gate=PaddingGate(),
        output_gate=PaddingGate(activation='linear'),
        return_sequences=True,
    )([x_joint, h(x_joint)])
    x_hr = ConvGRU(
        x_hr.shape[2:],
        update_gate=PaddingGate(),
        reset_gate=PaddingGate(),
        output_gate=PaddingGate(activation='linear'),
        return_sequences=True,
    )([x_hr, h(x_hr)])

    x_avg_joint = TimeDistributed(GlobalAveragePooling2D())(x_joint)
    x_avg_hr = TimeDistributed(GlobalAveragePooling2D())(x_hr)

    x = Concatenate()([x_avg_joint, x_avg_hr])
    # x = TimeDistributed(SNDense(256))(x)
    x = TimeDistributed(Dense(256))(x)
    x = LeakyReLU(0.2)(x)

    # disc_out = TimeDistributed(SNDense(1))(x)
    disc_out = TimeDistributed(Dense(1))(x)

    disc = Model(inputs=[low_res, high_res], outputs=disc_out, name='discriminator')
    return disc


def initial_state_model(num_preproc=3, in_channels=1, noise_channels=None):
    initial_frame_in = Input(shape=(None, None, in_channels))
    noise_initial = Input(shape=(None, None, noise_channels), name="noise_initial")

    h = ReflectionPadding2D(padding=(1, 1))(initial_frame_in)
    h = Conv2D(256 - noise_initial.shape[-1], kernel_size=(3, 3))(h)
    h = Concatenate()([h, noise_initial])
    for i in range(num_preproc):
        h = res_block(256, activation='relu')(h)

    return Model(
        inputs=[initial_frame_in, noise_initial],
        outputs=h
    )


def generator_initialized(gen, init_model,
                          in_channels=1, num_timesteps=8, noise_channels=None):
    noise_initial = Input(shape=(None, None, noise_channels),
                          name="noise_init")
    noise_update = Input(shape=(num_timesteps, None, None, noise_channels),
                         name="noise_update")
    low_res = Input(shape=(num_timesteps, None, None, in_channels),
                    name="low_res_image")
    inputs = [low_res, noise_initial, noise_update]

    initial_state = init_model([low_res[:, 0, ...], noise_initial])
    img_out = gen([low_res, initial_state, noise_update])

    model = Model(inputs=inputs, outputs=img_out, name='generator_initialized')

    return model
