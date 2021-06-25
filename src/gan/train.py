import os.path
from pathlib import Path

import tensorflow as tf

from data.data_generator import BatchGenerator, NaiveDecoder
from gan.ganbase import GAN
from gan.models import make_generator_model, make_discriminator_model, initial_state_model, generator_initialized

EPOCHS = 150
BATCH_SIZE = 8

this_dir = os.path.dirname(os.path.abspath(__file__))

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def generator_optimizer():
    return tf.keras.optimizers.Adam()


def discriminator_optimizer():
    return tf.keras.optimizers.Adam()


def build_gan(verbose=False,
              n_inputs=8,  # predictors
              n_outputs=2,  # u and v components of wind
              n_noise=4,  # random effects
              n_timesteps=24,  # timesteps in a sequence
              low_res_size=(3, 3),
              high_res_size=(1, 1),
              gen_disc_ratio=1):
    generator = make_generator_model(num_timesteps=n_timesteps, in_channels=n_inputs, out_channels=n_outputs,
                                     image_size=low_res_size, noise_channels=n_noise, num_res_blocks=1)
    if verbose:
        print(generator.summary(line_length=200))
    init_model = initial_state_model(in_channels=n_inputs, noise_channels=n_noise)
    generator = generator_initialized(generator, init_model, in_channels=n_inputs, num_timesteps=n_timesteps,
                                      noise_channels=n_noise)
    generator_opt = generator_optimizer()
    discriminator = make_discriminator_model(in_channels=n_inputs, out_channels=n_outputs, num_timesteps=n_timesteps,
                                             high_res_size=high_res_size,
                                             low_res_size=low_res_size)
    discriminator_opt = discriminator_optimizer()
    if verbose:
        print(generator.summary(line_length=200))
        print(discriminator.summary(line_length=200))

    gan = GAN(generator, discriminator, noise_dim=4, initial_state=init_model, gen_disc_ratio=gen_disc_ratio)
    gan.compile(generator_opt, discriminator_opt, generator_loss, discriminator_loss)
    print(f"Generator: {gan.generator.count_params():,} weights")
    print(f"Discriminator: {gan.discriminator.count_params():,} weights")
    print(f"Total: {gan.generator.count_params() + gan.discriminator.count_params():,} weights")
    return gan


if __name__ == '__main__':
    DATA_ROOT = Path(__file__).parent.parent.parent / 'data'
    ERA5_DATA_FOLDER = DATA_ROOT / 'ERA5'
    COSMO1_DATA_FOLDER = DATA_ROOT / 'COSMO1'
    DEM_DATA_FILE = DATA_ROOT / 'dem/Switzerland-90m-DEM.tif'
    PROCESSED_DATA_FOLDER = DATA_ROOT / 'img_prediction_files'

    SEQUENCE_LENGTH = 6
    IMG_SIZE = 30
    BATCH_SIZE = 16

    batches = BatchGenerator(PROCESSED_DATA_FOLDER, NaiveDecoder(), SEQUENCE_LENGTH, IMG_SIZE, BATCH_SIZE)
    gan = build_gan(n_inputs=len(batches.input_variables), n_timesteps=SEQUENCE_LENGTH, gen_disc_ratio=8,
                    low_res_size=(IMG_SIZE, IMG_SIZE), high_res_size=(IMG_SIZE, IMG_SIZE))
    gan.train(batches, 1, 1, verbose=True)
