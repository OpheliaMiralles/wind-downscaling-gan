import os.path
from itertools import islice

import numpy as np
import pandas as pd
import tensorflow as tf

from data_generator import BatchGenerator, NaiveDecoder
from ganbase import GAN
from models import make_generator_model, make_discriminator_model, initial_state_model, generator_initialized

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
    return tf.keras.optimizers.Adam(1e-4)


def discriminator_optimizer():
    return tf.keras.optimizers.Adam(1e-4)


def build_gan(verbose=False):
    n_inputs = 8  # predictors
    n_outputs = 2  # u and v components of wind
    n_noise = 4  # random effects
    n_timesteps = 24  # timesteps in a sequence
    low_res_size = (3, 3)
    high_res_size = (1, 1)
    generator = make_generator_model(num_timesteps=n_timesteps, in_channels=n_inputs, out_channels=n_outputs,
                                     image_size=low_res_size,
                                     noise_channels=n_noise)
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

    gan = GAN(generator, discriminator, noise_dim=4, initial_state=init_model)
    gan.compile(generator_opt, discriminator_opt, generator_loss, discriminator_loss)
    return gan


def retrieve_data(num_days=None):
    data_path = '/Users/Boubou/Documents/GitHub/WindDownscaling_EPFL_UNIBE/data/point_prediction_files/train'
    start_train_period = pd.to_datetime('2016-01-10')
    end_train_period = pd.to_datetime('2019-12-31')
    x_train = []
    y_train = []
    date_range = islice(pd.date_range(start_train_period, end_train_period), num_days)
    for d in date_range:
        d_str = d.strftime('%Y%m%d')
        x = f'{data_path}/x_train_{d_str}.npy'
        y = f'{data_path}/y_train_{d_str}.npy'
        x_train.append(np.load(x))
        y_train.append(np.load(y))
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train


def run_train(gan, epochs, batches_per_epoch=None):
    x_train, y_train = retrieve_data()
    train_dataset = BatchGenerator(x_train, y_train, NaiveDecoder(),
                                   batch_size=BATCH_SIZE)
    gan.train(train_dataset, epochs, batches_per_epoch)
    return gan


if __name__ == '__main__':
    gan = build_gan()
    run_train(gan, 3, 3)
