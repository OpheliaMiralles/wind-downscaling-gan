import os.path
from pathlib import Path

import tensorflow as tf

from data.data_generator import BatchGenerator, NaiveDecoder

EPOCHS = 150
BATCH_SIZE = 8

this_dir = os.path.dirname(os.path.abspath(__file__))

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    return -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)
    # real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # total_loss = real_loss + fake_lossÒ
    # return total_loss / 2


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def generator_optimizer():
    return tf.keras.optimizers.Adam()


def discriminator_optimizer():
    return tf.keras.optimizers.Adam(0.1)

