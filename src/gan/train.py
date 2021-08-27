import tensorflow as tf


def discriminator_loss(real_output, fake_output):
    return -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)


def generator_optimizer():
    return tf.keras.optimizers.Adam(lr=0.0001)


def discriminator_optimizer():
    return tf.keras.optimizers.Adam(lr=0.0001)
