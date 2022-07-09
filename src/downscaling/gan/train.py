from typing import Callable

import tensorflow as tf

from downscaling.gan.metrics import wind_speed_weighted_rmse

generator_losses = [wind_speed_weighted_rmse]
scaling_factors = [1.]


def discriminator_loss(real_output, fake_output):
    return -(tf.reduce_mean(real_output) - tf.reduce_mean(fake_output))


def discriminator_adversarial_loss(real_output, fake_output):
    return -(tf.reduce_mean(real_output) - tf.reduce_mean(fake_output))


class reconstruction_loss:
    def __init__(self, feature_extractor: Callable[[tf.Tensor], tf.Tensor], coefficient: float = 1):
        self.feature_extractor = feature_extractor
        self.coefficient = coefficient

    def __call__(self, low_res, high_res):
        delta = self.feature_extractor(low_res) - self.feature_extractor(high_res)
        return self.coefficient * tf.reduce_mean(tf.sqrt(tf.reduce_sum(delta ** 2, axis=-1)))


def generator_loss(real_output, fake_output):
    return tf.reduce_mean([sf * loss(real_output, fake_output) for loss, sf in
                           zip(generator_losses, scaling_factors)])


def generator_optimizer():
    return tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.9, epsilon=0.1)
    # return tf.keras.optimizers.RMSprop(learning_rate=5e-5)


class discriminator_score_real(tf.keras.metrics.Mean):
    def __init__(self, name='d_real', **kwargs):
        super(discriminator_score_real, self).__init__(name=name, **kwargs)
        self.score = self.add_weight(name='real_score', initializer='zeros')

    def update_state(self, real_output, fake_output, sample_weight=None):
        return super(discriminator_score_real, self).update_state(real_output, sample_weight)


class discriminator_score_fake(tf.keras.metrics.Mean):
    def __init__(self, name='d_fake', **kwargs):
        super(discriminator_score_fake, self).__init__(name=name, **kwargs)
        self.score = self.add_weight(name='fake_score', initializer='zeros')

    def update_state(self, real_output, fake_output, sample_weight=None):
        return super(discriminator_score_fake, self).update_state(fake_output, sample_weight)


def discriminator_optimizer():
    return tf.keras.optimizers.Adam(lr=4e-4, beta_1=0.5, beta_2=0.9, epsilon=0.1)
    # return tf.keras.optimizers.RMSprop(learning_rate=5e-5)
