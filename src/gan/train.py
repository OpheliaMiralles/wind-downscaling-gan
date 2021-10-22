import tensorflow as tf

from gan.metrics import log_spectral_distance, extreme_weighted_rmse

generator_losses = [log_spectral_distance, extreme_weighted_rmse]
scaling_factors = [1., 10.]


def discriminator_loss(real_output, fake_output):
    return -(tf.reduce_mean(real_output) - tf.reduce_mean(fake_output))


def generator_loss(real_output, fake_output):
    return tf.reduce_mean([sf * loss(real_output, fake_output) for loss, sf in
                           zip(generator_losses, scaling_factors)])


def generator_optimizer():
    return tf.keras.optimizers.Adam(lr=1e-4, beta_1=0, beta_2=0.9, epsilon=0.1)


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
    return tf.keras.optimizers.Adam(lr=4e-4, beta_1=0, beta_2=0.9, epsilon=0.1)
