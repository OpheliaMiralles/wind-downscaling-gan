import tensorflow as tf


def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)


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


def generator_optimizer():
    return tf.keras.optimizers.Adam(lr=1e-4)


def discriminator_optimizer():
    return tf.keras.optimizers.Adam(lr=4e-4)
