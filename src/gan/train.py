import tensorflow as tf

from gan.metrics import angular_cosine_distance, weighted_extreme_rmse, wind_speed_rmse, wind_speed_weighted_rmse


def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)


def generator_loss(real_output, fake_output):
    return tf.reduce_mean([loss(real_output, fake_output) for loss in
                           [angular_cosine_distance, weighted_extreme_rmse, wind_speed_rmse, wind_speed_weighted_rmse]])


def generator_optimizer():
    return tf.keras.optimizers.Adam(lr=1e-4)


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
    return tf.keras.optimizers.Adam(lr=4e-4)
