import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


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


def wind_speed_weighted_rmse(real_output, fake_output):
    # Only for cases where we output both wind speed components
    u, v = real_output[..., 0], real_output[..., 1]
    u_hat, v_hat = fake_output[..., 0], fake_output[..., 1]
    estimated_wind_speed = tf.math.sqrt(u_hat ** 2 + v_hat ** 2)
    realized_wind_speed = tf.math.sqrt(u ** 2 + v ** 2)
    epsilon = 4  # See Jerome Dujardin thesis
    t = 0.425  # See Jerome Dujardin thesis
    beta = tf.math.divide(epsilon + realized_wind_speed, epsilon + estimated_wind_speed)
    tau = tf.where(estimated_wind_speed >= realized_wind_speed, t, 1 - t)
    result = tau * ((u_hat - beta * u) ** 2 + (v_hat - beta * v) ** 2)
    ws_weighted_rmse = tf.sqrt(tf.reduce_mean(result, axis=(1, 2, 3)))
    return ws_weighted_rmse


WindSpeedWeightedRMSE = lambda: tfa.metrics.MeanMetricWrapper(wind_speed_weighted_rmse, name='ws_weighted_rmse')


def weighted_extreme_rmse(real_output, fake_output):
    sq = real_output ** 2
    # Weights proportional to extremeness of winds
    weights = tf.math.divide_no_nan(sq, tf.reduce_sum(sq))
    result = weights * (real_output - fake_output) ** 2
    weighted_rmse = tf.sqrt(tf.reduce_sum(result, axis=(1, 2, 3, 4)))
    return weighted_rmse


WeightedRMSEForExtremes = lambda: tfa.metrics.MeanMetricWrapper(weighted_extreme_rmse, name='extreme_rmse')


def log_spectral_distance(real_output, fake_output):
    power_spectra_real = tf.transpose(tf.abs(tf.transpose(tf.signal.rfft2d(real_output), perm=[0,1,4,2,3])) ** 2, perm=[0,1,3,4,2])
    power_spectra_fake = tf.transpose(tf.abs(tf.transpose(tf.signal.rfft2d(fake_output), perm=[0,1,4,2,3])) ** 2, perm=[0,1,3,4,2])
    ratio = tf.math.divide_no_nan(power_spectra_real, power_spectra_fake)

    def log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return tf.math.divide_no_nan(numerator, denominator)

    result = (10 * log10(ratio)) ** 2
    lsd = tf.sqrt(tf.reduce_mean(result, axis=(1, 2, 3, 4)))
    return lsd


LogSpectralDistance = lambda: tfa.metrics.MeanMetricWrapper(log_spectral_distance, name='lsd')
