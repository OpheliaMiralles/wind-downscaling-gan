import tensorflow as tf


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


class LogSpectralDistance(tf.keras.metrics.Metric):
    def __init__(self, name='lsd', **kwargs):
        super(LogSpectralDistance, self).__init__(name=name, **kwargs)
        self.score = self.add_weight(name='score', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, real_output, fake_output, sample_weight=None):
        batch_size = real_output.shape[0]
        if batch_size is None:
            return
        power_spectra_real = tf.abs(tf.signal.fftshift((tf.signal.rfft3d(real_output)))) ** 2 / batch_size
        power_spectra_fake = tf.abs(tf.signal.fftshift((tf.signal.rfft3d(fake_output)))) ** 2 / batch_size
        ratio = tf.math.divide_no_nan(power_spectra_real, power_spectra_fake)

        def log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return tf.math.divide_no_nan(numerator, denominator)

        result = (10 * log10(ratio)) ** 2
        lsd = tf.sqrt(tf.reduce_mean(result, axis=(1, 2, 3)))
        print(lsd)
        self.score.assign_add(tf.reduce_sum(lsd))
        self.count.assign_add(batch_size)

    def result(self):
        return tf.math.divide_no_nan(self.score, self.count)


class WeightedRMSEForExtremes(tf.keras.metrics.Metric):
    def __init__(self, name='extreme_rmse', **kwargs):
        super(WeightedRMSEForExtremes, self).__init__(name=name, **kwargs)
        self.score = self.add_weight(name='score', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, real_output, fake_output, sample_weight=None):
        batch_size = real_output.shape[0]
        if batch_size is None:
            return
        sq = real_output ** 2
        # Weights proportional to extremeness of winds
        weights = tf.math.divide_no_nan(sq, tf.reduce_sum(sq))
        result = weights * (real_output - fake_output) ** 2
        weighted_rmse = tf.sqrt(tf.reduce_sum(result, axis=(1, 2, 3, 4)))

        self.score.assign_add(tf.reduce_sum(weighted_rmse))
        self.count.assign_add(batch_size)

    def result(self):
        return tf.sqrt(tf.math.divide_no_nan(self.score, self.count))

