import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import TimeDistributed


class AutoEncoder(Model):
    def __init__(self, nb_channels_in, nb_channels_out, img_size, time_steps):
        super(AutoEncoder, self).__init__()
        interm1, interm2 = np.linspace(nb_channels_out, nb_channels_in, 4, dtype=int)[1:3]
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(time_steps, img_size, img_size, nb_channels_in)),
            TimeDistributed(layers.Conv2D(8 * nb_channels_in, (3, 3), activation=LeakyReLU(0.2), padding='same')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(layers.Conv2D(2 * nb_channels_in, (3, 3), activation=LeakyReLU(0.2), padding='same')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(layers.Conv2D(interm1, (3, 3), activation=LeakyReLU(0.2), padding='same')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(layers.Conv2D(nb_channels_out, (3, 3), activation=LeakyReLU(0.2), padding='same')),
        ], 'encoder')

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(time_steps, img_size, img_size, nb_channels_out)),
            TimeDistributed(layers.Conv2D(interm1, (3, 3), activation=LeakyReLU(0.2), padding='same')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(layers.Conv2D(interm2, (3, 3), activation=LeakyReLU(0.2), padding='same')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(layers.Conv2D(nb_channels_in, (3, 3), activation=LeakyReLU(0.2), padding='same')),
        ], 'decoder')

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class WeightedMeanSquaredError(losses.Loss):
    def __init__(self, weights=None, name='weighted_mean_squared_error'):
        super(WeightedMeanSquaredError, self).__init__(name=name)
        self.weights = tf.convert_to_tensor(weights) if weights is not None else None

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        if self.weights is not None:
            weighted_loss = tf.reduce_sum(self.weights * tf.reduce_mean((y_pred - y_true) ** 2, axis=(1, 2, 3)),
                                          axis=-1)
        else:
            weighted_loss = tf.reduce_mean((y_pred - y_true) ** 2, axis=(1, 2, 3, 4))
        return tf.sqrt(weighted_loss)
