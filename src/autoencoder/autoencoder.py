import shutil
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import TimeDistributed


class AutoEncoder(Sequential):
    def __init__(self, nb_channels_in, nb_channels_out, img_size, time_steps):
        interm1, interm2 = np.linspace(nb_channels_out, nb_channels_in, 4, dtype=int)[1:3]
        self.nb_channels_in = nb_channels_in
        self.nb_channels_out = nb_channels_out
        self.img_size = img_size
        self.time_steps = time_steps
        self.interm1 = interm1
        self.interm2 = interm2
        encoder = self.make_encoder()
        decoder = self.make_decoder()
        super(AutoEncoder, self).__init__([encoder, decoder], name='autoencoder')
        self.encoder = encoder
        self.decoder = decoder

    def make_encoder(self):
        all_inputs = layers.Input(shape=(self.time_steps, self.img_size, self.img_size, self.nb_channels_in),
                                  name='all_inputs')
        x = TimeDistributed(layers.Conv2D(8 * self.nb_channels_in, (3, 3), activation=LeakyReLU(0.2), padding='same'))(
            all_inputs)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(layers.Conv2D(2 * self.nb_channels_in, (3, 3), activation=LeakyReLU(0.2), padding='same'))(
            x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(layers.Conv2D(self.interm1, (3, 3), activation=LeakyReLU(0.2), padding='same'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        out = TimeDistributed(layers.Conv2D(self.nb_channels_out, (3, 3), activation=LeakyReLU(0.2), padding='same'),
                              name='reduced_inputs')(x)
        return Model(inputs=all_inputs, outputs=out, name='encoder')

    def make_decoder(self):
        reduced_inputs = layers.Input(shape=(self.time_steps, self.img_size, self.img_size, self.nb_channels_out),
                                      name='reduced_inputs')
        x = TimeDistributed(layers.Conv2D(self.interm1, (3, 3), activation=LeakyReLU(0.2), padding='same'))(
            reduced_inputs)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(layers.Conv2D(self.interm2, (3, 3), activation=LeakyReLU(0.2), padding='same'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        out = TimeDistributed(layers.Conv2D(self.nb_channels_in, (3, 3), activation=LeakyReLU(0.2), padding='same'),
                              name='predicted_inputs')(x)
        return Model(inputs=reduced_inputs, outputs=out, name='decoder')


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

