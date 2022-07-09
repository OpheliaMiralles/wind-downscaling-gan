import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras import losses
from tensorflow.keras.models import Model, Sequential
from tensorflow_addons.layers import SpectralNormalization

from downscaling.gan.metrics import opposite_cosine_similarity
from downscaling.tf_utils import img_size, channels


class AutoEncoder(Sequential):
    def __init__(self, img_size, time_steps, latent_dimension, batch_size):
        self.latent_dimension = latent_dimension
        self.img_size = img_size
        self.time_steps = time_steps
        self.batch_size = batch_size
        encoder = self.make_encoder()
        decoder = self.make_decoder()
        super(AutoEncoder, self).__init__([encoder, decoder], name='autoencoder')
        self.encoder = encoder
        self.decoder = decoder

    def make_encoder(self):
        input_layer = kl.Input(shape=(self.time_steps, self.img_size, self.img_size, 2), name='all_inputs')
        x = input_layer
        while img_size(x) >= 7:
            x = kl.TimeDistributed(kl.ZeroPadding2D())(x)
            x = kl.TimeDistributed(
                SpectralNormalization(kl.Conv2D(channels(x) * 2, (5, 5), strides=3, activation=kl.LeakyReLU(0.2))), name=f'conv_{img_size(x)}')(x)
            x = kl.LayerNormalization()(x)
        x = kl.TimeDistributed(kl.Flatten())(x)
        if x.shape[-1] > 2 * self.latent_dimension:
            middle = (x.shape[-1] + self.latent_dimension) // 2
            x = kl.TimeDistributed(kl.Dense(middle))(x)
        out = kl.TimeDistributed(kl.Dense(self.latent_dimension, activation='linear', name='reduced_inputs'))(x)
        return Model(inputs=input_layer, outputs=out, name='encoder')

    def make_decoder(self):
        reduced_inputs = kl.Input(shape=(self.time_steps, self.latent_dimension), name='reduced_inputs')
        x = kl.TimeDistributed(kl.Dense(self.latent_dimension * 6, activation='linear'))(reduced_inputs)
        x = kl.TimeDistributed(kl.Dense(self.latent_dimension * 12, activation='linear'))(x)
        x = tf.reshape(x, [-1, self.time_steps, 6, 6, self.latent_dimension // 3])
        while img_size(x) < self.img_size // 2:
            x = kl.TimeDistributed(kl.UpSampling2D(size=(2, 2), interpolation='bilinear'))(x)
            new_channels = channels(x) // 2 if channels(x) >= 4 else 2
            x = kl.TimeDistributed(
                kl.Conv2DTranspose(new_channels, (5, 5), padding='same', activation=kl.LeakyReLU(0.2)))(x)
            x = kl.BatchNormalization()(x)
        new_channels = channels(x) // 2 if channels(x) >= 4 else 2
        x = kl.TimeDistributed(kl.Conv2DTranspose(new_channels, (2, 2), strides=2, activation=kl.LeakyReLU(0.2)))(x)
        out = kl.TimeDistributed(kl.Conv2D(2, (3, 3), padding='same', activation='linear'), name='predicted_inputs')(x)
        return Model(inputs=reduced_inputs, outputs=out, name='decoder')


class WeightedVectorLoss(losses.Loss):
    def __init__(self, weights=None, name='weighted_vector_loss'):
        super(WeightedVectorLoss, self).__init__(name=name)
        self.weights = tf.convert_to_tensor(weights) if weights is not None else tf.constant([0.5, 0.5])

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        rmse = tf.sqrt(tf.reduce_sum(tf.reduce_mean((y_pred - y_true) ** 2, axis=(1, 2, 3)), axis=-1))
        ocs = opposite_cosine_similarity(y_true, y_pred)
        return tf.reduce_sum(tf.stack([rmse, ocs], axis=-1) * self.weights, axis=-1)
