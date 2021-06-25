from pathlib import Path

import tensorflow as tf
import tensorflow.keras.callbacks as cb
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import TimeDistributed, ConvLSTM2D


class AutoEncoder(Model):
    def __init__(self, nb_channels_in, nb_channels_out, img_size, time_steps):
        super(AutoEncoder, self).__init__()
        interm1, interm2 = np.linspace(nb_channels_out, nb_channels_in, 4, dtype=int)[1:3]
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(time_steps, img_size, img_size, nb_channels_in)),
            TimeDistributed(layers.Conv2D(interm2, (3, 3), activation=LeakyReLU(0.2), padding='same')),
            TimeDistributed(BatchNormalization()),  # Uses moving averages instead of current mean and std
            ConvLSTM2D(filters=interm1, kernel_size=3, strides=1, padding='same', activation='tanh',
                       recurrent_activation='sigmoid', return_sequences=True),
            TimeDistributed(layers.Conv2D(nb_channels_out, (3, 3), activation=LeakyReLU(0.2), padding='same')),
            TimeDistributed(BatchNormalization())
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(time_steps, img_size, img_size, nb_channels_out)),
            ConvLSTM2D(filters=interm1, kernel_size=3, strides=1, padding='same', activation='tanh',
                       recurrent_activation='sigmoid', return_sequences=True),
            TimeDistributed(
                layers.Conv2DTranspose(nb_channels_in, kernel_size=3, activation=LeakyReLU(0.2), padding='same')),
            TimeDistributed(BatchNormalization())
        ])

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


if __name__ == '__main__':
    import numpy as np
    from data.data_generator import BatchGenerator, NaiveDecoder
    import pandas as pd

    ERA5_PREDICTORS_SURFACE = ('u10', 'v10', 'blh', 'fsr', 'sp', 'sshf',
                               'd2m', 't2m',
                               'u100', 'v100')
    ERA5_PREDICTORS_Z500 = ('d', 'z', 'u', 'v', 'w', 'vo')
    TOPO_PREDICTORS = ('tpi_500', 'ridge_index_norm', 'ridge_index_dir',
                       'we_derivative', 'sn_derivative',
                       'slope', 'aspect')
    DATA_ROOT = Path(__file__).parent.parent.parent / 'data'
    ERA5_DATA_FOLDER = DATA_ROOT / 'ERA5'
    COSMO1_DATA_FOLDER = DATA_ROOT / 'COSMO1'
    DEM_DATA_FILE = DATA_ROOT / 'dem/Switzerland-90m-DEM.tif'
    PROCESSED_DATA_FOLDER = DATA_ROOT / 'img_prediction_files'
    START_DATE = pd.to_datetime('2017-09-01')
    END_DATE = pd.to_datetime('2018-04-01')
    nb_batches = len(pd.date_range(START_DATE, END_DATE))  # We want every day we can get
    SEQUENCE_LENGTH = 6
    IMG_SIZE = 64
    BATCH_SIZE = 8

    from data.download_ERA5 import download_ERA5

    download_ERA5(ERA5_DATA_FOLDER, START_DATE, END_DATE)
    print('Done')
    username = 'ophelia'
    password = 'rae69JMK!'
    from data import download_COSMO1

    download_COSMO1(username, password, COSMO1_DATA_FOLDER, START_DATE, END_DATE)
    from data.data_processing import process_imgs

    process_imgs(PROCESSED_DATA_FOLDER, ERA5_DATA_FOLDER, COSMO1_DATA_FOLDER, DEM_DATA_FILE.parent,
                 START_DATE, END_DATE, surface_variables_included=ERA5_PREDICTORS_SURFACE,
                 z500_variables_included=ERA5_PREDICTORS_Z500, topo_variables_included=TOPO_PREDICTORS)
    print('Done')
    from data.data_generator import BatchGenerator, NaiveDecoder

    batch = BatchGenerator(path_to_data=PROCESSED_DATA_FOLDER, decoder=NaiveDecoder(normalize=True),
                           sequence_length=SEQUENCE_LENGTH,
                           patch_length_pixel=IMG_SIZE, batch_size=BATCH_SIZE,
                           input_variables=ERA5_PREDICTORS_SURFACE + ERA5_PREDICTORS_Z500 + TOPO_PREDICTORS,
                           start_date=START_DATE, end_date=END_DATE)
    all_inputs = ERA5_PREDICTORS_SURFACE + ERA5_PREDICTORS_Z500 + TOPO_PREDICTORS
    batches = []
    for b in range(nb_batches):
        print(f'Creating batch {b}/{nb_batches}')
        batches.append(next(batch)[0])
    inputs = np.concatenate(batches, axis=0)

    autoencoder = AutoEncoder(nb_channels_in=len(all_inputs), nb_channels_out=4, time_steps=SEQUENCE_LENGTH,
                              img_size=IMG_SIZE)
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()
    w_u10_v10 = 0.15
    weights = np.array([w_u10_v10, w_u10_v10] + [(1 - 2 * w_u10_v10) / (len(all_inputs) - 2)] * (len(all_inputs) - 2),
                       dtype=np.float32)
    autoencoder.compile('adam', loss=WeightedMeanSquaredError(weights))
    checkpoint_path = DATA_ROOT.parent / 'src/autoencoder' / 'checkpoints/weights.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint_callback = cb.ModelCheckpoint(checkpoint_path, monitor='loss')
    callbacks = [
        cb.TensorBoard(),
        cb.ProgbarLogger('steps'),
        cb.EarlyStopping(min_delta=5e-3, patience=10),
        cb.TerminateOnNaN(),
        checkpoint_callback,
    ]
    history = autoencoder.fit(inputs, inputs, batch_size=BATCH_SIZE, epochs=300, steps_per_epoch=64,
                              callbacks=callbacks,
                              validation_split=0.25)

    # x_bis = autoencoder(x)
    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12.5, 12.5))
    # ax1.imshow(x[0, :, :, :, -1].mean(axis=0))
    # ax2.imshow(np.array(x_bis[0, :, :, :, -1]).mean(axis=0))
