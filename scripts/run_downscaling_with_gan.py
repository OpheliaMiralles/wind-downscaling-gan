from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.callbacks as cb

from autoencoder.autoencoder import AutoEncoder
from data.data_generator import BatchGenerator, NaiveDecoder
from data.data_generator import FlexibleNoiseGenerator
from gan import train, metrics
from gan.ganbase import GAN
from gan.models import make_generator, make_discriminator

DATA_ROOT = Path('./data')
ERA5_DATA_FOLDER = DATA_ROOT / 'ERA5'
COSMO1_DATA_FOLDER = DATA_ROOT / 'COSMO1'
DEM_DATA_FILE = DATA_ROOT / 'dem/Switzerland-90m-DEM.tif'
PROCESSED_DATA_FOLDER = DATA_ROOT / 'img_prediction_files'

DATA_ROOT.mkdir(parents=True, exist_ok=True)
ERA5_DATA_FOLDER.mkdir(exist_ok=True)
COSMO1_DATA_FOLDER.mkdir(exist_ok=True)
DEM_DATA_FILE.parent.mkdir(exist_ok=True)
PROCESSED_DATA_FOLDER.mkdir(exist_ok=True)


def train_with_all_data(sequence_length=6,
                        img_size=256,
                        batch_size=8,
                        noise_channels=100,
                        cosmoblurred=False,
                        run_id=datetime.today().strftime('%Y%m%d_%H%M'),
                        use_autoencoder=False,
                        saving_frequency=10,
                        nb_epochs=500):
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    TOPO_PREDICTORS = ['tpi_500', 'ridge_index_norm', 'ridge_index_dir',
                       'we_derivative', 'sn_derivative',
                       'slope', 'aspect']
    HOMEMADE_PREDICTORS = ['e_plus', 'e_minus']
    ERA5_PREDICTORS_SURFACE = ['u10', 'v10', 'blh', 'fsr', 'sp', 'sshf']
    ERA5_PREDICTORS_Z500 = ['z']
    if cosmoblurred:
        ALL_INPUTS = ['U_10M', 'V_10M'] + HOMEMADE_PREDICTORS + TOPO_PREDICTORS
        input_pattern = 'x_cosmo_{date}'
        run_id = f'{run_id}_cosmo_blurred'
    else:
        ALL_INPUTS = ERA5_PREDICTORS_Z500 + ERA5_PREDICTORS_SURFACE + TOPO_PREDICTORS + HOMEMADE_PREDICTORS
        input_pattern = 'x_{date}'
    ALL_OUTPUTS = ['U_10M', 'V_10M']
    AVAIL_DATES = [pd.to_datetime(v.stem.split('_')[-1]) for v in PROCESSED_DATA_FOLDER.glob('x_cosmo_*')]
    START_DATE = min(AVAIL_DATES)
    END_DATE = max(AVAIL_DATES)
    NUM_DAYS = (END_DATE - START_DATE).days + 1
    BATCH_WORKERS = 8
    AUTOENCODER_OUTPUT_FEATURES = 8
    # Data processing and batch creation
    batch_gen = BatchGenerator(path_to_data=PROCESSED_DATA_FOLDER, decoder=NaiveDecoder(normalize=True),
                               sequence_length=sequence_length,
                               input_pattern=input_pattern,
                               patch_length_pixel=img_size, batch_size=batch_size,
                               input_variables=ALL_INPUTS,
                               output_variables=ALL_OUTPUTS,
                               start_date=START_DATE, end_date=END_DATE,
                               num_workers=BATCH_WORKERS)

    inputs = []
    outputs = []
    with batch_gen as batch:
        for b in range(NUM_DAYS):
            print(f'Creating batch {b + 1}/{NUM_DAYS}')
            x, y = next(batch)
            inputs.append(x)
            outputs.append(y)
    inputs = np.concatenate(inputs, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    print(f"Inputs: {inputs.shape}")
    print(f"Outputs: {outputs.shape}")
    INPUT_CHANNELS = len(ALL_INPUTS)
    OUT_CHANNELS = len(ALL_OUTPUTS)
    if use_autoencoder:
        checkpoint_path_weights = Path('./checkpoints/autoencoder/weights.ckpt')
        if not checkpoint_path_weights.exists():
            print("No autoencoder weights found!")
        else:
            autoencoder = AutoEncoder(nb_channels_in=len(ALL_INPUTS), nb_channels_out=AUTOENCODER_OUTPUT_FEATURES,
                                      time_steps=sequence_length, img_size=img_size)
            autoencoder.load_weights(checkpoint_path_weights)

            print("Reducing data dimension")
            inputs = autoencoder.encoder.predict(inputs)
            INPUT_CHANNELS = AUTOENCODER_OUTPUT_FEATURES
    # Creating GAN
    generator = make_generator(image_size=img_size, in_channels=INPUT_CHANNELS,
                               noise_channels=noise_channels, out_channels=OUT_CHANNELS,
                               n_timesteps=sequence_length)
    print(f"Generator: {generator.count_params():,} weights")
    discriminator = make_discriminator(low_res_size=img_size, high_res_size=img_size, low_res_channels=INPUT_CHANNELS,
                                       high_res_channels=OUT_CHANNELS, n_timesteps=sequence_length)
    print(f"Discriminator: {discriminator.count_params():,} weights")
    noise_shape = (batch_size, sequence_length, img_size, img_size, noise_channels)
    gan = GAN(generator, discriminator, noise_generator=FlexibleNoiseGenerator(noise_shape, std=1))
    print(f"Total: {gan.generator.count_params() + gan.discriminator.count_params():,} weights")
    gan.compile(generator_optimizer=train.generator_optimizer(),
                generator_metrics=[tf.keras.metrics.RootMeanSquaredError(), metrics.LogSpectralDistance(),
                                   metrics.WeightedRMSEForExtremes(), metrics.WindSpeedWeightedRMSE()],
                discriminator_optimizer=train.discriminator_optimizer(),
                discriminator_loss=train.discriminator_loss,
                metrics=[metrics.discriminator_score_fake(), metrics.discriminator_score_real()])
    # Saving results
    checkpoint_path_weights = Path('./checkpoints/gan') / run_id / 'weights-{epoch:02d}.ckpt'
    checkpoint_path_weights.parent.mkdir(exist_ok=True, parents=True)
    log_path = Path('./logs/gan') / run_id
    log_path.parent.mkdir(exist_ok=True, parents=True)
    callbacks = [
        cb.TensorBoard(log_path, write_images=True, update_freq='batch', profile_batch=(2, 4)),
        cb.ProgbarLogger('steps'),
        cb.TerminateOnNaN(),
        cb.ModelCheckpoint(str(checkpoint_path_weights), save_best_only=False, period=saving_frequency,
                           save_weights_only=True),
    ]
    gan.fit(inputs, outputs, callbacks=callbacks, epochs=nb_epochs, batch_size=batch_size, validation_split=0.25)
