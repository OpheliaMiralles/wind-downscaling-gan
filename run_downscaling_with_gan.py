import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from silence_tensorflow import silence_tensorflow

silence_tensorflow()
import tensorflow as tf
import tensorflow.keras.callbacks as cb
import numpy as np
from data.data_generator import BatchGenerator, NaiveDecoder, LocalFileProvider, S3FileProvider
from data.data_generator import FlexibleNoiseGenerator
from gan import train, metrics
from gan.ganbase import GAN
from gan.models import make_generator, make_discriminator

DATA_ROOT = Path(os.getenv('DATA_ROOT', './data'))
PROCESSED_DATA_FOLDER = DATA_ROOT / 'img_prediction_files'


def train_with_all_data(sequence_length=6,
                        img_size=128,
                        batch_size=16,
                        noise_channels=20,
                        cosmoblurred=False,
                        run_id=datetime.today().strftime('%Y%m%d_%H%M'),
                        saving_frequency=10,
                        nb_epochs=500,
                        batch_workers=None,
                        data_provider: str = 'local',
                        eager_batches=False):
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    TOPO_PREDICTORS = []
    HOMEMADE_PREDICTORS = ['e_plus', 'e_minus']
    ERA5_PREDICTORS_SURFACE = ['u10', 'v10'] #, 'blh', 'fsr', 'sp', 'sshf']
    ERA5_PREDICTORS_Z500 = [] #['z']
    if cosmoblurred:
        ALL_INPUTS = ['U_10M', 'V_10M'] + TOPO_PREDICTORS + HOMEMADE_PREDICTORS
        input_pattern = 'x_cosmo_{date}.nc'
        run_id = f'{run_id}_cosmo_blurred'
    else:
        ALL_INPUTS = ERA5_PREDICTORS_Z500 + ERA5_PREDICTORS_SURFACE + TOPO_PREDICTORS + HOMEMADE_PREDICTORS
        input_pattern = 'x_{date}.nc'
    ALL_OUTPUTS = ['U_10M', 'V_10M']
    if batch_workers is None:
        batch_workers = os.cpu_count()
    # Data processing and batch creation
    if data_provider == 'local':
        input_provider = LocalFileProvider(PROCESSED_DATA_FOLDER, input_pattern)
        output_provider = LocalFileProvider(PROCESSED_DATA_FOLDER, 'y_{date}.nc')
    elif data_provider == 's3':
        input_provider = S3FileProvider('wind-downscaling', 'img_prediction_files', pattern=input_pattern)
        output_provider = S3FileProvider('wind-downscaling', 'img_prediction_files', pattern='y_{date}.nc')
    else:
        raise ValueError(f'Wrong value for data provider {data_provider}: please choose between s3 and local')
    AVAIL_DATES = [pd.to_datetime(v) for v in
                   set(input_provider.available_dates).intersection(output_provider.available_dates)]
    START_DATE = min(AVAIL_DATES)
    END_DATE = max(AVAIL_DATES)
    NUM_DAYS = (END_DATE - START_DATE).days + 1
    batch_gen_training = BatchGenerator(input_provider, output_provider,
                                        decoder=NaiveDecoder(normalize=True),
                                        sequence_length=sequence_length,
                                        patch_length_pixel=img_size, batch_size=batch_size,
                                        input_variables=ALL_INPUTS,
                                        output_variables=ALL_OUTPUTS,
                                        start_date=START_DATE, end_date=END_DATE,
                                        num_workers=batch_workers)
    INPUT_CHANNELS = len(ALL_INPUTS)
    OUT_CHANNELS = len(ALL_OUTPUTS)
    if eager_batches:
        inputs = []
        outputs = []
        with batch_gen_training as batch:
            for b in range(NUM_DAYS):
                print(f'Creating batch {b + 1}/{NUM_DAYS}')
                x, y = next(batch)
                inputs.append(x)
                outputs.append(y)
        x = np.concatenate(inputs)
        y = np.concatenate(outputs)
    else:
        x = batch_gen_training
        y = None
    # Creating GAN
    generator = make_generator(image_size=img_size, in_channels=INPUT_CHANNELS,
                               noise_channels=noise_channels, out_channels=OUT_CHANNELS,
                               n_timesteps=sequence_length)
    print(f"Generator: {generator.count_params():,} weights")
    discriminator = make_discriminator(low_res_size=img_size, high_res_size=img_size, low_res_channels=INPUT_CHANNELS,
                                       high_res_channels=OUT_CHANNELS, n_timesteps=sequence_length)
    print(f"Discriminator: {discriminator.count_params():,} weights")
    noise_shape = (batch_size, sequence_length, img_size, img_size, noise_channels)
    gan = GAN(generator, discriminator, noise_generator=FlexibleNoiseGenerator(noise_shape, std=0.01))
    print(f"Total: {gan.generator.count_params() + gan.discriminator.count_params():,} weights")
    gan.compile(generator_optimizer=train.generator_optimizer(),
                generator_metrics=[metrics.AngularCosineDistance(),
                                   metrics.LogSpectralDistance(),
                                   metrics.WeightedRMSEForExtremes(),
                                   metrics.WindSpeedWeightedRMSE()],
                discriminator_optimizer=train.discriminator_optimizer(),
                discriminator_loss=train.discriminator_loss,
                metrics=[metrics.discriminator_score_fake(), metrics.discriminator_score_real()])
    # Saving results
    checkpoint_path_weights = Path(
        './checkpoints/gan') / run_id / 'weights-{epoch:02d}.ckpt'
    checkpoint_path_weights.parent.mkdir(exist_ok=True, parents=True)
    log_path = Path('./logs/gan') / run_id
    log_path.parent.mkdir(exist_ok=True, parents=True)
    callbacks = [
        cb.TensorBoard(log_path, write_images=True, update_freq='batch'),
        cb.ProgbarLogger('steps'),
        cb.TerminateOnNaN(),
        cb.ModelCheckpoint(str(checkpoint_path_weights), save_best_only=False, period=saving_frequency,
                           save_weights_only=True),
    ]
    gan.fit(x=x, y=y, callbacks=callbacks, epochs=nb_epochs,
            workers=batch_workers, use_multiprocessing=True)


if __name__ == '__main__':
    train_with_all_data(cosmoblurred=False, data_provider='s3', eager_batches=True)
