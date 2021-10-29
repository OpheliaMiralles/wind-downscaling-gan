import os
from math import floor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

from data.data_generator import BatchGenerator, NaiveDecoder, LocalFileProvider, S3FileProvider
from data.data_generator import FlexibleNoiseGenerator
from gan import train, metrics
from gan.ganbase import GAN
from gan.models import make_generator, make_discriminator, make_generator_no_noise

DATA_ROOT = Path('./data')
PROCESSED_DATA_FOLDER = DATA_ROOT / 'img_prediction_files'


def plot_prediction(run_id, start_date, end_date, sequence_length=6,
                    img_size=256,
                    batch_size=8,
                    noise_channels=100,
                    validation_split=0.25,
                    cosmoblurred=False,
                    batch_workers=None,
                    data_provider: str = 'local',
                    saving_frequency=10,
                    nb_epochs=500,
                    gen_only=False
                    ):
    TOPO_PREDICTORS = ['tpi_500', 'slope', 'aspect']
    HOMEMADE_PREDICTORS = ['e_plus', 'e_minus', 'w_speed', 'w_angle']
    ERA5_PREDICTORS_SURFACE = ['u10', 'v10', 'blh', 'fsr', 'sp', 'sshf']
    ERA5_PREDICTORS_Z500 = ['z']
    if cosmoblurred:
        ALL_INPUTS = ['U_10M', 'V_10M'] + HOMEMADE_PREDICTORS + TOPO_PREDICTORS
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
    START_DATE = pd.to_datetime(start_date)
    END_DATE = pd.to_datetime(end_date)
    NUM_DAYS = (END_DATE - START_DATE).days + 1
    END_TRAIN = START_DATE + pd.to_timedelta(int(floor((1 - validation_split) * NUM_DAYS)), unit='days')
    START_VAL = END_TRAIN + pd.to_timedelta(1, unit='day')
    NUM_VAL_DAYS = (END_DATE - START_VAL).days + 1
    batch_gen = BatchGenerator(input_provider, output_provider,
                               decoder=NaiveDecoder(normalize=True),
                               sequence_length=sequence_length,
                               patch_length_pixel=img_size, batch_size=batch_size,
                               input_variables=ALL_INPUTS,
                               output_variables=ALL_OUTPUTS,
                               start_date=START_DATE, end_date=END_DATE,
                               num_workers=batch_workers)
    inputs = []
    outputs = []
    print('Creating a validation set for final plots')
    with batch_gen as batch:
        for b in range(NUM_VAL_DAYS):
            print(f'Creating batch {b + 1}/{NUM_VAL_DAYS}')
            x, y = next(batch)
            inputs.append(x)
            outputs.append(y)
    inputs = np.concatenate(inputs)
    outputs = np.concatenate(outputs)
    INPUT_CHANNELS = len(ALL_INPUTS)
    OUT_CHANNELS = len(ALL_OUTPUTS)

    # Plotting some results
    def show(images, dims=1, legends=None):
        fig, axes = plt.subplots(ncols=len(images), figsize=(10, 10))
        for ax, im in zip(axes, images):
            for i in range(dims):
                label = legends[i] if legends is not None else ''
                ax.imshow(im[0, :, :, i], cmap='jet')
                ax.set_title(label)
                ax.axis('off')
        return fig

    if not gen_only:
        # Creating GAN
        generator = make_generator(image_size=img_size, in_channels=INPUT_CHANNELS,
                                   noise_channels=noise_channels, out_channels=OUT_CHANNELS,
                                   n_timesteps=sequence_length)
        print(f"Generator: {generator.count_params():,} weights")
        discriminator = make_discriminator(low_res_size=img_size, high_res_size=img_size,
                                           low_res_channels=INPUT_CHANNELS,
                                           high_res_channels=OUT_CHANNELS, n_timesteps=sequence_length)
        print(f"Discriminator: {discriminator.count_params():,} weights")
        noise_shape = (batch_size, sequence_length, img_size, img_size, noise_channels)
        gan = GAN(generator, discriminator, noise_generator=FlexibleNoiseGenerator(noise_shape, std=1))
        print(f"Total: {gan.generator.count_params() + gan.discriminator.count_params():,} weights")
        gan.compile(generator_optimizer=train.generator_optimizer(),
                    generator_metrics=[metrics.AngularCosineDistance(),
                                       metrics.LogSpectralDistance(),
                                       metrics.WeightedRMSEForExtremes(),
                                       metrics.WindSpeedWeightedRMSE(),
                                       metrics.SpatialKS()],
                    discriminator_optimizer=train.discriminator_optimizer(),
                    discriminator_loss=train.discriminator_loss,
                    metrics=[metrics.discriminator_score_fake(), metrics.discriminator_score_real()])
        # Saving results
        checkpoint_path_weights = Path(
            './checkpoints/gan') / run_id / 'weights-{epoch:02d}.ckpt'
        checkpoint_path_weights.parent.mkdir(exist_ok=True, parents=True)

        for epoch in np.arange(saving_frequency, nb_epochs, saving_frequency):
            gan.load_weights(str(checkpoint_path_weights).format(epoch=epoch))
            noise = FlexibleNoiseGenerator(noise_shape)()
            for i in range(0, batch_size * 6, batch_size):
                results = gan.generator.predict([inputs[i:i + batch_size], noise])
                j = 0
                for inp, out, res in zip(inputs[i:i + batch_size], outputs[i:i + batch_size],
                                         results):
                    plot_path = Path(
                        './plots/gan_pred') / run_id / str(epoch) / f'inp_{i}_batch_{j}.png'
                    plot_path.parent.mkdir(exist_ok=True, parents=True)
                    fig = show([inp, out, res])
                    fig.savefig(plot_path)
                    j += 1

    else:
        generator = make_generator_no_noise(image_size=img_size, in_channels=INPUT_CHANNELS,
                                            out_channels=OUT_CHANNELS,
                                            n_timesteps=sequence_length)
        print(f"Generator: {generator.count_params():,} weights")
        generator.compile(optimizer=train.generator_optimizer(),
                          metrics=[metrics.AngularCosineDistance(),
                                   metrics.LogSpectralDistance(),
                                   metrics.WeightedRMSEForExtremes(),
                                   metrics.WindSpeedWeightedRMSE(),
                                   metrics.SpatialKS()],
                          loss=train.generator_loss)
        # Saving results
        checkpoint_path_weights = Path('./checkpoints/generator') / run_id / 'weights-{epoch:02d}.ckpt'

        for epoch in np.arange(saving_frequency, nb_epochs, saving_frequency):
            generator.load_weights(str(checkpoint_path_weights).format(epoch=epoch))
            for i in range(0, batch_size * 6, batch_size):
                results = generator.predict(inputs[i:i + batch_size])
                j = 0
                for inp, out, res in zip(inputs[i:i + batch_size], outputs[i:i + batch_size],
                                         results):
                    plot_path = Path(
                        './plots/gen_pred') / run_id / str(epoch) / f'inp_{i}_batch_{j}.png'
                    plot_path.parent.mkdir(exist_ok=True, parents=True)
                    fig = show([inp, out, res])
                    fig.savefig(plot_path)
                    j += 1
