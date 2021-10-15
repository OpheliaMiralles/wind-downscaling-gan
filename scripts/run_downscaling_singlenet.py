import os
from datetime import datetime
from math import floor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from silence_tensorflow import silence_tensorflow

silence_tensorflow()
import tensorflow as tf
import tensorflow.keras.callbacks as cb

from data.data_generator import BatchGenerator, NaiveDecoder
from gan import train, metrics
from gan.models import make_generator_no_noise

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
                        validation_split=0.25,
                        cosmoblurred=False,
                        run_id=datetime.today().strftime('%Y%m%d_%H%M'),
                        saving_frequency=10,
                        nb_epochs=500,
                        batch_workers=None):
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    TOPO_PREDICTORS = ['tpi_500', 'slope', 'aspect']
    HOMEMADE_PREDICTORS = ['e_plus', 'e_minus', 'w_speed', 'w_angle']
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
    END_TRAIN = START_DATE + pd.to_timedelta(int(floor((1 - validation_split) * NUM_DAYS)), unit='days')
    START_VAL = END_TRAIN + pd.to_timedelta(1, unit='day')
    NUM_VAL_DAYS = (END_DATE - START_VAL).days + 1
    if batch_workers is None:
        batch_workers = os.cpu_count()
    # Data processing and batch creation
    batch_gen_training = BatchGenerator(path_to_data=PROCESSED_DATA_FOLDER, decoder=NaiveDecoder(normalize=True),
                                        sequence_length=sequence_length,
                                        input_pattern=input_pattern,
                                        patch_length_pixel=img_size, batch_size=batch_size,
                                        input_variables=ALL_INPUTS,
                                        output_variables=ALL_OUTPUTS,
                                        start_date=START_DATE, end_date=END_DATE,
                                        num_workers=batch_workers)
    batch_gen_validation = BatchGenerator(path_to_data=PROCESSED_DATA_FOLDER, decoder=NaiveDecoder(normalize=True),
                                          sequence_length=sequence_length,
                                          input_pattern=input_pattern,
                                          patch_length_pixel=img_size, batch_size=batch_size,
                                          input_variables=ALL_INPUTS,
                                          output_variables=ALL_OUTPUTS,
                                          start_date=START_VAL, end_date=END_DATE,
                                          num_workers=batch_workers)
    inputs = []
    outputs = []
    print('Creating a validation set for final plots')
    with batch_gen_validation as batch:
        for b in range(NUM_VAL_DAYS):
            print(f'Creating batch {b + 1}/{NUM_VAL_DAYS}')
            x, y = next(batch)
            inputs.append(x)
            outputs.append(y)
    inputs = np.concatenate(inputs)
    outputs = np.concatenate(outputs)
    INPUT_CHANNELS = len(ALL_INPUTS)
    OUT_CHANNELS = len(ALL_OUTPUTS)
    # Creating NN
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
    checkpoint_path_weights.parent.mkdir(exist_ok=True, parents=True)
    log_path = Path('./logs/generator') / run_id
    log_path.parent.mkdir(exist_ok=True, parents=True)
    callbacks = [
        cb.TensorBoard(log_path, write_images=True, update_freq='batch'),
        cb.ProgbarLogger('steps'),
        cb.TerminateOnNaN(),
        cb.ModelCheckpoint(str(checkpoint_path_weights), save_best_only=False, period=saving_frequency,
                           save_weights_only=True),
    ]
    generator.fit(x=batch_gen_training, callbacks=callbacks, epochs=nb_epochs,
                  workers=batch_workers, use_multiprocessing=True)

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


if __name__ == '__main__':
    train_with_all_data(cosmoblurred=True)
