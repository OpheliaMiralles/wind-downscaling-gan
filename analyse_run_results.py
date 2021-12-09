import os
from pathlib import Path

import cartopy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.crs import epsg
from matplotlib.colors import LogNorm
from silence_tensorflow import silence_tensorflow

from data.data_processing import HigherResPlateCarree
from gan.metrics import WindSpeedRMSE, WindSpeedWeightedRMSE, WeightedRMSEForExtremes, LogSpectralDistance, \
    SpatialKS, AngularCosineDistance, cosine_similarity_from_xarray, tanh_wind_speed_weighted_rmse_from_xarray

silence_tensorflow()

from data.data_generator import BatchGenerator, NaiveDecoder, LocalFileProvider, S3FileProvider
from data.data_generator import FlexibleNoiseGenerator
from gan import train, metrics
from gan.ganbase import GAN
from gan.models import make_generator, make_discriminator

TEST_METRICS = [WindSpeedRMSE, WindSpeedWeightedRMSE, WeightedRMSEForExtremes,
                LogSpectralDistance, SpatialKS, AngularCosineDistance]
DATA_ROOT = Path(os.getenv('DATA_ROOT', './data'))
CHECKPOINT_ROOT = Path(os.getenv('CHECKPOINT_ROOT', './checkpoints'))
PROCESSED_DATA_FOLDER = DATA_ROOT / 'img_prediction_files'
# variables used in the run
TOPO_PREDICTORS = []
HOMEMADE_PREDICTORS = []
ERA5_PREDICTORS_SURFACE = ['u10', 'v10']
ERA5_PREDICTORS_Z500 = []
# CRS used for plots
crs_cosmo = epsg(21781)


def get_network_from_config(len_inputs=2,
                            len_outputs=2,
                            sequence_length=6,
                            img_size=128,
                            batch_size=16,
                            noise_channels=20,
                            noise_std=0.1
                            ):
    # Creating GAN
    generator = make_generator(image_size=img_size, in_channels=len_inputs,
                               noise_channels=noise_channels, out_channels=len_outputs,
                               n_timesteps=sequence_length)
    print(f"Generator: {generator.count_params():,} weights")
    discriminator = make_discriminator(low_res_size=img_size, high_res_size=img_size,
                                       low_res_channels=len_inputs,
                                       high_res_channels=len_outputs, n_timesteps=sequence_length)
    print(f"Discriminator: {discriminator.count_params():,} weights")
    noise_shape = (batch_size, sequence_length, img_size, img_size, noise_channels)
    gan = GAN(generator, discriminator, noise_generator=FlexibleNoiseGenerator(noise_shape, std=noise_std))
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
    return gan


def get_data_providers(data_provider='local', cosmoblurred=False):
    input_pattern = 'x_cosmo_{date}.nc' if cosmoblurred else 'x_{date}.nc'
    if data_provider == 'local':
        input_provider = LocalFileProvider(PROCESSED_DATA_FOLDER, input_pattern)
        output_provider = LocalFileProvider(PROCESSED_DATA_FOLDER, 'y_{date}.nc')
    elif data_provider == 's3':
        input_provider = S3FileProvider('wind-downscaling', 'img_prediction_files', pattern=input_pattern)
        output_provider = S3FileProvider('wind-downscaling', 'img_prediction_files', pattern='y_{date}.nc')
    else:
        raise ValueError(f'Wrong value for data provider {data_provider}: please choose between s3 and local')

    return input_provider, output_provider


def plot_prediction_by_batch(run_id, start_date, end_date, sequence_length=6,
                             img_size=128,
                             batch_size=16,
                             noise_channels=20,
                             noise_std=0.1,
                             cosmoblurred=False,
                             batch_workers=None,
                             data_provider: str = 'local',
                             saving_frequency=10,
                             nb_epochs=500,
                             ):
    ALL_OUTPUTS = ['U_10M', 'V_10M']
    if batch_workers is None:
        batch_workers = os.cpu_count()
    ALL_INPUTS = TOPO_PREDICTORS + HOMEMADE_PREDICTORS
    ALL_INPUTS += ['U_10M', 'V_10M'] if cosmoblurred else ERA5_PREDICTORS_Z500 + ERA5_PREDICTORS_SURFACE
    input_provider, output_provider = get_data_providers(data_provider=data_provider, cosmoblurred=cosmoblurred)
    run_id = f'{run_id}_cosmo_blurred' if cosmoblurred else run_id
    START_DATE = pd.to_datetime(start_date)
    END_DATE = pd.to_datetime(end_date)
    NUM_DAYS = (END_DATE - START_DATE).days + 1
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
    with batch_gen as batch:
        for b in range(NUM_DAYS):
            print(f'Creating batch {b + 1}/{NUM_DAYS}')
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

    network = get_network_from_config(len_inputs=INPUT_CHANNELS, len_outputs=OUT_CHANNELS,
                                      sequence_length=sequence_length, img_size=img_size, batch_size=batch_size,
                                      noise_channels=noise_channels, noise_std=noise_std)
    str_net = 'gan'
    gen = network.generator
    # Saving results
    checkpoint_path_weights = Path(f'{CHECKPOINT_ROOT}/{str_net}') / run_id / 'weights-{epoch:02d}.ckpt'

    for epoch in np.arange(saving_frequency, nb_epochs, saving_frequency):
        network.load_weights(str(checkpoint_path_weights).format(epoch=epoch))
        for i in range(0, batch_size * 6, batch_size):
            results = gen.predict([inputs[i:i + batch_size],
                                   network.noise_generator(bs=inputs[i:i + batch_size].shape[0], channels=noise_channels)])
            j = 0
            for inp, out, res in zip(inputs[i:i + batch_size], outputs[i:i + batch_size],
                                     results):
                plot_path = Path(
                    f'./plots/{str_net}') / run_id / str(epoch) / f'inp_{i}_batch_{j}.png'
                plot_path.parent.mkdir(exist_ok=True, parents=True)
                fig = show([inp, out, res])
                fig.savefig(plot_path)
                j += 1


def get_predicted_map_Switzerland(network, input_image, img_size=128, sequence_length=6, cosmoblurred_sample=False,
                                  noise_channels=20, handle_borders='overlap', overlapping_bonus=0):
    """

    :param network: the network model
    :param input_image: dataset (nc file) containing unprocessed inputs
    :param input_vars: selected input variables for model
    :param img_size: patch size as set up in model for the training
    :param sequence_length: time steps as set up in model for training
    :param noise_channels: noise channels as set up in model for training
    :param handle_borders: either 'crop' or 'overlap'
    :param overlapping_bonus: number of columns added to the integer part of the ratios pixel_lat/img_size and pixel_lon/img_size with the overlap option
    :return:
    """
    input_vars = TOPO_PREDICTORS + HOMEMADE_PREDICTORS
    input_vars += ['U_10M', 'V_10M'] if cosmoblurred_sample else ERA5_PREDICTORS_Z500 + ERA5_PREDICTORS_SURFACE
    range_long = (5.73, 10.67)
    range_lat = (45.69, 47.97)
    if handle_borders == 'crop':
        df_in = input_image.to_dataframe()
        df_in = df_in[
            (df_in["lon_1"] <= range_long[1]) & (df_in["lon_1"] >= range_long[0]) & (df_in["lat_1"] <= range_lat[1]) & (
                    df_in["lat_1"] >= range_lat[0])]
        x_good = sorted(list(set(df_in.index.get_level_values("x_1"))))
        y_good = sorted(list(set(df_in.index.get_level_values("y_1"))))
        ds_in = input_image.sel(x_1=x_good, y_1=y_good).copy()
    else:
        ds_in = input_image.copy()
    if 'elevation' in input_vars:
        ds_in['elevation'] = ds_in['elevation'] / 1e3
    variables_of_interest = ['u10', 'v10']
    pixels_lat, pixels_lon, time_window = ds_in.dims['x_1'], ds_in.dims['y_1'], ds_in.dims['time']
    ntimeseq = time_window // sequence_length
    if handle_borders == 'crop':
        nrows, ncols = np.math.floor(pixels_lon / img_size), np.math.floor(pixels_lat / img_size)
        squares = {(i, j, k): ds_in.isel(time=slice(k * sequence_length, (k + 1) * sequence_length),
                                         x_1=slice(j * img_size, (j + 1) * img_size),
                                         y_1=slice((ncols - i - 1) * img_size, (ncols - i - 2) * img_size, -1)
                                         )[input_vars]
                   for j in range(ncols)
                   for i in range(nrows)
                   for k in range(ntimeseq)}
    elif handle_borders == 'overlap':
        nrows, ncols = overlapping_bonus + np.math.ceil(pixels_lon / img_size), overlapping_bonus + np.math.ceil(
            pixels_lat / img_size)  # ceil and not floor, we want to cover the whole map
        xdist, ydist = (pixels_lat - img_size) // (ncols - 1), (pixels_lon - img_size) // (nrows - 1)
        leftovers_x, leftovers_y = pixels_lat - ((ncols - 1) * xdist + img_size), pixels_lon - (
                (nrows - 1) * ydist + img_size)
        x_vec_leftovers, y_vec_leftovers = np.concatenate(
            [[0], np.ones(leftovers_x), np.zeros(ncols - leftovers_x - 1)]).cumsum(), np.concatenate(
            [[0], np.ones(leftovers_y), np.zeros(nrows - leftovers_y - 1)]).cumsum()
        slices_start_x, slices_start_y = [int(i * xdist + x) for (i, x) in zip(range(ncols), x_vec_leftovers)], [
            int(j * ydist + y) for (j, y) in zip(range(nrows), y_vec_leftovers)]
        squares = {(sx, sy, k): ds_in.isel(time=slice(k * sequence_length, (k + 1) * sequence_length),
                                           x_1=slice(sx, sx + img_size),
                                           y_1=slice(sy + img_size - 1, sy - 1, -1) if sy != 0 else slice(img_size, 0, -1)
                                           )[input_vars]
                   for sx in slices_start_x
                   for sy in slices_start_y
                   for k in range(ntimeseq)}
    else:
        raise ValueError('Please chose one of "crop" or "overlap" for handling of image borders')
    positions = {(i, j, k): index for index, (i, j, k) in enumerate(squares)}
    tensors = np.stack([im.to_array().to_numpy() for k, im in squares.items()], axis=0)
    tensors = np.transpose(tensors, [0, 2, 3, 4, 1])
    tensors = (tensors - np.nanmean(tensors, axis=(0, 1, 2), keepdims=True)) / np.nanstd(tensors, axis=(0, 1, 2),
                                                                                         keepdims=True)
    gen = network.generator
    predictions = gen.predict([tensors, network.noise_generator(bs=tensors.shape[0], channels=noise_channels)])
    predicted_squares = {
        (i, j, k): xr.Dataset(
            {v: xr.DataArray(predictions[positions[(i, j, k)], ..., variables_of_interest.index(v)],
                             coords=squares[(i, j, k)].coords, name=v) for v in
             variables_of_interest},
            coords=squares[(i, j, k)].coords)
        for (i, j, k) in squares
    }
    if handle_borders == 'crop':
        predicted_data = xr.combine_by_coords(list(predicted_squares.values()))
    elif handle_borders == 'overlap':
        predicted_squares = {v: predicted_squares[v].isel(x_1=slice(2, -2), y_1=slice(2, -2)) for v in predicted_squares}
        bigdata = pd.concat([s.to_dataframe() for s in predicted_squares.values()])
        unique = bigdata.groupby(level=['time', 'y_1', 'x_1']).mean()
        predicted_data = xr.Dataset.from_dataframe(unique).set_coords(['lon_1', 'lat_1', 'y', 'x'])
    return predicted_data


def plot_predicted_maps_Switzerland(run_id, date, epoch, hour,
                                    sequence_length=6,
                                    img_size=128,
                                    batch_size=16,
                                    noise_channels=20,
                                    noise_std=0.1,
                                    cosmoblurred_sample=False,
                                    cosmoblurred_run=False,
                                    data_provider: str = 'local',
                                    handle_borders='overlap',
                                    overlapping_bonus=0
                                    ):
    ALL_OUTPUTS = ['U_10M', 'V_10M']
    ALL_INPUTS = TOPO_PREDICTORS + HOMEMADE_PREDICTORS
    ALL_INPUTS += ['U_10M', 'V_10M'] if cosmoblurred_sample else ERA5_PREDICTORS_Z500 + ERA5_PREDICTORS_SURFACE
    input_provider, output_provider = get_data_providers(data_provider=data_provider, cosmoblurred=cosmoblurred_sample)
    run_id = f'{run_id}_cosmo_blurred' if cosmoblurred_run else run_id
    network = get_network_from_config(len_inputs=len(ALL_INPUTS), len_outputs=len(ALL_OUTPUTS),
                                      sequence_length=sequence_length, img_size=img_size, batch_size=batch_size,
                                      noise_channels=noise_channels, noise_std=noise_std)
    str_net = 'gan'
    # Saving results
    checkpoint_path_weights = Path(f'{CHECKPOINT_ROOT}/{str_net}') / run_id / f'weights-{epoch:02d}.ckpt'
    network.load_weights(str(checkpoint_path_weights))
    with input_provider.provide(date) as input_file, output_provider.provide(date) as output_file:
        input_image = xr.open_dataset(input_file)
        output_image = xr.open_dataset(output_file)
    predicted = get_predicted_map_Switzerland(network, input_image, sequence_length=sequence_length, img_size=img_size,
                                              noise_channels=noise_channels, cosmoblurred_sample=cosmoblurred_sample, handle_borders=handle_borders,
                                              overlapping_bonus=overlapping_bonus)
    range_long = (5.8, 10.6)
    range_lat = (45.75, 47.9)
    fig = plt.figure(constrained_layout=True, figsize=(15, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    axes=[]
    for i in range(3):
        axes.append([])
        for j in range(3):
            ax = fig.add_subplot(gs[i, j], projection=HigherResPlateCarree())
            axes[i].append(ax)
    for variable_to_plot, i in zip(['u10', 'v10'], [0,1]):
        ax = axes[i]
        cosmo_var = 'U_10M' if variable_to_plot == 'u10' else 'V_10M'
        text = 'U-component' if variable_to_plot == 'u10' else 'V-component'
        cbar_kwargs = {"orientation": "horizontal", "shrink": 0.25,
                       "label": f"10-meter {text} (m.s-1)"}
        inp = input_image.isel(time=hour).get(cosmo_var) if cosmoblurred_sample else input_image.isel(time=hour).get(
            variable_to_plot)
        cosmo = output_image.isel(time=hour).get(cosmo_var)
        pred = predicted.isel(time=hour).get(variable_to_plot)
        mini = np.nanmin(output_image.isel(time=hour).get(cosmo_var).__array__())
        maxi = np.nanmax(output_image.isel(time=hour).get(cosmo_var).__array__())
        vmin, vmax = -max(abs(mini), abs(maxi)), max(abs(mini), abs(maxi))
        inp.plot(cmap='jet', ax=ax[0], transform=crs_cosmo, vmin=vmin, vmax=vmax, add_colorbar=False)
        pr = cosmo.plot(cmap='jet', ax=ax[1], transform=crs_cosmo, add_colorbar=False)
        pred.plot(cmap='jet', ax=ax[2], transform=crs_cosmo, vmin=vmin, vmax=vmax, add_colorbar=False)
        title_inp = 'COSMO1 blurred data' if cosmoblurred_sample else 'ERA5 reanalysis data'
        ax[0].set_title(title_inp)
        ax[1].set_title('COSMO-1 data')
        ax[2].set_title('Predicted')
        fig.colorbar(pr, ax=ax, **cbar_kwargs)
    ax = axes[2]
    dem = input_image.elevation.isel(time=hour)
    dem.plot(ax=ax[0], transform=crs_cosmo, cmap=plt.cm.terrain, norm=LogNorm(vmin=58, vmax=4473),
             cbar_kwargs={"orientation": "horizontal", "shrink": 0.7, "label": f"terrain height (m)"})
    ax[0].set_title('DEM')
    ax[0].add_feature(cartopy.feature.RIVERS.with_scale('10m'), color=plt.cm.terrain(0.))
    ax[0].add_feature(cartopy.feature.LAKES.with_scale('10m'), color=plt.cm.terrain(0.))
    for metric, ax, name, cmap, vmin in zip([cosine_similarity_from_xarray, tanh_wind_speed_weighted_rmse_from_xarray], [ax[1], ax[2]], ['Cosine Similarity', 'Tanh of Wind Speed Weighted RMSE'],
                                            ['brg', 'Reds'], [-1., 0.]):
        real_output = output_image[['U_10M', 'V_10M']].sel(x_1=np.unique(predicted.x_1[:]), y_1=np.unique(predicted.y_1[:]))
        fake_output = predicted[['u10', 'v10']]
        dist = metric(real_output, fake_output)
        dist.mean(dim='time').plot(cmap=cmap, ax=ax, transform=crs_cosmo, vmin=vmin, vmax=1., cbar_kwargs={"orientation": "horizontal", "shrink": 0.7})
        ax.set_title(name)
    for ax in [item for sublist in axes for item in sublist]:
        ax.set_extent([range_long[0], range_long[1], range_lat[0], range_lat[1]])
        ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
    return fig

def compute_metrics_val_set(run_id, start_date, end_date, sequence_length=6,
                            img_size=128,
                            batch_size=16,
                            noise_channels=20,
                            noise_std=0.1,
                            cosmoblurred=False,
                            batch_workers=None,
                            data_provider: str = 'local',
                            saving_frequency=10,
                            nb_epochs=500,
                            ):
    ALL_OUTPUTS = ['U_10M', 'V_10M']
    if batch_workers is None:
        batch_workers = os.cpu_count()
    ALL_INPUTS = TOPO_PREDICTORS + HOMEMADE_PREDICTORS
    ALL_INPUTS += ['U_10M', 'V_10M'] if cosmoblurred else ERA5_PREDICTORS_Z500 + ERA5_PREDICTORS_SURFACE
    input_provider, output_provider = get_data_providers(data_provider=data_provider, cosmoblurred=cosmoblurred)
    run_id = f'{run_id}_cosmo_blurred' if cosmoblurred else run_id
    START_DATE = pd.to_datetime(start_date)
    END_DATE = pd.to_datetime(end_date)
    NUM_DAYS = (END_DATE - START_DATE).days + 1
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
    with batch_gen as batch:
        for b in range(NUM_DAYS):
            print(f'Creating batch {b + 1}/{NUM_DAYS}')
            x, y = next(batch)
            inputs.append(x)
            outputs.append(y)
    inputs = np.concatenate(inputs)
    outputs = np.concatenate(outputs)
    INPUT_CHANNELS = len(ALL_INPUTS)
    OUT_CHANNELS = len(ALL_OUTPUTS)
    network = get_network_from_config(len_inputs=INPUT_CHANNELS, len_outputs=OUT_CHANNELS,
                                      sequence_length=sequence_length, img_size=img_size, batch_size=batch_size,
                                      noise_channels=noise_channels, noise_std=noise_std)
    str_net = 'gan'
    gen = network.generator
    # Saving results
    checkpoint_path_weights = Path(f'{CHECKPOINT_ROOT}/{str_net}') / run_id / 'weights-{epoch:02d}.ckpt'
    metrics_data = []
    for epoch in np.arange(saving_frequency, nb_epochs, saving_frequency):
        print(f'Loading weights for epoch {epoch}')
        network.load_weights(str(checkpoint_path_weights).format(epoch=epoch))
        preds = gen.predict([inputs, network.noise_generator(bs=inputs.shape[0], channels=noise_channels)])
        print(f'Computing metrics for epoch {epoch}')
        metrics_data.append(
            pd.DataFrame([float(m()(outputs, preds)) for m in TEST_METRICS], index=[m().name for m in TEST_METRICS],
                         columns=[epoch]).T)
    metrics_data = pd.concat(metrics_data)
    return metrics_data


def plot_metrics_data(metrics_data_csv):
    """

    :param metrics_data_csv: data is computed using the above function
    :return:
    """
    metrics_data_csv = metrics_data_csv.rename(columns={"ws_rmse": "Wind Speed RMSE",
                                                        "ws_weighted_rmse": "Wind Speed Weighted RMSE",
                                                        "extreme_rmse": "Extreme RMSE",
                                                        "spatial_ks": "Spatially Convolved KS Statistics",
                                                        "acd": "Angular Cosine Distance",
                                                        "lsd": "Log Spectral Distance"})
    if 'Unnamed: 0' in metrics_data_csv.columns:
        metrics_data_csv = metrics_data_csv.rename(columns={"Unnamed: 0": "epoch"}).set_index("epoch")
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    for c, ax in zip(metrics_data_csv.columns, axes):
        df = metrics_data_csv[c]
        title = c
        three_min = df.sort_values().iloc[:3]
        ax.plot(df, color="royalblue", ls='dotted')
        ax.scatter(three_min.index, three_min, color="navy", marker='^')
        ax.set_xlabel("Epoch")
        ax.set_title(title)
    fig.tight_layout()
    return fig


def get_targets_and_predictions_for_timerange(network, start_date, end_date, cosmoblurred_sample=False, img_size=128, sequence_length=6,
                                              noise_channels=20, handle_borders='overlap', overlapping_bonus=0):
    dates = pd.date_range(start_date, end_date, freq='1d')
    predictions = []
    out = []
    for date in dates:
        d_str = date.strftime('%Y%m%d')
        print(d_str)
        input_image = xr.open_dataset(PROCESSED_DATA_FOLDER / f'x_{d_str}.nc') if not cosmoblurred_sample else xr.open_dataset(PROCESSED_DATA_FOLDER / f'x_cosmo_{d_str}.nc')
        output_image = xr.open_dataset(PROCESSED_DATA_FOLDER / f'y_{d_str}.nc')
        preds = get_predicted_map_Switzerland(network, input_image, cosmoblurred_sample=cosmoblurred_sample, img_size=img_size, sequence_length=sequence_length,
                                              noise_channels=noise_channels, handle_borders=handle_borders, overlapping_bonus=overlapping_bonus)
        predictions.append(preds)
        out.append(output_image)
    all_out = xr.concat(out, dim='time')
    all_pred = xr.concat(predictions, dim='time')
    all_out = all_out.assign(hour=all_out.time.dt.hour)
    all_pred = all_pred.assign(hour=all_pred.time.dt.hour)
    return all_out, all_pred


def plot_mean_daily_pattern(targets, predictions):
    out_daily_pattern = targets.groupby('hour').mean(dim='time')
    pred_daily_pattern = predictions.groupby('hour').mean(dim='time')
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))
    for ax, name, colorset in zip(axes, ['U-component', 'V-component'], [('salmon', 'goldenrod'), ('navy', 'royalblue')]):
        if name == 'U-component':
            out_mean_daily_pattern = out_daily_pattern.mean(dim=['x_1', 'y_1']).U_10M
            pred_mean_daily_pattern = pred_daily_pattern.mean(dim=['x_1', 'y_1']).u10
        else:
            out_mean_daily_pattern = out_daily_pattern.mean(dim=['x_1', 'y_1']).V_10M
            pred_mean_daily_pattern = pred_daily_pattern.mean(dim=['x_1', 'y_1']).v10
        out_mean_daily_pattern.plot(ax=ax[0], color=colorset[0], label=name)
        ax[0].set_title('COSMO 1 target')
        pred_mean_daily_pattern.plot(ax=ax[1], color=colorset[1], label=name)
        ax[1].set_title('Prediction')
    fig.tight_layout()
    return fig


def plot_mean_spatial_pattern(targets, predictions):
    out_mean_spatial_pattern = targets.median(dim='time')
    pred_mean_spatial_pattern = predictions.median(dim='time')
    range_long = (5.8, 10.6)
    range_lat = (45.75, 47.9)
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 10), subplot_kw={'projection': HigherResPlateCarree()})
    for ax, name in zip(axes, ['U-component', 'V-component']):
        if name == 'U-component':
            out_mean_spatial_pattern_spec = out_mean_spatial_pattern.U_10M
            pred_mean_spatial_pattern_spec = pred_mean_spatial_pattern.u10
        else:
            out_mean_spatial_pattern_spec = out_mean_spatial_pattern.V_10M
            pred_mean_spatial_pattern_spec = pred_mean_spatial_pattern.v10
        vmin = -max(np.abs(out_mean_spatial_pattern_spec.min()), out_mean_spatial_pattern_spec.max())
        vmax = max(np.abs(out_mean_spatial_pattern_spec.min()), out_mean_spatial_pattern_spec.max())
        pred_mean_spatial_pattern_spec.plot(cmap='jet', ax=ax[1], transform=crs_cosmo, vmin=vmin, vmax=vmax, cbar_kwargs={"orientation": "horizontal", "shrink": 0.7,
                                                                                                                          "label": f"10-meter {name} (m.s-1)"})
        out_mean_spatial_pattern_spec.plot(cmap='jet', ax=ax[0], transform=crs_cosmo, vmin=vmin, vmax=vmax, cbar_kwargs={"orientation": "horizontal", "shrink": 0.7,
                                                                                                                         "label": f"10-meter {name} (m.s-1)"})
        ax[0].set_title('COSMO 1 target')
        ax[1].set_title('Prediction')
        for a in ax:
            a.set_extent([range_long[0], range_long[1], range_lat[0], range_lat[1]])
            a.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
    fig.suptitle('Mean spatial wind pattern')
    fig.tight_layout()
    return fig

