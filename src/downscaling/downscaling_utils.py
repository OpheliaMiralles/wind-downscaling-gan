import cartopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from silence_tensorflow import silence_tensorflow

from data.data_processing import HigherResPlateCarree

silence_tensorflow()

from data.data_generator import FlexibleNoiseGenerator
from gan import train, metrics
from gan.ganbase import GAN
from gan.models import make_generator, make_discriminator

WEIGHTS_PATH = 'weights-55.ckpt'
SEQUENCE_LENGTH = 24
IMG_SIZE = 96
BATCH_SIZE = 8
NOISE_CHANNELS = 20
NOISE_STD = 0.1
NB_INPUTS = 3
NB_OUTPUTS = 2


def process_topo(raster_topo: xr.DataArray, high_res_template: xr.Dataset):
    lon_coord, lat_coord = [c for c in high_res_template.coords if c.startswith('lon')][0], [c for c in high_res_template.coords if c.startswith('lat')][0]
    dem = raster_topo.isel(band=0, drop=True)
    inputs_topo = xr.DataArray(dem,
                               coords=dem.coords,
                               name='elevation').to_dataset().sel(x=high_res_template.get(lon_coord), y=high_res_template.get(lat_coord), method='nearest').drop(['x', 'y'])
    return inputs_topo


def process_era5(ds_era5: xr.Dataset, high_res_template: xr.Dataset):
    lon_coord, lat_coord = [c for c in high_res_template.coords if c.startswith('lon')][0], [c for c in high_res_template.coords if c.startswith('lat')][0]
    inputs_surface = ds_era5[['u10', 'v10']].sel(longitude=high_res_template.get(lon_coord), latitude=high_res_template.get(lat_coord), method='nearest').drop(['longitude', 'latitude'])
    return inputs_surface


def build_high_res_template_from_era5(ds_era5: xr.Dataset, range_lon=None, range_lat=None):
    upsampling_lat = 26
    upsampling_lon = 18
    if not range_lon:
        range_lon = (float(ds_era5.longitude.min()), float(ds_era5.longitude.max()))
    else:
        ds_era5 = ds_era5.sel(longitude=slice(range_lon[0], range_lon[1]))
    if not range_lat:
        range_lat = (float(ds_era5.latitude.min()), float(ds_era5.latitude.max()))
    else:
        ds_era5 = ds_era5.sel(latitude=slice(range_lat[1], range_lat[0]))
    nb_lon = ds_era5.dims['longitude']
    nb_lat = ds_era5.dims['latitude']
    new_longitudes = np.linspace(range_lon[0], range_lon[1], upsampling_lon * nb_lon)
    new_latitudes = np.linspace(range_lat[0], range_lat[1], upsampling_lat * nb_lat)
    high_res_template = ds_era5.coords.to_dataset().assign_coords({'lon_1': new_longitudes, 'lat_1': new_latitudes}).drop(['longitude', 'latitude'])
    return high_res_template


def get_network():
    # Creating GAN
    generator = make_generator(image_size=IMG_SIZE, in_channels=NB_INPUTS,
                               noise_channels=NOISE_CHANNELS, out_channels=NB_OUTPUTS,
                               n_timesteps=SEQUENCE_LENGTH)
    print(f"Generator: {generator.count_params():,} weights")
    discriminator = make_discriminator(low_res_size=IMG_SIZE, high_res_size=IMG_SIZE,
                                       low_res_channels=NB_INPUTS,
                                       high_res_channels=NB_OUTPUTS, n_timesteps=SEQUENCE_LENGTH)
    print(f"Discriminator: {discriminator.count_params():,} weights")
    noise_shape = (BATCH_SIZE, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, NOISE_CHANNELS)
    gan = GAN(generator, discriminator, noise_generator=FlexibleNoiseGenerator(noise_shape, std=NOISE_STD))
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
    gan.load_weights(str(WEIGHTS_PATH))
    return gan


def predict(inputs_era5: xr.Dataset, inputs_topo: xr.Dataset, high_res_template: xr.Dataset):
    lat_coord_hr, lon_coord_hr = [c for c in high_res_template.dims if c.startswith('lat') or c.startswith('y')][0], [c for c in high_res_template.dims if c.startswith('lon') or c.startswith('x')][0]
    time_var_topo = inputs_topo.expand_dims({'time': inputs_era5.time})
    inputs = xr.merge([inputs_era5, time_var_topo])
    inputs = inputs.drop(c for c in inputs.coords if c not in ['time'] + [lat_coord_hr, lon_coord_hr])
    network = get_network()
    ds_in = inputs.copy()
    ds_in['elevation'] = ds_in['elevation'] / 1e3
    variables_of_interest = ['u10', 'v10']
    pixels_lat, pixels_lon, time_window = ds_in.dims[lat_coord_hr], ds_in.dims[lon_coord_hr], ds_in.dims['time']
    ntimeseq = time_window // SEQUENCE_LENGTH
    ncols, nrows = np.math.ceil(pixels_lon / IMG_SIZE), np.math.ceil(
        pixels_lat / IMG_SIZE)  # ceil and not floor, we want to cover the whole map
    ydist, xdist = (pixels_lat - IMG_SIZE) // (nrows - 1), (pixels_lon - IMG_SIZE) // (ncols - 1)
    leftovers_y, leftovers_x = pixels_lat - ((nrows - 1) * ydist + IMG_SIZE), pixels_lon - (
            (ncols - 1) * xdist + IMG_SIZE)
    x_vec_leftovers, y_vec_leftovers = np.concatenate(
        [[0], np.ones(leftovers_x), np.zeros(ncols - leftovers_x - 1)]).cumsum(), np.concatenate(
        [[0], np.ones(leftovers_y), np.zeros(nrows - leftovers_y - 1)]).cumsum()
    slices_start_x, slices_start_y = [int(i * xdist + x) for (i, x) in zip(range(ncols), x_vec_leftovers)], [
        int(j * ydist + y) for (j, y) in zip(range(nrows), y_vec_leftovers)]
    squares = {(sx, sy, k): ds_in.isel({"time": slice(k * SEQUENCE_LENGTH, (k + 1) * SEQUENCE_LENGTH),
                                        lon_coord_hr: slice(sx, sx + IMG_SIZE),
                                        lat_coord_hr: slice(sy + IMG_SIZE - 1, sy - 1, -1) if sy != 0 else slice(IMG_SIZE, 0, -1)}
                                       )[['u10', 'v10', 'elevation']]
               for sx in slices_start_x
               for sy in slices_start_y
               for k in range(ntimeseq)}
    positions = {(i, j, k): index for index, (i, j, k) in enumerate(squares)}
    tensors = np.stack([im.to_array().to_numpy() for k, im in squares.items()], axis=0)
    tensors = np.transpose(tensors, [0, 2, 3, 4, 1])
    tensors = (tensors - np.nanmean(tensors, axis=(0, 1, 2), keepdims=True)) / np.nanstd(tensors, axis=(0, 1, 2),
                                                                                         keepdims=True)
    gen = network.generator
    noise = network.noise_generator(bs=tensors.shape[0], channels=NOISE_CHANNELS)
    predictions = gen.predict([tensors, noise])
    predicted_squares = {
        (i, j, k): xr.Dataset(
            {v: xr.DataArray(predictions[positions[(i, j, k)], ..., variables_of_interest.index(v)],
                             coords=squares[(i, j, k)].coords, name=v) for v in
             variables_of_interest},
            coords=squares[(i, j, k)].coords)
        for (i, j, k) in squares
    }
    predicted_squares = {v: predicted_squares[v].isel({lon_coord_hr: slice(2, -2), lat_coord_hr: slice(2, -2)}) for v in predicted_squares}
    bigdata = pd.concat([s.to_dataframe() for s in predicted_squares.values()])
    unique = bigdata.groupby(level=['time'] + [lat_coord_hr, lon_coord_hr]).mean()
    predicted_data = xr.Dataset.from_dataframe(unique)
    return predicted_data


def downscale(era5: xr.Dataset, raster_topo: xr.DataArray, range_lon=None, range_lat=None):
    high_res_template = build_high_res_template_from_era5(era5, range_lon=range_lon, range_lat=range_lat)
    inputs_era5 = process_era5(era5, high_res_template)
    inputs_topo = process_topo(raster_topo, high_res_template)
    prediction = predict(inputs_era5, inputs_topo, high_res_template)
    return prediction


def plot_wind_fields(ds: xr.Dataset, cmap='bwr', title='', range_lon=None, range_lat=None, high_res_crs=None):
    fig = plt.figure(constrained_layout=True, figsize=(15, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    axes = []
    for i in range(1):
        axes.append([])
        for j in range(2):
            ax = fig.add_subplot(gs[i, j], projection=HigherResPlateCarree())
            axes[i].append(ax)
    axes = axes[0]
    for variable_to_plot, i in zip(['u10', 'v10'], [0, 1]):
        ax = axes[i]
        text = 'U-component' if variable_to_plot == 'u10' else 'V-component'
        cbar_kwargs = {"orientation": "horizontal", "shrink": 0.5,
                       "label": f"10-meter {text} (m.s-1)"}
        var = ds.get(variable_to_plot)
        mini = np.nanmin(var.__array__())
        maxi = np.nanmax(var.__array__())
        vmin, vmax = -max(abs(mini), abs(maxi)), max(abs(mini), abs(maxi))
        if not high_res_crs:
            high_res_crs = HigherResPlateCarree()
        pr = var.plot(cmap=cmap, ax=ax, vmin=vmin, vmax=vmax, transform=high_res_crs, add_colorbar=False)
        ax.set_title(title)
        fig.colorbar(pr, ax=ax, **cbar_kwargs)
    for ax in axes:
        if range_lon is not None and range_lat is not None:
            ax.set_extent([range_lon[0], range_lon[1], range_lat[0], range_lat[1]])
        borders = cartopy.feature.NaturalEarthFeature(category='cultural',
                                                      name='admin_0_boundary_lines_land',
                                                      scale='10m', facecolor='none')
        ax.add_feature(borders, edgecolor='black')
        ax.coastlines(resolution='10m', color='black')
    return fig


def plot_elevation(raster_topo: xr.DataArray, range_lon=None, range_lat=None):
    dem = raster_topo.isel(band=0, drop=True)
    ds_topo = xr.DataArray(dem,
                           coords=dem.coords,
                           name='elevation').to_dataset()
    fig, ax = plt.subplots(constrained_layout=True, figsize=(7.5, 5), subplot_kw={'projection': HigherResPlateCarree()})
    ds_topo.elevation.plot(ax=ax, transform=HigherResPlateCarree(), cmap=plt.cm.terrain, norm=LogNorm(vmin=58, vmax=4473),
                           cbar_kwargs={"orientation": "horizontal", "shrink": 0.7, "label": f"terrain height (m)"})
    ax.set_title('DEM')
    ax.add_feature(cartopy.feature.RIVERS.with_scale('10m'), color=plt.cm.terrain(0.))
    ax.add_feature(cartopy.feature.LAKES.with_scale('10m'), color=plt.cm.terrain(0.))
    if range_lon is not None and range_lat is not None:
        ax.set_extent([range_lon[0], range_lon[1], range_lat[0], range_lat[1]])
    ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
    return fig
