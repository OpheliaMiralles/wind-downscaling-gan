import pathlib
import re
from glob import glob
from io import StringIO
from typing import Union, Tuple

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from topo_descriptors import topo, helpers


class HigherResPlateCarree(ccrs.PlateCarree):
    @property
    def threshold(self):
        return super().threshold / 100


def distance_from_coordinates(z1: Tuple, z2: Tuple):
    """

    :param z1: tuple of longitudes for the 2 places
    :param z2: tuple of latitudes for the 2 places
    :return: distance between the 2 places in km
    """
    lon1, lat1 = z1
    lon2, lat2 = z2
    # Harvestine formula
    r = 6371  # radius of Earth (KM)
    p = np.pi / 180
    a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (
            1 - np.cos((lon2 - lon1) * p)) / 2
    d = 2 * r * np.arcsin(np.sqrt(a))
    return d


def process_topographic_variables_file(path_to_file: pathlib.Path):
    dem = xr.open_rasterio(path_to_file)
    dem = dem.isel(band=0, drop=True)
    # TPI 500m
    scale_meters = 500
    scale_pixel, res_meters = helpers.scale_to_pixel(scale_meters, dem)
    tpi = topo.tpi(dem, scale_pixel)
    # Gradient
    gradient = topo.gradient(dem, 1 / 4 * scale_meters, res_meters)
    we_der, sn_der, slope, aspect = gradient
    # Norm and second the direction of Ridge index
    vr = topo.valley_ridge(dem, scale_pixel, mode='ridge')

    # Sx
    def Sx(azimuth=0, radius=500):
        pass

    all = (tpi, *vr, *gradient)
    names = ('tpi_500', 'ridge_index_norm', 'ridge_index_dir',
             'we_derivative', 'sn_derivative',
             'slope', 'aspect')
    for data, name in zip(all, names):
        da = xr.DataArray(data,
                          coords=dem.coords,
                          name=name)
        filename = f"topo_{name}.nc"
        da.to_dataset().to_netcdf(pathlib.Path(path_to_file.parent, filename))
    return tpi, vr


def process_wind_variables_file_from_MeteoSwiss(path_to_file: pathlib.Path):
    df = pd.read_csv(path_to_file, sep=';').rename(columns={"stn": 'station',
                                                            'time': 'datetime_raw',
                                                            'fkl010h0': "wind_speed_mps",
                                                            'dkl010h0': "wind_direction_degrees"})
    df = df[df["station"] != 'stn']
    df["wind_speed_mps"] = df["wind_speed_mps"].replace('-', np.NaN).astype(float)
    df["wind_direction_degrees"] = df["wind_direction_degrees"].replace('-', np.NaN).astype(float)
    df = df.assign(datetime=lambda x: pd.to_datetime(x["datetime_raw"], format='%Y%m%d%H')) \
        .assign(theta_radians=lambda x: x["wind_direction_degrees"] / 180) \
        .assign(u10=lambda x: x["wind_speed_mps"] * np.cos(x["theta_radians"])) \
        .assign(v10=lambda x: x["wind_speed_mps"] * np.sin(x["theta_radians"])) \
        .assign(hour=lambda x: x["datetime_raw"].astype(str).str[-2:]) \
        .assign(month=lambda x: x["datetime_raw"].astype(str).str[4:6])
    return df


def process_station_txt_file_from_MeteoSwiss(path_to_file: pathlib.Path):
    s = re.sub(' {2,}', '\t', path_to_file.read_text('latin1'))
    stations = pd.read_csv(StringIO(s), sep='\t')
    stations = stations.reset_index().drop_duplicates('index')
    if 'Name' in stations.columns:
        stations = stations.drop(columns=['Name'])
    elif 'Nom' in stations.columns:
        stations = stations.drop(columns=['Nom'])
    stations = stations.rename(
        columns={'  ': 'station',
                 'index': 'station',
                 'stn': 'station_name',
                 'Parameter': 'data_source',
                 'Source de donnÈes': 'lon/lat',
                 'Source de données': 'lon/lat',
                 'Source de donnees': 'lon/lat',
                 'Data source': 'lon/lat',
                 'Longitude/Latitude': 'coordinates_km',
                 'CoordonnÈes [km] Altitude [m]': 'altitude_m',
                 'Coordonnees [km] Altitude [m]': 'altitude_m',
                 'Coordonnées [km] Altitude [m]': 'altitude_m',
                 'Coordinates [km] Elevation [m]': 'altitude_m'})
    stations = stations.assign(lon=lambda x: x['lon/lat']).assign(lat=lambda x: x['lon/lat']).assign(
        x_km=lambda x: x['coordinates_km']).assign(y_km=lambda x: x['coordinates_km'])
    stations['lon'] = stations['lon'].apply(
        lambda x: float(x.split('/')[0].split('d')[0]) + float(x.split('/')[0].replace("'", '').split('d')[-1]) / 60)
    stations['lat'] = stations['lat'].apply(
        lambda x: float(x.split('/')[1].split('d')[0]) + float(x.split('/')[1].replace("'", '').split('d')[-1]) / 60)
    stations['x_km'] = stations['x_km'].apply(lambda x: float(x.split('/')[0]))
    stations['y_km'] = stations['y_km'].apply(lambda x: float(x.split('/')[1]))
    if 'lon/lat' in stations.columns:
        stations = stations.drop(columns=['lon/lat'])
    if 'coordinates_km' in stations.columns:
        stations = stations.drop(columns=['coordinates_km'])
    return stations


def dataset_times_to_datetime(data: Dataset, times: np.ndarray):
    time_unit, year, month, day = re.match('(\w+) since (\d{4})-(\d{1,2})-(\d{1,2})',
                                           data.variables['time'].units).groups()
    return pd.to_datetime(f"{year}-{month}-{day}") + pd.to_timedelta(np.array(times), time_unit)


def get_array_from_dataset(data: Dataset, variable_str: str, range_lon: Tuple[float, float] = None,
                           range_lat: Tuple[float, float] = None,
                           time_frame: Tuple[Union[str, None], Union[str, None]] = None,
                           unit_correction=1.):
    """

    :param data: dataset, netCF4 should work
    :param range_lon: min and max longitude coordinates of specific location of interest if any (degrees east)
    :param range_lat: min and max latitude coordinates of specific location of interest if any (degrees north)
    :param time_frame: timeframe of interest as tuple or list of str, None is for the whole dataset, while None on a
    bound means no limit on this side of the time window
    :param unit_correction: if needed, to rescale the values in a desired unit
    :return: the relevant array for a given variable
    """
    long_name = [v for v in data.variables if v.startswith('lon')][0]  # can contain longitude boundaries
    lat_name = [v for v in data.variables if v.startswith('lat')][0]  # can contain latitude boundaries
    longs = data.variables[long_name][:]
    lats = data.variables[lat_name][:]
    if range_lon is not None:
        min_lon, max_lon = range_lon
        mask_long = (longs <= max_lon) & (longs >= min_lon)
    else:
        mask_long = True
    if range_lat is not None:
        min_lat, max_lat = range_lat
        mask_lat = (lats <= max_lat) & (lats >= min_lat)
    else:
        mask_lat = True
    datetimes = dataset_times_to_datetime(data, data.variables['time'][:])
    if time_frame is not None:
        min_time, max_time = time_frame
        if max_time is None:
            mask_time = (datetimes >= min_time)
        elif min_time is None:
            mask_time = (datetimes <= max_time)
        else:
            mask_time = (datetimes <= max_time) & (datetimes >= min_time)
    else:
        mask_time = True
    values = data.variables[f"{variable_str}"][:]
    values.set_fill_value(np.nan)
    values_date = np.squeeze(values.filled()[mask_time, :, :])
    if values_date.ndim > 2:
        values_date_place = np.squeeze(np.squeeze(values_date[:, :, mask_long])[:, mask_lat, :]) * unit_correction
    else:
        values_date_place = np.squeeze(np.squeeze(values_date[:, mask_long])[mask_lat]) * unit_correction
    if len(longs.shape) > 1 and len(lats.shape) > 1:
        longs_place = np.squeeze(longs[:, mask_long][mask_lat])
        lats_place = np.squeeze(lats[:, mask_long][mask_lat])
    else:
        longs_place = np.squeeze(longs[mask_long])
        lats_place = np.squeeze(lats[mask_lat])
    datetimes = np.squeeze(np.array(datetimes)[mask_time])
    return values_date_place, longs_place, lats_place, datetimes


def plot_colormap_from_array(data: np.ndarray,
                             longitudes: np.ndarray,
                             latitudes: np.ndarray,
                             range_values_to_display=None,
                             ax=None,
                             title='',
                             unit=''):
    if ax is None:
        proj = HigherResPlateCarree()
        ax = plt.axes(projection=proj)
    ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()])
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
    c_scheme = ax.pcolormesh(longitudes, latitudes, data, transform=HigherResPlateCarree(), cmap='jet')
    plt.colorbar(c_scheme, location='bottom', pad=0.05,
                 label=unit, ax=ax)
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, color='black')
    if range_values_to_display is not None:
        min_value_to_display, max_value_to_display = range_values_to_display
        ax.clim(min_value_to_display, max_value_to_display)
    ax.set_title(title)
    return ax


def plot_barbs_from_array(u: np.ndarray, v: np.ndarray, longitudes: np.ndarray, latitudes: np.ndarray,
                          ax=None,
                          title=''):
    if ax is None:
        proj = HigherResPlateCarree()
        ax = plt.axes(projection=proj)
    ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()])
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, color='goldenrod')
    ax.add_feature(cartopy.feature.LAKES.with_scale('10m'), color='royalblue')
    ax.add_feature(cartopy.feature.OCEAN, color='skyblue')
    ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
    ax.add_feature(cartopy.feature.RIVERS.with_scale('10m'), color='grey')
    ax.barbs(longitudes, latitudes, u, v,
             pivot='middle', barbcolor='navy', fill_empty=True)
    ax.set_title(title)
    return ax


def plot_wind_components_from_array(u: np.ndarray, v: np.ndarray, longitudes: np.ndarray, latitudes: np.ndarray,
                                    range_values_to_display=None,
                                    ax=None,
                                    title=''):
    if ax is None:
        proj = HigherResPlateCarree()
        ax = plt.axes(projection=proj)
    ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()])
    speed = np.sqrt(u * u + v * v)
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, color='goldenrod')
    ax.add_feature(cartopy.feature.LAKES.with_scale('10m'), color='royalblue')
    ax.add_feature(cartopy.feature.OCEAN, color='skyblue')
    ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
    ax.add_feature(cartopy.feature.RIVERS.with_scale('10m'), color='grey')
    ax.quiver(longitudes, latitudes,
              u, v, speed,
              cmap=plt.cm.autumn, transform=ccrs.PlateCarree())
    if range_values_to_display is not None:
        min_value_to_display, max_value_to_display = range_values_to_display
        ax.clim(min_value_to_display, max_value_to_display)
    ax.set_title(title)
    return ax


def plot_colormap_from_dataset(data: Dataset, time_index: int, variable_str: str, range_lon=None, range_lat=None,
                               unit_correction=1.,
                               range_values_to_display=None,
                               ax=None):
    if ax is None:
        proj = HigherResPlateCarree()
        ax = plt.axes(projection=proj)
    interest_time = dataset_times_to_datetime(data, data.variables['time'][time_index])
    values, longs_place, lats_place, datetimes = get_array_from_dataset(data, variable_str, range_lon, range_lat,
                                                                        time_frame=(interest_time, interest_time),
                                                                        unit_correction=unit_correction)
    unit = f'{data.variables[f"{variable_str}"].units.replace("**", "^")}'
    title = f'{data.variables[f"{variable_str}"].long_name} \n {interest_time}'
    ax = plot_colormap_from_array(values, longs_place, lats_place, range_values_to_display,
                                  ax=ax, title=title, unit=unit)
    return ax


def plot_wind_components_from_dataset(data: Dataset, time_index: int, variable1_str: str, variable2_str: str,
                                      range_lon=None, range_lat=None,
                                      unit_correction=1.,
                                      range_values_to_display=None,
                                      ax=None):
    if ax is None:
        proj = HigherResPlateCarree()
        ax = plt.axes(projection=proj)
    interest_time = dataset_times_to_datetime(data, data.variables['time'][time_index])
    v1, longs_place, lats_place, datetimes = get_array_from_dataset(data, variable1_str, range_lon, range_lat,
                                                                    time_frame=(interest_time, interest_time),
                                                                    unit_correction=unit_correction)
    v2, _, _, _ = get_array_from_dataset(data, variable2_str, range_lon, range_lat,
                                         time_frame=(interest_time, interest_time),
                                         unit_correction=unit_correction)
    title = f'{data.variables[f"{variable1_str}"].long_name}\n {data.variables[f"{variable2_str}"].long_name} \n {interest_time}'
    ax = plot_wind_components_from_array(v1, v2, longs_place, lats_place, range_values_to_display, ax=ax,
                                         title=title)
    return ax


def plot_wind_components_from_different_datasets(d1: Dataset, d2: Dataset, time_index: int, variable1_str: str,
                                                 variable2_str: str,
                                                 range_lon=None, range_lat=None,
                                                 unit_correction=1.,
                                                 range_values_to_display=None,
                                                 ax=None):
    if ax is None:
        proj = HigherResPlateCarree()
        ax = plt.axes(projection=proj)
    interest_time = dataset_times_to_datetime(d1, d1.variables['time'][time_index])
    v1, longs_place, lats_place, datetimes = get_array_from_dataset(d1, variable1_str, range_lon, range_lat,
                                                                    time_frame=(interest_time, interest_time),
                                                                    unit_correction=unit_correction)
    v2, _, _, _ = get_array_from_dataset(d2, variable2_str, range_lon, range_lat,
                                         time_frame=(interest_time, interest_time),
                                         unit_correction=unit_correction)
    title = f'{d1.variables[f"{variable1_str}"].long_name}\n {d2.variables[f"{variable2_str}"].long_name} \n {interest_time}'
    ax = plot_wind_components_from_array(v1, v2, longs_place, lats_place, range_values_to_display, ax=ax,
                                         title=title)
    return ax


def plot_ERA5_wind_fields_timelapse(start_date, end_date, data_path, plot_path, range_lon=None, range_lat=None):
    from PIL import Image
    files = glob(f'{data_path}/*surface*.nc')
    dates = sorted([pd.to_datetime(file.split('/')[-1].split('.')[0].split('_')[0]) for file in files])
    dates = [d for d in dates if d >= start_date and d <= end_date]
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        file = f'{data_path}/{date_str}_era5_surface_hourly.nc'
        data = Dataset(file, mode='r')  # read the data
        subplot_kw = {'projection': HigherResPlateCarree()}
        fig, ax = plt.subplots(ncols=1, subplot_kw=subplot_kw, figsize=(25, 12.5))
        fig.subplots_adjust(wspace=0.1, left=0.05, right=0.95)
        plot_wind_components_from_dataset(data, 0, 'u10', 'v10',
                                          range_lon=range_lon, range_lat=range_lat,
                                          ax=ax)
        plt.savefig(f'{plot_path}/{date}.png')
    images = []
    for date in dates:
        images.append(
            Image.open(f'{plot_path}/{date}.png'))
    images[0].save(f'{plot_path}/era5_timelapse.gif',
                   save_all=True, append_images=images[1:], duration = 300, loop = 0)
