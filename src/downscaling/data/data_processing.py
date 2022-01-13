import os
import pathlib
from typing import Tuple

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from topo_descriptors import topo, helpers


class HigherResPlateCarree(ccrs.PlateCarree):
    @property
    def threshold(self):
        return super().threshold / 100


def distance_from_coordinates(z1: Tuple, z2: Tuple):
    """

    :param z1: tuple of lon, lat for the first place
    :param z2: tuple of lon, lat for the second place
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


def process_topographic_variables_file(path_to_file: str):
    path_to_file = pathlib.Path(path_to_file)
    names = ('elevation', 'tpi_500',
             'we_derivative', 'sn_derivative',
             'slope', 'aspect')
    if all((path_to_file.parent / f"topo_{name}.nc").exists() for name in names):
        print("Already processed all topo files")
        return
    dem = xr.open_rasterio(path_to_file)
    dem = dem.isel(band=0, drop=True)
    ind_nans, dem = helpers.fill_na(dem)
    # TPI 500m
    scale_meters = 500
    scale_pixel, res_meters = helpers.scale_to_pixel(scale_meters, dem)
    tpi = topo.tpi(dem, scale_pixel)
    # Gradient
    gradient = topo.gradient(dem, 1 / 4 * scale_pixel, res_meters)
    # Norm and second the direction of Ridge index
    variables = (dem, tpi, *gradient)
    for data, name in zip(variables, names):
        da = xr.DataArray(data,
                          coords=dem.coords,
                          name=name)
        filename = f"topo_{name}.nc"
        da.to_dataset().to_netcdf(pathlib.Path(path_to_file.parent, filename))


def compute_time_varying_topo_pred(u, v, slope, aspect):
    delta = np.arctan2(-v, -u) - aspect
    alpha = np.arctan(np.tan(slope) * np.cos(delta))
    e_plus = xr.where(np.sin(alpha) > 0, np.sin(alpha), 0)
    e_minus = xr.where(np.sin(alpha) < 0, np.sin(alpha), 0)
    return e_plus, e_minus


def compute_wind_speed_and_angle(u, v):
    w_speed = np.sqrt(u ** 2 + v ** 2)
    w_angle = np.arctan2(v, u)
    return w_speed, w_angle


def process_imgs(path_to_processed_files: str, ERA5_data_path: str, COSMO1_data_path: str, DEM_data_path: str,
                 start_date, end_date,
                 surface_variables_included=('u10', 'v10', 'blh', 'fsr', 'sp'),
                 z500_variables_included=('z', 'vo', 'd'),
                 topo_variables_included=('elevation', 'tpi_500', 'ridge_index_norm', 'ridge_index_dir',
                                          'we_derivative', 'sn_derivative',
                                          'slope', 'aspect'),
                 cosmo_variables_included=('U_10M', 'V_10M'),
                 homemade_variables_included=('e_plus', 'e_minus', 'w_speed', 'w_angle')):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    print(f'Reading DEM data files')
    inputs_topo = xr.open_mfdataset(pathlib.Path(DEM_data_path).glob('topo*.nc'))
    topo_vars_to_drop = [v for v in inputs_topo.variables if
                         v not in topo_variables_included + tuple(inputs_topo._coord_names)]
    for d in pd.date_range(start_date, end_date):
        # Downloading saved inputs
        d_str = d.strftime('%Y%m%d')
        x_path = pathlib.Path(path_to_processed_files, f'x_{d_str}').with_suffix('.nc')
        y_path = pathlib.Path(path_to_processed_files, f'y_{d_str}').with_suffix('.nc')
        all_inputs_there = False
        if os.path.isfile(x_path):
            inputs = xr.open_mfdataset(str(x_path))
            variables = [v for v in inputs.variables if v not in ['lat_1', 'lon_1', 'time', 'x_1', 'y_1',
                                                                  'longitude', 'latitude',
                                                                  'x', 'y']]
            all_inputs = set(z500_variables_included + surface_variables_included + topo_variables_included)
            if set(np.intersect1d(list(all_inputs), variables)) == all_inputs:
                all_inputs_there = True
                print(f'Inputs and outputs for date {d_str} have already been processed.')
        if not all_inputs_there:
            print(f'Reading data files for day {d}')
            print(f'Reading COSMO1 data files')
            cosmo = xr.open_mfdataset(pathlib.Path(COSMO1_data_path).glob(f'*{d_str}*.nc')).sel(time=d_str)
            cosmo_vars_to_drop = [v for v in cosmo.variables if
                                  v not in cosmo_variables_included + tuple(cosmo._coord_names)]
            outputs = cosmo.drop_vars(cosmo_vars_to_drop)
            print('Adjusting resolution to COSMO1 image size')
            print(f'Reading and Interpolating linearly ERA5 data to fit the topographic data resolution')
            inputs_surface = xr.open_mfdataset(pathlib.Path(ERA5_data_path).glob(f'{d_str}*surface*.nc')).sel(
                time=d_str).sel(longitude=cosmo.lon_1, latitude=cosmo.lat_1, method='nearest')
            surface_vars_to_drop = [v for v in inputs_surface.variables if
                                    v not in surface_variables_included + tuple(inputs_surface._coord_names)]
            inputs_surface = inputs_surface.drop_vars(surface_vars_to_drop)
            inputs_z500 = xr.open_mfdataset(pathlib.Path(ERA5_data_path).glob(f'{d_str}*z500*.nc')).sel(
                time=d_str).sel(longitude=cosmo.lon_1, latitude=cosmo.lat_1, method='nearest')
            z500_vars_to_drop = [v for v in inputs_z500.variables if
                                 v not in z500_variables_included + tuple(inputs_z500._coord_names)]
            inputs_z500 = inputs_z500.drop_vars(z500_vars_to_drop)
            # Replicate static inputs for time series concordance
            time_steps = inputs_surface.time.shape
            print(
                f'Replicating {time_steps[0]} times the static topographic inputs to respect the time series data format for inputs')
            temp_inputs_topo = inputs_topo \
                .sel(x=cosmo.lon_1, y=cosmo.lat_1, method='nearest')
            static_inputs = temp_inputs_topo.expand_dims({'time': inputs_surface.time})
            print(f'Merging input data into a dataset...')
            full_data = xr.merge([inputs_surface, inputs_z500, static_inputs])
            if 'e_plus' in homemade_variables_included:
                e_plus, e_minus = compute_time_varying_topo_pred(full_data.u10, full_data.v10,
                                                                 full_data.slope, full_data.aspect)
                full_data = full_data.assign({'e_plus': e_plus, 'e_minus': e_minus})
            if 'w_speed' in homemade_variables_included:
                w_speed, w_angle = compute_wind_speed_and_angle(full_data.u10, full_data.v10)
                full_data = full_data.assign({'w_speed': w_speed, 'w_angle': w_angle})
            full_data = full_data.drop_vars(topo_vars_to_drop)
            full_data.to_netcdf(x_path)
            if not os.path.isfile(y_path):
                outputs.to_netcdf(y_path)


def process_imgs_cosmoblurred(path_to_processed_files: str, COSMO1_data_path: str, DEM_data_path: str,
                              start_date, end_date,
                              topo_variables_included=('elevation', 'tpi_500', 'ridge_index_norm', 'ridge_index_dir',
                                                       'we_derivative', 'sn_derivative',
                                                       'slope', 'aspect'),
                              cosmo_variables_included=('U_10M', 'V_10M'),
                              homemade_variables_included=('e_plus', 'e_minus', 'w_speed', 'w_angle'),
                              blurring=7):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    print(f'Reading DEM data files')
    inputs_topo = xr.open_mfdataset(pathlib.Path(DEM_data_path).glob('topo*.nc'))
    topo_vars_to_drop = [v for v in inputs_topo.variables if
                         v not in topo_variables_included + tuple(inputs_topo._coord_names)]
    for d in pd.date_range(start_date, end_date):
        # Downloading saved inputs
        d_str = d.strftime('%Y%m%d')
        x_path = pathlib.Path(path_to_processed_files, f'x_cosmo_{d_str}').with_suffix('.nc')
        y_path = pathlib.Path(path_to_processed_files, f'y_{d_str}').with_suffix('.nc')
        all_inputs_there = False
        if os.path.isfile(x_path):
            inputs = xr.open_mfdataset(str(x_path))
            variables = [v for v in inputs.variables if v not in ['lat_1', 'lon_1', 'time', 'x_1', 'y_1',
                                                                  'x', 'y']]
            all_inputs = set(cosmo_variables_included + topo_variables_included)
            if set(np.intersect1d(list(all_inputs), variables)) == all_inputs:
                all_inputs_there = True
                print(f'Inputs and outputs for date {d_str} have already been processed.')
        if not all_inputs_there:
            print(f'Reading data files for day {d}')
            print(f'Reading COSMO1 data files')
            cosmo = xr.open_mfdataset(pathlib.Path(COSMO1_data_path).glob(f'*{d_str}*.nc')).sel(time=d_str)
            cosmo_vars_to_drop = [v for v in cosmo.variables if
                                  v not in cosmo_variables_included + tuple(cosmo._coord_names)]
            cosmo_outputs = cosmo.drop_vars(cosmo_vars_to_drop)
            cosmo_inputs = cosmo.drop_vars(cosmo_vars_to_drop)
            print('Blurring COSMO1 image to obtain inputs')
            cosmo_inputs = cosmo_inputs.map(gaussian_filter, sigma=blurring)
            print('Adjusting resolution of topographic descriptors to COSMO1 image size')
            temp_inputs_topo = inputs_topo \
                .sel(x=cosmo.lon_1, y=cosmo.lat_1, method='nearest')
            time_steps = cosmo.time.shape
            print(
                f'Replicating {time_steps[0]} times the static topographic inputs to respect the time series data format for inputs')
            static_inputs = temp_inputs_topo.expand_dims({'time': cosmo.time})

            print(f'Merging input data into a dataset...')
            full_data = xr.merge([cosmo_inputs, static_inputs])
            if 'e_plus' in homemade_variables_included:
                e_plus, e_minus = compute_time_varying_topo_pred(full_data.U_10M, full_data.V_10M,
                                                                 full_data.slope, full_data.aspect)
                full_data = full_data.assign({'e_plus': e_plus, 'e_minus': e_minus})
            if 'w_speed' in homemade_variables_included:
                w_speed, w_angle = compute_wind_speed_and_angle(full_data.U_10M, full_data.V_10M)
                full_data = full_data.assign({'w_speed': w_speed, 'w_angle': w_angle})
            full_data = full_data.drop_vars(topo_vars_to_drop)
            full_data.to_netcdf(x_path)
            if not os.path.isfile(y_path):
                cosmo_outputs.to_netcdf(y_path)
