import os
import pathlib
import re
from io import StringIO
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


class PointPredictionProcessing(object):
    def __init__(self, path_to_input_surface: str,
                 path_to_input_z500: str,
                 path_to_static_input: str,
                 output: pd.DataFrame,
                 patch_size: int,
                 variables_to_predict=('u10', 'v10'),
                 static_predictors=('tpi_500', 'ridge_index_norm', 'ridge_index_dir',
                                    'we_derivative', 'sn_derivative',
                                    'slope', 'aspect'),
                 predictors_surface=('blh', 'fsr', 'sp', 'sshf',
                                     'd2m', 't2m',
                                     'u100', 'v100', 'u10', 'v10'),
                 predictors_z500=('d', 'z', 'u', 'v', 'w', 'vo'),
                 start_period=pd.to_datetime('2016-01-10'),
                 end_period=pd.to_datetime('2020-12-31'),
                 num_days=None):
        """

        :param path_to_input: input files, likely images in low resolution (.nc individual files). the path can
        ether be the exact path to directory where the inputs are stored, or a pattern to individual files.
        :param path_to_output: output, likely csv containing values for specific stations
        :param patch_size: for point prediction, we need to build a patch around the point we aim at predicting.
                the parameter patches_size sets the size of the neighborhood (in pixel) around points that will be the low-res reference.
        :param num_days: number days to process, None for all
        """
        self.input_surface = xr.open_mfdataset(path_to_input_surface)
        self.input_z500 = xr.open_mfdataset(path_to_input_z500)
        self.static_input = xr.open_mfdataset(path_to_static_input)
        self.output = output
        self.output.date = pd.to_datetime(self.output.date)
        self.output = self.output[(self.output.date >= start_period) & (self.output.date < end_period)]
        self.start_date = start_period
        self.end_date = end_period
        self.patch_size = patch_size
        self.predictors_surface = predictors_surface
        self.predictors_z500 = predictors_z500
        self.static_predictors = static_predictors
        self.variables_to_predict = variables_to_predict
        self.num_days = num_days

    def create_patch_for_point(self, date, lon_point, lat_point):
        date = date.strftime("%Y-%m-%d")

        def inputs_rectangle_around_point(lon_point=lon_point, lat_point=lat_point):
            inputs_to_concatenate = []
            for df, predictors in zip([self.input_surface, self.input_z500],
                                      [self.predictors_surface, self.predictors_z500]):
                distances_lon, distances_lat = np.abs(np.array(df.longitude.data) - lon_point), np.abs(
                    np.array(df.latitude.data) - lat_point)
                nearest_lon, nearest_lat = np.argmin(distances_lon), \
                                           np.argmin(distances_lat)
                lon1, lon2 = nearest_lon - self.patch_size // 2, nearest_lon + self.patch_size // 2
                lat1, lat2 = nearest_lat - self.patch_size // 2, nearest_lat + self.patch_size // 2
                inputs = df.sel(time=date).isel(longitude=slice(lon1, lon2 + 1), latitude=slice(lat1, lat2 + 1))
                inputs = inputs.drop_vars([v for v in inputs.variables if v not in predictors])
                new_dim = (inputs.dims['time'], inputs.dims['longitude'], inputs.dims['latitude'], len(inputs))
                inputs = inputs.to_dataframe()[list(predictors)].to_numpy().reshape(new_dim)
                inputs_to_concatenate.append(inputs)
            return np.concatenate(inputs_to_concatenate, axis=3)

        inputs = inputs_rectangle_around_point()

        def static_inputs_rectangle_around_point(time_steps=inputs.shape[0], lon_point=lon_point, lat_point=lat_point):
            distances_lon, distances_lat = np.abs(np.array(self.static_input.x.data) - lon_point), np.abs(
                np.array(self.static_input.y.data) - lat_point)
            nearest_lon, nearest_lat = np.argmin(distances_lon), \
                                       np.argmin(distances_lat)
            lon1, lon2 = nearest_lon - self.patch_size // 2, nearest_lon + self.patch_size // 2
            lat1, lat2 = nearest_lat - self.patch_size // 2, nearest_lat + self.patch_size // 2
            static_inputs = self.static_input.isel(x=slice(lon1, lon2 + 1),
                                                   y=slice(lat1, lat2 + 1))
            static_inputs = static_inputs.drop_vars(
                [v for v in static_inputs.variables if v not in self.static_predictors])
            new_dim_static = (static_inputs.dims['x'], static_inputs.dims['y'], len(static_inputs))
            static_inputs = static_inputs.to_dataframe()[list(self.static_predictors)].to_numpy().reshape(
                new_dim_static)
            static_inputs = np.repeat(static_inputs[np.newaxis, ...], time_steps, axis=0)
            return static_inputs

        static_inputs = static_inputs_rectangle_around_point()

        dims_i1 = inputs.shape[:-1]
        dims_i2 = static_inputs.shape[:-1]
        if dims_i1 == dims_i2:
            return np.concatenate([inputs, static_inputs], axis=3)
        else:
            return None

    def get_input_and_output_arrays(self):
        inputs = []
        outputs = []
        lon_column = [c for c in self.output.columns if c.startswith('lon')][0]
        lat_column = [c for c in self.output.columns if c.startswith('lat')][0]
        station_column = [c for c in self.output.columns if c.startswith('station')][0]
        if 'date' not in self.output.columns:
            raise ValueError('date column not found in output, cannot synchronize data.')
        input_dates = np.intersect1d(pd.to_datetime(self.input_surface.time.data).normalize().unique().sort_values(),
                                     pd.to_datetime(self.input_z500.time.data).normalize().unique().sort_values())
        if self.num_days is not None:
            input_dates = input_dates[:self.num_days]
        # one file per day
        for date in pd.DatetimeIndex(self.output.date).unique().sort_values():
            # checks if data in input
            if date in input_dates:
                print(f"Processing {date}")
                num_stations = 0
                output_for_date = self.output[self.output.date == date]
                output_for_date['datetime'] = pd.to_datetime(output_for_date['datetime'])
                output_for_date = output_for_date.sort_values('datetime')
                complete_stations = output_for_date.groupby(station_column)['datetime'].nunique() == 24
                output_for_date = output_for_date[
                    output_for_date[station_column].isin(complete_stations[complete_stations].index)]
                for s in sorted(output_for_date[station_column].unique()):
                    output = output_for_date[output_for_date[station_column] == s]
                    longitude = float(output[lon_column].unique())
                    latitude = float(output[lat_column].unique())
                    new_output = np.asarray(output[list(self.variables_to_predict)]).reshape(len(output), 1, 1, len(
                        self.variables_to_predict))
                    hours, lons, lats, _ = new_output.shape
                    if hours != 24:
                        continue
                    new_input = self.create_patch_for_point(date, longitude, latitude)
                    if new_input is None:
                        continue
                    _, lons, lats, _ = new_input.shape
                    if lons != self.patch_size or lats != self.patch_size:
                        continue
                    inputs.append(new_input)
                    outputs.append(new_output)
                    num_stations += 1
                print(f"Processed {num_stations} stations")
            else:
                print(f'Date {date} has no complete match in inputs, cannot synchronize data.')
        return np.asarray(inputs), np.asarray(outputs)

    def get_input_output_dataframe_from_array(self, path_to_arrays, aggregation_method='nearest'):
        obs = self.output
        obs['date'] = pd.to_datetime(obs['date'])
        start_train_period = pd.to_datetime('2016-01-10')
        end_train_period = pd.to_datetime('2016-06-10')
        data = []
        for d in pd.date_range(start_train_period, end_train_period):
            d_str = d.strftime('%Y%m%d')
            x = f'{path_to_arrays}/x_train_{d_str}.npy'
            y = f'{path_to_arrays}/y_train_{d_str}.npy'
            if aggregation_method == 'nearest':
                xarr = np.load(x)[:, :, 1, 1, ...]
            elif aggregation_method == 'mean':
                xarr = np.mean(np.load(x), axis=(2, 3))
            else:
                print('Defaulting aggregation to nearest neighbor...')
                xarr = np.load(x)[:, :, 1, 1, ...]
            m, n, r = xarr.shape
            xarr = np.column_stack((np.repeat(np.arange(m), n), xarr.reshape(m * n, -1)))
            yarr = np.load(y)[:, :, 0, 0, ...]
            m, n, r = yarr.shape
            yarr = np.column_stack((np.repeat(np.arange(m), n), yarr.reshape(m * n, -1)))
            interest_var = tuple(f'{v}_hr' for v in self.variables_to_predict)
            out_df = pd.DataFrame(np.concatenate([xarr, yarr[:, 1:]], axis=-1), columns=list(tuple('station') +
                                                                                             self.predictors_surface +
                                                                                             self.predictors_z500 +
                                                                                             self.static_predictors +
                                                                                             interest_var))
            out_df['hour'] = np.repeat(np.arange(n), m)
            out_df['date'] = d
            out_df = out_df.assign(datetime=lambda x: x['date'] + pd.to_timedelta(x['hour'], unit='H'))
            output_for_date = obs[obs['date'] == d]
            station_column = 'station'
            complete_stations = output_for_date.groupby(station_column)['datetime'].nunique() == 24
            output_for_date = output_for_date[
                output_for_date[station_column].isin(complete_stations[complete_stations].index)]
            out_df['station'] = output_for_date.reset_index()[station_column]
            data.append(out_df)
        data = pd.concat(data)
        return data


class PointPredictionProcessingCOSMO(object):
    def __init__(self, path_to_static_input: str,
                 path_to_cosmo_input: str,
                 output: pd.DataFrame,
                 patch_size: int = 8,
                 variables_to_predict=('u10', 'v10'),
                 static_predictors=('tpi_500', 'ridge_index_norm', 'ridge_index_dir',
                                    'we_derivative', 'sn_derivative',
                                    'slope', 'aspect'),
                 predictors_cosmo=('U_10M', 'V_10M'),
                 start_period=pd.to_datetime('2016-01-10'),
                 end_period=pd.to_datetime('2020-12-31'),
                 path_to_output=None):
        """

        :param path_to_input: input files, likely images in low resolution (.nc individual files). the path can
        ether be the exact path to directory where the inputs are stored, or a pattern to individual files.
        :param path_to_output: output, likely csv containing values for specific stations
        :param patch_size: for point prediction, we need to build a patch around the point we aim at predicting.
                the parameter patches_size sets the size of the neighborhood (in pixel) around points that will be the low-res reference.
        """
        self.input_cosmo = xr.open_mfdataset(path_to_cosmo_input)
        self.static_input = xr.open_mfdataset(path_to_static_input)
        self.output = output
        self.output.date = pd.to_datetime(self.output.date)
        self.output = self.output[(self.output.date >= start_period) & (self.output.date < end_period)]
        self.start_date = start_period
        self.end_date = end_period
        self.patch_size = patch_size
        self.predictors_cosmo = predictors_cosmo
        self.static_predictors = static_predictors
        self.variables_to_predict = variables_to_predict
        self.path_to_output = path_to_output or pathlib.Path(
            path_to_cosmo_input).parent.parent / 'point_prediction_files'

    def create_patch_for_point(self, date, lon_point, lat_point):
        date = date.strftime("%Y-%m-%d")

        def inputs_rectangle_around_point():
            distances = np.array([distance_from_coordinates((lon_point, lat_point), (u, v)) \
                                  for u, v in zip(self.input_cosmo.lon_1[:],
                                                  self.input_cosmo.lat_1[:])])
            y_ind, x_ind = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
            x1, x2 = x_ind - self.patch_size // 2, x_ind + self.patch_size // 2
            y1, y2 = y_ind - self.patch_size // 2, y_ind + self.patch_size // 2
            mean = self.input_cosmo.sel(time=date).isel(x_1=slice(x1, x2 + 1), y_1=slice(y1, y2 + 1)).mean(
                dim=['x_1', 'y_1'])
            nn = self.input_cosmo.sel(time=date).isel(x_1=x_ind, y_1=y_ind)
            mean = mean.drop_vars([v for v in mean.variables if v not in self.predictors_cosmo]).to_dataframe()[
                list(self.predictors_cosmo)].assign(date=pd.to_datetime(date)).reset_index() \
                .rename(columns={'time': 'hour', 'U_10M': 'U_10M_mean', 'V_10M': 'V_10M_mean'}).assign(
                datetime=lambda x: x['date'] + pd.to_timedelta(x['hour'], unit='H')).set_index(
                ['datetime', 'date', 'hour'])
            nn = nn.drop_vars([v for v in nn.variables if v not in self.predictors_cosmo]).to_dataframe()[
                list(self.predictors_cosmo)].assign(date=pd.to_datetime(date)).reset_index() \
                .rename(columns={'time': 'hour'}).assign(
                datetime=lambda x: x['date'] + pd.to_timedelta(x['hour'], unit='H')).set_index(
                ['datetime', 'date', 'hour'])
            return pd.concat([mean, nn], axis=1)

        inputs = inputs_rectangle_around_point()

        def static_inputs_rectangle_around_point(time_steps=inputs.shape[0], lon_point=lon_point, lat_point=lat_point):
            distances_lon, distances_lat = np.abs(np.array(self.static_input.x.data) - lon_point), np.abs(
                np.array(self.static_input.y.data) - lat_point)
            nearest_lon, nearest_lat = np.argmin(distances_lon), \
                                       np.argmin(distances_lat)
            lon1, lon2 = nearest_lon - self.patch_size // 2, nearest_lon + self.patch_size // 2
            lat1, lat2 = nearest_lat - self.patch_size // 2, nearest_lat + self.patch_size // 2
            static_inputs = self.static_input.isel(x=slice(lon1, lon2 + 1),
                                                   y=slice(lat1, lat2 + 1))
            static_inputs = static_inputs.drop_vars(
                [v for v in static_inputs.variables if v not in self.static_predictors])
            static_inputs = static_inputs.to_dataframe()[list(self.static_predictors)]
            static_inputs_nn = static_inputs.loc[self.patch_size // 2, self.patch_size // 2]
            static_inputs_nn = pd.concat(
                [static_inputs_nn.to_frame(name=t).T for t in range(time_steps)]).reset_index().rename(
                columns={'index': 'hour'}).assign(date=pd.to_datetime(date)) \
                .assign(datetime=lambda x: x['date'] + pd.to_timedelta(x['hour'], unit='H')).set_index(
                ['datetime', 'date', 'hour'])
            return static_inputs_nn

        static_inputs = static_inputs_rectangle_around_point()

        if len(static_inputs) == len(inputs):
            return pd.concat([inputs, static_inputs], axis=1)
        else:
            return None

    def get_input_and_output_arrays(self):
        lon_column = [c for c in self.output.columns if c.startswith('lon')][0]
        lat_column = [c for c in self.output.columns if c.startswith('lat')][0]
        station_column = [c for c in self.output.columns if c.startswith('station')][0]
        if 'date' not in self.output.columns:
            raise ValueError('date column not found in output, cannot synchronize data.')
        input_dates = np.array(pd.to_datetime(self.input_cosmo.time.data).normalize().unique().sort_values())
        # one file per day
        for date in pd.DatetimeIndex(self.output.date).unique().sort_values():
            data = []
            # checks if data in input
            if date in input_dates:
                print(f"Processing {date}")
                date_str = date.strftime('%Y%m%d')
                if not os.path.isfile(f'{self.path_to_output}/x_{date_str}.csv'):
                    num_stations = 0
                    output_for_date = self.output[self.output.date == date]
                    output_for_date['datetime'] = pd.to_datetime(output_for_date['datetime'])
                    output_for_date = output_for_date.sort_values('datetime')
                    complete_stations = output_for_date.groupby(station_column)['datetime'].nunique() == 24
                    output_for_date = output_for_date[
                        output_for_date[station_column].isin(complete_stations[complete_stations].index)]
                    for s in sorted(output_for_date[station_column].unique()):
                        output = output_for_date[output_for_date[station_column] == s]
                        longitude = float(output[lon_column].unique())
                        latitude = float(output[lat_column].unique())
                        new_output = output.assign(station_column=s).set_index(['datetime', station_column])[
                            list(self.variables_to_predict)].dropna(subset=['u10', 'v10'])
                        hours = len(new_output)
                        if hours != 24:
                            continue
                        new_input = self.create_patch_for_point(date, longitude, latitude)
                        if new_input is None:
                            continue
                        new_input[station_column] = s
                        new_input = new_input.reset_index().set_index(['datetime', station_column])
                        data_stat = pd.concat([new_input, new_output], axis=1)
                        data.append(data_stat)
                        num_stations += 1
                    print(f"Processed {num_stations} stations")
                    pd.concat(data, axis=0).to_csv(f'{self.path_to_output}/x_{date_str}.csv')
                else:
                    print(f'Date {date} has been processed already.')
            else:
                print(f'Date {date} has no complete match in inputs, cannot synchronize data.')

    def get_input_output_dataframe_from_array(self, path_to_arrays, aggregation_method='nearest'):
        obs = self.output
        obs['date'] = pd.to_datetime(obs['date'])
        start_train_period = self.start_date
        end_train_period = self.end_date
        data = []
        for d in pd.date_range(start_train_period, end_train_period):
            d_str = d.strftime('%Y%m%d')
            x = f'{path_to_arrays}/x_{d_str}.npy'
            y = f'{path_to_arrays}/y_{d_str}.npy'
            if aggregation_method == 'nearest':
                xarr = np.load(x)[:, :, 1, 1, ...]
            elif aggregation_method == 'mean':
                xarr = np.mean(np.load(x), axis=(2, 3))
            else:
                print('Defaulting aggregation to nearest neighbor...')
                xarr = np.load(x)[:, :, 1, 1, ...]
            m, n, r = xarr.shape
            xarr = np.column_stack((np.repeat(np.arange(m), n), xarr.reshape(m * n, -1)))
            yarr = np.load(y)[:, :, 0, 0, ...]
            m, n, r = yarr.shape
            yarr = np.column_stack((np.repeat(np.arange(m), n), yarr.reshape(m * n, -1)))
            interest_var = tuple(f'{v}_hr' for v in self.variables_to_predict)
            out_df = pd.DataFrame(np.concatenate([xarr, yarr[:, 1:]], axis=-1), columns=list(tuple('station') +
                                                                                             self.static_predictors +
                                                                                             self.predictors_cosmo +
                                                                                             interest_var))
            out_df['hour'] = np.repeat(np.arange(n), m)
            out_df['date'] = d
            out_df = out_df.assign(datetime=lambda x: x['date'] + pd.to_timedelta(x['hour'], unit='H'))
            output_for_date = obs[obs['date'] == d]
            station_column = 'station'
            complete_stations = output_for_date.groupby(station_column)['datetime'].nunique() == 24
            output_for_date = output_for_date[
                output_for_date[station_column].isin(complete_stations[complete_stations].index)]
            out_df['station'] = output_for_date.reset_index()[station_column]
            data.append(out_df)
        data = pd.concat(data)
        return data


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


def process_topographic_variables_file(path_to_file: str):
    path_to_file = pathlib.Path(path_to_file)
    names = ('tpi_500', 'ridge_index_norm', 'ridge_index_dir',
             'we_derivative', 'sn_derivative',
             'slope', 'aspect')
    if all((path_to_file.parent / f"topo_{name}.nc").exists() for name in names):
        print("Already processed all topo files")
        return
    dem = xr.open_rasterio(path_to_file)
    dem = dem.isel(band=0, drop=True)
    # TPI 500m
    scale_meters = 500
    scale_pixel, res_meters = helpers.scale_to_pixel(scale_meters, dem)
    tpi = topo.tpi(dem, scale_pixel)
    # Gradient
    gradient = topo.gradient(dem, 1 / 4 * scale_meters, res_meters)
    # Norm and second the direction of Ridge index
    vr = topo.valley_ridge(dem, scale_pixel, mode='ridge')

    # Sx
    def Sx(azimuth=0, radius=500):
        pass

    variables = (tpi, *vr, *gradient)
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
                 topo_variables_included=('tpi_500', 'ridge_index_norm'),
                 cosmo_variables_included=('U_10M', 'V_10M')):
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
            temp_inputs_topo = inputs_topo \
                .sel(x=cosmo.lon_1, y=cosmo.lat_1, method='nearest').drop_vars(topo_vars_to_drop)
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
            static_inputs = temp_inputs_topo.expand_dims({'time': inputs_surface.time})

            print(f'Merging input data into a dataset...')
            full_data = xr.merge([inputs_surface, inputs_z500, static_inputs])
            e_plus, e_minus = compute_time_varying_topo_pred(full_data.u10, full_data.v10,
                                                             full_data.slope, full_data.aspect)
            w_speed, w_angle = compute_wind_speed_and_angle(full_data.u10, full_data.v10)
            full_data = full_data.assign({'e_plus': e_plus, 'e_minus': e_minus, 'w_speed': w_speed, 'w_angle': w_angle})
            full_data.to_netcdf(x_path)
            if not os.path.isfile(y_path):
                outputs.to_netcdf(y_path)


def process_imgs_cosmo_exclusive(path_to_processed_files: str, COSMO1_data_path: str, DEM_data_path: str,
                                 start_date, end_date,
                                 topo_variables_included=('tpi_500', 'ridge_index_norm'),
                                 cosmo_variables_included=('U_10M', 'V_10M'), blurring=4):
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
            cosmo = xr.open_mfdataset(pathlib.Path(COSMO1_data_path).glob(f'{d_str}*.nc')).sel(time=d_str)
            cosmo_vars_to_drop = [v for v in cosmo.variables if
                                  v not in cosmo_variables_included + tuple(cosmo._coord_names)]
            cosmo_outputs = cosmo.drop_vars(cosmo_vars_to_drop)
            cosmo_inputs = cosmo.drop_vars(cosmo_vars_to_drop)
            print('Blurring COSMO1 image to obtain inputs')
            cosmo_inputs = cosmo_inputs.map(gaussian_filter, sigma=blurring)
            print('Adjusting resolution of topographic descriptors to COSMO1 image size')
            temp_inputs_topo = inputs_topo \
                .sel(x=cosmo.lon_1, y=cosmo.lat_1, method='nearest').drop_vars(topo_vars_to_drop)
            time_steps = cosmo.time.shape
            print(
                f'Replicating {time_steps[0]} times the static topographic inputs to respect the time series data format for inputs')
            static_inputs = temp_inputs_topo.expand_dims({'time': cosmo.time})

            print(f'Merging input data into a dataset...')
            full_data = xr.merge([cosmo_inputs, static_inputs])
            e_plus, e_minus = compute_time_varying_topo_pred(full_data.U_10M, full_data.V_10M,
                                                             full_data.slope, full_data.aspect)
            w_speed, w_angle = compute_wind_speed_and_angle(full_data.U_10M, full_data.V_10M)
            full_data = full_data.assign({'e_plus': e_plus, 'e_minus': e_minus, 'w_speed': w_speed, 'w_angle': w_angle})
            full_data.to_netcdf(x_path)
            if not os.path.isfile(y_path):
                cosmo_outputs.to_netcdf(y_path)
